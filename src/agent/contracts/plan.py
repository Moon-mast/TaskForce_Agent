"""计划契约:Plan-and-Execute 的 todolist 形状(plan_0.1 冻结)。

本文件作用:
    定义两套形状——PlanDraft(LLM 结构化输出,只有"计划本身")与 Plan(主图运行时状态,
    带 status/task_id/result_digest 等代码维护的字段),外加校验与状态推进的纯函数方法。
    推进逻辑放模型上而非节点里:supervisor 只做编排,不被调度细节撑爆。

使用位置:
    - agent/planner.py:plan_draft 生成 PlanDraft 并 to_plan();plan_confirm 读 Plan 派发首批;
    - agent/supervisor.py:_advance_plan 按 task_id 对账、级联跳过、收口判定;
    - agent/state.py:plan 通道存 Plan.model_dump();
    - tests/test_plan_contract.py / test_plan_progress.py。
"""
import re
from typing import Literal

from pydantic import BaseModel, Field

from agent.contracts.summary import ResultSummary

# 步骤 id 只允许 s<数字>:依赖引用是字符串,形状固定才谈得上"编号比较"式的防环
_STEP_ID_RE = re.compile(r"^s(\d+)$")

# 进度快照符号(REPL/API 直接展示,见 03-调度推进与展示 §5)
_MARKS = {
    "pending": "☐",
    "running": "▶",
    "done": "☑",
    "failed": "✗",
    "skipped": "⊘"
}

StepStatus = Literal["pending", "running", "done", "failed", "skipped"]
Assignee = Literal["retriever", "research", "executor"]


class PlanStepDraft(BaseModel):
    """单步计划(LLM 输出用):只有"计划本身",不含执行状态。"""

    id: str = Field(description="步骤编号,s1 起(依赖引用的锚点)")
    title: str = Field(description="todolist 上的一句话标题")
    detail: str = Field(description="自包含任务描述:目标+输入+输出要求+成功标准")
    assignee: Assignee = Field(description="派给哪个子智能体")
    depends_on: list[str] = Field(
        default_factory=list,
        description="依赖的步骤 id,只能引用编号更小的步"
    )

class PlanStep(BaseModel):
    """运行时步骤:前半是计划内容,后半是代码维护的执行状态。"""

    id: str
    title: str
    detail: str
    assignee: Assignee
    depends_on: list[str] = Field(default_factory=list)

    # ---- 运行时字段:LLM 不填,由 planner/推进块维护 ----
    status: StepStatus = "pending"
    task_id: str = Field(default="", description="派发后回填,与 ResultSummary.task_id 对账")
    result_digest: str = Field(default="", description="完成后的结论摘要(收口与展示用)")

class Plan(BaseModel):
    """运行时计划:AgenState.plan 的模型侧;推进方法都在这,节点保持薄。"""

    goal: str
    steps: list[PlanStep]

    def ready_steps(self) -> list[PlanStep]:
        """可派发的步:pending 且依赖全部 done(保持原顺序,供分批截断)。"""
        done = {s.id for s in self.steps if s.status == "done"}
        return [
            s
            for s in self.steps
            if s.status == "pending" and all(d in done for d in s.depends_on)
        ]

    def mark_by_task(self, task_id: str, summary: ResultSummary) -> bool:
        """按 task_id 对账回收结果:failed 记 failed(硬失败,级联跳过下游),
        其余(success/partial/need_clarification)记 done——"缺澄清"不是失败,
        结论摘要在 result_digest 里,补救步(s2 联网补充)据此继续。

        返回是否命中——对账靠 task_id(派发时回填),不解析结果内容。
        """
        for s in self.steps:
            if s.task_id and s.task_id == task_id:
                s.status = "failed" if summary.status == "failed" else "done"
                s.result_digest = summary.conclusion
                return True
        return False

    def cascade_skip(self) -> None:
        """失败级联:依赖 failed/skipped 的 pending 步标 skipped,迭代到不动点。"""
        changed = True
        while changed:
            changed = False
            blocked = {s.id for s in self.steps if s.status in ("failed", "skipped")}
            for s in self.steps:
                if s.status == "pending" and any(d in blocked for d in s.depends_on):
                    s.status = "skipped"
                    changed = True

    def finished(self) -> bool:
        """全终态判定(done/failed/skipped 皆为终态:无 pending 亦无 running)。"""
        return all(s.status in ("done", "failed", "skipped") for s in self.steps)

    def snapshot(self) -> str:
        """进度快照(Markdown 兼容:列表项才是硬换行,标题独立段)。

        渲染两用:消息流经 rich Markdown(段落内单换行会被折叠成空格,
        2026-09-25 实测快照挤成一行);interrupt 的 Panel 走纯文本,列表前缀无害。
        digest 内的换行替换为空格,防列表项断裂。"""
        lines = [f"📋 计划:{self.goal}", ""]
        for s in self.steps:
            digest = (" —— " + s.result_digest.replace("\n", " ")) if s.result_digest else ""
            lines.append(f"- {_MARKS[s.status]} {s.id} {s.title}({s.assignee}){digest}")
        return "\n".join(lines)

class PlanDraft(BaseModel):
    """计划草案:planner 的 with_structured_output 目标类型。

    max_length=5 与 MAX_PARALLEL_SUBAGENTS=3、recursion_limit 预算对齐(01 §1.2)。
    """

    goal: str = Field(description="用户目标回显(意图锚点,防重构失真)")
    steps: list[PlanStepDraft] = Field(min_length=1, max_length=5)

    def validate_structure(self) -> None:
        """结构校验,失败抛 ValueError(调用方兜底 goto answer,与 route_node 同风格)。

        "只向前引用"(依赖编号严格小于自身)是结构性防环:深链只能在 1..n 内,天然无环,
        不需要图算法,也顺带把"自我依赖"拦下(等于自身满足 >= 条件)。
        """
        nums: dict[str, int] = {}
        for s in self.steps:
            m = _STEP_ID_RE.match(s.id)
            if not m:
                raise ValueError(f"步骤 id 必须形如 s1/s2,收到:{s.id!r}")
            if s.id in nums:
                raise ValueError(f"步骤 id 重复:{s.id}")
            nums[s.id] = int(m.group(1))
        for s in self.steps:
            for dep in s.depends_on:
                if dep not in nums:
                    raise ValueError(f"步骤 {s.id} 依赖不存在的步骤:{dep}")
                if nums[dep] >= nums[s.id]:
                    raise ValueError(f"步骤 {s.id} 只能依赖编号更小的步骤,{dep} 不合法(防环)")

    def to_plan(self) -> "Plan":
        """草案 -> 运行时计划:运行时字段取默认(全 pending、无 task_id)。"""
        return Plan(
            goal=self.goal,
            steps=[PlanStep(**s.model_dump()) for s in self.steps]
        )




