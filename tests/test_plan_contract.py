"""contracts/plan.py 契约层测试(M1 验收):纯函数,不依赖 LLM/DB/线程。

跑法:uv run pytest tests/test_plan_contract.py -q
(誊写 contracts/plan.py 之前此文件必然 ImportError——那是预期的红,implement 后转绿)
"""

import pytest
from pydantic import ValidationError

from agent.contracts import Plan, PlanDraft, PlanStep, ResultSummary


def _step(sid: str, deps: list[str] | None = None, assignee: str = "research") -> dict:
    return {
        "id": sid,
        "title": f"步骤{sid}的任务",
        "detail": "自包含描述:目标+输入+输出要求+成功标准",
        "assignee": assignee,
        "depends_on": deps or [],
    }


def _draft(*steps: dict) -> PlanDraft:
    return PlanDraft(goal="对比两家公司 2024 年毛利率", steps=list(steps))


def _summary(
    task_id: str = "t1", status: str = "success", conclusion: str = "已完成"
) -> ResultSummary:
    return ResultSummary(
        agent="research", task_id=task_id, task="任务回显", status=status, conclusion=conclusion
    )


# ---------------- validate_structure:四条规则 ----------------

def test_validate_ok():
    draft = _draft(_step("s1"), _step("s2", deps=["s1"]))
    draft.validate_structure()  # 不抛即通过


def test_validate_rejects_bad_id_format():
    draft = _draft(_step("step1"))
    with pytest.raises(ValueError, match="形如 s1"):
        draft.validate_structure()


def test_validate_rejects_duplicate_id():
    draft = _draft(_step("s1"), _step("s1"))
    with pytest.raises(ValueError, match="重复"):
        draft.validate_structure()


def test_validate_rejects_dangling_dep():
    draft = _draft(_step("s1", deps=["s9"]))
    with pytest.raises(ValueError, match="不存在"):
        draft.validate_structure()


def test_validate_rejects_backward_dep():
    """退回引用=环的等价物,s1 依赖 s2 必须拒绝。"""
    draft = _draft(_step("s1", deps=["s2"]), _step("s2"))
    with pytest.raises(ValueError, match="编号更小"):
        draft.validate_structure()


def test_validate_rejects_too_many_steps():
    """步数上限由 pydantic max_length 拦下(validate_structure 之前)。"""
    with pytest.raises(ValidationError):
        _draft(_step("s1"), _step("s2"), _step("s3"), _step("s4"), _step("s5"), _step("s6"))


def test_validate_rejects_empty_steps():
    with pytest.raises(ValidationError):
        _draft()


# ---------------- to_plan:运行时字段取默认 ----------------

def test_to_plan_defaults():
    plan = _draft(_step("s1"), _step("s2", deps=["s1"])).to_plan()
    assert isinstance(plan, Plan)
    assert plan.goal == "对比两家公司 2024 年毛利率"
    assert [s.id for s in plan.steps] == ["s1", "s2"]
    for s in plan.steps:
        assert s.status == "pending" and s.task_id == "" and s.result_digest == ""


def test_to_plan_keeps_content_fields():
    plan = _draft(_step("s1", assignee="executor")).to_plan()
    assert plan.steps[0].assignee == "executor"
    assert plan.steps[0].depends_on == []


# ---------------- Plan.ready_steps:依赖解锁 ----------------

def _plan(*steps: PlanStep) -> Plan:
    return Plan(goal="目标", steps=list(steps))


def test_ready_steps_initial():
    """首批=无依赖步;有依赖的步不该进 ready。"""
    p = _plan(
        PlanStep(**_step("s1")),
        PlanStep(**_step("s2", deps=["s1"])),
    )
    assert [s.id for s in p.ready_steps()] == ["s1"]


def test_ready_steps_unlocks_after_done():
    s1 = PlanStep(**_step("s1"))
    s2 = PlanStep(**_step("s2", deps=["s1"]))
    s1.status = "done"
    p = _plan(s1, s2)
    assert [s.id for s in p.ready_steps()] == ["s2"]


def test_ready_steps_excludes_running_and_done():
    s1 = PlanStep(**_step("s1"))
    s1.status = "running"
    p = _plan(s1)
    assert p.ready_steps() == []


# ---------------- mark_by_task:task_id 对账 ----------------

def test_mark_by_task_success_becomes_done():
    s1 = PlanStep(**_step("s1"), status="running", task_id="t1")
    p = _plan(s1)
    assert p.mark_by_task("t1", _summary("t1", "success", "查到了毛利率")) is True
    assert s1.status == "done" and s1.result_digest == "查到了毛利率"


def test_mark_by_task_partial_counts_as_done():
    s1 = PlanStep(**_step("s1"), status="running", task_id="t1")
    _plan(s1).mark_by_task("t1", _summary("t1", "partial"))
    assert s1.status == "done"


def test_mark_by_task_failed_only_hard_failure():
    """failed 记 failed(级联跳下游);need_clarification 记 done——"缺澄清"不是硬失败,
    digest 带缺口说明,补救步(如"联网补充核实")据此继续。2026-09-24 冒烟修正。"""
    s1 = PlanStep(**_step("s1"), status="running", task_id="t1")
    s2 = PlanStep(**_step("s2"), status="running", task_id="t2")
    p = _plan(s1, s2)
    p.mark_by_task("t1", _summary("t1", "failed"))
    p.mark_by_task("t2", _summary("t2", "need_clarification", conclusion="知识库无相关材料"))
    assert s1.status == "failed"
    assert s2.status == "done" and s2.result_digest == "知识库无相关材料"


def test_mark_by_task_unknown_task_id_returns_false():
    s1 = PlanStep(**_step("s1"), status="running", task_id="t1")
    assert _plan(s1).mark_by_task("other", _summary("other")) is False
    assert s1.status == "running"  # 不该被误改


# ---------------- cascade_skip:失败级联 ----------------

def test_cascade_skip_single_level():
    s1 = PlanStep(**_step("s1"), status="failed")
    s2 = PlanStep(**_step("s2", deps=["s1"]))
    p = _plan(s1, s2)
    p.cascade_skip()
    assert s2.status == "skipped"


def test_cascade_skip_transitive():
    """s1 失败 -> s2、s3 依次被跳过(迭代到不动点)。"""
    s1 = PlanStep(**_step("s1"), status="failed")
    s2 = PlanStep(**_step("s2", deps=["s1"]))
    s3 = PlanStep(**_step("s3", deps=["s2"]))
    s4 = PlanStep(**_step("s4"))  # 无依赖分支不受影响
    p = _plan(s1, s2, s3, s4)
    p.cascade_skip()
    assert [s.status for s in (s1, s2, s3, s4)] == ["failed", "skipped", "skipped", "pending"]


def test_cascade_skip_keeps_running_untouched():
    """running 步不在级联范围(它自己的结果还没回来)。"""
    s1 = PlanStep(**_step("s1"), status="failed")
    s2 = PlanStep(**_step("s2", deps=["s1"]), status="running", task_id="t1")
    p = _plan(s1, s2)
    p.cascade_skip()
    assert s2.status == "running"


# ---------------- finished:终态判定 ----------------

def test_finished_false_while_pending_or_running():
    assert _plan(PlanStep(**_step("s1"))).finished() is False
    assert _plan(PlanStep(**_step("s1"), status="running")).finished() is False


def test_finished_true_when_all_terminal():
    """done/failed/skipped 都是终态——全失败的计划也算"跑完",交 answer 如实汇总。"""
    p = _plan(
        PlanStep(**_step("s1"), status="done"),
        PlanStep(**_step("s2"), status="failed"),
        PlanStep(**_step("s3"), status="skipped"),
    )
    assert p.finished() is True


# ---------------- snapshot:展示形状 ----------------

def test_snapshot_contains_marks_and_digest():
    s1 = PlanStep(**_step("s1"), status="done", result_digest="毛利率 24.4%")
    s2 = PlanStep(**_step("s2", deps=["s1"]), status="running")
    s3 = PlanStep(**_step("s3"), status="skipped")
    text = _plan(s1, s2, s3).snapshot()
    assert "📋 计划" in text
    assert "☑ s1" in text and "毛利率 24.4%" in text
    assert "▶ s2" in text
    assert "⊘ s3" in text
