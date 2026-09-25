"""plan_draft 节点测试(M3 验收):结构化生成 / 校验兜底 / method 透传 / 计划互斥。

节点函数是纯逻辑,直接调用断言返回的 Command,不必建图(interrupt 往返在
test_plan_confirm.py 里用整图测)。
"""

from langchain_core.messages import AIMessage, HumanMessage

from agent.contracts import PlanDraft
from agent.planner import plan_draft_node


class FakeStructuredLLM:
    """按 schema 分派返回值的 fake:PlanDraft 阶段返回草案,其余给路由/兜底回答。"""

    def __init__(self, draft=None, exc=None):
        self._draft, self._exc = draft, exc
        self.schema = None
        self.kwargs = None

    def with_structured_output(self, schema, **kwargs):
        self.schema, self.kwargs = schema, kwargs
        return self

    def invoke(self, messages):
        if self.schema is PlanDraft:
            if self._exc is not None:
                raise self._exc
            return self._draft
        return AIMessage(content="兜底回答")


def _state(plan=None, text: str = "对比宁德时代与比亚迪 2024 年毛利率"):
    return {
        "messages": [HumanMessage(content=text)],
        "last_route": {},
        "subagent_results": [],
        "plan": plan,
    }


def _draft(*specs) -> PlanDraft:
    steps = [
        {
            "id": sid,
            "title": f"步骤{sid}",
            "detail": "自包含描述:目标+输入+输出要求+成功标准",
            "assignee": "research",
            "depends_on": deps,
        }
        for sid, deps in specs
    ]
    return PlanDraft(goal="对比毛利率", steps=steps)


def test_plan_draft_generates_plan_and_gotos_confirm():
    llm = FakeStructuredLLM(draft=_draft(("s1", []), ("s2", ["s1"])))
    cmd = plan_draft_node(_state(), llm)
    assert cmd.goto == "plan_confirm"
    assert cmd.update["plan"]["goal"] == "对比毛利率"
    assert [s["status"] for s in cmd.update["plan"]["steps"]] == ["pending", "pending"]
    assert cmd.update["plan"]["steps"][1]["depends_on"] == ["s1"]


def test_plan_draft_passes_function_calling():
    """method 必须显式:DeepSeek 端点不支持 json_schema,默认会 400 后静默兜底。"""
    llm = FakeStructuredLLM(draft=_draft(("s1", [])))
    plan_draft_node(_state(), llm)
    assert llm.kwargs.get("method") == "function_calling"


def test_plan_draft_invalid_structure_falls_back_to_answer():
    llm = FakeStructuredLLM(draft=_draft(("s1", ["s9"])))  # 悬空依赖
    cmd = plan_draft_node(_state(), llm)
    assert cmd.goto == "answer"
    assert "plan" not in (cmd.update or {})


def test_plan_draft_llm_exception_falls_back_to_answer():
    llm = FakeStructuredLLM(exc=RuntimeError("端点 400"))
    cmd = plan_draft_node(_state(), llm)
    assert cmd.goto == "answer"


def test_plan_draft_blocked_when_plan_active():
    """已有计划进行中:不生成新计划,且不该调 LLM(防旧 plan 对账被覆盖)。"""
    llm = FakeStructuredLLM(draft=_draft(("s1", [])))
    cmd = plan_draft_node(_state(plan={"goal": "旧计划", "steps": []}), llm)
    assert cmd.goto == "answer"
    assert llm.schema is None
