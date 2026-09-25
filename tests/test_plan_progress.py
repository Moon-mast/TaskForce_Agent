"""supervisor 推进块测试(M4 验收):唤醒轮短路推进 / 收口 / 失败级联 / 用户轮插话。

整图 + InMemorySaver(参照 test_ask.py 模式);FakeTasks.complete() 模拟后台完成,
让 drain_done 有结果可回收——不需要真线程池。
"""

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.build import build_graph
from agent.contracts import PlanDraft, ResultSummary, Route
from agent.service import AUTO_NOTICE


class FakeTasks:
    """假 TaskManager:submit 记录并返回固定 id;complete() 注入结果供 drain_done 消费。"""

    def __init__(self):
        self.submits = []
        self._results = {}
        self._n = 0

    def submit(self, jobs, thread_id=""):
        self.submits.append((thread_id, list(jobs)))
        ids = []
        for _ in jobs:
            self._n += 1
            ids.append(f"t{self._n}")
        return ids

    def complete(self, task_id: str, status: str = "success", conclusion: str = "完成"):
        """模拟后台任务完成:结果进任务表,下一轮 drain_done 取走(消费即清)。"""
        self._results[task_id] = ResultSummary(
            agent="research", task_id=task_id, task="任务回显", status=status, conclusion=conclusion
        )

    def drain_done(self, thread_id=None):
        out = list(self._results.values())
        self._results.clear()
        return out

    def has_done(self, thread_id=None):
        return bool(self._results)


class FakePlanLLM:
    """fake:首轮路由 next="plan";plan_draft 阶段返回草案;插话/汇总轮路由 answer。

    schema 消费即重置:否则 answer 轮会把 PlanDraft 当回答返回(消息类型错)。
    """

    def __init__(self, draft):
        self._draft = draft
        self._routed = False
        self.schema = None

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        return self

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        schema, self.schema = self.schema, None
        if schema is PlanDraft:
            return self._draft
        if schema is Route:
            if not self._routed:
                self._routed = True
                return Route(next="plan")  # 首轮:识别为需拆分
            return Route(next="answer")  # 插话/后续轮:answer
        return AIMessage(content="已按计划汇总。")


def _draft3() -> PlanDraft:
    """三步链 s1 -> s2 -> s3(全依赖),便于测解锁与级联。"""
    return PlanDraft(
        goal="三步调研",
        steps=[
            {"id": "s1", "title": "取数", "detail": "自包含描述", "assignee": "research"},
            {
                "id": "s2",
                "title": "计算",
                "detail": "自包含描述",
                "assignee": "executor",
                "depends_on": ["s1"],
            },
            {
                "id": "s3",
                "title": "复核",
                "detail": "自包含描述",
                "assignee": "research",
                "depends_on": ["s2"],
            },
        ],
    )


def _setup(thread_id: str):
    """规划 + 确认:计划挂起 -> 批准 -> 首批 s1 派发(t1)。返回 (graph, config, tasks)。"""
    tasks = FakeTasks()
    graph = build_graph(FakePlanLLM(_draft3()), InMemorySaver(), tasks=tasks)
    config = {"recursion_limit": 40, "configurable": {"thread_id": thread_id}}
    list(graph.stream({"messages": [("user", "做个三步调研")]}, config, stream_mode="updates"))
    list(graph.stream(Command(resume="y"), config, stream_mode="updates"))
    return graph, config, tasks


def _wake(graph, config):
    """模拟一次唤醒轮:watcher 发 AUTO_NOTICE 起新一轮(与 REPL/API 同款触发语)。"""
    list(graph.stream({"messages": [("user", AUTO_NOTICE)]}, config, stream_mode="updates"))


def test_wake_round_unlocks_next_batch():
    """s1 完成 -> 唤醒轮对账 done、解锁 s2 派发,短路结束(末条是进度快照)。"""
    graph, config, tasks = _setup("prog-1")
    st = graph.get_state(config)
    assert st.values["plan"]["steps"][0]["task_id"] == "t1"

    tasks.complete("t1", conclusion="数据到手")
    _wake(graph, config)
    st = graph.get_state(config)
    plan = st.values["plan"]
    assert plan["steps"][0]["status"] == "done"
    assert plan["steps"][0]["result_digest"] == "数据到手"
    assert plan["steps"][1]["status"] == "running" and plan["steps"][1]["task_id"] == "t2"
    assert "计划推进" in st.values["messages"][-1].content  # 短路轮的进度消息


def test_wake_round_finishes_plan_to_answer():
    """三步依次完成 -> 第三轮唤醒收口:plan 清账、answer 汇总。"""
    graph, config, tasks = _setup("prog-2")
    for tid in ("t1", "t2", "t3"):
        tasks.complete(tid)
        _wake(graph, config)
    st = graph.get_state(config)
    assert st.values["plan"] is None  # 收口清账
    assert "已按计划汇总" in st.values["messages"][-1].content  # answer 的汇总回答


def test_failed_step_cascades_to_finish():
    """s1 失败 -> s2/s3 级联 skipped -> finished -> 收口(快照里能看到 ✗ 与 ⊘)。"""
    graph, config, tasks = _setup("prog-3")
    tasks.complete("t1", status="failed", conclusion="检索失败")
    _wake(graph, config)
    st = graph.get_state(config)
    assert st.values["plan"] is None  # 全部终态,直接收口
    texts = [m.content for m in st.values["messages"] if isinstance(m.content, str)]
    assert any("计划执行完毕" in t for t in texts)
    assert any("✗ s1" in t and "⊘ s2" in t and "⊘ s3" in t for t in texts)


def test_user_round_advances_then_routes():
    """计划进行中用户插话:先推进(解锁 s2),再照常路由处理插话——两不误。"""
    graph, config, tasks = _setup("prog-4")
    tasks.complete("t1", conclusion="数据到手")
    list(graph.stream({"messages": [("user", "顺便问个别的")]}, config, stream_mode="updates"))
    st = graph.get_state(config)
    plan = st.values["plan"]
    assert plan["steps"][0]["status"] == "done" and plan["steps"][1]["status"] == "running"
    # 推进并入后路由照常:插话被路由 answer 并作答(fake 返回的兜底回答)
    assert st.values["messages"][-1].content == "已按计划汇总。"
