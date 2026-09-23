"""子图构建注册表(c7, 见 ROADMAP §7):三个子图的唯一装配出处与按 llm 复用的编译缓存。

架构整理前的双装配问题:build_graph 急切编译一份直挂、TaskManager._BUILDERS 又懒构建
第二份实例 —— 一个 Web 进程为 executor 支付两遍冷启动(MCP 连接发现 ×2), 两份实例也
可能装到不同批的工具。注册表按 llm 强引用缓存、按 agent 惰性编译:直挂路径与后台
invoke 路径取同一批编译实例;未派发的 agent 仍不编译(TaskManager 懒构建语义保留)。

ADR-0009 双通道语义不变:直挂节点仍是可 invoke 的编译子图(Send 同步备胎可达),
TaskManager 后台线程 invoke 的也是同一实例。

使用位置:
    - agent/build.py:build_graph() 直挂三子图;
    - agent/tasks.py:TaskManager 后台派发取同批实例 + 未知 agent 兜底判定(BUILDERS);
    - tests/test_tasks.py:注册表缓存与双路径同实例测试。
"""
import threading

from agent.subagents.executor import build_executor_graph
from agent.subagents.research import build_research_graph
from agent.subagents.retriever import build_retriever_graph

BUILDERS = {
    "retriever": build_retriever_graph,
    "research": build_research_graph,
    "executor": build_executor_graph,
}

# id(llm) -> (llm 强引用, {agent: 编译实例}):强引用既防 id 复用错撞,也保证缓存
# 命中时 llm 一定还活着;不同 llm(测试各自 fake)各用各的槽,互不串。
_cache: dict[int, tuple[object, dict[str, object]]] = {}
_lock = threading.Lock()


def get_subgraph(llm, agent: str):
    """取指定子图的编译实例:同 llm 同 agent 只编译一次,不同 agent 各自惰性编译。"""
    key = id(llm)
    with _lock:
        ref, graphs = _cache.get(key, (None, None))
        if ref is not llm:  # 槽空或极端防御下的 id 复用:整槽重置
            ref, graphs = llm, {}
            _cache[key] = (ref, graphs)
        if agent not in graphs:
            graphs[agent] = BUILDERS[agent](llm)
        return graphs[agent]
