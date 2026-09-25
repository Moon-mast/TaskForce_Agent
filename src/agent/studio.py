"""LangGraph Studio(langgraph dev)本地调试入口:与 cli/api 并列的第三种装配点。

与 cli/repl.py、api/main.py 的差别(仅为调试便利,不改图结构):
- 不挂本地 PostgresSaver:Studio 平台托管持久化(自带 in-memory checkpointer),
  挂本地连接池会与平台侧的 checkpoint 读写打架;
- 不接 watcher 自动唤醒:plan 推进由交互轮触发——supervisor 每轮照常 drain_done 回收,
  在 Studio 里手动再发一句(或点继续)即可推进下一批,验证图结构不受影响;
- 后台派发仍走 build_graph 内自建的 TaskManager(方案 A),线程池在 Studio 进程内照常工作。

langgraph.json 的 graphs 字段指向本模块的工厂;配置来自 .env(settings/config.py
按绝对路径锚定项目根,与 Studio 子进程 CWD 无关)。
"""

from agent.build import build_graph, make_llm


def make_studio_graph():
    """无参图工厂:平台负责注入 configurable 与持久化,这里只做装配。"""
    return build_graph(make_llm())
