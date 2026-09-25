"""REPL 流式渲染:Spinner 状态机 + Live Markdown 流式渲染 + 轻量事件轨迹。

从 repl.py 拆出(原子化):run_turn 的 on_* 回调与 rich Live/Spinner 的启动、
清理收敛为一个 StreamRenderer;主循环每轮 finally 调 finish() 保证收干净——
挂起/异常路径残留 Live 会吞终端回显(troubleshooting/06 #4),on_interrupt
不打印问题避免双重(troubleshooting/06 #3)。

视觉语言(参考 Claude Code / pi 的终端风格):
- 回复不带 "Agent ›" 标签直出:首行只打一个淡色 ✦ 作声音标识,前后空行与对话分隔;
- 路由/派发等系统事件用 dim 的 ● 事件行,不与回复抢注意力(answer/ask 不打——
  回复与提问本身就是输出);
- 状态行 = spinner + 动词,文案随路由节点切换(思考中/规划中/子智能体执行中…)。
"""

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from agent.contracts import Route

# 各路由节点对应的 spinner 文案(on_route 时切换)
_STATUS_BY_NODE = {
    "answer": "思考中…",
    "plan": "规划中…",
    "memory": "整理记忆中…",
    "ask": "整理问题中…",
    "dispatch": "子智能体执行中(检索/调研约 1-2 分钟)…",
}


def print_route(console: Console, route) -> None:
    """轻量事件轨迹:dispatch 打印任务清单;plan/memory 打一行 dim 事件;
    answer/ask 不打(回复与提问本身就是输出,避免噪音)。"""
    r = route if isinstance(route, Route) else Route(**route)
    if r.next == "dispatch":
        for t in (r.tasks or []):
            console.print(f"[dim]●[/] 派发 [cyan]{t.agent}[/][dim] · {t.reason}[/dim]")
    elif r.next in ("plan", "memory"):
        console.print(f"[dim]● {r.next}[/dim]")


class StreamRenderer:
    """run_turn 流式回调与 rich 状态句柄的统一持有者(全局仅一个 status/live)。"""

    def __init__(self, console: Console):
        self.console = console
        self._status = None   # console.status:子智能体执行中的转圈反馈
        self._live = None     # Live:answer 流式 Markdown 渲染
        self._buf: list[str] = []

    def start_status(self, text: str = "思考中…") -> None:
        """启动/更新转圈:未启动则创建,已运行则只换文案。"""
        if self._status is None:
            self._status = self.console.status(f"[cyan]{text}", spinner="dots")
            self._status.start()
        else:
            self._status.update(f"[cyan]{text}")

    def stop_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def new_turn(self) -> None:
        """新一轮开始:清空 Markdown 缓冲。"""
        self._buf.clear()

    def on_token(self, token: str) -> None:
        """流式回调:首个 token 到达时停转圈、打 ✦ 声音标识,起 Live 持续重绘 Markdown。"""
        self.stop_status()
        self._buf.append(token)
        if self._live is None:
            self.console.print()
            self.console.print("[dim cyan]✦[/dim cyan]")
            self._live = Live(console=self.console, refresh_per_second=12,
                              vertical_overflow="visible", transient=False)
            self._live.start()
        self._live.update(Markdown("".join(self._buf)))

    def on_route(self, route) -> None:
        """路由回调:spinner 文案切换为目标节点的动词,并打印轻量事件轨迹。"""
        r = route if isinstance(route, Route) else Route(**route)
        self.start_status(_STATUS_BY_NODE.get(r.next, "思考中…"))
        print_route(self.console, r)

    def on_interrupt(self, intrs) -> None:
        """ask 挂起:问题打印统一由主循环挂起检测负责(跨重启恢复同样生效),
        此处不打印——否则挂起当轮与下一轮检测各打一次,出现双重提问(实测坑)。"""

    def finish(self) -> bool:
        """每轮收尾:停转圈 + 收 Live;返回本轮是否走过流式渲染。"""
        self.stop_status()
        if self._live is not None:
            self._live.stop()   # transient=False:最终 Markdown 帧保留在屏上
            self._live = None
            self.console.print()  # 回复后空行,与下一轮输入分隔
            return True
        return False

    def render_final(self, content: str) -> None:
        """未走流式时的整段 Markdown 渲染(ask/memory 桩节点、兜底消息)。"""
        content = (content or "").strip()
        if content:
            self.console.print()
            self.console.print("[dim cyan]✦[/dim cyan]")
            self.console.print(Markdown(content))
            self.console.print()
