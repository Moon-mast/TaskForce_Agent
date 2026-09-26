"""REPL 流式渲染:块级增量 Markdown 渲染 + Spinner 状态机 + 非流式 Markdown 兜底。

从 repl.py 拆出(原子化):run_turn 的 on_* 回调与 Spinner 的启动、清理收敛为
一个 StreamRenderer;主循环每轮 finally 调 finish() 保证收干净——挂起/异常路径
残留 Spinner 会吞终端回显(troubleshooting/06 #4),on_interrupt 不打印问题避免
双重(troubleshooting/06 #3)。

视觉语言(参考 Claude Code / pi):
- 流式回答按**块级增量**渲染:新完成的段落序列(以空行为块边界,``` 围栏内不断块)
  以 rich Markdown 静态打印,每段只打印一次——无重绘、无堆叠,长内容不超限。
  历史教训:Live(vertical_overflow=visible)全量重绘在内容超终端高度时光标回退
  擦不掉历史行,每个刷新周期全量重打(2026-09-25 茅台汇总实测滚屏堆叠);
  逐 token 打字机则牺牲格式——块级增量是"流式 + 格式 + 不重复"的平衡点。
- 工具循环中间帧(带 tool_calls)在 service 层过滤,不进对话流。
- 路由/派发事件用 dim ● 行;spinner 文案随路由节点切换。
"""

from rich.console import Console
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
    """run_turn 流式回调与 rich 状态句柄的统一持有者(全局仅一个 status)。"""

    def __init__(self, console: Console):
        self.console = console
        self._status = None           # console.status:转圈反馈
        self._buf: list[str] = []     # 流式 token 累积(拼接后即完整回答)
        self._renderedChars = 0       # 已渲染到 buf 的字符游标(增量渲染游标)
        self._streamedOnce = False    # 本轮是否已渲染过内容(新标识符按 camelStyle 驼峰)

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
        """新一轮开始:清空缓冲与游标。"""
        self._buf.clear()
        self._renderedChars = 0
        self._streamedOnce = False

    def on_token(self, token: str) -> None:
        """流式回调:token 进缓冲,随后渲染已完成的块(无完整块则等待下一个边界)。
        工具循环中间帧已在 service 层过滤,不会到达这里。"""
        self.stop_status()
        self._buf.append(token)
        self._flushCompleteBlocks()

    def _flushCompleteBlocks(self, final: bool = False) -> None:
        """增量渲染 buf 中已完成的块序列。

        块边界 = 空行(``` 围栏内不断块,防代码块被切散);渲染区间 = 上次游标到
        最后一个块边界,整段作为一个 Markdown 打印(内部多段/列表/标题均可)。
        final=True 时渲染到缓冲末尾(含未闭合尾块)。每段只打印一次,无重绘。"""
        text = "".join(self._buf)
        pending = text[self._renderedChars:]
        if not pending.strip():
            return
        cut = len(pending)
        if not final:
            cut = 0
            inFence = False
            consumed = 0
            for ln in pending.split("\n"):
                if ln.strip().startswith("```"):
                    inFence = not inFence
                consumed += len(ln) + 1
                if not inFence and ln.strip() == "":
                    cut = consumed
            if cut == 0:
                return  # 尾块未闭合:等更多 token 或 finish
        chunk = pending[:cut].strip("\n")
        if not chunk.strip():
            self._renderedChars += cut
            return
        self.stop_status()
        if not self._streamedOnce:
            self._streamedOnce = True
            self.console.print()
            self.console.print("[dim cyan]✦[/dim cyan]")
        self.console.print(Markdown(chunk))
        self.console.print()
        self._renderedChars += cut

    def on_route(self, route) -> None:
        """路由回调:spinner 文案切换为目标节点的动词,并打印轻量事件轨迹。"""
        r = route if isinstance(route, Route) else Route(**route)
        self.start_status(_STATUS_BY_NODE.get(r.next, "思考中…"))
        print_route(self.console, r)

    def on_interrupt(self, intrs) -> None:
        """ask 挂起:问题打印统一由主循环挂起检测负责(跨重启恢复同样生效),
        此处不打印——否则挂起当轮与下一轮检测各打一次,出现双重提问(实测坑)。"""

    def finish(self) -> bool:
        """每轮收尾:停转圈 + 渲染未闭合尾块;返回本轮是否渲染过内容。"""
        self.stop_status()
        self._flushCompleteBlocks(final=True)
        if self._streamedOnce:
            self.console.print()  # 回复后空行,与下一轮输入分隔
            return True
        return False

    def render_final(self, content: str) -> None:
        """非流式路径的整段 Markdown 渲染(桩节点、兜底、计划快照等)。"""
        content = (content or "").strip()
        if content:
            self.console.print()
            self.console.print("[dim cyan]✦[/dim cyan]")
            self.console.print(Markdown(content))
            self.console.print()
