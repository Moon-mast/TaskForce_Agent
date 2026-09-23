"""长期记忆命令(模块 06):/memory list(全量倒序)| /memory delete <key>。

查询与 namespace 收编 agent/memory_ctx(2026-09-23 架构整理 c4, ROADMAP §7):
本命令只负责渲染,语义与 API /memory 同源(memory_ctx.list_memories/delete_memory)。
"""

from rich.table import Table

from cli.context import ReplContext


def cmd_memory(args: str, ctx: ReplContext) -> None:
    """长期记忆管理:列表 / 删除(纯渲染,语义全在 memory_ctx)。"""
    sub, _, rest = args.strip().partition(" ")
    console = ctx.console
    try:
        from agent import memory_ctx  # 局部导入:首次 /memory 才触达记忆层

        if sub == "list":
            items = memory_ctx.list_memories()
            if not items:
                console.print("[dim](长期记忆为空)[/dim]")
                return
            table = Table(title=f"长期记忆({len(items)} 条)")
            table.add_column("key", style="dim")
            table.add_column("内容")
            table.add_column("来源")
            table.add_column("时间")
            for it in items:
                table.add_row(it["key"], it["content"], it["source"],
                              (it["created_at"] or "")[:16])
            console.print(table)
        elif sub == "delete":
            key = rest.strip()
            memory_ctx.delete_memory(key)
            console.print(f"[green]已删除[/] {key}")
        else:
            console.print("用法:[bold]/memory[/] list | [bold]/memory[/] delete <key>")
    except Exception as e:
        console.print("[red]出错[/]", e)
