"""知识库命令:/kb upload <文件路径> | /kb list | /kb delete <doc_id>。

管理语义(判重 / 20MB 上限 / 存在性判定)收编 `rag_v01` facade(2026-09-23 架构整理 c4,
ROADMAP §7),与 API /knowledge 是同一实现;本命令只负责路径解析与渲染。
展示上少一列"上传时间":新内核是内容寻址, 文档只是子块的聚合, 没有上传时间这个概念。
"""

from pathlib import Path

from rich.table import Table

from cli.context import ReplContext


def cmd_kb(args: str, ctx: ReplContext) -> None:
    """知识库管理:上传/列表/删除(纯渲染,语义全在 rag_v01 facade)。"""
    sub, _, rest = args.strip().partition(" ")
    console = ctx.console
    try:
        import rag_v01  # facade 无 import 副作用(不连 Milvus);首次调用才碰库

        if sub == "upload":
            # 用户习惯给路径包引号(防空格):剥掉首尾引号再建 Path,否则后缀名带引号匹配失败
            path = Path(rest.strip().strip("\"'"))
            try:
                content = path.read_bytes()
            except OSError as e:
                console.print(f"[red]读取失败[/] {e}")
                return
            r = rag_v01.upload(content, path.name)
            if not r["ok"]:
                console.print(f"[red]入库失败[/] {r['error']}")
                return
            if r["created"]:
                console.print(f"[green]已上传[/] {path.name} → doc_id={r['doc_id']}")
            else:
                console.print(f"[yellow]内容重复[/yellow],已复用已有文档(doc_id={r['doc_id']})")
        elif sub == "list":
            rows = rag_v01.list_docs()
            if not rows:
                console.print("[dim](知识库为空)[/dim]")
                return
            table = Table(title=f"知识库({len(rows)} 个文档)")
            table.add_column("doc_id", style="dim")
            table.add_column("文件名")
            table.add_column("子块数", justify="right")
            for row in rows:
                table.add_row(row["doc_id"], row["filename"], str(row["chunks"]))
            console.print(table)
        elif sub == "delete":
            doc_id = rest.strip()
            if not rag_v01.delete_doc(doc_id):
                console.print(f"[red]文档不存在[/] {doc_id}")
                return
            console.print(f"[green]已删除[/] {doc_id}")
        else:
            console.print("用法:[bold]/kb[/] upload <文件路径> | [bold]/kb[/] list"
                          " | [bold]/kb[/] delete <doc_id>")
    except Exception as e:
        console.print("[red]出错[/]", e)
