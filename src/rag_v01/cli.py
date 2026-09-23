"""cli.py —— rag_v01 的命令行入口(07 篇 §4.2)。

子命令(与旧 src/tools/rag/cli.py 同风格: 手写 argv 解析 + USAGE + 出错 SystemExit(1)):
    ingest <路径...>            批量入库(文件或目录, 目录递归)
    search <query> [--top-k N]  检索, 打印命中的父块与最佳子块
    list                        列出库里的文档(运维)
    delete <doc_id>             按 doc_id 删除一份文档(运维)
`evaluate` 子命令随模块 06 落地。

`list` / `delete` 直调 store.py 的运维面、不进 facade —— 它们是运维动作, 不是检索能力(07 篇 §2.1)。
"""
from __future__ import annotations

import sys

# Windows 控制台默认 GBK, 直接 print 中文/表格 Markdown 会 UnicodeEncodeError(01/07 篇的实测旧账)。
# 必须在**任何 print 之前**做, 且**只在这个入口模块**做 —— 库模块不许动全局 stdout(它要能整体搬走)。
if hasattr(sys.stdout, "reconfigure"):  # pytest 的捕获对象没有这个方法, 所以要判存在
    sys.stdout.reconfigure(encoding="utf-8")

# 业务模块一律**按子命令懒加载**(见各 `_cmd_*`): CLI 一次只跑一个子命令, 不该把 docling(连带 torch)
# 与 pymilvus 全拉进来 —— 评估环境里没有可用的 torch, 顶层 import 会让 evaluate 子命令起不来。

USAGE = """用法: python -m rag_v01.cli <子命令> [参数]
  ingest <路径...>             批量入库(文件或目录, 目录递归)
  search <query> [--top-k N]   检索, 打印命中的父块与最佳子块
  list                         列出库里的文档
  delete <doc_id>              按 doc_id 删除一份文档
  evaluate [--testset 路径] [--report-dir 目录] [--limit N] [--samples 路径] [--label 名]
                               跑 ragas 四指标评估(需评估环境, 见 06 篇); --label 用于新旧对比"""


def main() -> None:
    """按第一个参数分发子命令; 参数错/库里报错都打人话并以退出码 1 结束。"""
    args = sys.argv[1:]
    if not args or args[0] not in ("ingest", "search", "list", "delete", "evaluate"):
        print(USAGE)
        raise SystemExit(1)
    command, rest = args[0], args[1:]
    try:
        if command == "ingest":
            _cmd_ingest(rest)
        elif command == "search":
            _cmd_search(rest)
        elif command == "list":
            _cmd_list()
        elif command == "delete":
            _cmd_delete(rest)
        else:
            _cmd_evaluate(rest)
    except ValueError as exc:  # 参数错与库里的可读报错: 不冒堆栈, 让人看清是什么用错了
        print(f"错误: {exc}")
        raise SystemExit(1) from exc


def _cmd_ingest(args: list[str]) -> None:
    """批量入库并打印 IngestReport 摘要(成功/失败/父子块计数/警告)。"""
    from .pipeline import run_ingest

    if not args:
        raise ValueError("ingest 需要至少一个路径")
    report = run_ingest(args)
    print(
        f"入库: 成功 {len(report.ok_sources)} 份 / 失败 {len(report.failed)} 份, "
        f"父块 {report.total_parents} 个, 子块 {report.total_children} 个"
    )
    for source, reason in sorted(report.failed.items()):
        print(f"  失败 {source}: {reason}")
    for source in report.ok_sources:
        for note in report.warnings.get(source, []):
            print(f"  警告 {source}: {note}")


def _cmd_search(args: list[str]) -> None:
    """检索并打印: RRF 分、两路名次、来源#chunk_id、最佳子块文本前 80 字。"""
    from .retrieve import search as run_search

    top_k, words = _split_top_k(args)
    query = " ".join(words)
    if not query:
        raise ValueError("search 需要一句 query")
    rows = run_search(query, top_k=top_k)
    if not rows:
        print("没有命中任何内容(库是空的? 还是 query 与语料无关?)")
        return
    for row in rows:
        best = row.hits[0]
        dense = best.dense_rank if best.dense_rank is not None else "-"
        sparse = best.sparse_rank if best.sparse_rank is not None else "-"
        print(
            f"[{best.rrf_score:.6f} d{dense}/s{sparse}] {row.source}#{best.chunk_id} "
            f"({len(row.hits)} 条命中, {row.heading_path or '无章节'})"
        )
        print(f"    {best.text[:80]}")


def _split_top_k(args: list[str]) -> tuple[int | None, list[str]]:
    """把 `--top-k N` 从参数里摘出来, 剩下的词按顺序拼成 query(query 里可以有空格)。"""
    top_k: int | None = None
    words: list[str] = []
    rest = list(args)
    while rest:
        token = rest.pop(0)
        if token != "--top-k":
            words.append(token)
            continue
        if not rest:
            raise ValueError("--top-k 后面要跟一个数字")
        raw = rest.pop(0)
        try:
            top_k = int(raw)
        except ValueError as exc:
            raise ValueError(f"--top-k 的参数不是整数: {raw!r}") from exc
    return top_k, words


def _cmd_list() -> None:
    """列出库里的文档(运维): 子块数 + doc_id + source。"""
    from .store import list_docs

    docs = list_docs()
    if not docs:
        print("库是空的。")
        return
    print(f"{'子块':>6}  {'doc_id':<18} source")
    for doc in docs:
        print(f"{doc['children']:>6}  {doc['doc_id']:<18} {doc['source']}")


def _cmd_delete(args: list[str]) -> None:
    """按 doc_id 删除一份文档(运维): 两个 collection 一起删, 支持一次多个 id。"""
    from .store import delete_doc

    if not args:
        raise ValueError("delete 需要 doc_id(可用 list 查看)")
    for doc_id in args:
        delete_doc(doc_id)
        print(f"已删除 {doc_id}")


_EVAL_VALUE_FLAGS = {  # 这些后面要跟一个值

    "--testset": "testset",
    "--report-dir": "report_dir",
    "--samples": "samples",
    "--label": "system_label",  # 系统标签: 新旧对比时旧系统用 --label tools_rag_v1
}
_EVAL_FLAGS = (*_EVAL_VALUE_FLAGS, "--limit")


def _cmd_evaluate(args: list[str]) -> None:
    """跑四指标评估并打印结果; `--limit` 用于先试跑看账单(06 篇 §4.6)。"""
    from .evaluation import run_eval

    options = _parse_evaluate_args(args)
    report = run_eval(
        options["testset"],
        options["report_dir"],
        samples_path=options["samples"],
        system_label=options["system_label"],
        limit=options["limit"],
    )
    print(f"评估完成: {report.question_count} 条题, 报告在 {report.report_dir}")
    for name, value in report.metrics.items():
        print(f"  {name}: {value:.4f}" if value is not None else f"  {name}: NaN")


def _parse_evaluate_args(args: list[str]) -> dict:
    """解析 `evaluate` 的选项(风格与 `--top-k` 一致: 手写、未知项直接报错)。"""
    options: dict = {
        "testset": None,
        "report_dir": "eval_reports",
        "samples": None,
        "limit": None,
        "system_label": "rag_v01",
    }
    rest = list(args)
    while rest:
        token = rest.pop(0)
        if token not in _EVAL_FLAGS:
            raise ValueError(
                f"evaluate 不认识参数 {token!r}"
                "(可用: --testset / --report-dir / --samples / --limit / --label)"
            )
        if not rest:
            raise ValueError(f"{token} 后面要给一个值")
        value = rest.pop(0)
        if token == "--limit":
            try:
                options["limit"] = int(value)
            except ValueError as exc:
                raise ValueError(f"--limit 的参数不是整数: {value!r}") from exc
        else:
            options[_EVAL_VALUE_FLAGS[token]] = value
    return options


if __name__ == "__main__":
    main()
