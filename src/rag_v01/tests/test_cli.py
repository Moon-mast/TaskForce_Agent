"""模块 07 入口层单测: facade(`__init__.py`) + `cli.py`。

不碰 Milvus / docling: `cli.run_ingest` / `cli.run_search` / `cli.store.*` 全部 monkeypatch,
输出用 capsys 断言 —— 顺便证明中文在 stdout 上打得出来。

facade 的"无 import 副作用"用**子进程**验: 同一进程里别的测试早就 import 过 docling 了,
进程内看一眼 `sys.modules` 无论真假都说明不了问题。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from rag_v01 import cli
from rag_v01.contracts import ChildHit, IngestReport, RetrievedChunk

MODULE_ROOT = Path(__file__).resolve().parents[2]  # .../src (rag_v01 包的父目录)


def report(**kw) -> IngestReport:
    base = {
        "ok_sources": ["a.md"],
        "failed": {},
        "total_parents": 2,
        "total_children": 3,
    }
    base.update(kw)
    return IngestReport(**base)


def one_row() -> RetrievedChunk:
    return RetrievedChunk(
        parent_id="doc:p001",
        parent_text="父块全文",
        rrf_rank=1,
        hits=[
            ChildHit(
                chunk_id="doc:p001:c001",
                text="命中文本" * 40,
                chunk_type="text",
                dense_rank=1,
                sparse_rank=2,
                rrf_score=0.032266,
            )
        ],
        doc_id="doc",
        source="a.md",
        page_no=3,
        heading_path="一 > 二",
    )


@pytest.fixture
def run(monkeypatch, capsys):
    """按 argv 驱动 `main()`, 返回 `(退出码, stdout)`。"""

    def _run(*argv: str) -> tuple[int, str]:
        monkeypatch.setattr(sys, "argv", ["rag_v01.cli", *argv])
        code = 0
        try:
            cli.main()
        except SystemExit as exc:  # main 用 SystemExit 表达失败, 这里把它变成返回值
            code = exc.code if isinstance(exc.code, int) else 1
        return code, capsys.readouterr().out

    return _run


# ---------------------------------------------------------------- 分发


def test_no_args_prints_usage_and_exits_1(run):
    code, out = run()

    assert code == 1
    assert out.startswith("用法: python -m rag_v01.cli")


def test_unknown_command_prints_usage_and_exits_1(run):
    code, out = run("搜一下")

    assert code == 1
    assert "用法:" in out


# ---------------------------------------------------------------- ingest


def test_ingest_prints_summary_failures_and_warnings(run, monkeypatch):
    seen: list[list[str]] = []

    def fake_ingest(paths):
        seen.append(list(paths))
        return report(
            ok_sources=["a.md"],
            failed={"bad.pdf": "ParseError: 解析失败"},
            warnings={"a.md": ["清洗统计: {'deduped': 2}", "c001: text 超上限, 已跳过"]},
        )

    monkeypatch.setattr("rag_v01.pipeline.run_ingest", fake_ingest)

    code, out = run("ingest", "corpus", "extra.md")

    assert code == 0
    assert seen == [["corpus", "extra.md"]]
    assert "成功 1 份 / 失败 1 份" in out
    assert "父块 2 个, 子块 3 个" in out
    assert "失败 bad.pdf: ParseError: 解析失败" in out
    assert "警告 a.md: c001: text 超上限, 已跳过" in out


def test_ingest_without_path_exits_1(run, monkeypatch):
    monkeypatch.setattr("rag_v01.pipeline.run_ingest", lambda paths: pytest.fail("不该被调用"))

    code, out = run("ingest")

    assert code == 1
    assert "错误: ingest 需要至少一个路径" in out


# ---------------------------------------------------------------- search


def test_search_prints_score_ranks_and_snippet(run, monkeypatch):
    seen: list[tuple[str, int | None]] = []

    def fake_search(query, top_k=None):
        seen.append((query, top_k))
        return [one_row()]

    monkeypatch.setattr("rag_v01.retrieve.search", fake_search)

    code, out = run("search", "向量数据库怎么用")

    assert code == 0
    assert seen == [("向量数据库怎么用", None)]  # 不给 --top-k 就走 config 默认
    assert "[0.032266 d1/s2] a.md#doc:p001:c001 (1 条命中, 一 > 二)" in out
    snippet = out.strip().splitlines()[-1].strip()
    assert snippet == "命中文本" * 20  # 前 80 字(每个"命中文本"4 字 × 20)


def test_search_with_no_hits_prints_hint(run, monkeypatch):
    monkeypatch.setattr("rag_v01.retrieve.search", lambda query, top_k=None: [])

    code, out = run("search", "无关的问题")

    assert code == 0
    assert "没有命中任何内容" in out


@pytest.mark.parametrize(
    "argv",
    [
        ("search", "向量", "数据库", "--top-k", "3"),
        ("search", "--top-k", "3", "向量", "数据库"),
    ],
)
def test_search_parses_top_k_and_joins_query(run, monkeypatch, argv):
    """query 本身含空格, `--top-k` 放前放后都要认。"""
    seen: list[tuple[str, int | None]] = []
    def fake_search(query, top_k=None):
        seen.append((query, top_k))
        return []

    monkeypatch.setattr("rag_v01.retrieve.search", fake_search)

    code, _ = run(*argv)

    assert code == 0
    assert seen == [("向量 数据库", 3)]


def test_search_top_k_without_value_exits_1(run):
    code, out = run("search", "向量", "--top-k")

    assert code == 1
    assert "--top-k 后面要跟一个数字" in out


def test_search_top_k_with_non_integer_exits_1(run):
    code, out = run("search", "向量", "--top-k", "三")

    assert code == 1
    assert "不是整数" in out


def test_search_without_query_exits_1(run):
    code, out = run("search", "--top-k", "3")

    assert code == 1
    assert "需要一句 query" in out


def test_search_propagates_library_errors_as_usage_errors(run, monkeypatch):
    """库里抛的可读错误(如维度不一致)不该冒堆栈, 而是"错误: ..."+退出码 1。"""

    def boom(query, top_k=None):
        raise ValueError("top_k 必须为正整数, 收到 0")

    monkeypatch.setattr("rag_v01.retrieve.search", boom)

    code, out = run("search", "向量")

    assert code == 1
    assert "错误: top_k 必须为正整数" in out


# ---------------------------------------------------------------- 运维子命令


def test_list_prints_table(run, monkeypatch):
    monkeypatch.setattr(
        "rag_v01.store.list_docs",
        lambda: [
            {"doc_id": "aaa", "source": "a.md", "children": 3},
            {"doc_id": "bbb", "source": "b.md", "children": 1},
        ],
    )

    code, out = run("list")

    assert code == 0
    assert "aaa" in out and "a.md" in out and "bbb" in out
    assert "子块" in out  # 表头


def test_list_empty_library(run, monkeypatch):
    monkeypatch.setattr("rag_v01.store.list_docs", lambda: [])

    code, out = run("list")

    assert code == 0
    assert "库是空的" in out


def test_delete_calls_store_for_each_id(run, monkeypatch):
    deleted: list[str] = []
    monkeypatch.setattr("rag_v01.store.delete_doc", lambda doc_id: deleted.append(doc_id))

    code, out = run("delete", "aaa", "bbb")

    assert code == 0
    assert deleted == ["aaa", "bbb"]
    assert "已删除 aaa" in out and "已删除 bbb" in out


def test_delete_without_id_exits_1(run, monkeypatch):
    monkeypatch.setattr("rag_v01.store.delete_doc", lambda doc_id: pytest.fail("不该被调用"))

    code, out = run("delete")

    assert code == 1
    assert "delete 需要 doc_id" in out


# ---------------------------------------------------------------- facade


def test_facade_exports_only_the_public_surface():
    """`__all__` 只列公开面(检索 ingest/search/evaluate + 管理 list_docs/upload/delete_doc
    /MAX_UPLOAD_BYTES —— 管理面 2026-09-23 c4 收编, 见 ROADMAP §7)。

    这里**不**断言 `rag_v01.__dict__` 里没有子模块: 只要本进程任何地方 import 过
    `rag_v01.retrieve`, Python 就会把它绑成父包的属性 —— 进程内看这个没有意义,
    "import 不带副作用"由下面那条子进程用例负责。
    """
    import rag_v01

    assert rag_v01.__all__ == [
        "ingest",
        "search",
        "evaluate",
        "list_docs",
        "upload",
        "delete_doc",
        "MAX_UPLOAD_BYTES",
        "ChildHit",
        "EvalReport",
        "IngestReport",
        "RetrievedChunk",
    ]


def test_facade_ingest_stays_callable_after_being_used(monkeypatch):
    """回归用例(这个 bug 真发生过): 流水线模块若与 facade 函数**同名**, 一旦该子模块被 import,
    Python 会把父包上的 `ingest` 属性从函数换成模块 —— 第一次调用正常, 第二次就
    `TypeError: 'module' object is not callable`。所以流水线模块叫 `pipeline.py`。"""
    import rag_v01

    calls: list[list[str]] = []
    monkeypatch.setattr(
        "rag_v01.pipeline.run_ingest",
        lambda paths: calls.append(list(paths)) or IngestReport(ok_sources=[], failed={}),
    )

    assert callable(rag_v01.ingest)
    rag_v01.ingest(["a.md"])
    assert callable(rag_v01.ingest)  # 调过一次之后仍是函数, 不是 module
    rag_v01.ingest(["b.md"])

    assert calls == [["a.md"], ["b.md"]]


def test_import_rag_v01_has_no_heavy_side_effects():
    """07 篇 §五的空跑: `import rag_v01` 不加载 docling / pymilvus / torch, 也不连 Milvus。"""
    code = (
        "import sys, rag_v01;"
        "print(','.join(rag_v01.__all__));"
        "print('heavy=[' + ','.join(m for m in"
        " ('docling', 'pymilvus', 'torch', 'sentence_transformers') if m in sys.modules) + ']')"
    )
    env = {**os.environ, "PYTHONPATH": str(MODULE_ROOT)}
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stderr
    exports, heavy = proc.stdout.strip().splitlines()
    assert exports == (
        "ingest,search,evaluate,list_docs,upload,delete_doc,"
        "MAX_UPLOAD_BYTES,ChildHit,EvalReport,IngestReport,RetrievedChunk"
    )
    assert heavy == "heavy=[]"  # 一个重依赖都没被拉进来(管理面新增导出同样是惰性 import)


def test_package_is_portable_when_copied_away(tmp_path):
    """07 篇 §2.3 的第一条硬保证「可整体搬走」的实测: 把包拷到别处、只把那份拷贝放进 sys.path,
    能导入、能用纯函数、且**不 import 项目任何包**。

    为什么值得一条测试: 只有包内 import 全写成相对形式时它才成立; 谁哪天写了一句绝对自引用或
    `from settings...`, 单元测试照样全绿、拷走却散架 —— 这条把它钉住。
    """
    import shutil

    target = tmp_path / "rag_v01"
    shutil.copytree(
        MODULE_ROOT / "rag_v01",
        target,
        ignore=shutil.ignore_patterns("__pycache__", "tests", "data"),
    )
    code = (
        "import rag_v01;"
        "from rag_v01.chunk import split;"          # 纯函数(不碰外部服务)
        "from rag_v01.contracts import ChildChunk;"  # 契约可导入
        "print(rag_v01.__all__[0], split.__name__, ChildChunk.__name__)"
    )
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},  # 只给拷贝所在目录
        cwd=str(tmp_path),  # 也不靠当前工作目录
        timeout=180,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ingest split ChildChunk"
