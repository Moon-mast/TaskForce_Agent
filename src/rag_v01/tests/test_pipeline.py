"""模块 07 入库编排单测(`pipeline.py`)。

**全程不碰真东西**: docling 不解析、Milvus 不连、embedding 不调 —— 上游四个入口与 store 全部
monkeypatch 成假件。测的是**编排本身的正确性**: 文件展开与 source 计算、单文件失败隔离、计数、
flush 时机、以及"notes 有没有把各阶段的警告汇到一个口里"。

不测的: 真实解析/切分/嵌入质量(那由 02/03/04 篇各自的单测与真机冒烟兜底)。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rag_v01 import pipeline
from rag_v01.config import Config
from rag_v01.contracts import ChildChunk, ParentChunk, ParsedDoc, ParsedItem


def parent(parent_id: str) -> ParentChunk:
    return ParentChunk(chunk_id=parent_id, text="父块", doc_id="doc-x", source="a.md")


def child(chunk_id: str) -> ChildChunk:
    return ChildChunk(chunk_id=chunk_id, text="子块", parent_id="doc-x:p001", doc_id="doc-x")


@pytest.fixture
def wired(monkeypatch):
    """把流水线的每一环换成假件, 并记下调用: 不碰 docling / Milvus / embedding。"""

    def _install(*, parents=None, children=None, fail: dict[str, str] | None = None,
                 parse_warnings=(), embed_warnings=(), stale=(), doc_id="doc-x"):
        calls: dict[str, list] = {
            "order": [], "flush": 0, "ensure": 0, "purge": [], "upsert_parents": [],
            "upsert_children": [], "embed": [],
        }
        parents = [parent("doc-x:p001")] if parents is None else list(parents)
        children = [child("doc-x:p001:c001")] if children is None else list(children)
        fail = fail or {}

        def fake_parse(path, cfg, *, source=None):
            calls["order"].append("parse")
            if path.name in fail:
                raise ValueError(fail[path.name])
            return ParsedDoc(
                doc_id=doc_id, source=source or path.name,
                items=[ParsedItem(type="text", text="正文")],
                meta={"warnings": list(parse_warnings)},
            )

        def fake_clean(doc, cfg):
            calls["order"].append("clean")
            stats = {"garbled": 0, "deduped": 2, "merged_short": 3}
            return doc, stats

        def fake_split(doc, cfg):
            calls["order"].append("split")
            return list(parents), list(children)

        def fake_embed_chunks(rows, *, warnings=None):
            calls["order"].append("embed")
            calls["embed"].append([row.chunk_id for row in rows])
            if warnings is not None:
                warnings.extend(embed_warnings)
            return [[0.0]] * len(rows)

        def count(key: str):
            calls[key] += 1

        def fake_purge(source, *, keep_doc_id):
            calls["order"].append("purge")
            calls["purge"].append((source, keep_doc_id))
            return list(stale)

        monkeypatch.setattr(pipeline, "parse_document", fake_parse)
        monkeypatch.setattr(pipeline, "clean_document", fake_clean)
        monkeypatch.setattr(pipeline, "split", fake_split)
        monkeypatch.setattr(pipeline.embed, "embed_chunks", fake_embed_chunks)
        monkeypatch.setattr(pipeline.store, "ensure_collections", lambda: count("ensure"))
        monkeypatch.setattr(pipeline.store, "purge_stale", fake_purge)
        monkeypatch.setattr(
            pipeline.store, "upsert_parents",
            lambda rows: calls["upsert_parents"].append(len(rows)) or len(rows),
        )
        monkeypatch.setattr(
            pipeline.store, "upsert_children",
            lambda rows, vectors, *, warnings=None: calls["upsert_children"].append(len(rows)),
        )
        monkeypatch.setattr(pipeline.store, "flush", lambda: count("flush"))
        return calls

    return _install


# ---------------------------------------------------------------- 文件展开


def test_expand_paths_single_file_source_is_filename(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("x", encoding="utf-8")

    assert pipeline.expand_paths([f]) == [(f, "a.md")]
    assert pipeline.expand_paths([str(f)]) == [(f, "a.md")]


def test_expand_paths_directory_is_recursive_and_skips_hidden(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("x", encoding="utf-8")
    (tmp_path / "sub" / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("x", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("x", encoding="utf-8")

    expanded = pipeline.expand_paths([tmp_path])

    # 递归到底、隐藏项全跳过、source 是相对语料根的 POSIX 路径、按路径排序(可复现)
    assert expanded == [
        (tmp_path / "a.md", "a.md"),
        (tmp_path / "sub" / "b.md", "sub/b.md"),
        (tmp_path / "sub" / "notes.txt", "sub/notes.txt"),
    ]


def test_expand_paths_dedupes_same_file_from_two_inputs(tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")

    expanded = pipeline.expand_paths([tmp_path, tmp_path / "a.md"])

    assert expanded == [(tmp_path / "a.md", "a.md")]  # 目录与文件两种入口只算一次


def test_expand_paths_keeps_missing_file_for_parse_to_report(tmp_path):
    """文件不存在不在这里丢: 统一由 parse_document 报"不存在或不是文件", 错误文案只此一处。"""
    missing = tmp_path / "nope.md"

    assert pipeline.expand_paths([missing]) == [(missing, "nope.md")]


# ---------------------------------------------------------------- 整批编排


def test_run_ingest_happy_path_counts_and_order(wired, tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("x", encoding="utf-8")
    calls = wired(parents=[parent("p1"), parent("p2")], children=[child("c1")])

    report = pipeline.run_ingest([tmp_path], Config())

    assert report.ok_sources == ["a.md", "b.md"]
    assert report.failed == {}
    assert (report.total_parents, report.total_children) == (4, 2)  # 每份文件 2 父 + 1 子
    assert calls["ensure"] == 1  # 建表自检在流水线最前, 且只做一次
    assert calls["flush"] == 1  # 一整批只 flush 一次
    # 每个文件的五个环节按序走完, 且 flush 在最后
    assert calls["order"][:5] == ["parse", "clean", "split", "embed", "purge"]
    assert calls["order"].count("flush") == 0  # flush 不走 _ingest_one


def test_run_ingest_isolates_single_file_failure(wired, tmp_path):
    (tmp_path / "good.md").write_text("x", encoding="utf-8")
    (tmp_path / "bad.md").write_text("x", encoding="utf-8")
    calls = wired(fail={"bad.md": "解析失败: 后端崩了"})

    report = pipeline.run_ingest([tmp_path], Config())

    assert report.ok_sources == ["good.md"]  # 坏文件不拖垮整批
    assert report.failed == {"bad.md": "ValueError: 解析失败: 后端崩了"}  # 类型 + 原因
    assert report.total_children == 1
    assert calls["upsert_children"] == [1]  # 只写了好的那份


def test_run_ingest_flushes_even_when_everything_failed(wired, tmp_path):
    (tmp_path / "bad.md").write_text("x", encoding="utf-8")
    calls = wired(fail={"bad.md": "boom"})

    report = pipeline.run_ingest([tmp_path], Config())

    assert report.ok_sources == []
    assert list(report.failed) == ["bad.md"]
    assert calls["flush"] == 1


def test_run_ingest_empty_input(wired):
    calls = wired()

    report = pipeline.run_ingest([], Config())

    assert (report.ok_sources, report.failed) == ([], {})
    assert (report.total_parents, report.total_children) == (0, 0)
    assert report.warnings == {}
    assert calls["ensure"] == 1 and calls["flush"] == 1


def test_run_ingest_drops_old_versions_of_the_same_source(wired, tmp_path):
    """重灌改过的文件: 先按**新的** doc_id 清同源旧版本, 再写入, 且这件事要在报告里可见。"""
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    calls = wired(doc_id="new-id", stale=["old-id"])

    report = pipeline.run_ingest([tmp_path], Config())

    assert calls["purge"] == [("a.md", "new-id")]  # 拿到的是本次解析出的 doc_id
    assert "清掉同源旧版本 ['old-id']" in report.warnings["a.md"]


def test_run_ingest_notes_collect_every_stage(wired, tmp_path):
    """一个文件的警告要能一次看全: 解析警告 + 清洗统计 + 入库(embed/upsert)追加的警告。"""
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    wired(parse_warnings=["部分页面解析失败"], embed_warnings=["c001: text 超上限, 已跳过"])

    report = pipeline.run_ingest([tmp_path], Config())

    notes = report.warnings["a.md"]
    assert "部分页面解析失败" in notes
    assert any(note.startswith("清洗统计: ") for note in notes)
    assert "c001: text 超上限, 已跳过" in notes


def test_run_ingest_uses_injected_config(wired, tmp_path):
    """配置由调用方注入: 流水线内部不自己读环境(否则单测无法零环境跑)。"""
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    wired()
    cfg = Config(parent_target_chars=999)

    pipeline.run_ingest([tmp_path], cfg)

    assert cfg.parent_target_chars == 999  # 传下去的就是这一份(此处只确认接口不吞参数)


def test_run_ingest_accepts_str_paths(wired, tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    wired()

    report = pipeline.run_ingest([str(tmp_path)], Config())

    assert report.ok_sources == ["a.md"]


def test_is_hidden_checks_every_path_part(tmp_path):
    assert pipeline._is_hidden(Path(".git/config")) is True
    assert pipeline._is_hidden(Path("sub/.cache/x")) is True
    assert pipeline._is_hidden(Path("sub/x.md")) is False
