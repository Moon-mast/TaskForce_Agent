"""里程碑 B 步骤 2 单测:kb_search 工厂适配层(make_kb_search 注入假后端,不连 Milvus)。

假件直接用 `rag_v01.contracts` 的真类型构造 —— 这样契约字段一旦改名,这里立刻红,
比手搓 dict 更早暴露"适配层读错了字段"。
c6 起注入点是工厂参数,测试不再打桩模块内部。
"""

import json

from rag_v01.contracts import ChildHit, RetrievedChunk
from tools.rag.kb_search import SNIPPET_LIMIT, hit_key, make_kb_search


def _row(content="父块正文" * 3, child="命中子块", *, doc_id="d1", source="a.md",
         seq=7, score=0.031):
    """造一条 RetrievedChunk:字段名与契约一致,故意让 seq 体现在 chunk_id 里。"""
    return RetrievedChunk(
        parent_id=f"{doc_id}:p001",
        parent_text=content,
        rrf_rank=1,
        hits=[
            ChildHit(
                chunk_id=f"{doc_id}:p001:c{seq:03d}",
                text=child,
                chunk_type="text",
                dense_rank=1,
                sparse_rank=2,
                rrf_score=score,
            )
        ],
        doc_id=doc_id,
        source=source,
    )


def _tool(rows, calls):
    """工厂注入假后端:记录调用参数,返回给定命中。"""
    def fake(query, top_k):
        calls.append((query, top_k))
        return rows[:top_k]

    return make_kb_search(fake)


def test_tool_name_stays_kb_search():
    """工具名必须是 kb_search —— 提示词 prompts/subagents/retriever.md 写死了这个名字。"""
    assert make_kb_search().name == "kb_search"
    assert _tool([], []).name == "kb_search"  # 注入后端也不改名


def test_contract_keys_and_mapping():
    """五个键一个不少,且各自来自新契约的哪个字段。"""
    calls: list = []
    tool = _tool([_row()], calls)
    hits = json.loads(tool.invoke({"query": "昆玉河"}))
    assert len(hits) == 1
    h = hits[0]
    assert set(h) == {"doc_id", "filename", "seq", "content", "score"}
    assert h["doc_id"] == "d1"           # ← RetrievedChunk.doc_id
    assert h["filename"] == "a.md"       # ← RetrievedChunk.source
    assert h["seq"] == 7                 # ← chunk_id 的 :c007
    assert h["score"] == 0.031           # ← hits[0].rrf_score
    assert h["content"].startswith("父块正文")  # ← 父块在前


def test_content_is_parent_plus_best_child():
    """content = 父块全文 + 分隔线 + 最佳子块(旧工具只回切片,这是升级点)。"""
    tool = _tool([_row(content="父块全文", child="最佳子块")], [])
    h = json.loads(tool.invoke({"query": "q"}))[0]
    assert h["content"] == "父块全文\n---\n最佳子块"


def test_content_truncated_to_snippet_limit():
    tool = _tool([_row(content="长" * 1000, child="也长" * 100)], [])
    h = json.loads(tool.invoke({"query": "q"}))[0]
    assert len(h["content"]) == SNIPPET_LIMIT


def test_empty_result_is_empty_array():
    tool = _tool([], [])
    assert tool.invoke({"query": "没有的东西"}) == "[]"


def test_top_k_clamped_high_and_low():
    """异常 top_k 钳到 1-10:大值防拖垮检索,0/负值防"不检索"(0 走 or 5 的兜底)。"""
    rows = [_row(doc_id=f"d{i}", seq=i) for i in range(20)]
    calls: list = []
    tool = _tool(rows, calls)
    assert len(json.loads(tool.invoke({"query": "q", "top_k": 99}))) == 10
    assert calls[-1] == ("q", 10)
    assert len(json.loads(tool.invoke({"query": "q", "top_k": -3}))) == 1
    assert calls[-1] == ("q", 1)
    tool.invoke({"query": "q", "top_k": 0})
    assert calls[-1] == ("q", 5)


def test_seq_falls_back_to_zero_on_odd_chunk_id():
    """chunk_id 不合约定时 seq 回 0 而不是 None —— 去重键 (doc_id, seq) 不能出现 null。"""
    row = _row()
    row.hits[0] = ChildHit(chunk_id="没有编号", text="x", chunk_type="text")
    assert json.loads(_tool([row], []).invoke({"query": "q"}))[0]["seq"] == 0


def test_hit_key_is_dedup_key_single_source():
    """c6:去重键唯一出处 —— 消费方(_collect_hits)与生产方(seq 写出)对齐的是同一个函数。"""
    assert hit_key({"doc_id": "d1", "seq": 7}) == ("d1", 7)
    assert hit_key({"doc_id": "d1"}) == ("d1", None)  # 缺键显式出 None,而不是各自 get 默认值漂移


def test_default_factory_delegates_to_rag_v01_facade(monkeypatch):
    """不注入后端时,默认检索必须转调 rag_v01 的 facade,并把钳制后的 top_k 传下去。

    这里替掉的是 `rag_v01.search`(包级 facade),顺带证明适配层走的是公开入口而不是内部模块。
    """
    seen = []
    monkeypatch.setattr(
        "rag_v01.search", lambda query, top_k=None: seen.append((query, top_k)) or []
    )
    make_kb_search().invoke({"query": "玉渊潭", "top_k": 3})
    assert seen == [("玉渊潭", 3)]
