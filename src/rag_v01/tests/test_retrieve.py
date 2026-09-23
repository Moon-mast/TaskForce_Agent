"""模块 05 融合层单测(步骤 1: `rrf_fuse` 纯函数 + 两个契约类型)。

口径:
- `rrf_fuse` 是**纯函数**, 单测直接手算期望分(`1/(k+rank)` 的和)逐条断言, 不连 Milvus 也不用
  假的 store —— 这正是 05 篇 §2.3 选手写 RRF 而不是原生 `hybrid_search` 的收益之一(原生版要测
  融合逻辑必须起库);
- 并列分与可复现性是重点: `1/(k+rank)` 极易凑出同分, 少了 `chunk_id` 兜底键, 结果就依赖输入
  顺序, ragas 回归也就没法比(05 篇坑位 5)。
"""
from __future__ import annotations

from dataclasses import fields

import pytest

from rag_v01 import retrieve
from rag_v01.config import Config
from rag_v01.contracts import ChildHit, RetrievedChunk
from rag_v01.retrieve import rrf_fuse
from rag_v01.store import Hit, ParentRow

K = 60
TOP_K = 5


def hit(chunk_id: str, *, text: str | None = None, parent_id: str | None = None) -> Hit:
    """造一条 store 层的 Hit: 只填 rrf_fuse 会读的字段, 其余给默认值。"""
    return Hit(
        chunk_id=chunk_id,
        parent_id=parent_id if parent_id is not None else f"{chunk_id}:p001",
        text=text if text is not None else f"文本-{chunk_id}",
        chunk_type="text",
        score=0.5,
        metadata={"source": "a.md"},
    )


def fuse(dense, sparse, *, k: int = K, top_k: int = TOP_K):
    return rrf_fuse(dense, sparse, k=k, top_k=top_k)


# ---------------------------------------------------------------- 契约类型


def test_contract_field_names_are_frozen():
    """字段名是跨模块契约, 改名等于悄悄改协议: 在这里钉死(与 03 篇九字段同一套做法)。"""
    assert [f.name for f in fields(ChildHit)] == [
        "chunk_id",
        "text",
        "chunk_type",
        "dense_rank",
        "sparse_rank",
        "rrf_score",
    ]
    assert [f.name for f in fields(RetrievedChunk)] == [
        "parent_id",
        "parent_text",
        "rrf_rank",
        "hits",
        "doc_id",
        "source",
        "page_no",
        "heading_path",
        "metadata",
    ]


def test_retrieved_chunk_hits_list_is_mutable_in_place():
    """`hits` 要在同父合并时原地追加(05 篇 §4.2): frozen 只挡属性赋值, 不挡列表内容。"""
    row = RetrievedChunk(parent_id="p1", parent_text="全文", rrf_rank=1, hits=[])
    row.hits.append(ChildHit(chunk_id="c1", text="x", chunk_type="text", rrf_score=1 / 61))

    assert len(row.hits) == 1
    with pytest.raises(Exception):  # noqa: B017  (dataclasses.FrozenInstanceError)
        row.rrf_rank = 2  # type: ignore[misc]


# ---------------------------------------------------------------- 名次与分数


def test_ranks_are_one_based_and_scores_are_reciprocal():
    out = fuse([hit("c1"), hit("c2")], [])

    assert [(f.chunk_id, f.dense_rank, f.sparse_rank) for f in out] == [
        ("c1", 1, None),
        ("c2", 2, None),
    ]
    assert out[0].rrf_score == pytest.approx(round(1 / (K + 1), 6))
    assert out[1].rrf_score == pytest.approx(round(1 / (K + 2), 6))


def test_both_routes_add_votes():
    """05 篇 §五的手算例: dense 第 1 + sparse 第 3 → `1/61 + 1/63`。"""
    out = fuse([hit("c1")], [hit("x1"), hit("x2"), hit("c1")])

    # 单路命中的 x1/x2 也在候选里; 但叠加两票的 c1(0.03227) 稳稳压过它们(0.01613/0.01587)
    assert [f.chunk_id for f in out] == ["c1", "x1", "x2"]
    assert (out[0].dense_rank, out[0].sparse_rank) == (1, 3)
    assert out[0].rrf_score == pytest.approx(round(1 / 61 + 1 / 63, 6))


def test_same_chunk_on_both_routes_appears_once():
    out = fuse([hit("c1")], [hit("c1")])

    assert len(out) == 1
    assert (out[0].dense_rank, out[0].sparse_rank) == (1, 1)
    assert out[0].rrf_score == pytest.approx(round(2 / 61, 6))


def test_double_route_hit_outranks_single_route_top_hit():
    """融合的意义: 两路都靠前要胜过一路第一 —— 否则混合检索没有存在价值。"""
    dense = [hit("both"), hit("d_only")]
    sparse = [hit("s_only"), hit("both")]

    out = fuse(dense, sparse)

    assert out[0].chunk_id == "both"  # 1/61 + 1/62 胜过 1/61
    assert out[0].rrf_score > out[1].rrf_score


def test_single_route_empty_degenerates_to_that_route_order():
    """一路为空不是特例: 缺席那路贡献 0, 公式自己退化成该路名次排序(05 篇 §4.3)。"""
    sparse_only = [hit("s1"), hit("s2"), hit("s3")]

    out = fuse([], sparse_only)

    assert [f.chunk_id for f in out] == ["s1", "s2", "s3"]
    assert all(f.dense_rank is None and f.sparse_rank == i for i, f in enumerate(out, start=1))


def test_both_routes_empty_returns_empty():
    assert fuse([], []) == []


def test_top_k_truncates_after_fusion():
    dense = [hit(f"c{i}") for i in range(10)]

    out = fuse(dense, [], top_k=3)

    assert [f.chunk_id for f in out] == ["c0", "c1", "c2"]


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_top_k_raises(bad):
    with pytest.raises(ValueError, match="top_k"):
        fuse([hit("c1")], [], top_k=bad)


# ---------------------------------------------------------------- 并列与可复现


def test_ties_are_broken_by_chunk_id():
    """`dense 第1` 与 `sparse 第1` 同分, 顺序必须由内容决定, 不能由输入顺序决定。"""
    out = fuse([hit("bbb")], [hit("aaa")])

    assert [f.chunk_id for f in out] == ["aaa", "bbb"]
    assert out[0].rrf_score == out[1].rrf_score


def test_tie_order_is_independent_of_input_order():
    """同分候选在两路之间换来换去, **顺序与分数**必须一致(名次归属会变, 那是输入决定的)。

    这是 ragas 回归能比对的前提: 换个输入顺序就换排序, 前后两次评估没有可比性(坑位 5)。
    """
    a = fuse([hit("bbb")], [hit("aaa")])
    b = fuse([hit("aaa")], [hit("bbb")])

    assert [f.chunk_id for f in a] == [f.chunk_id for f in b] == ["aaa", "bbb"]
    assert [f.rrf_score for f in a] == [f.rrf_score for f in b]


def test_rounding_happens_after_sorting():
    """两个候选原始分只差极小值时, round 成同分也不该影响顺序(排序在 round 之前)。"""
    dense = [hit("first"), hit("second")]

    out = fuse(dense, [], k=1_000_000)  # k 极大 → 相邻名次的分差被压到 1e-12 量级

    assert [f.chunk_id for f in out] == ["first", "second"]
    assert out[0].rrf_score == out[1].rrf_score  # round(..., 6) 之后确实并列


# ---------------------------------------------------------------- 透传字段


def test_passthrough_fields_come_from_the_first_route_that_saw_the_chunk():
    """双路命中同一条时内容理应相同; `setdefault` 保证取**先见到**的那条而非被后一条覆盖。"""
    out = fuse([hit("c1", text="来自 dense")], [hit("c1", text="来自 sparse")])

    assert out[0].text == "来自 dense"

    out2 = fuse([hit("c1", text="来自 dense")], [hit("c1", parent_id="other:p002")])

    assert out2[0].parent_id == "c1:p001"


def test_passthrough_fields_are_carried():
    out = fuse([hit("c1", text="正文", parent_id="doc:p003")], [])

    assert (out[0].text, out[0].parent_id, out[0].chunk_type) == ("正文", "doc:p003", "text")


# ---------------------------------------------------------------- 步骤 2: search 编排


def parent_row(
    parent_id: str,
    *,
    text: str = "父块全文",
    page_no: int = 3,
    source: str = "a.md",
    heading_path: str = "一 > 二",
) -> ParentRow:
    return ParentRow(
        parent_id=parent_id,
        text=text,
        doc_id="doc-1",
        metadata={
            "source": source,
            "page_no": page_no,
            "heading_path": heading_path,
            "char_len": len(text),
        },
    )


@pytest.fixture
def wired(monkeypatch):
    """把 `search` 的四个依赖全换成假的: 不连 Milvus、不调 embedding。

    patch 的是 `retrieve.store.*`(模块属性)而不是 retrieve 里的名字 —— 所以 retrieve.py 必须写成
    `store.dense_search(...)` 才打得中, 这也是 05 篇 §4.2 定的单测姿势。
    """

    def _install(*, dense=(), sparse=(), parents=None, cfg=None):
        calls: dict[str, list] = {"embed": [], "dense": [], "sparse": [], "parents": []}
        dense_hits, sparse_hits = list(dense), list(sparse)
        parent_rows = {} if parents is None else dict(parents)
        config = cfg or Config()

        def fake_embed(query: str) -> list[float]:
            calls["embed"].append(query)
            return [0.1, 0.2]

        def fake_dense(vec, limit):
            calls["dense"].append({"vec": vec, "limit": limit})
            return list(dense_hits)

        def fake_sparse(query, limit):
            calls["sparse"].append({"query": query, "limit": limit})
            return list(sparse_hits)

        def fake_parents(ids):
            calls["parents"].append(list(ids))
            return {key: value for key, value in parent_rows.items() if key in set(ids)}

        monkeypatch.setattr(retrieve, "embed_query", fake_embed)
        monkeypatch.setattr(retrieve.store, "dense_search", fake_dense)
        monkeypatch.setattr(retrieve.store, "sparse_search", fake_sparse)
        monkeypatch.setattr(retrieve.store, "get_parents", fake_parents)
        monkeypatch.setattr(retrieve, "_cfg", lambda: config)
        return calls

    return _install


def test_search_wires_config_into_both_routes(wired):
    cfg = Config()  # candidate_top_n=20, rrf_k=60, retrieve_top_k=5
    calls = wired(
        dense=[hit("c1", parent_id="p1")],
        sparse=[hit("c2", parent_id="p2"), hit("c3", parent_id="p3")],
        parents={"p1": parent_row("p1"), "p2": parent_row("p2"), "p3": parent_row("p3")},
        cfg=cfg,
    )

    out = retrieve.search("向量数据库怎么用")

    assert calls["embed"] == ["向量数据库怎么用"]
    assert [c["limit"] for c in calls["dense"]] == [cfg.candidate_top_n]
    assert [c["limit"] for c in calls["sparse"]] == [cfg.candidate_top_n]
    assert calls["sparse"][0]["query"] == "向量数据库怎么用"  # 原文直传, 不自己分词
    assert all(c["vec"] == [0.1, 0.2] for c in calls["dense"])
    assert [row.parent_id for row in out] == ["p1", "p2", "p3"]  # 1/61 > 1/62 > 1/63
    assert out[0].hits[0].rrf_score == pytest.approx(round(1 / 61, 6))  # k 真的接到了 60


def test_search_merges_same_parent_and_keeps_hit_details(wired):
    dense = [
        hit("c1", parent_id="p1"),
        hit("c2", parent_id="p1"),
        hit("c3", parent_id="p1"),
    ]
    calls = wired(dense=dense, parents={"p1": parent_row("p1")})

    out = retrieve.search("q")

    assert len(out) == 1  # 3 个同父子块 → 1 条父块
    assert [h.chunk_id for h in out[0].hits] == ["c1", "c2", "c3"]  # 明细按融合名次追加
    assert out[0].rrf_rank == 1
    assert calls["parents"] == [["p1"]]  # 只查一次, 且 id 已去重


def test_rrf_rank_is_contiguous_after_merging(wired):
    """交替命中 p1/p2: 最终名次是连续 1..N(不是子块的融合槽位)。"""
    dense = [hit("c1", parent_id="p1"), hit("c2", parent_id="p2"), hit("c3", parent_id="p1")]
    wired(dense=dense, parents={"p1": parent_row("p1"), "p2": parent_row("p2")})

    out = retrieve.search("q")

    assert [(row.parent_id, row.rrf_rank, len(row.hits)) for row in out] == [
        ("p1", 1, 2),
        ("p2", 2, 1),
    ]


def test_parent_ids_are_deduplicated_in_one_lookup(wired):
    dense = [hit("c1", parent_id="p1"), hit("c2", parent_id="p1"), hit("c3", parent_id="p2")]
    calls = wired(dense=dense, parents={"p1": parent_row("p1"), "p2": parent_row("p2")})

    retrieve.search("q")

    assert calls["parents"] == [["p1", "p2"]]  # 保序去重: 没有 p1 出现两次


def test_both_routes_empty_returns_empty_without_parent_lookup(wired):
    calls = wired()

    assert retrieve.search("q") == []
    assert calls["parents"] == []


def test_single_route_empty_still_returns_results(wired):
    """一路为空不是特例: 公式退化成该路名次排序(05 篇 §4.3)。"""
    wired(sparse=[hit("c1", parent_id="p1")], parents={"p1": parent_row("p1")})

    out = retrieve.search("q")

    assert len(out) == 1
    assert out[0].hits[0].dense_rank is None
    assert out[0].hits[0].sparse_rank == 1


def test_search_combines_ranks_into_one_hit(wired):
    wired(
        dense=[hit("c1", parent_id="p1")],
        sparse=[hit("x1", parent_id="p9"), hit("c1", parent_id="p1")],
        parents={"p1": parent_row("p1"), "p9": parent_row("p9")},
    )

    out = retrieve.search("q")

    best = out[0]
    assert best.parent_id == "p1"
    assert (best.hits[0].dense_rank, best.hits[0].sparse_rank) == (1, 2)
    assert best.hits[0].rrf_score == pytest.approx(round(1 / 61 + 1 / 62, 6))


def test_missing_parent_is_skipped_and_warned(wired):
    """数据不一致(children 有、parents 没有)时跳过该条并留痕, 其余结果照常返回(坑位 6)。"""
    warnings: list[str] = []
    wired(
        dense=[hit("c1", parent_id="p1"), hit("c2", parent_id="gone")],
        parents={"p1": parent_row("p1")},
    )

    out = retrieve.search("q", warnings=warnings)

    assert [row.parent_id for row in out] == ["p1"]
    assert out[0].rrf_rank == 1
    assert len(warnings) == 1
    assert "gone" in warnings[0]


def test_missing_parent_is_silent_without_warnings_holder(wired):
    wired(dense=[hit("c1", parent_id="gone")], parents={})

    assert retrieve.search("q") == []


def test_page_no_sentinel_becomes_none_but_metadata_keeps_it(wired):
    """契约上"无页码"是 None; `-1` 是存储哨兵, 只在这一层还原。"""
    wired(dense=[hit("c1", parent_id="p1")], parents={"p1": parent_row("p1", page_no=-1)})

    out = retrieve.search("q")

    assert out[0].page_no is None
    assert out[0].metadata["page_no"] == -1  # 存储事实仍可见

    wired(dense=[hit("c2", parent_id="p2")], parents={"p2": parent_row("p2", page_no=7)})


def test_transparent_fields_come_from_the_parent_row(wired):
    wired(
        dense=[hit("c1", parent_id="p1")],
        parents={
            "p1": parent_row("p1", text="完整段落", source="b.docx", heading_path="二 > 三")
        },
    )

    out = retrieve.search("q")[0]

    assert (out.parent_text, out.source, out.heading_path, out.doc_id) == (
        "完整段落",
        "b.docx",
        "二 > 三",
        "doc-1",
    )


def test_top_k_override_limits_results(wired):
    wired(
        dense=[hit(f"c{i}", parent_id=f"p{i}") for i in range(3)],
        parents={f"p{i}": parent_row(f"p{i}") for i in range(3)},
    )

    out = retrieve.search("q", top_k=2)

    assert [row.parent_id for row in out] == ["p0", "p1"]


def test_non_positive_top_k_fails_before_embedding(wired):
    calls = wired()

    with pytest.raises(ValueError, match="top_k"):
        retrieve.search("q", top_k=0)

    assert calls["embed"] == []  # fail fast: 一次 embedding 调用都不该花


def test_infrastructure_error_is_not_swallowed_as_empty(monkeypatch):
    """基础设施故障必须炸出来: `[]` 的语义只留给"库里确实没有相关内容"(05 篇 §4.3)。"""
    monkeypatch.setattr(retrieve, "embed_query", lambda q: [0.1])
    monkeypatch.setattr(retrieve, "_cfg", lambda: Config())

    def boom(vec, limit):
        raise RuntimeError("Milvus 连不上")

    monkeypatch.setattr(retrieve.store, "dense_search", boom)

    with pytest.raises(RuntimeError, match="连不上"):
        retrieve.search("q")
