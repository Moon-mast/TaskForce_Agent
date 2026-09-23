"""模块 03 切块层单测: 纯函数, 不碰 docling / 网络 / Milvus —— 内存造 ParsedDoc 即可。

两个刻意的测试口径:
1. 父块目标带由 `cfg.parent_target_chars` 按 ±20% 推出来 —— 多数用例把目标调到 100
   (上限 120), 于是"跨节闭合 / 超上限闭合 / 退化切分"都能用几十个字符的假文本测到,
   不必写一屏长文;
2. `created_at` 走注入的假时钟: 除了它, 切两遍必须逐字段相同(§五确定性验收)。
"""
from __future__ import annotations

import itertools
from dataclasses import fields, replace

from rag_v01.chunk import split
from rag_v01.config import Config
from rag_v01.contracts import (
    TYPE_IMAGE,
    TYPE_TABLE,
    TYPE_TEXT,
    ChildChunk,
    ParentChunk,
    ParsedDoc,
    ParsedItem,
)

DOC_ID = "0123456789abcdef"
FIXED_TIME = "2026-09-21T00:00:00+00:00"
NINE_FIELDS = [
    "chunk_id",
    "text",
    "parent_id",
    "doc_id",
    "source",
    "page_no",
    "heading_path",
    "chunk_type",
    "char_len",
    "created_at",
    "image_path",          # 透传字段(非九字段之一)
]

TABLE_MD = "| 项目 | 值 |\n|---|---|\n| 维度 | 1024 |\n| 索引 | HNSW |"


# ---------------------------------------------------------------- 造数据

def _cfg(**overrides: int) -> Config:
    """默认把父块目标压到 100(带 80~120), 便于用短文本触发各条规则。"""
    base: dict[str, int] = {"parent_target_chars": 100, "child_target_chars": 450}
    base.update(overrides)
    return Config(**base)


def _heading(text: str, level: int, heading_path: str = "", page: int | None = None) -> ParsedItem:
    return ParsedItem(TYPE_TEXT, text, heading_path, page, level)


def _text(text: str, heading_path: str = "", page: int | None = None) -> ParsedItem:
    return ParsedItem(TYPE_TEXT, text, heading_path, page)


def _table(md: str, heading_path: str = "") -> ParsedItem:
    return ParsedItem(TYPE_TABLE, md, heading_path)


def _image(placeholder: str, image_path: str | None = None, heading_path: str = "") -> ParsedItem:
    return ParsedItem(TYPE_IMAGE, placeholder, heading_path, None, None, image_path)


def _doc(items: list[ParsedItem], source: str = "a.md") -> ParsedDoc:
    return ParsedDoc(doc_id=DOC_ID, source=source, items=items)


def _run(doc: ParsedDoc, cfg: Config | None = None) -> tuple[list[ParentChunk], list[ChildChunk]]:
    return split(doc, cfg or _cfg(), clock=lambda: FIXED_TIME)


# ---------------------------------------------------------------- 父块: 三条规则

def test_parent_closes_on_same_level_heading() -> None:
    """规则 A: 同级标题 → 闭合, 新块从新标题起。父块绝不跨节。"""
    parents, _ = _run(_doc([
        _heading("第一章 总览", 1),
        _text("甲" * 40, heading_path="第一章 总览"),
        _heading("第二章 细节", 1),
        _text("乙" * 40, heading_path="第二章 细节"),
    ]))
    assert len(parents) == 2
    assert parents[0].heading_path == "第一章 总览"       # 开块标题自身也进面包屑
    assert parents[1].heading_path == "第二章 细节"
    assert "第二章" not in parents[0].text


def test_parent_absorbs_deeper_heading() -> None:
    """更深标题属于"同一父块的一段", 不闭合 —— 这正是父块"大而全"的来源。"""
    parents, _ = _run(_doc([
        _heading("第一章", 1),
        _text("甲" * 40, heading_path="第一章"),
        _heading("1.1 小节", 2, heading_path="第一章"),
        _text("乙" * 40, heading_path="第一章 > 1.1 小节"),
    ]))
    assert len(parents) == 1
    assert "1.1 小节" in parents[0].text                    # 标题进正文(BM25 只索引 text)
    assert parents[0].heading_path == "第一章"


def test_parent_closes_before_exceeding_max() -> None:
    """规则 B: 再加一个单元就超上限 → 先闭合再装。"""
    parents, _ = _run(_doc([
        _heading("S", 1),
        _text("x" * 80, heading_path="S"),
        _text("y" * 80, heading_path="S"),
    ]))
    assert [p.char_len for p in parents] == [83, 80]        # 83 = 标题 1 + 拼接符 2 + 正文 80
    assert parents[1].heading_path == "S"                   # 续块用单元自带的面包屑


def test_parent_keeps_last_buffer() -> None:
    """规则 C: 收尾把残余 buffer 成块(宁小勿丢)。"""
    parents, _ = _run(_doc([_text("只有一段正文")]))
    assert len(parents) == 1
    assert parents[0].page_no is None                       # md/docx 无页码
    assert parents[0].parent_id == ""                       # 父块自己没有父亲


def test_text_before_first_heading_is_its_own_block() -> None:
    """导言出现在第一个标题之前: 见到标题就收口, 不被并进第一章。"""
    parents, _ = _run(_doc([
        _text("这是没有标题的导言文字。"),
        _heading("第一章", 1),
        _text("甲" * 40, heading_path="第一章"),
    ]))
    assert len(parents) == 2
    assert "导言" in parents[0].text and "甲" not in parents[0].text


def test_heading_joins_oversized_body() -> None:
    """规则 B 也会留下孤立标题块(标题后紧跟超长正文): 标题并入退化块的第一片。"""
    body = "".join(f"第{index}句说明该段的背景与适用范围, 内容需要足够长。" for index in range(60))
    parents, _ = _run(_doc([_heading("第1章 导言", 1), _text(body, heading_path="第1章 导言")]))
    assert len(parents) > 2
    assert all(p.char_len <= 120 for p in parents)          # 上限 120(目标 100)
    assert parents[0].heading_path == "第1章 导言"           # 标题留下当锚点, 不是空壳块
    assert "".join(p.text for p in parents) == f"第1章 导言\n\n{body}"


def test_empty_section_does_not_steal_next_heading() -> None:
    """空章节(标题后直接跟下一个标题)不并入下一节 —— 否则内容会顶着上一节的面包屑。"""
    parents, _ = _run(_doc([
        _heading("第一章", 1),
        _heading("第二章", 1),
        _text("甲" * 40, heading_path="第二章"),
    ]))
    assert [p.heading_path for p in parents] == ["第一章", "第二章"]


def test_empty_document_yields_nothing() -> None:
    parents, children = _run(_doc([]))
    assert parents == [] and children == []
    # 清洗后空掉的条目既不产块也不占序号
    parents, _ = _run(_doc([_text(""), _text("有内容的一段")]))
    assert [p.chunk_id for p in parents] == [f"{DOC_ID}:p001"]


def test_degenerate_long_body_is_split_into_blocks() -> None:
    """超长无标题正文: 没有标题边界可用 → 退化为 ~上限 的递归切分, 一条一个父块。"""
    body = "".join(f"第{index}句填充内容, 用来把这一段撑到远超上限的长度。" for index in range(20))
    parents, _ = _run(_doc([_text(body)]))
    assert len(parents) > 2
    assert max(p.char_len for p in parents) <= 120          # 退化块不超上限(验收线)
    assert "".join(p.text for p in parents) == body         # 一个字都没丢


# ---------------------------------------------------------------- 原子块: 表格与图片

def test_table_is_atomic_own_block() -> None:
    """表格独占父块、整块不切: 行列结构(含表头分隔行)必须原样保留, 宁超上限也要完整。"""
    big_table = "| 项目 | 值 |\n|---|---|\n" + "\n".join(
        f"| 字段{index} | 取值{index} |" for index in range(12)
    )
    parents, children = _run(_doc([
        _heading("表节", 1),
        _text("甲" * 40, heading_path="表节"),
        _table(big_table, heading_path="表节"),
        _text("乙" * 40, heading_path="表节"),
    ]))
    assert [p.chunk_type for p in parents] == [TYPE_TEXT, TYPE_TABLE, TYPE_TEXT]
    table_parent = parents[1]
    assert table_parent.text == big_table
    assert table_parent.char_len > 120                      # 允许超上限(宁整勿切)

    kids = [c for c in children if c.parent_id == table_parent.chunk_id]
    assert len(kids) == 1                                   # 整表恰好一个子块
    assert kids[0].chunk_type == TYPE_TABLE
    assert kids[0].chunk_id.endswith(":c001")
    assert kids[0].text == big_table
    assert "|---|" in kids[0].text                          # 表头分隔行还在


def test_heading_carries_into_atomic_block() -> None:
    """标题后面直接就是表格/图片: 标题随原子块走, 不留"正文只有一行标题"的空壳父块。"""
    parents, children = _run(_doc([
        _heading("一、表格", 1),
        _table(TABLE_MD, heading_path="一、表格"),
    ]))
    assert len(parents) == 1
    assert parents[0].chunk_type == TYPE_TABLE                  # 带标题仍按表格类型走
    assert parents[0].text == f"一、表格\n\n{TABLE_MD}"
    assert parents[0].heading_path == "一、表格"
    assert [c.chunk_type for c in children] == [TYPE_TABLE]     # 整表仍是恰好一个子块

    parents2, _ = _run(_doc([
        _heading("二、图片", 1),
        _image("图: 架构示意", "docs_ab12/p0012_i000.png", "二、图片"),
    ]))
    assert parents2[0].chunk_type == TYPE_IMAGE
    assert parents2[0].image_path == "docs_ab12/p0012_i000.png"


def test_image_block_keeps_path_and_makes_one_child() -> None:
    parents, children = _run(_doc([_image("图: 架构示意", "docs_ab12/p0012_i000.png")]))
    assert parents[0].chunk_type == TYPE_IMAGE
    assert parents[0].image_path == "docs_ab12/p0012_i000.png"
    assert len(children) == 1
    assert children[0].chunk_type == TYPE_IMAGE
    assert children[0].image_path == parents[0].image_path  # 04 篇靠它读图做多模态嵌入


# ---------------------------------------------------------------- 子块: 递归切分与重叠

def test_children_are_split_and_capped() -> None:
    """子块目标 450: 父块 ~1200 字符应切出多块, 每块 ≤ 550(450 + 50 重叠 + 容差)。"""
    body = "".join(
        f"这是第{index}个句子, 用来把父块撑到需要切成多个子块的长度。" for index in range(60)
    )
    parents, children = _run(_doc([_text(body)]), _cfg(parent_target_chars=2000))
    assert len(parents) == 1
    assert len(children) >= 3
    assert all(c.char_len <= 550 for c in children)
    assert all(c.chunk_type == TYPE_TEXT for c in children)


def test_adjacent_children_overlap_by_design() -> None:
    """相邻子块重叠 40~50 字符(窗口 50, 对齐句界后仍 ≥ 40)。"""
    body = "".join(
        f"第{index}句的内容写得足够长, 好让重叠量落在验收区间里。" for index in range(60)
    )
    _, children = _run(_doc([_text(body)]), _cfg(parent_target_chars=2000))
    texts = [c.text for c in children]
    assert len(texts) >= 3
    for prev, nxt in itertools.pairwise(texts):
        assert 40 <= _overlap_len(prev, nxt) <= 50


def test_chinese_sentence_boundaries_are_respected() -> None:
    """坑位二: 闭引号跟在前句尾、小数不被切开、省略号整体成界。"""
    body = (
        "第一句先说明背景与范围, 内容需要足够长才能触发多次切分, 此处继续填充。"
        "他喊道:“就这样办。”然后转身离开房间, 又补了一句说明情况再走。"
        "圆周率取 3.14 参与计算, 但这个数字内部不该出现切分点。"
        "省略号…… 后面接着更多内容, 继续填充到长于目标长度以便观察。"
    )
    _, children = _run(_doc([_text(body)]), _cfg(child_target_chars=30))
    texts = [c.text for c in children]
    assert len(texts) >= 4
    assert any("3.14" in t for t in texts)                  # 小数整体落在同一块里
    for text in texts:
        assert text[0] not in "。！？；」』”…"                # 不以句界/闭引号开头
        assert text[-1] not in "“「『"                        # 不以开引号结尾


def test_single_small_paragraph_is_one_child() -> None:
    parents, children = _run(_doc([_text("很短的正文")]))
    assert len(children) == 1
    assert children[0].text == parents[0].text


# ---------------------------------------------------------------- ID 与元信息

def test_metadata_field_names_are_frozen() -> None:
    """九字段命名是全局定稿: 增删改名都会破坏 04/05 篇的落库与融合, 单测钉死。"""
    assert [f.name for f in fields(ChildChunk)] == NINE_FIELDS
    assert [f.name for f in fields(ParentChunk)] == NINE_FIELDS
    assert ChildChunk(chunk_id="x", text="三个字").char_len == 3     # char_len 由 text 派生
    assert ChildChunk(chunk_id="x", text="abc", char_len=99).char_len == 3


def test_ids_are_deterministic_and_linked() -> None:
    parents, children = _run(_doc([
        _heading("第一章", 1),
        _text("甲" * 40, heading_path="第一章"),
        _table(TABLE_MD, heading_path="第一章"),
        _text("乙" * 40, heading_path="第一章"),
    ]))
    assert [p.chunk_id for p in parents] == [f"{DOC_ID}:p{i:03d}" for i in (1, 2, 3)]
    assert [p.parent_id for p in parents] == ["", "", ""]   # 父块自身的 parent_id 为空

    ids = {p.chunk_id for p in parents}
    by_parent: dict[str, list[str]] = {}
    for child in children:
        assert child.parent_id in ids                       # 每个子块都能回溯到现存父块
        assert child.chunk_id.startswith(f"{child.parent_id}:c")
        by_parent.setdefault(child.parent_id, []).append(child.chunk_id)
    for parent_id, child_ids in by_parent.items():
        assert child_ids == [f"{parent_id}:c{i:03d}" for i in range(1, len(child_ids) + 1)]


def test_split_is_deterministic_except_created_at() -> None:
    doc = _doc([_heading("第一章", 1), _text("甲" * 40, heading_path="第一章"), _table(TABLE_MD)])
    first = _run(doc)
    second = split(doc, _cfg(), clock=lambda: "2099-01-01T00:00:00+00:00")
    strip = lambda chunks: [replace(c, created_at="") for c in chunks]      # noqa: E731
    assert strip(first[0]) == strip(second[0])
    assert strip(first[1]) == strip(second[1])
    assert {c.created_at for c in first[0] + first[1]} == {FIXED_TIME}      # 一次切块一个时间戳


def test_metadata_is_carried_through() -> None:
    parents, children = _run(_doc([
        _heading("第一章", 1, page=3),
        _text("甲" * 40, heading_path="第一章", page=3),
    ], source="corpus/a.pdf"))
    parent = parents[0]
    assert (parent.doc_id, parent.source, parent.page_no) == (DOC_ID, "corpus/a.pdf", 3)
    assert parent.heading_path == "第一章" and parent.chunk_type == TYPE_TEXT
    assert parent.char_len == len(parent.text) and parent.created_at == FIXED_TIME
    assert children[0].doc_id == DOC_ID and children[0].page_no == 3


def test_child_page_follows_its_own_unit() -> None:
    """父块横跨两页时, 子块页码要跟各自的内容单元走 —— 否则第 4 页命中会标成第 3 页。"""
    _, children = _run(
        _doc([
            _heading("第一章", 1, page=3),
            _text("甲" * 200, heading_path="第一章", page=3),
            _text("乙" * 200, heading_path="第一章", page=4),
        ]),
        _cfg(parent_target_chars=2000, child_target_chars=200),
    )
    assert children[0].page_no == 3
    assert children[-1].page_no == 4


# ---------------------------------------------------------------- 小工具

def _overlap_len(prev: str, nxt: str, cap: int = 60) -> int:
    """`nxt` 开头与 `prev` 尾部实际重合的字符数(0 = 没有重叠)。"""
    for length in range(min(cap, len(prev), len(nxt)), 0, -1):
        if prev.endswith(nxt[:length]):
            return length
    return 0
