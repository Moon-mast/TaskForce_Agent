"""模块 02 清洗层单测: 五条规则逐条覆盖(纯内存构造 ParsedDoc, 不碰 docling)。

规则执行顺序在实现里固定为: 乱码 → 空白 → 页眉页脚 → 去重 → 超短行合并。
每个用例只针对一条规则造输入, 断言"该删的删了、不该动的没动"。

两个造数据的注意点(规则之间会互相影响):
1. 规则 (5) 会合并 < 30 字符的正文行 → 与合并无关的用例都用 ≥ 30 字符的正文块;
2. 规则 (4) 会对相似度 ≥ 0.95 的相邻条目动手 → 需要"多条不同正文"的用例
   一律用 `_BODIES` 里彼此差异明显的句子, 不要用同一模板套不同标签(那会被当重复删掉)。
"""
from __future__ import annotations

from rag_v01.clean import clean_document
from rag_v01.config import Config
from rag_v01.contracts import TYPE_TABLE, TYPE_TEXT, ParsedDoc, ParsedItem

# 五段彼此差异明显、长度都 ≥ 30 的正文, 供"不该被去重/合并"的用例复用
_BODIES = [
    "文档解析阶段最关键的是保住结构树, 打平成 Markdown 再正则切会丢掉标题层级与页码。",
    "表格序列化选择 Markdown 而不是 CSV, 是为了让表头随身走, 检索时表头词能被精确命中。",
    "图片落盘按 doc_id 分目录, 同内容重跑会覆盖同名文件, 因此不会留下上一版的旧图。",
    "页眉页脚判定要求跨足够多的不同页码重复出现, 命中后整组剔除, 否则会留下残句。",
    "噪声处理的顺序不能随便换, 先归一化再比对, 剔除必须早于短行合并, 否则删不干净。",
]


def _item(
    text: str,
    type_: str = TYPE_TEXT,
    *,
    page: int | None = None,
    level: int | None = None,
    heading_path: str = "",
) -> ParsedItem:
    return ParsedItem(type_, text, heading_path, page, level)


def _doc(items: list[ParsedItem], meta: dict | None = None) -> ParsedDoc:
    return ParsedDoc(doc_id="d" * 16, source="sample.md", items=items, meta=meta or {})


# ---------------------------------------------------- 规则 (1) 乱码/不可见字符

def test_rule1_strips_invisible_and_cid_residue() -> None:
    doc = _doc([_item("正常\u200b文本\ufeff带软连\u00ad字符与(cid:12)残留")])
    out, stats = clean_document(doc, Config())
    assert out.items[0].text == "正常文本带软连字符与残留"
    assert stats["garbled"] >= 10          # \u200b \ufeff \u00ad 各 1 + "(cid:12)" 8


def test_rule1_warns_instead_of_silently_fixing() -> None:
    """解码级问题清洗救不了: 只记 warning, 不假装修好。"""
    out, _ = clean_document(_doc([_item("x\ufffd\ufffd\ufffd\ufffd")]), Config())
    assert any("OCR" in w for w in out.meta["warnings"])

    out2, _ = clean_document(_doc([_item("\ue001\ue002\ue003正文")]), Config())
    assert any("字体映射失败" in w for w in out2.meta["warnings"])
    assert out2.items[0].text == "正文"     # 私用区字符被清掉


# ------------------------------------------------------------ 规则 (2) 空白规整

def test_rule2_normalizes_whitespace() -> None:
    # 每条给不同的 heading_path: 规则 (5) 只合并同 heading_path 的相邻短行,
    # 不然这三条短文本会被并成一条, 就测不到规则 (2) 了
    doc = _doc([
        _item("a\r\n\r\n\r\n\r\nb", heading_path="A"),                # 3+ 空行 → 压成一个空行
        _item("  首尾要去掉  ", heading_path="B"),
        _item("全角\u3000空格与\u00a0NBSP   多空格", heading_path="C"),  # 全角/NBSP 归一 + 空格压缩
    ])
    out, _ = clean_document(doc, Config())
    assert [it.text for it in out.items] == [
        "a\n\nb",
        "首尾要去掉",
        "全角 空格与 NBSP 多空格",
    ]


def test_rule2_keeps_table_row_structure() -> None:
    """表格: 空行丢掉(否则切断 Markdown 表), 杂散行并回上一行, 行内多空格压缩。"""
    table = "| 名称 | 数量 |\n\n|---|---|\n残行\n| 苹果 |  3 |"
    out, _ = clean_document(_doc([_item(table, TYPE_TABLE)]), Config())
    assert out.items[0].text == "| 名称 | 数量 |\n|---|---| 残行\n| 苹果 | 3 |"


# -------------------------------------------------------- 规则 (3) 页眉页脚剔除

def test_rule3_drops_cross_page_header_footer() -> None:
    items: list[ParsedItem] = []
    for page, body in enumerate(_BODIES, start=1):     # 5 页 → 阈值 max(3, ceil(0.6 * 5)) = 3
        items.append(_item(f"XX 公司内部资料 第 {page} 页", page=page))
        items.append(_item(body, page=page))            # 正文不该被删
    out, stats = clean_document(_doc(items), Config())
    assert stats["dropped_hf"] == 5
    assert all("内部资料" not in it.text for it in out.items)
    assert len(out.items) == 5


def test_rule3_skips_headings_and_pageless_docs() -> None:
    headings = [_item("XX 公司内部资料", page=p, level=1) for p in range(1, 6)]
    out, stats = clean_document(_doc(headings), Config())
    assert stats["dropped_hf"] == 0 and len(out.items) == 5      # 标题豁免

    flat = [_item("逐页重复的短句") for _ in range(5)]             # 页码全 None(docx/md)
    _, stats2 = clean_document(_doc(flat), Config())
    assert stats2["dropped_hf"] == 0


# ------------------------------------------------------------ 规则 (4) 相邻去重

def test_rule4_dedups_adjacent_repeats() -> None:
    base = "这是 PDF 渲染抖动导致的重复条目, 内容写得足够长, 以便让相似度分支的比值足够高"
    variant = base[:-1] + "!"                           # 只差最后一个字符 → 相似度 ≥ 0.95
    doc = _doc([_item(base), _item(base), _item(variant), _item(_BODIES[0])])
    out, stats = clean_document(doc, Config())
    assert stats["deduped"] == 2
    assert [it.text for it in out.items] == [base, _BODIES[0]]


def test_rule4_keeps_same_named_headings_and_far_apart_repeats() -> None:
    doc = _doc([
        _item("第一章 概述", level=1),
        _item("第一章 概述", level=1),                    # 同名标题: 目录与正文各一处, 都留
        _item(_BODIES[0]),
        _item(_BODIES[1]),
        _item(_BODIES[2]),
        _item(_BODIES[0]),                               # 相隔 3 条 → 不算相邻重复
    ])
    _, stats = clean_document(doc, Config())
    assert stats["deduped"] == 0


# -------------------------------------------------------- 规则 (5) 超短行合并

def test_rule5_merges_short_lines() -> None:
    doc = _doc([
        _item("这是第一行正文, 长度足够不会被合并。"),
        _item("续行"),                                   # 中文断行: 直接拼, 不补空格
        _item("Second line of English text here"),
        _item("next"),                                   # 英文断行: 补一个空格
    ])
    out, stats = clean_document(doc, Config())
    assert [it.text for it in out.items] == [
        "这是第一行正文, 长度足够不会被合并。续行",
        "Second line of English text here next",
    ]
    assert stats["merged_short"] == 2


def test_rule5_skips_headings_and_different_heading_paths() -> None:
    doc = _doc([
        _item("小标题", level=2, heading_path="顶层"),
        _item("短"),                                      # 上一条是标题 → 不并
        _item(_BODIES[0], heading_path="第一章"),
        _item("第二节的短行", heading_path="第二章"),        # heading_path 不同 → 不并
    ])
    out, stats = clean_document(doc, Config())
    texts = [it.text for it in out.items]
    assert "短" in texts and "第二节的短行" in texts
    assert stats["merged_short"] == 0


def test_rule5_drops_bare_page_number_fragments() -> None:
    doc = _doc([
        _item(_BODIES[0]),
        _item("12"),                                      # 长度 ≤ 4 的纯数字行 → 删
        _item(_BODIES[1]),
    ])
    out, stats = clean_document(doc, Config())
    assert stats["dropped_pagenum"] == 1
    assert [it.text for it in out.items] == [_BODIES[0], _BODIES[1]]


# ------------------------------------------------------------------ 纯函数性质

def test_clean_is_pure_and_preserves_parse_warnings() -> None:
    doc = _doc(
        [_item("  a\r\n\r\n\r\nb  ")],
        meta={"warnings": ["部分页面解析失败"], "status": "PARTIAL_SUCCESS"},
    )
    before = doc.items[0].text
    out, _ = clean_document(doc, Config())
    assert doc.items[0].text == before                    # 输入没被就地改(frozen dataclass)
    assert out.items[0].text == "a\n\nb"
    assert out.meta["warnings"] == ["部分页面解析失败"]      # 解析期警告保留
    assert out.meta["status"] == "PARTIAL_SUCCESS"        # 其余 meta 原样透传
