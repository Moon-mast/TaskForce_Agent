"""模块 02 解析层单测: 不联网、不下载模型。

三层覆盖:
1. 契约与纯规则 —— `doc_id` 内容寻址、后缀白名单、异常文案、占位文本/图注小工具;
2. 遍历算法 —— 用鸭子类型的假 item 测标题栈与条目映射(完全不碰 docling);
3. 真格式冒烟 —— 真跑 docling 解析样例 md / docx(这两个后端都不需要下载模型)。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel
from PIL import Image

from rag_v01.config import Config
from rag_v01.contracts import TYPE_IMAGE, TYPE_TABLE, TYPE_TEXT
from rag_v01.parsers import ParseError, parse_document
from rag_v01.parsers.common import compute_doc_id, walk
from rag_v01.parsers.images import caption_nearby, placeholder_text

# ---------------------------------------------------------------- 假 item / 假 doc
# walk() 只用到 item 的这几个属性, 所以用鸭子类型的假对象就够 —— 遍历算法的单测
# 因此完全不依赖 docling 的解析结果(也不受后端版本漂移影响)。

class _FakeLayer:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeProv:
    def __init__(self, page_no: int | None) -> None:
        self.page_no = page_no


class _FakeItem:
    def __init__(self, label, text="", level=None, page=0, layer=None, table_md=None) -> None:
        self.label = label
        self.text = text
        self.level = level
        self.prov = [_FakeProv(page)] if page is not None else []   # prov 空 = 无页码(docx/md)
        self.content_layer = _FakeLayer(layer) if layer else None
        self._table_md = table_md

    def export_to_markdown(self, doc):        # noqa: ARG002 - 签名对齐 docling 的 TableItem
        return self._table_md or ""


class _FakeDoc:
    def __init__(self, items) -> None:
        self._items = items

    def iterate_items(self):
        return [(it, 0) for it in self._items]


def _walk(items):
    return walk(_FakeDoc(items), doc_id="d" * 16, cfg=Config(), local_images=False)


# ---------------------------------------------------------- 1. 契约与纯规则

def test_doc_id_is_content_addressed(tmp_path: Path) -> None:
    """同内容不同路径 → 同 doc_id(重跑幂等); 内容变 → id 变。"""
    (tmp_path / "sub").mkdir()
    a, b = tmp_path / "a.md", tmp_path / "sub" / "b.md"
    a.write_text("同样的内容", encoding="utf-8")
    b.write_text("同样的内容", encoding="utf-8")
    assert compute_doc_id(a) == compute_doc_id(b)
    c = tmp_path / "c.md"
    c.write_text("不同内容", encoding="utf-8")
    assert compute_doc_id(c) != compute_doc_id(a)


def test_rejects_doc_unknown_suffix_and_missing(tmp_path: Path) -> None:
    old = tmp_path / "old.doc"
    old.write_bytes(b"x")
    with pytest.raises(ParseError, match="另存为 .docx"):
        parse_document(old)
    txt = tmp_path / "a.txt"
    txt.write_text("x", encoding="utf-8")
    with pytest.raises(ParseError, match="不支持的后缀"):
        parse_document(txt)
    with pytest.raises(ParseError, match="不存在"):
        parse_document(tmp_path / "missing.md")


def test_placeholder_and_caption_helpers() -> None:
    expected = "[图片 2: doc/p0001_i002.png; 图注: 无]"
    assert placeholder_text(2, "doc/p0001_i002.png", None) == expected
    # 没有本地图/没有引用时也要给出可读占位, 不能把 None 拼进文本
    assert placeholder_text(1, None, "架构图") == "[图片 1: 无本地图; 图注: 架构图]"

    raw = [(_FakeItem(DocItemLabel.PICTURE), 0), (_FakeItem(DocItemLabel.TEXT, "图注文本"), 0)]
    assert caption_nearby(raw, 0) == "图注文本"
    assert caption_nearby([(_FakeItem(DocItemLabel.PICTURE), 0)], 0) is None


# ------------------------------------------------------------ 2. 遍历算法

def test_walk_heading_stack_builds_heading_path() -> None:
    """标题栈: 新标题弹掉同级与更深层级, heading_path 只含上级、不含自身。"""
    items = [
        _FakeItem(DocItemLabel.TITLE, "文档标题"),
        _FakeItem(DocItemLabel.TEXT, "正文一"),
        _FakeItem(DocItemLabel.SECTION_HEADER, "第一章", level=1),
        _FakeItem(DocItemLabel.SECTION_HEADER, "1.1 小节", level=2),
        _FakeItem(DocItemLabel.TEXT, "正文二"),
        _FakeItem(DocItemLabel.SECTION_HEADER, "第二章", level=1),
        _FakeItem(DocItemLabel.TEXT, "正文三"),
    ]
    out, warnings = _walk(items)
    assert [i.heading_path for i in out] == [
        "",
        "文档标题",
        "文档标题",
        "文档标题 > 第一章",
        "文档标题 > 第一章 > 1.1 小节",   # 二级标题之后的正文: 面包屑带全上级链
        "文档标题",                       # 回到一级标题: "1.1 小节" 已被弹栈
        "文档标题 > 第二章",
    ]
    assert [i.level for i in out] == [0, None, 1, 2, None, 1, None]
    assert out[0].is_heading and not out[1].is_heading
    assert warnings == []


def test_walk_drops_furniture_and_header_footer_labels() -> None:
    """页眉页脚踏两道判据: content_layer=FURNITURE(docling 判的) 与显式 label(兜底)。"""
    items = [
        _FakeItem(DocItemLabel.TEXT, "正文被误标", page=1, layer="FURNITURE"),
        _FakeItem(DocItemLabel.PAGE_HEADER, "页眉", page=1),
        _FakeItem(DocItemLabel.PAGE_FOOTER, "页脚", page=2),
        _FakeItem(DocItemLabel.FOOTNOTE, "脚注", page=2),
        _FakeItem(DocItemLabel.TEXT, "真正文", page=2),
    ]
    out, _ = _walk(items)
    assert [i.text for i in out] == ["真正文"]


def test_walk_table_is_single_item_and_picture_gets_caption() -> None:
    items = [
        _FakeItem(DocItemLabel.TABLE, table_md="| a | b |\n|---|---|\n| 1 | 2 |", page=3),
        _FakeItem(DocItemLabel.PICTURE, page=3),
        _FakeItem(DocItemLabel.TEXT, "图一 架构", page=3),
        _FakeItem(DocItemLabel.TEXT, "   ", page=3),        # 空白条目: 不进 items
    ]
    out, _ = _walk(items)
    assert [i.type for i in out] == [TYPE_TABLE, TYPE_IMAGE, TYPE_TEXT]
    assert out[0].text.startswith("| a | b |") and out[0].page_no == 3
    assert out[1].text == "[图片 1: 无本地图; 图注: 图一 架构]"    # 图注取自相邻 TEXT 条目
    assert out[1].page_no == 3 and out[2].text == "图一 架构"


def test_walk_table_falls_back_to_grid_when_serializer_fails() -> None:
    """序列化失败时用 table_cells 手拼 —— 至少要保住行结构与表头分隔行。"""

    class _Cell:
        def __init__(self, text, row, col) -> None:
            self.text = text
            self.start_row_offset_idx = row
            self.start_col_offset_idx = col

    class _Data:
        table_cells = [
            _Cell("名称", 0, 0), _Cell("数量", 0, 1),
            _Cell("苹果", 1, 0), _Cell("3", 1, 1),
        ]

    class _BrokenTable(_FakeItem):
        def __init__(self) -> None:
            super().__init__(DocItemLabel.TABLE)
            self.data = _Data()

        def export_to_markdown(self, doc):
            raise RuntimeError("boom")

    out, _ = _walk([_BrokenTable()])
    assert out[0].text.splitlines() == ["| 名称 | 数量 |", "| --- | --- |", "| 苹果 | 3 |"]


# ------------------------------------------------- 3. 真格式冒烟(无模型下载)

def test_parse_md_sample(samples_dir: Path, tmp_path: Path) -> None:
    doc = parse_document(samples_dir / "sample.md", Config(image_dir=str(tmp_path / "imgs")))
    assert doc.source == "sample.md" and len(doc.doc_id) == 16
    assert doc.items[0].is_heading and doc.items[0].level == 0     # md 的 # 是 TitleItem → 0 层

    table = next(i for i in doc.items if i.type == TYPE_TABLE)
    assert table.heading_path == "rag_0.1 样例文档 > 一、表格"
    assert "| 名称" in table.text and "苹果" in table.text and "---" in table.text

    image = next(i for i in doc.items if i.type == TYPE_IMAGE)
    assert "图注: 样例图的图注" in image.text
    # md 的图片是引用型且 docling 不保留引用(实测 2.128.0): 只留占位文本 + 图注, 无本地本体
    assert image.image_path is None
    assert all(i.page_no is None for i in doc.items)      # md 无页码概念
    assert all(i.text.strip() for i in doc.items)         # 不产出空 text 条目


def test_parse_docx_sample_saves_image(samples_dir: Path, tmp_path: Path) -> None:
    doc = parse_document(samples_dir / "sample.docx", Config(image_dir=str(tmp_path)))
    assert [i.level for i in doc.items if i.is_heading] == [1, 2, 2, 2]   # 来自 Heading 样式
    assert any(i.type == TYPE_TABLE for i in doc.items)

    image = next(i for i in doc.items if i.type == TYPE_IMAGE)
    saved = tmp_path / image.image_path          # image_path 是相对 RAG2_IMAGE_DIR 的路径
    assert saved.is_file()
    with Image.open(saved) as im:
        assert im.format == "PNG" and im.width > 0
    assert doc.meta["status"] == "SUCCESS" and doc.meta["pages"] is None
