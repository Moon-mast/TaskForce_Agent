"""格式无关的解析骨架: 转换调用 + "DoclingDocument → ParsedDoc" 遍历。

留在这里的都是三路共用的:
- `doc_id` / `source` 规则(03/04 篇要用同一套值);
- 遍历: 标题栈维护 `heading_path`、内容层过滤、表格整表转 Markdown、图片交给 images;
- 失败语义: 非 SUCCESS / PARTIAL_SUCCESS 一律 `ParseError`。

导入分层: `docling_core`(只有枚举, 轻)在模块顶层; `docling.document_converter`
(拖 torch, 重)在 `parse_with()` 里局部导入 —— 这样 clean.py 的单测和"内存造
ParsedDoc"的测试完全不碰重依赖。
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

from docling_core.types.doc import DocItemLabel

from ..contracts import TYPE_IMAGE, TYPE_TABLE, TYPE_TEXT, ParsedDoc, ParsedItem
from . import images


class ParseError(RuntimeError):
    """解析失败(不支持的后缀 / 文件损坏 / 后端异常)。ingest 逐文件捕获, 不中断整批。"""


_SKIP_LABELS = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.FOOTNOTE}
_HEADING_LABELS = {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}
_TITLE_LEVEL = 0   # TitleItem 没有 level 字段(实测 docling 2.128) → 约定记 0 层


def compute_doc_id(path: Path) -> str:
    """内容寻址: 同内容不同路径 → 同 doc_id(重跑幂等); 内容变 → id 变(旧记录按 source 清)。"""
    h = hashlib.sha256()
    with path.open("rb") as fh:                      # 二进制读: 不受平台换行/编码影响
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def page_no_of(item) -> int | None:
    """item 的页码: 取第一条 prov(溯源记录); docx/md 后端 prov 为空 → None。"""
    prov = getattr(item, "prov", None)
    return prov[0].page_no if prov else None


def parse_with(
    fmt, option, path: Path, cfg, *, source: str | None, local_images: bool
) -> ParsedDoc:
    """按格式模块给的 option 建 converter, 转换 + 遍历成 ParsedDoc。"""
    # 局部 import: 只有真正要走 docling 时才付"导入 torch"的代价
    from docling.datamodel.base_models import ConversionStatus
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter(allowed_formats=[fmt], format_options={fmt: option})
    started = time.perf_counter()
    try:
        result = converter.convert(str(path))
    except Exception as exc:   # 损坏文件 / 后端崩溃: 统一成人话异常, 不让长堆栈冒到上层
        raise ParseError(f"解析失败: {path.name}: {exc!r}") from exc
    elapsed = round(time.perf_counter() - started, 2)

    warnings: list[str] = []
    if result.status is ConversionStatus.PARTIAL_SUCCESS:
        warnings.append(f"部分页面解析失败(status={result.status.name})")   # 保留可用部分
    elif result.status is not ConversionStatus.SUCCESS:
        raise ParseError(f"解析未成功(status={result.status.name}): {path.name}")

    doc = result.document
    doc_id = compute_doc_id(path)                     # 只算一次: 这个 id 贯穿解析到入库
    items, walk_warnings = walk(doc, doc_id=doc_id, cfg=cfg, local_images=local_images)
    meta = {
        "pages": len(doc.pages) if getattr(doc, "pages", None) else None,
        "elapsed_s": elapsed,
        "status": result.status.name,
        "warnings": warnings + walk_warnings,
    }
    return ParsedDoc(doc_id=doc_id, source=source or path.name, items=items, meta=meta)


def walk(doc, *, doc_id: str, cfg, local_images: bool) -> tuple[list[ParsedItem], list[str]]:
    """遍历结构树 → (items, warnings)。

    核心是标题栈算法:
      遇到标题时先把它自己"及更深层级"的旧标题弹掉, 再压入自己 —— 于是栈里任一时刻
      都是"当前位置的上级链", 用 " > " 拼起来就是 heading_path(不含自身)。

    其它约定:
    - `content_layer == FURNITURE` 的条目直接丢(docling 已判定的页眉/页脚/脚注);
    - 表格整表一个 item(03 篇不再切); 序列化失败才退到手拼 grid;
    - 图片: 本地图落盘(image_path 记相对路径); md 的引用型既无本体也无引用(实测), 记 None;
    - 图注另有一个普通 text 条目(md 后端把 ![](alt) 的 alt 变成图片后的 TextItem, 实测),
      所以图注会同时出现在"图片占位文本"和它自己的 text 条目里 —— 这是有意的:
      图注文本要能被检索命中; 占位文本只是图片条的 text 载体。
    - 遍历只走 `doc.iterate_items()`; **不读 `doc.furniture`**(已 deprecated)。
    """
    raw = list(doc.iterate_items())          # 先物化: 找图片的图注要前后看邻居
    items: list[ParsedItem] = []
    headings: list[tuple[int, str]] = []     # [(level, title)]
    warnings: list[str] = []
    seq = 0

    for pos, (item, _tree_level) in enumerate(raw):
        layer = getattr(item, "content_layer", None)
        if layer is not None and layer.name == "FURNITURE":
            continue                                   # 第一道: docling 判定的页眉页脚
        label = getattr(item, "label", None)
        if label in _SKIP_LABELS:
            continue                                   # 第二道: 显式 label 兜底
        page = page_no_of(item)
        heading_path = " > ".join(title for _, title in headings)   # 非标题条目用它; 标题在下面重算

        if label in _HEADING_LABELS:
            level = (
                _TITLE_LEVEL if label is DocItemLabel.TITLE
                else int(getattr(item, "level", 1) or 1)
            )
            while headings and headings[-1][0] >= level:
                headings.pop()                          # 回退到父级
            # 弹栈之后再算面包屑: 标题条目只能含"祖先", 不能含上一个同级/更深的旧标题
            # (否则新章节会被算成上一节的子节, 03 篇的父块边界跟着错)
            heading_path = " > ".join(title for _, title in headings)
            headings.append((level, item.text))
            items.append(ParsedItem(TYPE_TEXT, item.text, heading_path, page, level=level))
            continue

        if label is DocItemLabel.TABLE:
            items.append(ParsedItem(TYPE_TABLE, table_markdown(item, doc), heading_path, page))
            continue

        if label is DocItemLabel.PICTURE:
            seq += 1
            local_path = (
                images.save_picture(item, doc, doc_id=doc_id, page_no=page, seq=seq, cfg=cfg)
                if local_images else None
            )
            if local_images and local_path is None:
                warnings.append(f"图片 {seq} 取不到本体(只留占位文本); page={page}")
            ref = local_path or images.original_ref(item)     # 取不到 → None(md 属这种情况)
            caption = images.caption_nearby(raw, pos)
            items.append(ParsedItem(
                TYPE_IMAGE, images.placeholder_text(seq, ref, caption), heading_path, page,
                image_path=ref,
            ))
            continue

        text = (getattr(item, "text", "") or "").strip()
        if text:
            items.append(ParsedItem(TYPE_TEXT, text, heading_path, page))

    return items, warnings


def table_markdown(item, doc) -> str:
    """整表 → Markdown(含表头分隔行)。

    实测(docling 2.128.0): 序列化器已自行处理单元格内换行(→空格)与竖线(→&#124;),
    所以这里**不做任何转义** —— 再转一遍会把列分隔符一起毁掉。
    合并单元格会被"摊平"(只在起点格写值): 检索可用, 展示别依赖列对齐。
    """
    try:
        md = (item.export_to_markdown(doc) or "").strip()
    except Exception:
        md = ""
    return md or _table_from_grid(item)


def _table_from_grid(item) -> str:
    """序列化失败的兜底: 从 TableData.table_cells 手拼 Markdown(保行列, 丢 span)。"""
    cells = getattr(getattr(item, "data", None), "table_cells", None) or []
    if not cells:
        return "[表格: 无法序列化]"
    rows: dict[int, list] = {}
    for c in cells:
        rows.setdefault(c.start_row_offset_idx, []).append(c)

    lines: list[str] = []
    for r in sorted(rows):
        row = sorted(rows[r], key=lambda c: c.start_col_offset_idx)
        lines.append("| " + " | ".join((c.text or "").replace("\n", " ") for c in row) + " |")
        if len(lines) == 1:                                    # Markdown 表必须有分隔行
            lines.append("| " + " | ".join("---" for _ in row) + " |")
    return "\n".join(lines)
