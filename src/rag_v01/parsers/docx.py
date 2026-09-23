"""DOCX 专属: OOXML 直读(无模型、无 OCR、无页码)。

三路里最干净的一路:
- 标题层级直接来自 Heading 样式(不靠模型猜);
- 表格是原生结构(含合并单元格);
- **没有页码概念** → `page_no` 恒为 None(落库时 04 篇写 -1 哨兵);
- 内嵌图片有本体, `get_image(doc)` 直接可用, 不需要 PDF 那套 `generate_picture_images`。
"""
from __future__ import annotations

from docling.datamodel.base_models import InputFormat
from docling.document_converter import WordFormatOption

FORMAT = InputFormat.DOCX
HAS_LOCAL_IMAGES = True          # 内嵌图片有本体 → images.save_picture 能落盘


def build_option(cfg) -> WordFormatOption:
    """默认后端就是 MsWord(规则驱动, 无模型下载); 没有 OCR / 表格模型这类开关, 所以不用 cfg。"""
    return WordFormatOption()
