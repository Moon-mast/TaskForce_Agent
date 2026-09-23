"""Markdown 专属: 后端解析(无模型、无页码、无图片本体)。

要点:
- 标题层级来自 `#` 数量, 但 docling 的行为要记牢(实测 2.128.0):
  `#` 一级标题被识别为 **TitleItem**(没有 level → 按 0 层处理),
  `##` 起才是 SectionHeaderItem(level = "#" 数 − 1);
- 表格只支持简单表(单元格 row_span/col_span 恒为 1, 首行当表头);
- 图片是**引用型**: 不搬文件、不联网抓取(backend_options 里显式关掉 fetch), 保证"解析结果
  只取决于文件内容"的确定性。**实测 2.128.0: 引用取不到**(`PictureItem.image` 恒为 None,
  三种 fetch 开关都试过), 所以这类条目 `image_path=None`, 只有占位文本 + 图注参与检索。
"""
from __future__ import annotations

from docling.datamodel.base_models import InputFormat
from docling.document_converter import MarkdownBackendOptions, MarkdownFormatOption

FORMAT = InputFormat.MD
HAS_LOCAL_IMAGES = False         # 引用型: 没有图片本体可落盘


def build_option(cfg) -> MarkdownFormatOption:
    """显式关掉一切抓取: 离线确定性优先(要图片本体就把图放到语料里当独立文件处理)。"""
    return MarkdownFormatOption(
        backend_options=MarkdownBackendOptions(
            fetch_images=False,
            enable_remote_fetch=False,
            enable_local_fetch=False,
        )
    )
