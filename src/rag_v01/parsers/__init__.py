"""解析层入口: 后缀白名单 → 对应格式模块 → `ParsedDoc`。

对外只有两个名字: `parse_document()` 与 `ParseError`。
三种格式的差异收在各自文件里(pdf.py / docx.py / md.py), 共享遍历在 common.py,
图片落盘在 images.py —— 加一种格式 = 加一个模块 + 在这里注册一行。
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config, load_config
from ..contracts import ParsedDoc
from . import docx as _docx
from . import md as _md
from . import pdf as _pdf
from .common import ParseError, parse_with

# 白名单: docling 支持的格式远不止这三种(还含 pptx/xlsx/html 等),
# 我们要的是"语料面可控", 所以在这里限制, 不依赖 docling 的判断。

_FORMATS = {".pdf": _pdf, ".docx": _docx, ".md": _md}

__all__ = ["parse_document", "ParseError"]

def parse_document(
        path:str|Path,
        cfg:Config|None = None,
        *,
        source:str|None=None
)->ParsedDoc:
    """解析单个文件。

    - `cfg`: 不传则读环境变量; 单测传显式 Config 即可零环境依赖;
    - `source`: 相对语料根的 POSIX 路径, 由 ingest 编排层算好传入; 不传就用文件名。
    失败一律抛 `ParseError`(.doc / 未知后缀 / 文件不存在 / docling 内部异常),
    由 ingest 逐文件捕获进 `IngestReport.failed`, 不中断整批。
    """
    cfg=cfg or load_config()          # 不传配置就读环境变量; 单测传显式 Config 可零环境依赖
    p=Path(path)
    suffix=p.suffix.lower()           # 白名单只看后缀, 且统一转小写(.MD 也要认)

    if suffix ==".doc":
        # 不自动转换: 引入 soffice 外部进程依赖不划算, 给人话提示即可
        raise ParseError(
            f"不支持 .doc(旧二进制格式): {p.name} —— 请先另存为 .docx"
        )
    fmt_mod=_FORMATS.get(suffix)      # 每个格式模块只交出 FORMAT + option + 有无图片本体三件事
    if fmt_mod is None:
        raise ParseError(f"不支持的后缀 {suffix!r}: 只接受 {sorted(_FORMATS)}")
    if not p.is_file():
        raise ParseError(f"文件不存在或不是文件: {p}")

    return parse_with(
        fmt_mod.FORMAT,               # 告诉 docling 用哪个后端
        fmt_mod.build_option(cfg),    # 该后端的 pipeline 参数(如 pdf 的 OCR / 表格模型开关)
        p,
        cfg,
        source=source,                # 相对语料根的路径, 由编排层算好传入(不传就用文件名)
        local_images=fmt_mod.HAS_LOCAL_IMAGES,   # md 是引用型: 没有本体可落盘
    )
