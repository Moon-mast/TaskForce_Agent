"""图片落盘与占位文本(与格式无关)。

目录约定(02 篇 §4.5):
    RAG2_IMAGE_DIR/<doc_id>/p<页号 4 位>_i<序号 3 位>.<png|jpg>

为什么这样排:
- 按 doc_id 分目录 —— doc_id 是内容 hash, 同内容重跑写同名文件, 天然覆盖、不残留旧图;
- docx/md 没有页码, 页号段写 0000 占位;
- 存"相对 RAG2_IMAGE_DIR 的路径", 整个图片目录搬走也不失效。

本文件不做任何外部模型调用(描述生成是可选开关, 见 02 篇 §4.5 / 01 篇 image_desc)。
"""
from __future__ import annotations

from pathlib import Path

from docling_core.types.doc import DocItemLabel


def save_picture(item, doc, *, doc_id: str, page_no: int | None, seq: int, cfg) -> str | None:
    """把 PictureItem 的图像本体落盘, 返回相对 RAG2_IMAGE_DIR 的路径; 取不到图返回 None。"""
    try:
        pil = item.get_image(doc)      # docling 2.128: 返回 PIL.Image, 取不到时给 None
    except Exception:
        pil = None
    if pil is None:
        return None

    ext = "jpg" if cfg.image_format.strip().lower() in {"jpg", "jpeg"} else "png"
    rel = f"{doc_id}/p{(page_no or 0):04d}_i{seq:03d}.{ext}"
    out = Path(cfg.image_dir) / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if ext == "jpg":
        pil.convert("RGB").save(out, format="JPEG", quality=92)   # JPEG 无透明通道, 必须先转 RGB
    else:
        pil.save(out, format="PNG")
    return Path(rel).as_posix()        # POSIX 风格: 元信息跨平台一致(Milvus / 报告里都用它)


def original_ref(item) -> str | None:
    """引用型图片(md)的原始引用: 取到才返回, 取不到返回 None。

    实测(docling 2.128.0): md 后端既不保留引用也不给图片本体 —— PictureItem.image 恒为 None,
    fetch_images / enable_local_fetch / enable_remote_fetch 三种组合都试过, uri 一样取不到。
    所以这里**不编造路径**(曾返回 "inline-image", 会让上层以为真有引用)。
    """
    img = getattr(item, "image", None)
    for attr in ("uri", "path", "name"):
        value = getattr(img, attr, None)
        if value:
            return str(value)
    return None


def caption_nearby(raw, pos: int, *, max_len: int = 60) -> str | None:
    """取图片的图注: 在阅读顺序里是图片前或后的一个普通 TextItem。

    实测 docling 2.128: 图注不挂在 PictureItem.captions 上(该字段存在但常为空),
    md 后端会把 ![](alt) 的 alt 变成图片后的 TextItem —— 所以按邻居找, 并卡长度上限,
    避免把正文段落误当图注。
    """
    for offset in (1, -1):
        idx = pos + offset
        if 0 <= idx < len(raw):
            neighbor, _ = raw[idx]
            if getattr(neighbor, "label", None) is DocItemLabel.TEXT:
                text = (getattr(neighbor, "text", "") or "").strip()
                if text and len(text) <= max_len:
                    return text
    return None


def placeholder_text(seq: int, ref: str | None, caption: str | None) -> str:
    """图片条的 text(恒非空): 稀疏路与文本展示的最小可读信息。

    语义主体由 04 篇的多模态向量承担; 无本地本体/无引用时写"无本地图", 不把 None 拼进文本。
    """
    return f"[图片 {seq}: {ref or '无本地图'}; 图注: {caption or '无'}]"
