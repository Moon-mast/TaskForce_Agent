"""噪声处理: 五条确定性规则(纯函数, 只吃 `ParsedDoc`)。

设计分工(02 篇 §2.5): docling 在解析期白拿的过滤(内容层 FURNITURE、显式 page_header 等 label)
放在 `parsers/common.py`; 这里做的是**确定性、可单测、不随库版本漂移**的规则。

执行顺序固定(不能换序): (1) 乱码/不可见字符 → (2) 空白规整 → (3) 页眉页脚剔除
→ (4) 相邻去重 → (5) 超短行合并。
为什么不能换序: 归一化必须先做(否则重复比对与短行判定都不准); 剔除必须先于合并
(否则页眉先被并进正文, 焊死后再也删不干净)。

本文件不 import docling、不读环境、不写文件 —— 单测可以直接内存造 `ParsedDoc`。
"""
from __future__ import annotations

import difflib
import math
import re
from dataclasses import replace

from .config import Config
from .contracts import TYPE_TABLE, TYPE_TEXT, ParsedDoc, ParsedItem

# ---------------------------------------------------------------- (1) 乱码/不可见字符

# C0/C1 控制符: 保留 \t(0x09) 与 \n(0x0a), 其余(含 DEL 与 C1 段)清掉
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# 零宽字符 / BOM / 软连字符: 删除(用删除表, 一次 translate 搞定)
_INVISIBLE = str.maketrans("", "", "\u200b\u200c\u200d\ufeff\u00ad")
# PDF 字体未嵌入时的经典残留, 如 "(cid:123)"
_CID = re.compile(r"\(cid:\d+\)")
# Unicode 私用区: 常被嵌入字体拿来表示图标, 占比高即"字体映射失败"
_PRIVATE_USE = re.compile(r"[\ue000-\uf8ff]")

# ---------------------------------------------------------------- (2) 空白规整

_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")

# ---------------------------------------------------------------- (3) 页眉页脚剔除

_HF_MAX_LEN = 50          # 归一化后超过这个长度就不当页眉页脚看(正文段落极少逐页原样重复)
_HF_MIN_PAGES = 3         # 跨页判定的下限
_HF_PAGE_RATIO = 0.6      # 命中页数占比阈值
_FULLWIDTH = str.maketrans({chr(c): chr(c - 0xFEE0) for c in range(0xFF01, 0xFF5F)})
_DIGITS = re.compile(r"\d+")

# ---------------------------------------------------------------- (5) 超短行合并

_PURE_NUMBER = re.compile(r"^\d{1,4}$")     # 长度 ≤ 4 的纯数字行 = PDF 页码残片


def clean_document(doc: ParsedDoc, cfg: Config) -> tuple[ParsedDoc, dict]:
    """按固定顺序跑五条规则, 返回 (清洁后的 ParsedDoc, 清洗统计)。

    纯函数: 不 IO、不原地修改(契约是 frozen dataclass, 改内容只能 `replace`)、
    不跨条目移动内容。统计 dict 由 `ingest.py` 汇总进 `IngestReport`。
    """
    warnings: list[str] = list(doc.meta.get("warnings", []))     # 保留解析阶段的警告
    items: list[ParsedItem] = []
    garbled = 0

    for item in doc.items:                                        # (1)
        cleaned, removed = _strip_invalid_chars(item, warnings)
        garbled += removed
        items.append(cleaned)

    items = [_normalize_ws(it) for it in items]                   # (2)
    items, dropped_hf = _drop_header_footer(items)                # (3)
    items, deduped = _dedup_adjacent(items)                       # (4)
    items, merged_short, dropped_pagenum = _merge_short_lines(items, cfg)   # (5)

    stats = {
        "garbled": garbled,              # 清掉的乱码/不可见字符数
        "dropped_hf": dropped_hf,        # 跨页重复的页眉页脚条目数
        "deduped": deduped,              # 相邻重复条目数
        "merged_short": merged_short,    # 被合并的超短行数
        "dropped_pagenum": dropped_pagenum,   # 删掉的纯数字残片数(02 篇规则 (5) 附带项)
    }
    meta = {**doc.meta, "warnings": warnings}
    return replace(doc, items=items, meta=meta), stats


# ---------------------------------------------------------------- 规则 (1)

def _strip_invalid_chars(item: ParsedItem, warnings: list[str]) -> tuple[ParsedItem, int]:
    """删控制符/零宽/BOM/软连字符/`(cid:n)`/私用区字符; 解码级问题只告警不静默。

    两类"救不了"的问题按占比写 warning(它们不是清洗能修的):
    - `U+FFFD` 替换符占比 > 1% → 解码问题, 该文档应走 OCR;
    - 私用区字符占比 > 10% → 字体映射失败, 条目内容不可信。
    """
    text = item.text
    cleaned = _CONTROL.sub("", text)
    cleaned = cleaned.translate(_INVISIBLE)
    cleaned = _CID.sub("", cleaned)

    pua_count = len(_PRIVATE_USE.findall(cleaned))
    if pua_count:
        ratio = pua_count / max(len(cleaned), 1)
        if ratio > 0.10:
            warnings.append(f"疑似字体映射失败(私用区字符占比 {ratio:.0%}): {text[:24]!r}")
        cleaned = _PRIVATE_USE.sub("", cleaned)

    if cleaned and cleaned.count("\ufffd") / len(cleaned) > 0.01:
        warnings.append(f"替换符(U+FFFD)占比过高, 疑似解码失败, 建议该文档走 OCR: {text[:24]!r}")

    return replace(item, text=cleaned), len(text) - len(cleaned)


# ---------------------------------------------------------------- 规则 (2)

def _normalize_ws(item: ParsedItem) -> ParsedItem:
    """空白规整: 换行统一/全角空格与 NBSP 归一/连续空格与空行压缩/首尾去空白。

    PDF 抽取常见"逐字空格 + 逐行换行", 不规整会让 03 篇子块的递归分隔符判断失准。
    **表格特殊**: 只规整行内空白并保行结构 —— 空行会把 Markdown 表切断。
    """
    text = item.text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ").replace("\u00a0", " ")

    if item.type == TYPE_TABLE:
        text = _normalize_table_text(text)
    else:
        text = _MULTI_SPACE.sub(" ", text)
        text = _MULTI_NEWLINE.sub("\n\n", text)
    return replace(item, text=text.strip())


def _normalize_table_text(text: str) -> str:
    """表格: 丢空行(不让它切断表)、行内多空格压成一个、非 `|` 开头的杂散行并回上一行。"""
    rows: list[str] = []
    for raw_line in text.split("\n"):
        line = _MULTI_SPACE.sub(" ", raw_line).strip()
        if not line:
            continue
        if rows and not line.startswith("|"):
            rows[-1] = f"{rows[-1]} {line}"
        else:
            rows.append(line)
    return "\n".join(rows)


# ---------------------------------------------------------------- 规则 (3)

def _drop_header_footer(items: list[ParsedItem]) -> tuple[list[ParsedItem], int]:
    """跨页重复判定: 同一段短文本出现在足够多的不同页码上 → 整组剔除。

    判据全满足才剔: `type=text`、`level is None`(标题豁免)、归一化后 ≤ 50 字符、
    命中 ≥ max(3, 60% 页) 个不同页码。**整组剔除**是刻意的: 只删命中的几页会留下残句。
    页码全为 None 的 docx/md 天然不满足"不同页", 直接跳过(符合预期)。
    """
    pages = {it.page_no for it in items if it.page_no is not None}
    if len(pages) < _HF_MIN_PAGES:
        return items, 0

    threshold = max(_HF_MIN_PAGES, math.ceil(_HF_PAGE_RATIO * len(pages)))
    buckets: dict[str, list[int]] = {}
    for idx, item in enumerate(items):
        if item.type != TYPE_TEXT or item.level is not None:
            continue
        key = _hf_key(item.text)
        if not key or len(key) > _HF_MAX_LEN:
            continue
        buckets.setdefault(key, []).append(idx)

    drop: set[int] = set()
    for idxs in buckets.values():
        hit_pages = {items[i].page_no for i in idxs} - {None}
        if len(hit_pages) >= threshold:
            drop.update(idxs)
    if not drop:
        return items, 0
    return [it for i, it in enumerate(items) if i not in drop], len(drop)


def _hf_key(text: str) -> str:
    """页眉页脚比对键: 去空白 + 全角转半角 + 数字串替 `#`(吃掉页码/编号差异)。"""
    return _DIGITS.sub("#", text.strip().translate(_FULLWIDTH))


# ---------------------------------------------------------------- 规则 (4)

def _dedup_adjacent(items: list[ParsedItem]) -> tuple[list[ParsedItem], int]:
    """相邻(±2 条内)重复条目去重: 归一化后相等, 或相似度 ≥ 0.95 且都不短于 20 字符。

    保守阈值是刻意的: 残留重复对检索的伤害 < 误删正文。标题不参与(目录与正文的同名标题
    必须都保留), 非相邻的重复也不动(那多半是不同章节的正常重复内容)。
    """
    kept: list[ParsedItem] = []
    removed = 0
    for item in items:
        duplicated = False
        for prev in kept[-2:]:
            if prev.type != item.type or prev.level is not None or item.level is not None:
                continue
            if _dedup_key(prev.text) == _dedup_key(item.text):
                duplicated = True
            elif (
                len(prev.text) >= 20
                and len(item.text) >= 20
                and difflib.SequenceMatcher(None, prev.text, item.text).ratio() >= 0.95
            ):
                duplicated = True
            if duplicated:
                break
        if duplicated:
            removed += 1
        else:
            kept.append(item)
    return kept, removed


def _dedup_key(text: str) -> str:
    """去空白与标点的比对键(PDF 渲染抖动只改这些, 不改字)。"""
    return re.sub(r"[\s\W_]+", "", text).lower()


# ---------------------------------------------------------------- 规则 (5)

def _merge_short_lines(items: list[ParsedItem], cfg: Config) -> tuple[list[ParsedItem], int, int]:
    """超短行合并: PDF 布局模型常把每个视觉行切成独立条目, 不合并会让子块在行边界乱切。

    判据全满足才并: 相邻两条同为 `type=text`、`level is None`、`heading_path` 完全相同、
    后一条短于 `cfg.short_line_max`。拼接时: 前一条末字符是 ASCII 字母/数字则补一个空格
    (英文断行), 否则直接拼(中文断行无空格)。标题天然豁免(它有 level)。
    另外: 长度 ≤ 4 的纯数字行单独出现时直接删(PDF 页码残片), 单独计数。
    """
    merged_items: list[ParsedItem] = []
    merged = 0
    dropped_pagenum = 0

    for item in items:
        if item.type == TYPE_TEXT and item.level is None and _PURE_NUMBER.match(item.text.strip()):
            dropped_pagenum += 1
            continue

        prev = merged_items[-1] if merged_items else None
        if (
            prev is not None
            and prev.type == TYPE_TEXT
            and item.type == TYPE_TEXT
            and prev.level is None
            and item.level is None
            and prev.heading_path == item.heading_path
            and len(item.text) < cfg.short_line_max
        ):
            merged_items[-1] = replace(prev, text=_join_lines(prev.text, item.text))
            merged += 1
            continue

        merged_items.append(item)

    return merged_items, merged, dropped_pagenum


def _join_lines(head: str, tail: str) -> str:
    """英文断行补空格, 中文断行直接拼(实测 PDF 的换行在中文里不带空格)。"""
    if head and head[-1].isascii() and head[-1].isalnum():
        return f"{head} {tail}"
    return head + tail
