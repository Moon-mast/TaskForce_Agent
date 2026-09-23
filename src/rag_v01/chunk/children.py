"""子块侧: 把父块的正文递归切成小块, 并装配子块的九字段。

这一侧只回答一个问题 —— **一段文本怎么切成 ~450 字符的小块**, 以及小块之间怎么重叠。
它不认识标题、不认识结构(标题前缀由 `common.compose` 贴回第一条), 拿到的是 `parents.py`
拼好的"标题前缀 + 正文"。

分隔符层级从粗到细: `段落(\n\n) → 换行(\n) → 中文句界(。！？；…) → 无标点硬切`。
句界为什么带尾随闭引号/括号: `他说。"` 的闭引号必须跟在前句尾; 英文句点不入集(会切坏 3.14、e.g.)。
"""
import itertools
import re
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..contracts import TYPE_TEXT, ChildChunk
from .common import Ctx, compose, page_at

# 分隔符层级(粗 → 细), 直接交给库: 段落 → 换行 → 中文句界。
# **末尾那个空串不能省**: 库靠它做"逐字符切 + 合并"的硬切兜底 —— 少了它, 遇到完全无标点的
# 长串(实测 'x'*300)库会整段不切, 切出来的子块超过上限三倍。
# 英文句点故意不入集: `3.14`、`e.g.` 会被误切, 中英混排里的英文长句宁可降级到硬切。
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ""]

# 库的 `keep_separator=True` 是 `sep + 下一片` 的拼法(实测下一片以 `。` / `。”` 开头),
# 这些字符要在切完后搬回前一片尾部 —— 不搬子块会以标点开头, "闭引号跟在前句尾"也破了。
_SEP_CHARS = '。！？；…\n」』”）)]】"'

# 重叠回退到句界时的匹配器(与 SEPARATORS 的句界口径一致)。
_SENTENCE_END = re.compile(r'[。！？；…]+[」』”）)\]】"]*')
# 重叠窗口内挑句界的容差: 宁可裸取整个窗口, 也不让对齐后的重叠缩到十几个字(验收线 40~50)。
_OVERLAP_TOLERANCE = 10


@lru_cache(maxsize=8)
def _splitter(chunk_size:int)->RecursiveCharacterTextSplitter:
    """按 chunk_size 缓存切分器(构造一次, 全流程复用)。
    四个参数都是刻意的:
    - `chunk_overlap=0`: 重叠交给 `add_overlap` —— 库的重叠按整片量化(参数 50 实测落 32~41、
    70 跳到 ~76), 落点由语料句长决定, 做不到我们要的 40~50;
    - `keep_separator=True`: 分隔符不能丢(丢了 `。` 文本就变了), 位置错位由 `_SEP_CHARS` 搬回;
    - `add_start_index=True`: 每片的起始下标, 用来回查页码(跨页父块的子块页码靠它);
    - `chunk_size` 传"目标字符数" —— 库默认按字符数算长度, 与我们的口径一致。
    """
    return RecursiveCharacterTextSplitter(
        separators=SEPARATORS,          # 降级顺序: 段落 → 换行 → 中文句界 → 空串(逐字符硬切)
        chunk_size=chunk_size,          # 目标字符数(库按字符算长度, 与我们的口径一致)
        chunk_overlap=0,                # 重叠自己加: 库的重叠按整片量化, 落不到 40~50
        keep_separator=True,            # 不丢分隔符(丢了 "。" 文本就变了), 错位由 _SEP_CHARS 搬回
        add_start_index=True,           # 每片带起始下标 → 子块回查页码要用它
    )

def split_pieces(body: str, target: int) -> list[tuple[str, int]]:
    """把一段正文切成 (片段文本, 在正文里的起点) 序列。

    为什么返回"文本 + 起点"而不是区间: 库切出来的文本已经去过首尾空白、分隔符也搬过位置,
    用区间回切原文是对不上的; 而起点只用来回查页码(`common.page_at`), 单独带出来最省事。
    库按 `chunk_size` **严格封顶**(实测: 450 → 最长 448, 逐字符兜底时正好 100), 所以手写版
    那个 slack 容差在这里不存在 —— 比目标略长的段落会被降级切开, 这是行为变化, 不影响验收线
    (450 + 50 重叠 ≤ 550)。
    """
    pieces: list[tuple[str, int]] = []
    for doc in _splitter(target).create_documents([body]):     # 一整段正文 → 一批 Document
        text, start = doc.page_content, doc.metadata["start_index"]   # 片文本 + 它在正文里的起点
        lead = ""
        while text[:1] and text[0] in _SEP_CHARS:      # 库把分隔符贴在片头: 逐字剥下来暂存进 lead
            lead, text = lead + text[0], text[1:]
        if pieces:
            pieces[-1] = (pieces[-1][0] + lead, pieces[-1][1])    # lead 还给前一片, 补在它尾巴上
            start += len(lead)                     # 自己少了 lead 个字符, 起点跟着往后挪
        else:
            text = lead + text                     # 首片前面没有可贴的, 原样保留
        if text:                                   # 库偶尔吐空片(实测末尾会有一个), 直接丢
            pieces.append((text, start))
    return pieces

def add_overlap(pieces: list[str], overlap: int) -> list[str]:
    """相邻子块加重叠: 上一块的尾巴拼到下一块头上(03 篇 §2.3)。"""
    if overlap <= 0 or len(pieces) < 2:
        return pieces                              # 只有一片(或不要重叠): 没有相邻关系可加
    out = [pieces[0]]                              # 第一条前面没有可借的尾巴
    for prev, cur in itertools.pairwise(pieces):   # 相邻成对: (1,2) / (2,3) / ...
        tail = _overlap_tail(prev, overlap)        # 借上一块的尾巴(已回退到句界)
        out.append(f"{tail}{cur}" if tail else cur)
    return out

def _overlap_tail(prev: str, overlap: int) -> str:
    """取上一子块尾部 ~`overlap` 字符, 并**回退到句子边界**(不把半句拼给下一块)。

    规则: 在窗口内找句界, 挑"对齐后长度 ≥ 窗口 - 容差"的那个, 取它之后的文本 —— 按句界对齐
    的同时重叠量仍落在 40~50(验收线), 不会为对齐缩到十几个字。窗口里没有合用句界(整窗就是
    半句话)就裸取整个窗口: 重叠量的价值高于句界整齐。
    上一块本身比窗口还短时直接不重叠 —— 整块重复一遍是噪声, 不是上下文。
    """
    if overlap <= 0 or len(prev) <= overlap:
        return ""                                  # 上一块比窗口还短: 整块复制过来是噪声, 不重叠
    window = prev[-overlap:]                       # 只看尾部这一个窗口, 重叠量由它封顶(≤50)
    aligned = [
        len(window) - match.end()                  # 句界之后还剩多少字符 = 对齐后的重叠长度
        for match in _SENTENCE_END.finditer(window)
        if len(window) - match.end() >= overlap - _OVERLAP_TOLERANCE   # 太短的不要(会把重叠缩水)
    ]
    if aligned:
        return window[len(window) - max(aligned) :].strip()   # 挑最长的一条: 最接近窗口大小
    return window.strip()                          # 窗内没有合用句界: 裸取整窗(重叠量 > 句界整齐)

def build_children(
    parent_id: str,
    chunk_type: str,
    head: str,
    core: str,
    core_start: int,
    offsets: list[tuple[int, int | None]],
    heading: str,
    image_path: str | None,
    ctx: Ctx,
) -> list[ChildChunk]:
    """把父块的正文切成子块, 并装配九字段(原子块整块即一个子块)。

    三条分支:
    - 原子块(表格/图片): 整块即一个子块, **不切也不加重叠** —— 一行行切开就毁了表结构;
    - 空壳标题块(core 为空): 用空串占位, `compose(head, "")` 正好返回标题本身 —— 直接拿 head
    当片段会拼成"标题 + 标题"重复两遍(这个坑我实测过);
    - 正文块: 交 `split_pieces` 切(库), 再 `add_overlap` 加重叠; 标题前缀只贴给第一条。

    子块页码按**加重叠之前**的起点回查(`split_pieces` 返回的起点): 重叠是从上一块尾巴借来的
    文本, 不属于这一块, 拿它查页会偏到上一页。
    """
    if chunk_type !=TYPE_TEXT:
        pieces:list[tuple[str,int]]=[(core,core_start)]        # 表格/图片: 整块一条, 不切也不重叠
    elif not core:
        pieces=[("",core_start)]                              # 空壳标题块: 空串占位, compose 得标题
    else :
        cut=split_pieces(core,ctx.cfg.child_target_chars)      # 库切分 → [(文本, 相对 core 的起点)]
        overlapped = add_overlap([text for text, _ in cut], ctx.cfg.child_overlap_chars)
        pieces = [
            (text, core_start + start)                         # 起点换算回"整段正文"的坐标系
            for text, (_, start) in zip(overlapped, cut, strict=True)
        ]
    children: list[ChildChunk] = []
    for piece, start in pieces:
        # 标题前缀只贴给第一条(同节的后续片不需要再带)。这里用 `not children` 判"是不是第一条"
        # 而不是 enumerate 下标 —— 空片段会被下面的 continue 跳过, 下标会失准
        text = compose(head, piece) if not children else piece
        if not text:
            continue                                           # 空文本不产块, 也不占子块序号
        children.append(
            ChildChunk(
                chunk_id=f"{parent_id}:c{len(children) + 1:03d}",
                text=text,
                parent_id=parent_id,
                doc_id=ctx.doc_id,
                source=ctx.source,
                page_no=page_at(offsets, start),
                heading_path=heading,
                chunk_type=chunk_type,
                created_at=ctx.created_at,
                image_path=image_path,
            )
        )
    return children
