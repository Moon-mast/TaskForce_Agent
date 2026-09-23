"""父块侧: 沿标题层级把内容单元聚合成大块(结构感知)。

这一侧只回答一个问题 —— **哪些内容单元该待在同一个父块里**。文本怎么切成小块是
`children.py` 的事, 本文件不认识"子块"这个概念, 只产出"单元组"。

三条硬约束里的两条在这里守:
1. **父块不跨节**: 遇到同级/更浅的标题就收口, 这样一个父块的 heading_path 单义;
2. **表格/图片是原子**: 一格都不切, 独占父块(与正文混在一块会让子块层的"表格不切"失去落点)。
"""
from __future__ import annotations

from ..config import Config
from ..contracts import TYPE_IMAGE, TYPE_TABLE, TYPE_TEXT, ParsedDoc, ParsedItem
from .children import split_pieces
from .common import UNIT_JOIN

# 父块目标带: 由 cfg.parent_target_chars 上下浮动 20% 得到(默认 1500 → 1200~1800)。
# 用比例而不是写死 1200/1800, 是为了让 cfg 一改, 下限与退化阈值跟着走, 不会自相矛盾。
PARENT_MIN_RATIO = 0.8
PARENT_MAX_RATIO = 1.2


def parent_bounds(cfg: Config) -> tuple[int, int]:
    """父块目标带的 (下限, 上限); 下限只用于统计达标率, 不作为切分约束。"""
    return (
        round(cfg.parent_target_chars * PARENT_MIN_RATIO),
        round(cfg.parent_target_chars * PARENT_MAX_RATIO),
    )


def aggregate(doc: ParsedDoc, parent_max: int) -> list[list[ParsedItem]]:
    """沿标题层级把内容单元聚合成父块(03 篇 §2.2), 返回按阅读顺序的单元组。

    三条规则(顺序不能换):
    - 规则 A 跨节即闭合: 遇到**同级或更浅**的标题就收口当前块, 新块从该标题起;
    - 规则 B 将超上限先闭合: 再加一个单元就超 `parent_max` → 收口后从该单元重开;
    - 规则 C 收尾不丢: 循环结束 `buf` 非空就成块(宁小勿丢, 短块正是"宁小勿跨节"的产物)。
    另有两条同样硬:
    - 表格/图片单元独占父块(先把前面攒的正文收口): 它是原子的;
    - 标题单元**进正文**而不是只进栈: BM25 只索引 text 字段, 标题词往往是该节最强的检索
      信号, 不写进正文就永远匹配不上; 子块也因此自带语义锚点。
    """
    blocks: list[list[ParsedItem]] = []
    buf: list[ParsedItem] = []
    size = 0
    level: int | None = None            # 当前父块所属节的标题层级(尚未见标题时为 None)

    for item in doc.items:
        if not item.text:
            continue                                   # 清洗可能留下空条目: 空文本会污染长度统计

        if item.is_heading:
            # 规则 A: 同级或更浅 → 跨节, 先收口当前块。`level is None` 那一支管的是"文档开头
            # 还没出现过标题"(导言): 它遇到任何标题都该收口, 否则导言会被并进第一章
            if buf and (level is None or item.level <= level):
                blocks.append(buf)
                buf, size = [], 0
            level = item.level                         # 记住当前块的层级, 给下一个标题比较用
            buf.append(item)                           # 标题进正文: BM25 只索引 text 字段
            size += len(item.text)
            continue

        if item.type in (TYPE_TABLE, TYPE_IMAGE):
            if buf:
                blocks.append(buf)                     # 先收口攒着的正文: 原子块不与正文同块
            blocks.append([item])                      # 原子块独占一块, 整块不切
            buf, size = [], 0
            continue

        # 规则 B: 判据是"将要超"而不是"已经超"(写成后者块长必然越界)。这里用累加长度近似块长:
        # 拼接符的 2 字符误差只让实际上限略高一点, 不影响达标率口径
        if buf and size + len(item.text) > parent_max:
            blocks.append(buf)
            buf, size = [], 0
        buf.append(item)
        size += len(item.text)

    if buf:
        blocks.append(buf)                             # 规则 C: 收尾的残余也成块(宁小勿丢)
    return absorb_heading_only(blocks)                 # 最后一道: 消掉"只有标题"的空壳块


def absorb_heading_only(blocks: list[list[ParsedItem]]) -> list[list[ParsedItem]]:
    """把‘只有标题’的父块交给下一块当语义锚点(标题后面直接是原子块或超长正文时会冒出来)。

    为什么必须处理: 这种块的正文只有一行标题, 它切出来的子块就是标题本身 —— 检索命中它
    既没有上下文也没有答案, 纯噪声; 而那个标题本该属于紧随其后的表格/正文。
    两条边界(别把"不跨节"的规矩弄丢):
    - 下一块自己以标题开头 → 不并: 那是"空章节后跟新章节", 并了会让新章节的内容顶着
        上一节的面包屑(面包屑按开块单元取, 就取错了);
    - 走到末尾还攒着 → 自己成块: 全文只有标题的文档也就这一份内容了(宁小勿丢)。
    """
    out:list[list[ParsedItem]]=[]
    pending:list[ParsedItem]=[]              # 攒着"只有标题"的块, 等下一块来接手
    for block in blocks:
        if all(unit.is_heading for unit in block):
            # 整块只有标题(可能同时攒了多级标题, 比如第一章紧跟 1.1) → 它没有正文,
            # 自己成块就只剩一行标题, 检索命中等于没上下文
            if pending:
                out.append(pending) #连续两个空标题块：前一个先落地，不合并
            pending=list(block)
            continue
        if pending and not block[0].is_heading:
            # 下一块从正文/原子块开始 → 标题归它, 当它的语义锚点(它切出的子块也自带节标题)
            out.append([*pending,*block])
        else:
            # 下一块自己以标题开头 = "空章节紧接着下一节": 不能并, 否则新节内容会顶着上一节的面包屑
            if pending:
                out.append(pending)
            out.append(block)
        pending=[]
    if pending:
        out.append(pending)                  # 收尾还攒着: 全文只有标题时就这一份内容(宁小勿丢)
    return out


def body_units(block: list[ParsedItem]) -> list[ParsedItem]:
    """块里承载内容的单元(去掉随块走的标题)。

    单独抽出来是因为有四处都要问"这个块的内容是什么": 拆正文、判类型、取图片路径、判原子性。
    分散写四遍, 改一次就得改四处。
    """
    return [unit for unit in block if not unit.is_heading]

def join_units(block: list[ParsedItem]) -> tuple[str, str, list[tuple[int, int | None]]]:
    """拆成 (标题前缀, 正文, 正文里每个单元的起点→页码)。

    为什么标题要单独摘出来、不跟着正文一起切:
    - 切分器的合并逻辑是"按片贪心、不回头", 标题混进去就有被单独关成一个几字符碎块的风险
      (换个切分器实现更是说不准) —— 标题该不该独立成块, 是本模块的结构决定, 不该由切分器裁决;
    - 偏移表只覆盖正文, 子块的页码回查就不会被标题那几个拼接符带偏。
    最终 chunk 文本仍然是"标题 + 正文"(由 `common.compose` 拼), 与"标题进正文"的定稿一致。
    """
    heads = [unit.text for unit in block if unit.is_heading]    # 标题: 只当前缀, 不进切分输入
    body=body_units(block)                                      # 正文: 唯一会被切分的部分
    parts:list[str]=[]
    offsets:list[tuple[int,int|None]]=[]                        # [(正文里的起点, 该单元的页码)]
    pos=0
    for index,unit in enumerate(body):
        if index:
            pos+=len(UNIT_JOIN)               # 除第一条外, 每条前面都隔了一个拼接符, 游标要跳过
        offsets.append((pos,unit.page_no))    # 记下这条正文拼进来后从正文的第几个字符开始
        parts.append(unit.text)
        pos+=len(unit.text)                   # 推进游标, 供下一条记录起点
    return UNIT_JOIN.join(heads),UNIT_JOIN.join(parts),offsets





def block_type(block: list[ParsedItem]) -> str:
    """原子块(单条表格/图片)保留自身类型, 其余一律 text。

    类型不是显示用的小标签: 子块层按它分支(表格/图片整块即一个子块, 不做递归切分),
    04 篇按它分支(图片块走多模态嵌入)。判错就会把表切碎、把图当文本嵌。
    判据是两条的合取 —— 块里**只有一条内容单元**, 且它**是表格或图片**;
    少了第一条, 一条普通正文也会被当成"原子的正文块"(目前看结果一样, 但语义已经错了)。
    """
    body=body_units(block)                                        # 先去掉随块走的标题, 只看内容单元
    if len(body)==1 and body[0].type in (TYPE_TABLE,TYPE_IMAGE):
        return body[0].type            # 原子块: 单条表格/图片, 类型要透给子块层与 04 篇
    return TYPE_TEXT                   # 其余(单条正文 / 多条正文混合)一律按文本块处理


def heading_of(block: list[ParsedItem]) -> str:
    """父块的面包屑: 取**开块那个单元**的位置。

    开块若是标题, 就把标题自身也接上 —— 标题条目的 heading_path 按 02 篇约定不含自身,
    直接用会丢掉"这是哪一节"; 开块若是正文(规则 B 切出来的续块), 它自带的面包屑已经是完整的。
    父块不跨节由规则 A 保证, 所以这份面包屑单义。
    """
    first = block[0]                                # 只看开块单元: 父块不跨节, 它就代表整块的位置
    if first.is_heading:
        # 开块是标题: 标题条目的 heading_path 按 02 篇约定不含自身, 要自己接上, 否则丢了"这是哪一节"
        return f"{first.heading_path} > {first.text}" if first.heading_path else first.text
    return first.heading_path                       # 开块是正文(规则 B 的续块): 面包屑已完整


def image_of(block: list[ParsedItem]) -> str | None:
    """图片块的落盘相对路径(透传字段; 表格/正文块为 None)。"""
    # 只有图片块才透传路径(04 篇按它读图做多模态嵌入); 表格/正文块一律 None
    return body_units(block)[0].image_path if block_type(block) == TYPE_IMAGE else None


def block_spans(
    head: str, body: str, chunk_type: str, block: list[ParsedItem], parent_max: int
) -> list[tuple[int, int]]:
    """一个父块要拆成几条(常态 1 条), 返回 (文本, 在正文里的起点)。

    特例: **单个超长的无标题正文**(导言、法条这类几千字符一段的正文)没有标题边界可用,
    退化为按上限切分, 产物各自成父块(03 篇 §2.2 退化规则) —— 复用子块切分器, 不另写一份。
    尺寸要**扣掉标题前缀并让出拼接符**: 标题会贴回第一片, 不扣父块就会超上限。
    """
    if chunk_type == TYPE_TEXT and len(body_units(block)) == 1 and len(body) > parent_max:
        # 单条正文本身就超上限(导言/法条这类): 没有标题边界可用, 只能借子块切分器补出边界
        size = max(parent_max - len(head) - len(UNIT_JOIN), 1)  # 扣掉标题与拼接符, 否则超上限
        return split_pieces(body, size)                         # 目标 = 上限, 产物各自成一个父块
    return [(body, 0)]                                          # 常态: 整段正文就是一条, 起点 0
