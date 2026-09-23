"""父子分块(03 篇): 对外唯一入口 `split`。

包内分工(依赖方向: 本文件 → 两侧 → common, 不成环):
- `common.py`: 一次切块内不变的量(`Ctx`)与共享小工具(`utc_now` / `page_at` / `trim` / `compose`);
- `parents.py`: 父块侧 —— 沿标题层级边界聚合出大块, 锚定原子块(表格/图片);
- `children.py`: 子块侧 —— 把大块递归切成小块并装配九字段;
- 本文件: **只做编排**, 一行算法都不放(算法放错地方, 下一个读代码的人就得满包找)。

两级分工(small-to-big, 03 篇 §2.1):
- **父块** 沿标题层级边界聚合, 目标 1200~1800 字符, 承担"送给 LLM 的完整上下文";
- **子块** 在父块内部按 `段落 → 换行 → 中文句界` 递归降级切到 ~450 字符(相邻重叠 ~50),
  承担"进索引做匹配"。

三条硬约束(每个文件都在守):
1. **父块不跨节**: 遇到同级/更浅标题就闭合, 这样一个父块的 heading_path 单义;
2. **表格/图片是原子**: 一格都不切(Markdown 表从中间切开就毁掉行列结构), 独占父块;
3. **确定性**: ID 只由 doc_id + 序号决定, 同一份 `ParsedDoc` 切两遍逐字段一致
   (唯一的非确定字段 created_at 走可注入的 `clock`)。
"""
from __future__ import annotations

from collections.abc import Callable

from ..config import load_config
from ..contracts import ChildChunk, ParentChunk, ParsedDoc
from .children import build_children
from .common import Ctx, compose, page_at, utc_now
from .parents import (
    aggregate,
    block_spans,
    block_type,
    heading_of,
    image_of,
    join_units,
    parent_bounds,
)

__all__ = ["split"]


def split(
    doc: ParsedDoc,
    cfg=None,
    *,
    clock: Callable[[], str] = utc_now,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """切块主入口: `ParsedDoc` → (父块序列, 子块序列), 序号从 1 起、按阅读顺序。

    `clock` 只为单测注入固定时间(`created_at` 是本模块唯一的非确定字段), 生产不用传。
    """
    ctx = Ctx(doc.doc_id, doc.source, clock(), cfg or load_config())
    parent_max = parent_bounds(ctx.cfg)[1]

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for block in aggregate(doc, parent_max):
        head, body, offsets = join_units(block)      # 一个父块 → 标题前缀 / 正文 / 正文起点→页码表
        chunk_type = block_type(block)               # 原子块(表格/图片)还是文本块: 子块层据此分支
        heading, image_path = heading_of(block), image_of(block)   # 整块共有的元信息, 透传给子块

        for index, (core, core_start) in enumerate(
            block_spans(head, body, chunk_type, block, parent_max)
        ):
            # 标题前缀只贴给**第一条**: 退化切分会把一个块拆成多条父块, 每条都贴一遍会让标题在
            # 正文里重复出现(拼起来对不上原文), 而标题本来只属于这段内容的开头。
            text = compose(head, core) if index == 0 else core
            if not text:
                continue                          # 清洗后空掉的条目: 不产块也不占序号
            # 父块的 chunk_id **就是** parent_id 那个 id 串(子块靠它回溯); 它自身的 parent_id 留空
            parent_id = f"{doc.doc_id}:p{len(parents) + 1:03d}"
            parents.append(
                ParentChunk(
                    chunk_id=parent_id,
                    text=text,
                    doc_id=ctx.doc_id,
                    source=ctx.source,
                    page_no=page_at(offsets, core_start),
                    heading_path=heading,
                    chunk_type=chunk_type,
                    created_at=ctx.created_at,
                    image_path=image_path,
                )
            )
            # 子块由 children 侧切好并装字段, 这里只负责把它们挂到刚建好的父块下
            children.extend(
                build_children(
                    parent_id,
                    chunk_type,
                    head,
                    core,
                    core_start,
                    offsets,
                    heading,
                    image_path,
                    ctx,
                )
            )

    return parents, children
