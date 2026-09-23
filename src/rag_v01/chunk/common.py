"""一次切块内不变的量(`Ctx`)与三个跨两侧共用的小工具。

为什么单独一个文件: 父块侧(`parents.py`)与子块侧(`children.py`)都要用到这几个东西,
放在任何一侧都会让另一侧反向 import —— 共享的放中间, 依赖方向永远是"两侧 → common"。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from ..config import Config

# 单元之间的拼接符: 用空行而不是换行 —— 子块切分器的第一层分隔符就是 "\n\n",
# 单元边界留着空行, 子块才能优先落在"段与段之间", 而不是从段落中间切。
UNIT_JOIN = "\n\n"


def utc_now() -> str:
    """created_at 的默认来源: UTC ISO 8601(秒精度够用, 且不带本地时区歧义)。"""
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class Ctx:
    """一次 `split` 里对所有 chunk 都相同的量(透传元信息), 攒成一个对象免得参数表拉长。

    这四项会原封不动写进**每一个** chunk(父块与子块都写), 所以它是"透传"的: 谁都不改它,
    只用它填九字段; `frozen=True` 就是防着某个函数图省事就地改一下。
    """

    doc_id: str        # 文件内容 sha256 前 16 位: 归属标识, 也是"按文档删除重灌"的键(04 篇)
    source: str        # 文件路径(相对语料根): 检索结果要能回答"依据来自哪个文件"
    created_at: str    # 这次切块的统一时间戳: 一次切块只取一次, 所有块共用一个值
    cfg: Config        # 切块口径(目标长度/重叠): 子块侧按它切, 父块侧按它算上限


def page_at(offsets: list[tuple[int, int | None]], pos: int) -> int | None:
    """片段起点落在哪个单元里 → 该单元的页码(单元页码全为 None 的 md/docx 自然是 None)。

    取"最后一个起点 ≤ pos"的单元; pos 越界就落到最后一条 —— 不报错, 给个最近的答案,
    因为页码只是溯源信息, 不值得为它让整条流水线失败。
    """
    page: int | None = None
    for start, unit_page in offsets:          # 表是按起点升序的, 一旦越过 pos 就没有更近的单元了
        if start > pos:
            break
        page = unit_page                      # 一路滚动记下最后一个"起点 ≤ pos"的单元页码
    return page

def compose(head: str, core: str) -> str:
    """标题前缀 + 被切出来的正文拼成最终文本(标题进正文)。

    四种组合都要能返回: 有前缀有正文、只有正文、只有前缀(空壳标题块)、都空(调用方会跳过)。
    """
    if head and core:
        return f"{head}{UNIT_JOIN}{core}"      # 常态: 标题 + 空行 + 正文, 空行让子块能在段间落刀
    return head or core                        # 只有标题(空壳标题块) 或 只有正文(无标题的块)
