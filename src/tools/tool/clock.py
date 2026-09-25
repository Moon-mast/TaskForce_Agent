"""当前时间工具:所有 ReAct 智能体共用,消除"模型无实时时钟"的盲区。

LLM 的训练数据里没有"现在";投资调研判断财报披露状态、信息时效都需要真实时钟。
以 @tool 形式供各智能体按需调用(单次调用拿精确到分的时间),不做 system 注入——
注入到分钟级会让 ADR-0011 的静态 system 前缀每分钟失效,缓存价值归零。
"""
from datetime import datetime

from langchain_core.tools import tool

_WEEK = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


@tool
def get_current_time() -> str:
    """获取当前本地日期与时间(含星期)。需要判断信息时效、财报/公告披露状态、
    计算时间跨度时调用;禁止凭训练数据猜测当前日期。"""
    d = datetime.now()
    return f"{d:%Y-%m-%d %H:%M} {_WEEK[d.weekday()]}"
