"""get_current_time 工具测试:返回格式与工具名(供全部 ReAct 智能体共用的时钟)。"""

from tools.tool.clock import get_current_time

_WEEK = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def test_returns_datetime_with_weekday():
    out = get_current_time.invoke({})
    assert out[:2] == "20"  # 日期以年份开头
    assert any(w in out for w in _WEEK)  # 含中文星期
    assert ":" in out  # 含时间部分


def test_tool_name():
    assert get_current_time.name == "get_current_time"
