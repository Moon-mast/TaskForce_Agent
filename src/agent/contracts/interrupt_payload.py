"""挂起载荷与合成消息前缀契约(架构整理 c5, 唯一出处, 见 ROADMAP §7)。

挂起载荷:ask/memory 节点经 interrupt 携带的自描述形状 {kind, text}, 双入口直读 kind
分发; classify_interrupt 兼容旧检查点的键名形状(question/proposal, 不迁移存量),
未知形状给 "unknown" 不静默 —— 识别知识从入口代码收编到契约。

合成消息前缀:主图内部合成消息([用户回答] / 子智能体结果已回收 / (系统通知))的标记
常量, 生产者(ask/supervisor/service)与消费者(历史过滤、取真实用户消息)一律 import,
禁止再写字面量 —— 改一处, 双入口与提示词口径同步。

使用位置:
    - agent/ask.py、agent/memory.py:按契约构造挂起载荷;
    - api/routers/chat.py、cli/repl.py:classify_interrupt 直读 kind 分发;
    - agent/supervisor.py、agent/service.py、agent/memory.py:前缀常量;
    - tests/test_interrupt_payload.py:契约与兼容性单测。
"""
from typing import Literal, TypedDict

InterruptKind = Literal["ask", "memory", "unknown"]


class InterruptPayload(TypedDict):
    """挂起载荷(c5 新形状):kind 自描述, 入口不再按键名猜类型。"""

    kind: Literal["ask", "memory"]
    text: str


def ask_payload(question: str) -> InterruptPayload:
    """ask 问询挂起载荷。"""
    return {"kind": "ask", "text": question}


def memory_payload(proposal: str) -> InterruptPayload:
    """memory 确认写入挂起载荷。"""
    return {"kind": "memory", "text": proposal}


def classify_interrupt(value) -> tuple[InterruptKind, str]:
    """识别挂起载荷 → (kind, text)。

    kind 优先(c5 新形状); 兼容旧检查点的键名形状(question/proposal);
    非 dict / 未知形状 → ("unknown", str(value)), 调用方显式兜底而不是猜。
    """
    v = value if isinstance(value, dict) else {}
    kind = v.get("kind")
    if kind in ("ask", "memory"):
        return kind, str(v.get("text", ""))
    if "question" in v:  # 旧形状兼容(存量检查点, 不迁移)
        return "ask", str(v["question"])
    if "proposal" in v:
        return "memory", str(v["proposal"])
    return "unknown", str(v)


# ---- 合成消息前缀(生产/消费两侧的唯一字面出处) ----

PREFIX_USER_ANSWER = "[用户回答]:"  # ask 节点恢复后追加的用户回答标记
PREFIX_SUBAGENT_RESULT = "子智能体结果已回收"  # supervisor 汇总前注入的子结果标记
PREFIX_SYSTEM_NOTICE = "(系统通知)"  # 自动汇总触发语前缀(agent.service.AUTO_NOTICE)

# 历史回填/渲染需过滤的内部合成 user 消息(api/chat.thread_messages 使用)
SYNTHETIC_USER_PREFIXES = (PREFIX_USER_ANSWER, PREFIX_SUBAGENT_RESULT, PREFIX_SYSTEM_NOTICE)
