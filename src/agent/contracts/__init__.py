"""共享契约对外唯一定处:其他包一律 from agent.contracts import ...

本文件作用:
    契约包的统一出口,把各契约 schema 与识别契约聚合到包顶层,收敛 import 路径
    (契约代码本体在 route.py / task_contract.py / summary.py / subgraph.py /
    interrupt_payload.py —— 后者是挂起载荷 kind 契约与合成消息前缀常量, c5)。

使用位置:
    - agent/(supervisor/build/answer/state/ask/memory/service)、agent/subagents/*、
      api/routers/chat.py、cli/repl.py、tests/* 均从本文件 import。
"""

from agent.contracts.interrupt_payload import (
    PREFIX_SUBAGENT_RESULT,
    PREFIX_SYSTEM_NOTICE,
    PREFIX_USER_ANSWER,
    SYNTHETIC_USER_PREFIXES,
    InterruptKind,
    InterruptPayload,
    ask_payload,
    classify_interrupt,
    memory_payload,
    plan_payload,
)
from agent.contracts.plan import Plan, PlanDraft, PlanStep, PlanStepDraft
from agent.contracts.route import Route, Task
from agent.contracts.subgraph import SubgraphContract
from agent.contracts.summary import ResultSummary
from agent.contracts.task_contract import TaskContract

__all__ = [
    "Route",
    "Task",
    "TaskContract",
    "ResultSummary",
    "SubgraphContract",
    "InterruptKind",
    "InterruptPayload",
    "ask_payload",
    "memory_payload",
    "plan_payload",
    "classify_interrupt",
    "PREFIX_USER_ANSWER",
    "PREFIX_SUBAGENT_RESULT",
    "PREFIX_SYSTEM_NOTICE",
    "SYNTHETIC_USER_PREFIXES",
    "Plan",
    "PlanStep",
    "PlanDraft",
    "PlanStepDraft",
]
