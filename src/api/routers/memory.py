"""记忆路由(模块 11 T3):GET /memory / DELETE /memory/{key}。

查询与 namespace 收编 agent/memory_ctx(2026-09-23 架构整理 c4, ROADMAP §7):
本路由只做 HTTP 渲染, 语义与 REPL /memory 同源(memory_ctx.list_memories/delete_memory,
namespace 单一出处 MEMORY_NS = ("memory", "default"))。
"""
from fastapi import APIRouter

from agent import memory_ctx

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("")
def list_memory():
    return {"items": memory_ctx.list_memories()}


@router.delete("/{key}")
def delete_memory(key: str):
    memory_ctx.delete_memory(key)
    return {"ok": True, "key": key}
