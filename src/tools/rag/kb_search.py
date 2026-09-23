"""kb_search:知识库检索工具(内核 rag_v01)—— 迁移期兼容旧 JSON 契约的适配层(07 篇 §4.3)。

**2026-09-22 起这是唯一的检索工具**(里程碑 B 步骤 5/6):旧实现(pgvector + rank_bm25)已下线,
本文件从 `kb_search_v2.py` 改名回来。内核换成了 Milvus + 双路 RRF,但对外仍是同一个工具名
`kb_search`、同一份 JSON 契约 —— 于是 `prompts/subagents/retriever.md`(静态前缀,前缀缓存
ADR-0011)与 `retriever.py` 的 `_collect_hits` 都不用改。契约是接收方定的,不是发送方定的:

接收方 `agent/subagents/retriever.py` 的 `_collect_hits` 按 `hit_key` = `(doc_id, seq)` 去重:

| 键 | 新来源 | 说明 |
|---|---|---|
| `doc_id` | `RetrievedChunk.doc_id` | 直接映射 |
| `filename` | `RetrievedChunk.source` | 直接映射(新系统里 source 是入库时的文件名) |
| `seq` | `chunk_id` 的 `:cNNN` 段 | 旧语义是子块在文档内的序号, 新形如 `doc:p001:c007` |
| `content` | 父块全文 + 分隔线 + `hits[0].text`, 截 500 | **升级点**: 旧工具只回 500 字切片 |
| `score` | `hits[0].rrf_score` | 父块内最佳子块的融合分, 同为"越大越相关", **无需取反** |

**2026-09-23 架构整理 c6**(ROADMAP §7):模块级单例改为工厂 `make_kb_search(search_backend)`
显式注入检索后端 —— retriever 装配时传入,测试直接传假件,不再 monkeypatch 模块私有符号
(原 test_retriever 裸赋值 + autouse 还原曾实测假件泄漏);工具名与 JSON 五键契约不变。
去重键定义也收编于此:`hit_key()` 与 `_seq_of()` 同居一个文件,消费方只调不拼。

使用位置:
    - agent/subagents/retriever.py:build_retriever_graph 经工厂装配,子智能体唯一知识库入口;
    - tests/test_kb_search.py、tests/test_retriever.py:工厂注入假检索。
"""

import json

from langchain_core.tools import tool

SNIPPET_LIMIT = 500  # 单条命中内容上限(字符),防单次工具结果撑爆子图上下文
_PARENT_SEP = "\n---\n"  # 父块与最佳子块的分隔线:让模型看出"这是整节上下文 + 这里是命中处"


def _default_search(query: str, top_k: int):
    """缺省检索后端:转调 rag_v01 的包级 facade(make_kb_search 未注入时使用)。

    `rag_v01` 在函数体内 import 而非模块顶层 —— 顶层 import 会把本次调用无关的解析栈
    (docling/torch)拖进 agent 进程;函数体内 import 后仍走 `sys.modules` 缓存,不重复付代价。
    """
    from rag_v01 import search

    return search(query, top_k=top_k)


def _seq_of(chunk_id: str) -> int:
    """`doc:p001:c007` → 7。取不到数字就回 0(旧契约里 seq 是整数,给 None 会让去重键错位)。"""
    tail = chunk_id.rsplit(":c", 1)
    if len(tail) == 2 and tail[1].isdigit():
        return int(tail[1])
    return 0


def hit_key(item: dict) -> tuple:
    """去重键 (doc_id, seq) 的唯一出处(c6):seq 在本文件从 chunk_id 提出并写进 JSON,
    消费方(retriever._collect_hits)只调本函数取键,不再各自拼元组。"""
    return (item.get("doc_id"), item.get("seq"))


def make_kb_search(search_backend=None):
    """工厂:绑定检索后端返回 kb_search 工具(c6 注入 seam)。

    search_backend: callable(query, top_k) -> list[RetrievedChunk];
    缺省 `_default_search`(惰性转调 rag_v01.search)。
    工具名与 ToolMessage 的 JSON 字符串契约不随后端变化。
    """
    backend = search_backend if search_backend is not None else _default_search

    @tool
    def kb_search(query: str, top_k: int = 5) -> str:
        """在知识库中检索用户上传的文档,返回最相关的片段列表(JSON 数组,无命中时为 [])。

        Args:
            query: 检索查询。用名词短语,一次一条;复合问题拆成多条分别检索。
            top_k: 返回命中条数上限。
        """
        top_k = min(max(int(top_k or 5), 1), 10)  # 钳制:防模型给异常值拖垮检索
        rows = backend(query, top_k)
        return json.dumps(
            [
                {
                    "doc_id": row.doc_id,
                    "filename": row.source,
                    "seq": _seq_of(row.hits[0].chunk_id),
                    "content": (row.parent_text + _PARENT_SEP + row.hits[0].text)[:SNIPPET_LIMIT],
                    "score": row.hits[0].rrf_score,
                }
                for row in rows
            ],
            ensure_ascii=False,
        )

    return kb_search
