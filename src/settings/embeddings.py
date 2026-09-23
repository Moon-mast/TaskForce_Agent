"""智谱 embedding 客户端(OpenAI 兼容):配置校验 + 实测维度(供建表 vector(N))。

2026-09-23 架构整理 c2 从 `tools/rag/embed.py` 整体下沉至此:它是长期记忆向量化的
基础设施, 原居 tools 包导致 `settings/db/store` 反向 import tools、与 "tools→settings"
单向依赖成环; 下沉后 `tools/rag` 只剩 kb_search 适配层(见 ROADMAP §7 登记)。
知识库内核另有独立 embedding(`rag_v01/embed.py`, 百炼 qwen3-vl-embedding), 两套各自独立。

使用位置:
    - settings/db/store.py:长期记忆写入/检索的向量化;
    - tests/test_rag_embed.py:维度与配置缺失报错测试。
"""

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from settings.config import get_settings

EMBEDDING_ERROR = (
    "Embedding 未配置:请在 .env 填写 "
    "EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL(智谱)"
)


def make_embeddings() -> OpenAIEmbeddings:
    s = get_settings()
    if not s.embedding_api_key:
        raise RuntimeError(EMBEDDING_ERROR)
    return OpenAIEmbeddings(
        base_url=s.embedding_base_url,
        api_key=s.embedding_api_key,
        model=s.embedding_model,
    )


@lru_cache(maxsize=1)
def embedding_dim() -> int:
    """实测 embedding 维度(进程内只探测一次),供建表 vector(N) 与一致性断言。"""
    vec = make_embeddings().embed_query("维度探测")
    return len(vec)


ZHIPU_BATCH_LIMIT = 64  # 智谱 embedding 单请求 input 数组上限(超限报错误码 1214)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量向量化;超过智谱 64 条/请求上限时自动分批,结果按原顺序拼接。"""
    if not texts:
        return []
    client = make_embeddings()
    return [
        vec
        for i in range(0, len(texts), ZHIPU_BATCH_LIMIT)
        for vec in client.embed_documents(texts[i:i + ZHIPU_BATCH_LIMIT])
    ]
