"""`settings/embeddings.py` 单测 —— 长期记忆向量化客户端(原 tools/rag/embed.py, 2026-09-23 下沉)。

2026-09-22 里程碑 B 删掉 parse/split/bm25/store/kb_search 时 embed 被保留(长期记忆在用);
2026-09-23 架构整理 c2 把它整体下沉 `settings/embeddings.py`(消除 settings→tools 反向依赖),
本文件随迁:只测不依赖 DB 的两条用例, 覆盖不能跟着文件搬家丢掉。

需要智谱 key 的用例未配置时自动 skip, 与项目其它外部依赖用例同一口径。
"""

import pytest


def _embedding_ready() -> bool:
    from settings.config import get_settings

    return bool(get_settings().embedding_api_key)


@pytest.mark.skipif(not _embedding_ready(), reason="EMBEDDING 未配置")
def test_embedding_dim_positive():
    """维度来自真实探针(建表 vector(N) 用它), 必须是个正数。"""
    from settings.embeddings import embedding_dim

    assert embedding_dim() > 0


def test_embed_texts_without_key_raises_actionable_error(monkeypatch):
    """没配 key 时给的是可操作的配置报错, 不是底层 401 —— 运维照着文案就能修。"""
    from settings import embeddings

    class _NoKey:
        embedding_api_key = ""

    monkeypatch.setattr(embeddings, "get_settings", lambda: _NoKey())
    with pytest.raises(RuntimeError, match="Embedding 未配置"):
        embeddings.make_embeddings()
