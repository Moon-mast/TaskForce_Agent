"""psycopg 连接辅助 + pgvector 扩展幂等创建(00 模块契约,其他模块禁止复制定义)。

使用位置:
    - settings/db/checkpointer.py:经 ConnectionPool 自行建池,不直接用本文件;
    - tests/test_memory.py:测试里直连清理临时表。

`get_conn` / `ensure_vector_ext` 在 2026-09-22 之后**没有生产路径的调用方**了 —— 原来的调用者是旧
`tools/rag/store.py`(已下线,知识库内核换成 `rag_v01` 走 Milvus),长期记忆表的建表由 langgraph
`PostgresStore.setup()` 自己负责。保留它们作为"新环境手工建库/迁移"的工具函数(00 模块契约里就有),
不是死代码清理的对象。
"""

import psycopg


def get_conn(database_url: str):
    """返回 psycopg 连接上下文管理器(退出自动提交/回滚并关闭)。"""
    return psycopg.connect(database_url)


def ensure_vector_ext(database_url: str) -> None:
    """幂等创建 pgvector 扩展(重复执行不报错)。"""
    with get_conn(database_url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
