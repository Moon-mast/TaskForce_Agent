"""知识库工具包:检索工具 `kb_search`(适配 rag_v01 内核)。

2026-09-22 里程碑 B 步骤 6:解析/切块/BM25/pgvector 存储与旧管理命令行已下线(内核换成
`rag_v01`:docling 解析 + 父子分块 + Milvus 双路 RRF)。
2026-09-23 架构整理 c2:embedding 客户端从本包下沉 `settings/embeddings.py`,
消除 settings→tools 反向依赖(见 ROADMAP §7),本包只剩 kb_search 适配层。
"""
