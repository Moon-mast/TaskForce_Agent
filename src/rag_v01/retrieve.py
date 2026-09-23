"""retrieve.py —— 混合检索编排: 两路召回 → RRF 融合 → 父块回溯去重(05 篇)。

分工(05 篇 §2.4): **融合排序在子块层做, 送给模型的内容在父块层取** —— 子块(~450 字符)保证
embedding 语义集中、BM25 词项命中精准; 父块(1200~1800 字符)提供完整语境。

本文件只 import 同级的 config / contracts / embed / store, 不 import 项目其它包, 整目录可搬走。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from . import store
from .config import Config, load_config
from .contracts import ChildHit, RetrievedChunk
from .embed import embed_query
from .store import Hit, ParentRow


@dataclass(frozen=True,slots=True)
class FusedChild:
    """融合后的一条子块候选(**retrieve.py 内部结构, 不进 contracts.py**)。

    比 store 的 `Hit` 多出来的是"两路各自的名次": 融合只吃名次, 而两个名次是调试与调参时最
    想看的中间量 —— 05 篇 §2.3 选客户端手写 RRF 而不是 Milvus 原生 `hybrid_search` 的核心理由
    就是"中间过程可见"。`dense_rank` / `sparse_rank` 用 `None` 表示"该路没进前 N", 不用 0
    哨兵(名次从 1 起, 0 会与"未命中"歧义)。
    """

    chunk_id: str
    parent_id: str
    text: str
    chunk_type: str
    dense_rank: int | None
    sparse_rank: int | None
    rrf_score: float

def rrf_fuse(
        dense_hits:list[Hit],
        sparse_hits:list[Hit],
        *,
        k:int,
        top_k:int
)->list[FusedChild]:
    """两路名次 RRF 融合, 返回按融合分降序的前 `top_k` 条候选(05 篇 §4.1)。纯函数, 可直接单测。

    为什么只吃名次: 两路原始分**量纲不可比**(dense 是有界相似度, sparse 是无界 BM25 分, 且随
    语料分布漂移), 而名次是分布无关的单调变换。公式 `Σ 1/(k+rank)` 的含义是"每路按名次发一张
    衰减选票", 双路都靠前的候选票数叠加自然胜出 —— 语义路负责泛化, 关键词路负责精确。

    两处刻意不写特判:
    - **单路为空天然退化为该路名次排序**: 缺席的那一路贡献 0, 公式自己就处理了;
    - **`k` / `top_k` 不给默认值**: 走 `RAG2_RRF_K` / `RAG2_RETRIEVE_TOP_K`, 只在 config.py 里
        有一处默认值。若这里再写一份 `k: int = 60`, 两处默认值迟早漂移(与 04 篇 `_cfg()` 懒加载
        同一个理由: 单一事实源)。
    """
    if top_k <= 0:
        raise ValueError(f"top_k 必须为正整数, 收到 {top_k}")

    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int | None]] = defaultdict(lambda: {"dense": None, "sparse": None})
    info: dict[str, Hit] = {}
    for route, hits in (("dense", dense_hits), ("sparse", sparse_hits)):
        for rank, hit in enumerate(hits, start=1):  # 名次从 1 起, 与 RRF 原论文一致
            scores[hit.chunk_id] += 1.0 / (k + rank)
            ranks[hit.chunk_id][route] = rank
            # 双路命中同一条时内容必然相同(同一个 collection 的同一行), 留先见到的那条即可;
            # 用 setdefault 是为了"保留第一条"而不是被后一条覆盖
            info.setdefault(hit.chunk_id, hit)

    # 同分按 chunk_id 兜底: `1/(k+rank)` 很容易凑出并列分, 少了这把钥匙结果就不可复现,
    # 而单测断言与评估回归都依赖可复现(05 篇坑位 5)。注意**先排序后 round**
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        FusedChild(
            chunk_id=chunk_id,
            parent_id=info[chunk_id].parent_id,
            text=info[chunk_id].text,
            chunk_type=info[chunk_id].chunk_type,
            dense_rank=ranks[chunk_id]["dense"],
            sparse_rank=ranks[chunk_id]["sparse"],
            rrf_score=round(score, 6),  # 只为输出整洁; 排序用的是未 round 的原值
        )
        for chunk_id, score in ordered[:top_k]
    ]

@lru_cache(maxsize=1)
def _cfg()->Config:
    """配置懒加载(与 embed.py / store.py 同款)。

    单测注假配置、入口先载 .env 都不受 import 顺序影响。
    """
    return load_config()

def search(
        query:str,
        top_k:int|None=None,
        *,
        warnings:list[str]|None=None
)->list[RetrievedChunk]:
    """一句话 query → 上下文列表(05 篇 §3.1 全链路)。`top_k` 默认取 `RAG2_RETRIEVE_TOP_K`。

    错误语义(05 篇 §4.3)是这一层的硬约定: **`[]` 只表示"库里确实没有相关内容"**。embedding
    后端连不上、Milvus 连不上这类基础设施故障一律向上抛, 绝不吞成空结果 —— 上层(06 篇评估、
    07 篇移植后的 agent)要靠这个区分"知识库答不了"和"服务挂了", 混同就会把故障掩盖成能力不足。
    """
    cfg=_cfg()
    want=cfg.retrieve_top_k if top_k is None else top_k
    if want<=0:
        raise ValueError(f"top_k 必须为正整数, 收到 {want}")  # 在花 embedding 调用之前就失败
    vec = embed_query(query)  # 与入库同一后端、同一模式(04 篇); 查询侧永远送文本
    dense = store.dense_search(vec, limit=cfg.candidate_top_n)  # top-20, ef 走 RAG2_HNSW_EF
    sparse = store.sparse_search(query, limit=cfg.candidate_top_n)  # 原文直传, 服务端 jieba 分词
    if not dense and not sparse:
        return []  # 两路皆空才算"没查到"; 单路为空由 RRF 公式自然退化成单路排序

    fused = rrf_fuse(dense, sparse, k=cfg.rrf_k, top_k=want)
    # 父块批量点查: 去重且保序(dict.fromkeys)——重复的 id 会让 filter 表达式白长一截
    parents = store.get_parents(list(dict.fromkeys(child.parent_id for child in fused)))

    merged: list[RetrievedChunk] = []
    index_by_parent: dict[str, int] = {}
    for child in fused:
        row = parents.get(child.parent_id)
        if row is None:
            # children 里每个 parent_id 都该在 parents 里有行(同一次 ingest 写入), 缺了就是数据
            # 不一致: 跳过这一条并留痕, 不让整次检索崩掉, 剩下的结果照常返回(05 篇坑位 6)
            if warnings is not None:
                warnings.append(f"{child.parent_id}: 父块查不到, 已跳过该条结果")
            continue
        at = index_by_parent.get(child.parent_id)
        if at is not None:
            # 同父去重**发生在截断之后**: 命中明细追加进已有条目, 不新增父块条目 —— 保证进入
            # 上下文的每个父块都确有命中子块, 且父块全文永不重复拼接(05 篇坑位 1)
            merged[at].hits.append(_child_hit(child))
            continue
        index_by_parent[child.parent_id] = len(merged)
        merged.append(_retrieved(child, row, rank=len(merged) + 1))
    return merged

def _child_hit(child: FusedChild) -> ChildHit:
    """内部结构 → 契约类型: 只搬契约里有的字段, `FusedChild` 不外泄。"""
    return ChildHit(
        chunk_id=child.chunk_id,
        text=child.text,
        chunk_type=child.chunk_type,
        dense_rank=child.dense_rank,
        sparse_rank=child.sparse_rank,
        rrf_score=child.rrf_score,
    )


def _retrieved(child: FusedChild, row: ParentRow, *, rank: int) -> RetrievedChunk:
    """新父块条目: 上下文主体取**父块**全文, 元信息也以父块为准(05 篇 §3.2 设计说明)。

    `rrf_rank` 用"在最终结果里的位置"(1 起, 连续)而不是子块的融合槽位: 同父合并后槽位会跳号,
    而契约上这个字段的语义是"本父块在最终结果中的名次"。
    """
    return RetrievedChunk(
        parent_id=child.parent_id,
        parent_text=row.text,
        rrf_rank=rank,
        hits=[_child_hit(child)],
        doc_id=row.doc_id,
        source=row.metadata.get("source", ""),
        page_no=_page_no(row.metadata),
        heading_path=row.metadata.get("heading_path", ""),
        metadata=row.metadata,
    )


def _page_no(metadata: dict) -> int | None:
    """parents 里无页码存的是 -1 哨兵(04 篇坑位 7c), 契约上要还原成 None。

    只在这一处转换: 上游(03 篇)的语义本来就是 None, 存储细节不该漏到调用方; 原始 `metadata`
    里仍是 -1, 需要看存储事实时还有它。
    """
    value = metadata.get("page_no")
    return None if value is None or value < 0 else int(value)
