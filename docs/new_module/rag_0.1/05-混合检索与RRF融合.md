# 05-混合检索与RRF融合

> 对应实现: `src/rag_v01/retrieve.py`。本篇是检索链路的编排层设计: 两路召回(top-20)、
> 客户端手写 RRF(k=60)融合、按 parent_id 回溯父块、同父去重合并, 最终产出 `list[RetrievedChunk]`。
> 建表与原子查询原语见 [./04-嵌入与Milvus索引.md](./04-嵌入与Milvus索引.md), 父子分块产出的
> parent_id 关系见 [./03-父子分块策略.md](./03-父子分块策略.md)。

## 一、模块目标与边界

**目标**: 给定一句自然语言 query, 从 Milvus 的 children / parents 两个 collection 中
取回"够模型用的上下文"——命中细节(子块级)+ 完整语境(父块级), 全程同步、零项目内依赖。

**边界**(不做什么):

- 不做 embedding 计算——query 向量化调 `embed.py` 的 `embed_query()`(04 篇);
- 不直接操作 pymilvus——dense/sparse 查询、按主键查 parents 等原子原语由 `store.py` 提供
  (04 篇), 本模块只做编排与融合;
- 不做 rerank 模型、不做查询改写(query 重写是后续可选增强, 当前直查);
- 不做答案生成——06 篇 evaluate.py 才拼上下文调 LLM。

**对外唯一入口**: `search(query, top_k=None) -> list[RetrievedChunk]`(落地时把骨架的 `top_k=5`
改成 `None` —— 默认值只允许存在于 config.py 一处, 见 §4.6), 由包级 facade(`__init__.py`)透出(07 篇)。

## 二、设计决策与理由

### 2.1 为什么两路各自取 top-20, 而不是各取 top-5 或 top-100

RRF 的融合力来自"候选池比最终结果深"。设最终取 top-5:

- **池深 = top-5**: 融合退化为两路各自前 5 的对齐求和, 一路独有的第 6~20 名优质结果
  (恰恰是混合检索要捞的"另一路盲区")永远进不了候选, 融合形同虚设;
- **池深 = top-100**: 长尾噪声进入候选; 且 `1/(60+rank)` 在 rank 大时贡献趋同
  (rank=80 与 rank=100 的单项分差 < 0.0004), 深名单的区分力已被 k=60 压平, 白付查询开销;
- **top-20**: 约为最终深度的 4 倍, 与 k=60 的分辨力匹配(rank 1→20 的单项分从 1/61≈0.0164
  平滑降到 1/80=0.0125, 头部名次间仍有区分度), 也是 RRF 工程实践的常用候选深度。

### 2.2 为什么用 RRF, 而不是分数加权融合

两路的原始分**量纲不可比**: dense 路(HNSW + COSINE)返回的是相似度, 有界; sparse 路
(Milvus 内置 BM25 Function)返回的是无界正分, 值随词频/文档长度分布漂移, 语料一换整体
分布就变。直接加权(`α·s_dense + β·s_sparse`)要调 α/β, 且调好的权重对分数分布敏感,
换 embedding 模型或语料规模变化后可能失效。

RRF 只取**名次**: `score(d) = Σ_路 1/(k + rank_路(d))`。名次是分布无关的单调变换——
无论原始分是 0.87 还是 23.5, 都只关心"它排第几"。推导直觉:

- 每一路给每个候选发一个随名次衰减的"选票", 第 1 名 1/(k+1), 第 2 名 1/(k+2)……;
- 两路都命中的候选票数**叠加**, 所以"双路都靠前"的子块自然胜出——这正是混合检索
  想要的性质: 语义路负责泛化(同义改写), 关键词路负责精确(术语、编号、专名);
- k=60 是原论文(Cormack et al., 2009)的经验值, 作用是**平滑**: k 越大, 相邻名次的
  票差越小, 融合越"民主"(不偏头部); k→0 则退化成"只看谁拿过第一"。top-20 深度下
  k=60 表现稳定, 且与 Milvus 原生 `RRFRanker` 的默认口径一致, 对照实验可比。

附带红利: 因为只吃名次, 本模块完全不需要关心 dense 路原始 distance 的方向语义与
SDK 版本差异(COSINE 下 Milvus 返回的是"越大越优"的相似度, 与欧氏距离类 metric 相反),
这一类坑被整体绕开。

### 2.3 为什么客户端手写 RRF, 而非 Milvus 原生 hybrid_search

Milvus 2.5+ 提供 `collection.hybrid_search(reqs=[...], rerank=RRFRanker(60), limit=5)`,
服务端一次完成双路查询 + 融合。**主实现不选它, 理由**:

1. **中间过程不可见**: 原生版只返回融合后的最终列表, **拿不到每个候选在两路各自的
   名次与得分构成**。教学与调参恰恰要观察"哪路召回强、某子块为什么排第一";
2. **可测性**: 手写 `rrf_fuse` 是纯函数, 构造两路名次表即可单测(手算期望分数),
   不依赖真实 Milvus;原生版要测融合逻辑必须起库;
3. **备胎兼容**: 全局定稿的备胎方案(进程内 rank_bm25 + jieba 客户端自算 BM25,
   应对 milvus-lite/低版本下内置 BM25 Function 异常)只替换 sparse 路的数据来源,
   手写融合层代码**结构不变**;选了原生 hybrid_search 则备胎切换时检索层要重写;
4. **可演进**: 想换 WeightedRanker 思路、加 per-路过滤、改 k, 都是改自己代码里的一行。

代价是两次 RPC + 客户端计算, 延迟略高于原生版——单机单用户场景可忽略。
对照写法见 §4.4, 仅作教学参照, 不进主实现。

### 2.4 为什么命中子块后要回溯父块

父子分块的分工就是"小块检索、大块喂模型"(03 篇): 检索粒度小(约 450 字符)以保证
embedding 语义集中、BM25 词项命中精准; 但 ~450 字符往往不足以独立成语境, 直接拼
top-5 子块容易断头断尾。parents collection 不建向量索引、仅按主键查(04 篇),
回溯成本是一次点查, 换来 1200~1800 字符的完整段落语境。**融合排序在子块层做,
送给模型的内容在父块层取**——两层各司其职。

### 2.5 备选方案对比小结

| 方案 | 结论 | 一句话理由 |
|---|---|---|
| 分数加权融合 | 弃 | 量纲不可比, 权重难调且随分布漂移 |
| Milvus 原生 hybrid_search + RRFRanker | 仅对照 | 服务端融合, 过程不可见、不可单测、备胎切换需重写 |
| 客户端手写 RRF(k=60) | **选定** | 名次无量纲、可单测、中间量全可见、备胎兼容 |
| 融合后再跑 rerank 模型 | 暂不做 | 单用户本地场景延迟预算紧, 留作后续增强 |

## 三、数据流与接口契约

### 3.1 检索全链路图

```
query(原文)
   │
   ├─> embed_query(query) ──> dense 向量(1024 维)                    [embed.py]
   │        │
   │        v
   │   children.dense_search(vec, limit=20)      # HNSW ANN, ef=64~128
   │        │
   │        v
   │   dense 命中列表(≤20 条, 按名次)
   │        │
   ├─> 原文直传 ─> Milvus 服务端 jieba 分词 + BM25 Function 生成查询稀疏向量
   │        │
   │        v
   │   children.sparse_search(query, limit=20)   # 内置 BM25
   │        │
   │        v
   │   sparse 命中列表(≤20 条, 按名次)
   │        │
   v        v
 rrf_fuse(dense, sparse, k=60, top_k=5)                             [retrieve.py]
   score(d) = 1/(60 + rank_dense(d)) + 1/(60 + rank_sparse(d))
   (仅单路命中只累加该项; 两路全空直接返回 [])
   │
   v
 融合后 top-5 子块(按 rrf_score 降序, 同分按 chunk_id 稳定排序)
   │
   v
 按 parent_id 回溯: parents 按主键批量点查(无向量索引)               [store.py]
   │
   v
 同父去重合并: top-5 中多个子块同父 → 合并为 1 个 RetrievedChunk
   │
   v
 list[RetrievedChunk](父块上下文 + 命中子块明细, 条数 ≤ top_k)
```

### 3.2 契约类型(contracts.py 中定义, 本篇消费)

```python
@dataclass
class ChildHit:              # 一条命中子块明细
    chunk_id: str
    text: str
    chunk_type: str          # text|table|image
    dense_rank: int | None   # dense 路名次(1 起); 该路未命中为 None
    sparse_rank: int | None  # sparse 路名次(1 起); 该路未命中为 None
    rrf_score: float

@dataclass
class RetrievedChunk:        # search() 的返回单元, 一个父块一条
    parent_id: str
    parent_text: str         # 父块全文(检索命中的上下文主体)
    rrf_rank: int            # 本父块在最终结果中的名次(1 起)
    hits: list[ChildHit]     # 本父块下进入 top-5 的命中子块明细(≥1 条)
    doc_id: str              # —— 以下为父块元信息透传(字段名与切块层逐字一致)
    source: str
    page_no: int | None
    heading_path: str        # 标题栈快照, 各级以 " > " 连接(03 篇 §3.1)
    metadata: dict           # 其余元信息(char_len、created_at、image 路径等)
```

`ChildHit` 是 `RetrievedChunk` 的辅助类型, 与六个主契约同放 `contracts.py`, 不计入"六契约"之列;
`FusedChild` / `Hit` / `ParentRow` 这类纯中间结构仍不进 `contracts.py`(04 篇 §3.2 同款约定)。

设计说明:

- `rrf_rank` 挂在父块上, 取**该父块在最终结果里的名次(1 起, 连续)** —— 父块在结果列表里的
  顺序就是它最佳子块的 RRF 顺序, 命中明细全部留在 `hits` 里, 不丢信息。落地时修正过一次措辞:
  §3.1 原写"取该父块最高名次子块的**融合槽位**", 与同父合并后的"最终名次"不是同一个数(槽位会
  跳号), 以字段注释为准取"最终名次";
- 元信息透传以**父块自身**的元信息为准(03 篇定义父块同样携带九字段, parent_id 为空);
  子块各自的 page_no 等如需细看, 由 `hits` 里的 text 与调试输出承担, 不在契约里膨胀;
- `dense_rank`/`sparse_rank` 用 `int | None` 而非 0 哨兵: 名次从 1 起, 0 会与
  "未命中"歧义, None 显式表达"该路没进前 20"。

### 3.3 依赖的模块接口

| 来源 | 接口 | 说明 |
|---|---|---|
| contracts.py | `RetrievedChunk` / `ChildHit` | 本篇产出契约 |
| embed.py | `embed_query(query) -> list[float]` | 1024 维, 与入库同一模型同一模式(04 篇); **查询侧永远送文本**, 与图片块同处一个向量空间(多模态 embedding), 文本 query 可跨模态命中图片块 |
| store.py | `dense_search(vec, limit) -> list[Hit]` | Hit 含 chunk_id/parent_id/text/chunk_type 等透传字段, 已按相关度降序 |
| store.py | `sparse_search(query, limit) -> list[Hit]` | 同上, BM25 路 |
| store.py | `get_parents(parent_ids) -> dict[str, ParentRow]` | parents 主键批量点查, 返回 parent_id → text+元信息 |
| config.py | 候选深度(20)、top_k(5)、RRF k(60)、ef(64~128) | RAG2_ 前缀环境变量, 变量名清单见 [./01-环境与依赖.md](./01-环境与依赖.md) |

## 四、实现要点(API + 骨架 + 坑位)

### 4.1 rrf_fuse 纯函数(本模块核心, 允许最详)

```python
from collections import defaultdict

def rrf_fuse(dense_hits: list[Hit], sparse_hits: list[Hit],
             k: int = 60, top_k: int = 5) -> list[FusedChild]:
    """两路名次 RRF 融合, 返回按 rrf_score 降序的前 top_k 个子块。纯函数, 可直接单测。"""
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int | None]] = defaultdict(
        lambda: {"dense": None, "sparse": None})
    for route, hits in (("dense", dense_hits), ("sparse", sparse_hits)):
        for rank, hit in enumerate(hits, start=1):     # 名次从 1 起
            scores[hit.chunk_id] += 1.0 / (k + rank)   # 未命中的一路天然贡献 0
            ranks[hit.chunk_id][route] = rank
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))  # 同分按 chunk_id 定序
    return [FusedChild(chunk_id=cid, rrf_score=round(s, 6),
                       dense_rank=ranks[cid]["dense"],
                       sparse_rank=ranks[cid]["sparse"])
            for cid, s in ordered[:top_k]]
```

要点: `FusedChild` 是 retrieve.py 的内部结构(不进 contracts.py), 还需从两路 Hit 里
带回 parent_id/text/chunk_type 等透传字段(骨架略); `round(s, 6)` 只为输出整洁,
排序发生在 round 之前。

### 4.2 search 编排骨架

```python
def search(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    vec = embed_query(query)                        # 04 篇; 连接 embedding 后端失败会抛错
    dense = store.dense_search(vec, limit=cfg.candidates)      # top-20, ef=cfg.ef
    sparse = store.sparse_search(query, limit=cfg.candidates)  # top-20, 服务端分词
    if not dense and not sparse:
        return []                                   # 真没查到: 返回空, 不报错
    fused = rrf_fuse(dense, sparse, k=cfg.rrf_k, top_k=top_k)
    parents = store.get_parents({f.parent_id for f in fused})  # 主键批量点查
    merged: list[RetrievedChunk] = []
    index_by_parent: dict[str, int] = {}
    for pos, f in enumerate(fused, start=1):        # 按融合名次遍历
        if f.parent_id in index_by_parent:          # 同父去重合并: 追加 hit, 不新增条目
            merged[index_by_parent[f.parent_id]].hits.append(f.as_child_hit())
            continue
        index_by_parent[f.parent_id] = len(merged)
        merged.append(f.as_retrieved(parents[f.parent_id], rrf_rank=pos))
    return merged
```

单测姿势: monkeypatch `store.dense_search` / `store.sparse_search` /
`store.get_parents` / `embed_query`, 用构造的 Hit 列表驱动, 全程不连真实 Milvus。

### 4.3 降级与空结果的错误语义

- **单路为空直通**: query 分词后全为停用词、或词项全部未命中时 sparse 路可能返回空;
  RRF 公式天然支持(缺席一路贡献 0), 剩下一路按自身名次排序即退化为单路检索,
  代码**无需特判分支**, 只有"两路皆空"才短路返回 `[]`——这是公式自带的优雅;
- **Milvus 连接失败 ≠ 空结果**: store.py 捕获 pymilvus 的连接类异常后向上抛
  `RuntimeError`(message 指引: Milvus 未启动? 见 01 篇 docker compose 片段),
  retrieve.search **不吞异常**。`[]` 的语义严格保留给"库里确实没有相关内容",
  基础设施故障必须炸出来——若混同, 上层(06 篇评估、07 篇移植后的 agent)会把
  "数据库挂了"误判成"知识库答不了", 掩盖故障;
- `top_k <= 0` 视为参数错误直接抛 `ValueError`(契约防御, 不静默返回空)。

### 4.4 对照: Milvus 原生 hybrid_search + RRFRanker(仅教学对照, 不作主实现)

```python
from pymilvus import AnnSearchRequest, RRFRanker

dense_req = AnnSearchRequest(
    data=[vec], anns_field="dense",
    param={"metric_type": "COSINE", "params": {"ef": 96}},   # ef 在 param["params"]
    limit=20)                                                # 每路候选深度
sparse_req = AnnSearchRequest(
    data=[query], anns_field="sparse",                       # 稀疏路直传原文
    param={"metric_type": "BM25"},                           # drop_ratio_search 默认 0, 不开
    limit=20)
res = collection.hybrid_search(
    reqs=[dense_req, sparse_req], rerank=RRFRanker(60), limit=5,
    output_fields=["chunk_id", "parent_id", "text", "chunk_type"])
```

与手写版的差异:

| 维度 | 原生 hybrid_search | 手写(主实现) |
|---|---|---|
| 融合位置 | Milvus 服务端 | 客户端 retrieve.py |
| RPC 次数 | 1 次(双路+融合) | 2 次 search + 1 次父块点查 |
| per-路名次/得分构成 | **不可见**, 只有融合后结果 | 全部可见(dense_rank/sparse_rank/rrf_score) |
| 单测 | 需起真实 Milvus | 纯函数 + monkeypatch 即可 |
| 备胎(rank_bm25)切换 | 检索层需重写 | 只换 sparse_hits 来源, 融合层不动 |
| 等价性 | RRFRanker(60) 与手写公式同构(名次从 0 还是 1 起等实现细节需实测对齐, 见坑位 5) | — |

**坑位 5 说明**: `RRFRanker` 内部名次约定(0 起还是 1 起、并列名次如何计)官方文档未
完全展开, 与手写版做对照实验时同一名次的候选可能差一个位次级别的微小分差。验证办法:
同一 query 两种方式各跑一次, 对比 top-5 的 chunk_id 序列; 序列一致或仅同分相邻互换
即可认定等价。此对照只用于验证手写实现的正确性, 不进主链路。

### 4.5 坑位清单

1. **同父多子的合并策略**: 合并发生在**截断之后**——先取 top-5 子块, 再按 parent_id
   去重, 所以最终父块条数 ≤ top_k(极端时 5 个子块同父, 只返回 1 条 RetrievedChunk)。
   这是有意的: 保证进入上下文的每个父块都确有命中子块、rrf_rank 语义清晰、父块文本
   绝不重复拼接。代价是多样性下降(一个强父块霸屏), 若评估(06 篇)发现 context_recall
   卡在这里, 调整旋钮是**加深候选**(如两路 top-30)或限制单父命中条数——均属参数级
   调整, 不改链路结构。`hits` 按融合名次追加, 明细不丢。
2. **ef 是查询期参数, 传入方式别搞错**: HNSW 建索引时只定 M=16 / efConstruction=200
   (04 篇), ef 在**每次 search 的 param 里传**: `param={"metric_type": "COSINE",
   "params": {"ef": 96}}`。两个子坑: (a) ef 必须 ≥ limit(我们 64~128 ≥ 20),
   ef < limit 时各版本行为不一致(有的自动抬高、有的召回不足), 不依赖;
   (b) 写错层级(把 ef 直接放 param 顶层)低版本 pymilvus 不报错但静默用默认 ef,
   单测里 mock 校验 param 结构可防。
3. **查询侧与入库侧分词一致性(主链路)**: 选 Milvus 内置 BM25 Function 的核心收益就是
   分词在**服务端**由 collection 上配置的 jieba analyzer 统一完成, 入库与查询天然同一
   份逻辑, 不存在两侧漂移。但要注意: (a) analyzer 参数配在 schema 上, **建表后改
   jieba 配置需重建 collection**(drop 后重灌), 所以 04 篇建表时的 analyzer_params 要
   一次定稿; (b) 切到备胎方案(进程内 rank_bm25 + jieba)后, 查询侧分词必须与入库侧
   调用**同一个函数同一参数**(如都用 `jieba.lcut` 或都用 `cut_for_search`, HMM 开关
   一致), 两边各写一份分词调用迟早漂移, 应抽成公共函数; (c) 入库文本经 clean.py
   清洗而 query **不做** clean(用户输入保持原样), 全角/半角差异会漏配——这是已知
   取舍, 评估若暴露再考虑查询侧做轻量归一化。
4. **连接失败与空结果的语义分层**: 见 §4.3, 不重复。移植(07 篇)时上层 agent 依赖
   "search 抛异常 = 基础设施故障, 返回 [] = 无相关知识"这一约定来决定降级话术。
5. **同分排序的确定性**: `1/(60+r)` 生成的分数大量并列(单路命中且同 rank 不可能,
   但"不同名次组合凑出同分"可能), 排序键追加 chunk_id 保证结果**可复现**——单测
   断言与评估回归都依赖这一点。
6. **父块点查的缺失防御**: 理论上 children 里每个 parent_id 都在 parents 里有对应行
   (同一次 ingest 事务性写入, 04 篇), 但防御式编程仍要在 `parents[f.parent_id]`
   处兜 KeyError: 找不到父块的子块**跳过并记 warning**(而非崩溃), 数据不一致时
   检索仍能返回其余结果, 问题留给下次 ingest 修复。落地时用 `parents.get(...)` + `continue`
   实现, warning 出口沿用项目里统一的 `warnings: list[str] | None = None` 关键字参数
   (与 `embed_chunks` / `upsert_children` 同一套, 07 篇的 ingest 报告直接复用这个口)。

### 4.6 落地记录(2026-09-21, 与骨架的出入)

1. **`k` / `top_k` 不给默认值**: 骨架的 `rrf_fuse(..., k=60, top_k=5)` 改成必填关键字参数, 取值
   由 `search()` 从 `RAG2_RRF_K` / `RAG2_RETRIEVE_TOP_K` 传入 —— 默认值只留 config.py 一处,
   两处写默认值迟早漂移(与 04 篇 `_cfg()` 懒加载同一理由);
2. **`search(query, top_k=None)`**: 同上, 默认走 `RAG2_RETRIEVE_TOP_K`;
3. **`rrf_rank` 取最终结果名次**(见 §3.2 设计说明);
4. **`store` 用模块导入**(`from . import store` + 调用点写 `store.dense_search(...)`), 这样单测
   monkeypatch `store.*` 才打得中 —— 若写成 `from .store import dense_search` 直接调用, 名字在
   import 时就绑死了, patch 模块属性不会生效(§4.2 的单测姿势要求的正是前者);
5. **`page_no` 的哨兵还原**: parents 里无页码存 `-1`(04 篇坑位 7c), `search` 这层转回契约语义的
   `None`(`metadata` 里仍是 `-1`), 存储细节不外漏。

## 五、验收标准(自测清单)

- [x] `rrf_fuse` 单测: 构造两路名次表, 手算期望分数(如某子块 dense 第 1 + sparse
      第 3, 期望 1/61 + 1/63)逐断言; 单路为空时退化为该路名次序; 两路皆空返回 `[]`;
- [x] 同父去重单测: 3 个 top-5 子块同 parent_id 时合并为 1 条 RetrievedChunk,
      `hits` 含 3 条明细且 rrf_rank 取首个融合位置; 结果条数 ≤ top_k;
- [x] 并列分数单测: 构造同分候选, 断言按 chunk_id 稳定排序(可复现);
- [x] 错误语义单测: monkeypatch store 抛连接类异常, 断言 search 向上抛 RuntimeError
      而非返回 `[]`; `top_k=0` 抛 ValueError;
- [x] monkeypatch 全套(不连真实 Milvus/LLM): search 编排链路打通, 返回结构符合
      contracts.py 的 RetrievedChunk;
- [x] 集成冒烟(手动, 起 Milvus 后): 真机跑通, 见下方执行记录 —— 07 篇的 `cli.py search`
      子命令届时会把它变成一条命令, 本期用临时脚本驱动 `search()` 完成;
- [x] `uv run ruff check src/rag_v01` 通过; `uv run pytest src/rag_v01/tests -q`
      → **137 passed**(模块 05 贡献 30 项)。

**执行记录(2026-09-21, 真 Milvus 2.6.0 + 真百炼多模态 embedding)**: 语料 = `sample1.md`(英文长文,
24 父块 / 39 子块, 含 5 表格 + 2 图片) + `data/rag2_samples/sample.md`(中文短文, 5 父块 / 5 子块,
含 1 表格 + 1 图片), 两份都已入库保留(供 06/07 篇复用; 清掉用 `delete_doc(doc_id)`: 前者
`71eb7293fa5da003`, 后者 `77939d44a6014367`)。查询侧结果:

| query | 结果(父块名次 / 命中构成) |
|---|---|
| 样例文档里的表格有哪些名称 | 前 4 名全是中文样例(`d1s1` / `d9s3` / `d14s4` / `d20s2`, 含 table 与 image 块), 第 5 名才是英文文档 |
| 图片的图注是什么 | 第 1 名是 image 块(`d2s1`), 第 2 名同章节 text(`d1s4`) |
| 怎么保证检索出来的内容准确 | 全 dense-only(语料里没有这些实词) |
| HNSW 索引的参数 | 全 dense-only(同上, 语料无 HNSW) |
| Table of Contents | 4 条同父子块合并成 1 条(`hits` 4 条), 另 1 条 sparse 精确命中(`s13`) |

结构自检全过: 双路命中的 4 条子块分数与手算 `1/(60+r_d) + 1/(60+r_s)` **逐条相等**; `rrf_rank`
连续; 父块不重复; 结果条数 ≤ top_k; `hits` 条条挂在已有父块上。

**调参诊断(2026-09-22, 同语料 20 条 query, 纯检索不调 judge)**: `ef` 64/96/128 三点结果**完全一致**(20/20), 候选深度 20→50 让 **11/20** 条 query 的 top-5 变化, `top_k` 5→3 按定义必然不同(不作证据)。含义: **在当前语料规模下要判"改没改好", 该动的是候选深度与 top_k, 不是 `ef`**(把 `ef` 的收益留到语料涨到 10⁴ 量级再测); 另外任何对比都要记住 06 篇坑位 12c 的判读阈值(judge 自身抖动 ~0.1), 纯检索侧的前后对比才是确定性的。

**这次冒烟抓到并修掉一个真缺陷(转 04 篇坑位 1)**: jieba analyzer 会把**空格**也切成 token, 而空格
几乎存在于每一块 —— 带空格的 query 因此在 sparse 路命中一大批无关子块、分数还几乎相同; RRF 只吃
名次, 噪声票与真信号票等值, 融合被噪声左右(修复前的 `HNSW 索引的参数怎么定` 一例: 5 个候选里
sparse 名次全落在 4~14, 且第一名是靠 `dense#1 + sparse#8(噪声)` 胜出的)。修法是在 `text` 字段的
`analyzer_params` 上加 `stop` filter(停用词 = 空白 + 标点 + 单字母 + 中文单字虚词, 共 138 个),
并在建表自检 `_check_children` 里加 `_check_analyzer` 拦住"加词表之前建的旧表"。修完 sparse 路
从"匹配一切"变成精确关键词匹配(见上表)。**注意**: analyzer 是 schema 级配置, 改词表必须
drop 重建 + 重新 ingest。

## 六、与其它模块的依赖关系

```
contracts.py(RetrievedChunk/ChildHit)   config.py(RAG2_ 前缀检索参数)
        │                                     │
        v                                     v
embed.py(embed_query) ──> retrieve.py(本篇) <── store.py(dense/sparse/get_parents)
                                 │
        ┌────────────────────────┼──────────────────────────┐
        v                        v                          v
__init__.py facade        evaluate.py(06 篇)          cli.py search 子命令
(search 三入口之一)      (检索上下文来源,            (人工冒烟与调试出口)
                          同集对比新旧系统)
```

- **上游**: 03 篇的父子分块决定了 parent_id 关系与子块粒度(约 450 字符), 是检索
  质量的地基; 04 篇的 HNSW 参数、BM25 analyzer、parents 表结构是本篇查询的物理基础;
- **下游**: 06 篇 ragas 评估的 context_precision / context_recall 直接度量本篇返回的
  上下文质量, 评估结论反过来调本篇的候选深度与 top_k; 07 篇移植时, 上层 agent 只依赖
  `search(query, top_k=5) -> list[RetrievedChunk]` 与 §4.3 的错误语义约定;
- **零项目依赖**: 本模块(含 store/embed)不 import agent/、tools/、settings/ 任何包,
  全同步, 保证目录整体拷走即可移植(07 篇)。
