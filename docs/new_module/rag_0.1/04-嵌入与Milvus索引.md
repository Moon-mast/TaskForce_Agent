# 04-嵌入与Milvus索引

> 对应实现: `src/rag_v01/embed.py` 与 `src/rag_v01/store.py`。
> 上游: [./03-父子分块策略.md](./03-父子分块策略.md) 产出的 `ParentChunk` / `ChildChunk`;
> 下游: [./05-混合检索与RRF融合.md](./05-混合检索与RRF融合.md) 的两路召回。
> Milvus 部署、依赖、`RAG2_` 变量与冒烟清单见 [./01-环境与依赖.md](./01-环境与依赖.md)。

## 一、模块目标与边界

**目标**:

1. **embed.py**: 双模式(local / api, 定稿走 api)把文本与图片转成 **1024 维** dense 向量(同一向量空间)并使用同一后端服务入库与查询, 带维度自检;
2. **store.py**: 建两个 collection(children / parents)、建 HNSW 与内置 BM25 索引、批量 upsert、按 doc_id 整文档删除(支撑全量重建)、对外提供 05 篇要的三个查询原语;
3. 全程同步(pymilvus 同步客户端), 不引入 async/await。

**边界**(不做什么):

- 不做解析、清洗、切块([./02-文档解析与噪声处理.md](./02-文档解析与噪声处理.md) / [./03-父子分块策略.md](./03-父子分块策略.md)), 不决定父子块大小;
- 不做融合排序与截断(05 篇), 不生成答案、不评分(06 篇);
- 不改 `src/tools/rag/` 下任何文件; **零项目依赖**: 不 import `agent/`、`tools/`、`settings/` 任何包, 配置只读 `RAG2_` 前缀环境变量(见 [./07-接口设计与移植方案.md](./07-接口设计与移植方案.md));
- 本篇只产出文档与骨架片段, 不落任何代码。

## 二、设计决策与理由(含备选对比)

### 2.1 Embedding 选型: 主推百炼 `qwen3-vl-embedding`(多模态, API)

| 模型 | 维度 | 输入/上下文 | 语言 | 获取 | 与本方案的契合度 |
|---|---|---|---|---|---|
| **qwen3-vl-embedding(选定)** | 2560 默认, 指定 **1024**(2560/2048/1536/1024/768/512/256) | text 32,000 token; image(URL/Base64, ≤10MB, JPEG/PNG/WEBP/BMP/TIFF 等); video(仅 URL) | 33 种(中文强) | API: 阿里云百炼, **DashScope SDK `MultiModalEmbedding.call`** | 文本与图片在**同一向量空间**——这是选它的核心理由: 02 篇抽取的图片块能以真向量进入 dense 路, 文本 query 可跨模态命中图片, 不再依赖 VLM 描述"翻译成文字"; 维度可指定 1024 与 schema 对齐; 单请求内容元素 ≤20(图片 ≤5); 价格: 图片/视频 1.8 元、文本 0.7 元每百万 token; 免费额度 100 万 token |
| qwen3.7-text-embedding-flash(纯文本等价替换) | 1024 默认 | text 128K token | 201 种 | API: 同一 API Key, OpenAI 兼容 | 便宜(0.125 元/百万 token)、文本上下文更长; 但**没有图片向量能力**, 图片块只能退化为占位/VLM 描述文本参与检索。若语料确定无图可换回 |
| BAAI/bge-m3(local 离线备选, **仅文本**) | 1024 | text 8192 token | 多语言(中文强, C-MTEB 第一梯队) | 本地 ST | 优势只剩"完全离线 + 无 token 成本"; 代价是 torch + 约 2GB 权重下载与 CPU 编码慢; **不支持图片向量**, 切 local 时图片块退化为占位文本 |
| 智谱 embedding-3(复用现有 key) | 服务端定, 需实测探测 | text | 中文强 | API | 纯文本等价替换备选。代价: (a) 维度由服务端定——旧 `src/tools/rag/embed.py:31-35` 正是靠"运行时探测维度"规避硬编码, 与本模块"dim 锁定写进 schema"的取舍冲突; (b) 单请求 input 数组上限(旧代码 `ZHIPU_BATCH_LIMIT = 64`, 超限报错误码 1214); (c) 向量空间不同, 换用必须整库重嵌 |

**结论**: 默认 `RAG2_EMBED_MODE=api` + `RAG2_EMBED_MODEL=qwen3-vl-embedding`, 经 `dimension=1024(SDK 顶层参数)` 与 schema 对齐; 文本与图片共用同一模型(查询侧永远送文本)。**关键约束: OpenAI 兼容端点不支持图片输入**, api 路必须走 DashScope SDK(`dashscope` 包), 不能再用 `openai.OpenAI` 直连。

**后续可选升级(本期不启用)**: (a) 融合向量 `enable_fusion=true` 可把"图片 + 其题注/上下文文本"合成 1 个向量(多图+文本亦可), 对图文混排强的语料通常优于单图向量——需用 [./06-ragas评估.md](./06-ragas评估.md) 评估集验证; (b) 纯文本路径下 `qwen3.7-text-embedding-flash` 的 `output_type=dense&sparse`(学习型稀疏)可替换 sparse 路的 BM25, 但会改变 RRF 两路特性, 同样先验证。

### 2.2 向量归一化与 metric=COSINE 的关系

- 数学上, `‖a‖=‖b‖=1` 时 `cos(a,b) = a·b`, 即"归一化 + IP"与 COSINE 的**排名完全等价**; 而 Milvus 的 COSINE 本身就会做归一化计算, 所以归一化对 COSINE 的排序**没有影响**。
- 那为什么仍显式归一化: (a) **双模式一致性**——有的 OpenAI 兼容端点已返回单位向量, 有的不, 显式归一化把 local / api 两个后端的输出拉到同一尺度; (b) **分数可读**——归一化后 COSINE 落在 [-1, 1], 命中样本大致 0.3~0.9, 调试输出与阈值直觉清晰; (c) **留退路**——将来若为省一次归一化改用 IP, 排名与行为不变。
- metric 选 **COSINE**(定稿): 不用 L2(距离语义与 sparse 路的 BM25 分反向, "越小越优 vs 越大越优"在调试与融合里极易搞错); 不用 IP(未归一化时数值无界, 且与归一化耦合)。
- 落地: local 侧 `encode(..., normalize_embeddings=True)`; api 侧手算 L2 归一化, 两模式共用同一个 `_l2_normalize`。

### 2.3 children / parents 为什么分两个 collection

children 承担**召回**(dense + sparse 双索引), parents 只承担**回溯取全文**(无向量索引、主键点查)。分工理由见 [./03-父子分块策略.md](./03-父子分块策略.md) §2.1 与 [./05-混合检索与RRF融合.md](./05-混合检索与RRF融合.md) §2.4。代价是文本冗余约 1.3 份(子块是父块的切分), 换来"命中后一次点查即得完整语境"。

**实测修正(2026-09-21)**: "parents 只有标量字段、没有任何向量字段"在 Milvus 上**建不出来** —— 服务端要求 collection 至少含一个向量字段(`1100 schema does not contain vector field`), 且**每个向量字段都必须建索引**才能 load(`65535 there is no vector index on field`), 而向量字段又**不支持 nullable**(`1100 vector type not support null`)。所以 parents 保留"只做点查"的定位, 但 schema 里加一个 `dense` FLOAT_VECTOR **占位字段**(dim=2, 恒写零向量, 从不检索) + 一个 `FLAT` 索引(暴力检索、不建图结构), 详见 §4.2 与坑位 9。

### 2.4 向量索引: HNSW vs 备选

| 方案 | 结论 | 理由 |
|---|---|---|
| **HNSW(选定)** | 主实现 | 图索引, 高召回低延迟; 内存换速度, 本规模(10⁴~10⁵ 子块)可全内存; 增量插入即可用, 删除/新增无需重训 |
| IVF_FLAT / IVF_SQ8 | 弃 | 要选 nlist/nprobe 且依赖训练集分布; 小规模下召回未必胜过 HNSW, 多一层"训练—再训练"的心智 |
| DiskANN | 弃 | 面向超大库 + SSD 的场景, 单机本地小库收益为负(依赖多、构建慢) |
| FLAT(暴力) | 仅评估对照 | 召回 100% 但 O(n); 建议在评估阶段用它量出"检索上限"(见 §五) |
| AUTOINDEX | 弃 | 省心但把参数交给服务端; 定稿要求显式 HNSW(M=16, efConstruction=200), 教学与调参都需要显式 |
| 稀疏路索引 | 非 HNSW | BM25 Function 生成的 sparse 向量官方口径配 `SPARSE_INVERTED_INDEX` + `metric_type="BM25"` |

### 2.5 BM25 为什么放在服务端(以及备胎)

主实现走 **Milvus 内置 BM25 Function + jieba analyzer**: (a) 入库与查询由同一份服务端分词逻辑处理, 天然无"两侧分词漂移"问题; (b) 不需要在客户端维护倒排索引与词频统计(进程退出即丢的状态很烦)。备胎(**全局定稿口径**): 若 milvus-lite / 低版本下 BM25 Function 与 jieba analyzer 实测异常, 改用进程内 `rank_bm25` + `jieba` 客户端自算——只替换 sparse 路的数据来源, [./05-混合检索与RRF融合.md](./05-混合检索与RRF融合.md) 的融合与编排代码结构不变(这正是主实现选手写 RRF 的收益之一)。备胎的代价: 自行维护语料统计, 且分词模式必须与 Milvus jieba 默认口径对齐(坑位 1)。

## 三、数据流与接口契约

### 3.1 链路图

```
03 篇产物: (list[ParentChunk], list[ChildChunk])
   │
   ├─ embed_documents(children) ──> list[list[float]]   # text 块送文本、image 块送图片(同一向量空间); 1024 维已归一化   [embed.py]
   │        │
   │        v
   │   store.upsert_children(children, vectors)  ──> children            # text 挂 BM25 Function,
   │   store.upsert_parents(parents)             ──> parents             # Milvus 自动生成 sparse
   │        │
   │        v
   │   store.flush()  (ingest 收尾)                                        [store.py]
   │
query ──> embed_query(query) ──> 1024 维向量
              │                        │
              v                        v
   store.dense_search(vec, limit=20)     store.sparse_search(query, limit=20)   # 05 篇调用
              │                        │
              └──────────┬─────────────┘
                         v
                store.get_parents(parent_ids)  # parents 主键批量点查(无向量索引)
```

### 3.2 对外接口(store.py / embed.py, 供 05 / ingest / evaluate 调用)

| 接口 | 签名 | 说明 |
|---|---|---|
| `embed_documents` | `(texts: list[str]) -> list[list[float]]` | 批量; 顺序与入参一致; 内部按批切分 |
| `embed_images` | `(paths: list[str]) -> list[list[float]]` | 图片块专用: 本地落盘图片转 Base64 后送多模态模型; 与文本同空间; 批量 ≤ `RAG2_EMBED_IMG_BATCH` |
| `embed_chunks` | `(rows: list[ChildChunk], *, warnings=None) -> list[list[float]]` | **07 篇 ingest 的入口**(落地补入): 按 `chunk_type` 分组(文本组含正文/表格/无本地图的图片块; 图片组需 `image_path` 有效且文件在盘上), 分别调 `encode` / `encode_images` 后**按原下标拼回**; 顺序错位是这个模块最难查的故障, 所以拼回处带条数校验 |
| `embed_query` | `(text: str) -> list[float]` | 单条; 与入库**同一后端同一模式** |
| `embedding_dim` | `() -> int` | 首次调用实测后缓存; 与 `RAG2_EMBED_DIM` 不一致即抛错 |
| `ensure_collections` | `() -> None` | 幂等建表: 不存在则建 schema + 索引; 存在则校验 dim 与索引存在性 |
| `upsert_parents` | `(rows: list[ParentChunk]) -> int` | 返回写入行数 |
| `upsert_children` | `(rows: list[ChildChunk], vectors: list[list[float]]) -> int` | payload **不含 sparse**(由 Function 生成) |
| `dense_search` | `(vec: list[float], limit: int) -> list[Hit]` | HNSW ANN, 查询期 ef 传入 |
| `sparse_search` | `(query: str, limit: int) -> list[Hit]` | 直传原文, 服务端分词 + BM25 |
| `get_parents` | `(parent_ids: list[str]) -> dict[str, ParentRow]` | parents 主键批量点查 |
| `delete_doc` | `(doc_id: str) -> None` | 两个 collection 按 doc_id 全删(全量重建入口) |
| `flush` | `() -> None` | ingest 收尾, 让新数据对检索可见 |

- 依赖 contracts.py 的类型: **读** `ParentChunk` / `ChildChunk`(字段与 [./03-父子分块策略.md](./03-父子分块策略.md) §3.1 九字段表逐字一致); `Hit` / `ParentRow` 是 store.py 内部结构, **不进 contracts.py**(与 05 篇 §3.3 的约定一致);
- `Hit` 字段: `chunk_id, parent_id, text, chunk_type, score, metadata`(05 篇按名次使用, 不依赖 score 量纲)。`metadata` 是**落地时补的**(决策 2): 图片块命中时 `text` 只是占位文本, 展示要看 `image_path`, 而 05/07 篇给出"依据来自哪个文件、第几页、哪个章节"也要 `source`/`page_no`/`heading_path` —— 不带它就得回表再查一次。`metadata` 不参与排序;
- `ParentRow` 字段: `parent_id, text, doc_id, metadata`。

### 3.3 本篇补充的环境变量(已回填 01 篇 §三表格; 落地时一并进 config.py)

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RAG2_EMBED_DIM` | `1024` | dense 维度基准; 经 `dimension=1024(SDK 顶层参数)` 请求(参数名 `dimension`, 默认 2560), 与 embed 自检、建表共用 |
| `RAG2_EMBED_BATCH` | `20` | 单次编码/请求的内容元素总数上限(文本 ≤20; local 批大小) |
| `RAG2_EMBED_IMG_BATCH` | `5` | 单请求图片条数上限(qwen3-vl-embedding 单请求图片 ≤5) |
| `RAG2_HNSW_EF` | `96` | 查询期 ef, 取值区间 64~128(定稿); 即 [./05-混合检索与RRF融合.md](./05-混合检索与RRF融合.md) §3.3 引用的 `cfg.ef` |

## 四、实现要点(API + 骨架 + 坑位)

> 事实核查说明(撰写时 2026-09 对照 Milvus 官方文档拉取核对): BM25 Function 的 schema 声明形态、sparse 索引用 `SPARSE_INVERTED_INDEX` + `metric_type="BM25"`、jieba tokenizer 的默认参数(`dict=["_default_"]`, `mode="search"`, `hmm=True`)、`client.run_analyzer(...)` 的用法、HNSW 的 `M`(默认 30, 建议区间 [5, 100])与 `efConstruction`(默认 360)范围均已逐条核到; 未逐字核到官方文档的点在坑位清单里标"待验证"并给验证办法, 不凭记忆写死。

### 4.1 embed.py 双模式骨架

```python
# embed.py —— 双模式 embedding(骨架, 说明性质)
# 配置统一经 config.py 读取(01 篇 §三), 模块内不直读 os.environ
_DIM, _BATCH, _IMG_BATCH = cfg.embed_dim, cfg.embed_batch, cfg.embed_img_batch   # RAG2_EMBED_DIM / _BATCH / _IMG_BATCH

def _l2_normalize(vec: list[float]) -> list[float]: ...

class _LocalBackend:                                  # 离线备选模式(sentence-transformers)
    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer      # 懒 import: 单测/纯 API 用户不必装 torch
        self.model = SentenceTransformer(model)                     # 首次会联网下载(见 01 篇 HF_ENDPOINT)
    def encode(self, texts: list[str]) -> list[list[float]]:
        arr = self.model.encode(texts, batch_size=_BATCH, normalize_embeddings=True)
        return [v.tolist() for v in arr]                            # numpy -> list, 便于 JSON 序列化

class _ApiBackend:                                    # 百炼多模态模式(DashScope SDK; OpenAI 兼容端点不支持图片)
    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        import dashscope                                     # 懒 import: 纯 local 模式不必装 dashscope
        from dashscope import MultiModalEmbedding
        dashscope.base_http_api_url = base_url
        self._mm, self.model, self.api_key = MultiModalEmbedding, model, api_key
    def _call(self, contents: list[dict]) -> list[list[float]]:
        resp = self._mm.call(                                # contents 例: {"text": ...} / {"image": "data:image/png;base64,..."}
            api_key=self.api_key, model=self.model, input=contents,
            dimension=_DIM,                                  # 顶层参数; parameters={"dimension":...} 形式实测静默失效(dashscope 1.27.6 返回默认 2560)
        )
        return [_l2_normalize(e["embedding"]) for e in resp.output["embeddings"]]  # 响应字段以官方示例为准
    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), _BATCH):               # 分批: 单请求内容元素 ≤20
            out += self._call([{"text": t} for t in texts[i:i + _BATCH]])
        return out
    def encode_images(self, paths: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(paths), _IMG_BATCH):           # 分批: 单请求图片 ≤5
            out += self._call([{"image": _to_data_uri(p)} for p in paths[i:i + _IMG_BATCH]])
        return out

def get_embedder():                                   # 单例懒加载(lru_cache); RAG2_EMBED_MODE=local|api 二选一
    ...
def embed_documents(texts: list[str]) -> list[list[float]]:
    ...
def embed_query(text: str) -> list[float]:
    ...
```

要点与坑:

- **维度自检**放在第一次成功拿到向量之后: `len(vec) != _DIM` 抛 `ValueError`, 报错文案必须带"实际维度 / 期望维度 / 提示改 `RAG2_EMBED_DIM` 并重建 collection"——Milvus 侧的维度错误往往延迟到建索引或插入阶段才炸, 信息晦涩;
- local 模式 `SentenceTransformer.encode` 默认返回 numpy 数组, 需 `.tolist()`; 大 batch 在 CPU 上很慢(约 450 字符 × 数千块可能十几分钟), 这是 local 仅作离线备选、定稿走 api 的直接原因;
- api 模式不重试、不限流: 重试与断点续灌属于 ingest.py 的职责, embed.py 只把原始异常向上抛(与 05 篇 §4.3 的错误语义分层一致);
- **查询侧永远送文本**: 与图片块在同一向量空间, 无需指令前缀; 融合模式(`enable_fusion`)本期不启用——一旦启用必须整库重灌, 不要只改一侧(否则检索静默变差, 很难查)。

**落地记录(2026-09-21)**: 上面骨架落地为 `embed.py` / `store.py` 时的四处出入, 以代码为准:

1. **配置改懒加载**: 骨架写的模块级 `_DIM = cfg.embed_dim` 换成 `@lru_cache` 的 `_cfg()`. 模块级取值会在 `import embed` 那一刻锁死环境(单测注假配置、入口先载 `.env` 再 import 都变成顺序问题);
2. **非 200 显式抛错**: DashScope SDK 失败时**不抛异常**, 只把 `status_code` / `code` / `message` / `request_id` 放进响应对象; 不显式抛, 上层会崩在 `output` 为 `None` 的 `TypeError` 上, 把真正原因丢掉;
3. **多一个 `supports_images` 能力声明**(类属性, api=True / local=False): `embed_chunks` 据此决定图片块走图片路还是文本路, 不用 `hasattr` 那种换后端就静默失效的动态判断;
4. **多一层 `embed_chunks` 编排**: 见 §3.2; `_to_data_uri` 在文件不存在时**主动抛异常**, "图没了就降级文本"的判断留给 `embed_chunks`(它先按 `RAG2_IMAGE_DIR` 拼真实路径再判存在)。

### 4.2 children schema 字段表(定稿: text 挂 BM25 Function, dense 走 HNSW)

| 字段 | Milvus 类型 | 说明 |
|---|---|---|
| `chunk_id` | VARCHAR(64), **主键**, `auto_id=False` | 03 篇规则生成的确定性 ID(`doc_id:pNNN:cNNN`) |
| `parent_id` | VARCHAR(64) | 回溯 parents 的外键 |
| `doc_id` | VARCHAR(64) | 按文档整删的过滤键(`doc_id == "..."`) |
| `source` | VARCHAR(512) | 相对语料根的文件路径(POSIX 风格) |
| `chunk_type` | VARCHAR(16) | `text` / `table` / `image` |
| `page_no` | INT64 | 无页码记 `-1`(见坑位 7 的 nullable 备选) |
| `text` | VARCHAR(8192), `enable_analyzer=True`, `analyzer_params={"tokenizer": "jieba", "filter": [{"type": "stop", "stop_words": [...]}]}` | 子块文本; BM25 Function 的输入。停用词表是**必须的**(见坑位 1 的实测修正), 共 138 个: 空白 + 标点 + 单字母 + 中文单字虚词 |
| `dense` | FLOAT_VECTOR, `dim=1024` | 多模态 embedding 输出(text/image 块同一空间; 客户端 L2 归一化; 与请求 `dimension=1024` 对齐), HNSW + COSINE |
| `sparse` | SPARSE_FLOAT_VECTOR | **由 Function 生成, 入库 payload 不写此字段** |
| `metadata` | JSON | `heading_path` / `char_len` / `created_at` 及其余透传元信息 |
| Function | `Function(name="text_bm25", function_type=FunctionType.BM25, input_field_names=["text"], output_field_names=["sparse"])` | 服务端 BM25 词项加权 |

**parents 字段表**(定稿口径 + 实测修正):

| 字段 | Milvus 类型 | 说明 |
|---|---|---|
| `parent_id` | VARCHAR(64), **主键** | 03 篇的 `doc_id:pNNN` |
| `text` | VARCHAR(65535) | 父块全文(1200~1800, 超长表格顶格留量) |
| `doc_id` | VARCHAR(64) | 整删过滤键 |
| `metadata` | JSON | 九字段里除 doc_id 外的其余项(含 source/page_no/heading_path/char_len/created_at) |
| `dense` | FLOAT_VECTOR, **dim=2** | **占位字段**(实测修正): 恒写零向量、从不检索, 只为满足"collection 必须有向量字段且必须有索引"; dim=2 是服务端允许的最小维度 |

parents 索引: `dense` 用 **FLAT**(不建图结构, 从不检索所以代价只是存储) + `doc_id` 用 **INVERTED** 标量索引(按文档整删/过滤用)。**parents 没有任何向量检索**: 只按主键或 `doc_id` 点查, 供 05 篇回溯取全文。collection 名固定 `children` / `parents`(全局定稿, 不得改名增前缀), 作为 config.py 内的模块常量, 不额外开环境变量。

### 4.3 建表骨架(pymilvus)

```python
# store.py —— 建表骨架(children; parents 去掉向量字段与 Function 即可)
from pymilvus import DataType, Function, FunctionType, MilvusClient

def _child_schema():
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
    schema.add_field("source", DataType.VARCHAR, max_length=512)
    schema.add_field("chunk_type", DataType.VARCHAR, max_length=16)
    schema.add_field("page_no", DataType.INT64)
    schema.add_field("text", DataType.VARCHAR, max_length=8192,
                     enable_analyzer=True,
                     analyzer_params=_analyzer_params())          # jieba + 停用词(见坑位 1)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)       # Function 输出字段
    schema.add_field("metadata", DataType.JSON)
    schema.add_function(Function(name="text_bm25", function_type=FunctionType.BM25,
                                 input_field_names=["text"],
                                 output_field_names=["sparse"]))
    return schema

def _child_index():
    idx = MilvusClient.prepare_index_params()
    idx.add_index(field_name="dense", index_type="HNSW", metric_type="COSINE",
                  params={"M": 16, "efConstruction": 200})
    idx.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                  metric_type="BM25")
    return idx

def ensure_collections() -> None:                                  # 幂等
    client = get_client()                                          # MilvusClient(uri=RAG2_MILVUS_URI)
    if not client.has_collection("children"):
        client.create_collection("children", schema=_child_schema(),
                                 index_params=_child_index())
        client.load_collection("children")                         # parents 同理(但它也必须带向量字段+索引, 见 §4.2)
```

坑: `create_collection` 的索引是**在空表上建的**, 之后新增的数据由 Milvus 在 segment 封存时自动建索引, 不需要每次 ingest 后手工重建; `has_collection` 只校验存在性, 换 embedding 模型时必须**人工判断**是否 drop(坑位 6)。

### 4.4 HNSW 参数论证: M=16 / efConstruction=200 / COSINE / ef 64~128

| 参数 | 定稿值 | 官方默认 | 作用与取舍 |
|---|---|---|---|
| `M` | 16 | 30(建议区间 5~100) | 每节点每层的最大连接数。越大图越密: 召回↑、内存↑、插入变慢。本规模(10⁴~10⁵ 子块)+ 1024 维, M=16(原论文默认)已能拿到高召回, 内存比 30 省约三成 |
| `efConstruction` | 200 | 360 | 建图时候选池大小。越大图质量越好、构建越慢。构建是一次性成本且语料小, 这里从默认下调**不是**性能取舍, 而是与论文/主流实践对齐便于对照; 真要调, 方向是往大 |
| `metric_type` | COSINE | — | 见 §2.2; 归一化后与 IP 等价, 但分数语义更直观 |
| `ef`(查询期) | 96(区间 64~128) | — | **召回/延迟最直接的旋钮**; 必须 ≥ `limit`(=20, 05 篇坑位 2)。ef 越大候选列表越深: 召回↑、延迟↑ |

**什么时候才需要动这些数**:

- 只嫌延迟高 → 降 `ef`(只改查询参数, 不动索引、不重建);
- 评估显示召回是瓶颈且 `ef` 已到 128 → 升 `M` / `efConstruction`, 执行 `release_collection` → `drop_index` → `create_index` → `load_collection`, **数据不动**;
- 语料涨到百万级或内存吃紧 → 降 `M`、考虑量化索引(超出本方案范围);
- 换 embedding 模型或维度 → 与上两条无关, 只能重建 collection(坑位 6)。
- 调参方法论: 用 [./06-ragas评估.md](./06-ragas评估.md) 的 30~50 条评估集当 probe 集, 固定语料, 在 `ef ∈ {64, 96, 128}` 三点扫 `context_recall` 与延迟, 取"召回不再明显上升"的拐点。
- **实测提醒(2026-09-22, 本机语料 597 个子块 / 20 条 query)**: `ef` 在 64 / 96 / 128 三点上**结果完全相同(20/20 条 query 的 top-5 一字不差)** —— 语料这么小时 HNSW 的候选池深浅根本不构成瓶颈, 扫 `ef` 是**白扫**; 同一次实验里真正会改变结果的旋钮是**候选深度**(`candidate_top_n` 20 → 50 时 11/20 条 query 的 top-5 变化)。所以: 要测 `ef` 的收益, 得先把语料拉到 10⁴ 量级; 在此之前调参只动候选深度与 `top_k`。

### 4.5 upsert / 按 doc_id 删除 / flush 与 load

```python
# store.py —— 写入与删除骨架
UPSERT_BATCH = 500          # 单次 upsert 行数上限(经验值, 见坑位 11)

def upsert_children(rows, vectors) -> int:
    data = [{"chunk_id": r.chunk_id, "parent_id": r.parent_id, "doc_id": r.doc_id,
             "source": r.source, "chunk_type": r.chunk_type,
             "page_no": r.page_no if r.page_no is not None else -1,
             "text": r.text, "dense": v, "metadata": r.metadata()}
            for r, v in zip(rows, vectors)]          # 注意: 不写 sparse, 由 Function 生成
    for i in range(0, len(data), UPSERT_BATCH):
        client.upsert("children", data=data[i:i + UPSERT_BATCH])
    return len(data)

def delete_doc(doc_id: str) -> None:                 # 全量重建 = delete_doc + 重新 ingest
    for coll in ("children", "parents"):
        client.delete(coll, filter=f'doc_id == "{doc_id}"')
    client.flush("children"); client.flush("parents")

def flush() -> None:                                 # ingest 流水线收尾时调用一次
    client.flush("children"); client.flush("parents")
```

- **upsert 语义**: 近似"主键存在则删旧插新"; 因 chunk_id 由内容确定性生成(03 篇 §2.4), 同一文档重复 ingest 是幂等的——这是"全量重建"能安全重跑的基础;
- **超长条目跳过而不是截断**(坑位 7b 的落地口径): `text` 超过 8192 字节(约 2700 汉字)的子块**不写库**并计入 `warnings`。截断会让 BM25 与 dense 都基于残缺文本, 属"静默变差"; 跳过是可被 ingest 报告看见的, 而父块(`VARCHAR(65535)`)里仍留着完整文本, 调小 `RAG2_CHILD_TARGET_CHARS` 或拆表后重灌即可(upsert 幂等);
- **删除是逻辑删除 + 时间戳过滤**: 删完立刻查就不该再命中; 物理空间回收要等 compaction(本地单机可不管, 靠自动 compaction);
- **flush 的时机**: 只在一次 ingest 结束时调一次, 不要每条一调(flush 是重操作, 会封 segment 并促发索引构建); 查询前不需要手工 load——`ensure_collections` 时已 load, 重启 Milvus 后需重新 load(`load_collection` 幂等, 可在启动路径里无脑调)。

### 4.6 坑位清单

1. **入库侧/检索侧 analyzer 一致性(主链路已根治, 备胎需自律)**: 走内置 BM25 Function 时, 两侧分词都由 collection 上的 `analyzer_params` 完成, 不存在漂移; 但 (a) analyzer 是 **schema 级**配置, 建表后改分词只能重建 collection; (b) 切备胎(rank_bm25)后, 客户端分词必须与 Milvus jieba 默认口径一致——官方 jieba tokenizer 默认 `dict=["_default_"]`, `mode="search"`, `hmm=True`, 等价于 `jieba.lcut_for_search(text, HMM=True)`; 抽成一个公共函数供入库与查询共用, 禁止两处各写一份。验证办法: 取一串含多字词/未登录词的中文, 分别过 `client.run_analyzer([...], {...})` 与本地 jieba, 逐 token 对比。

   **实测修正(2026-09-21, 模块 05 集成冒烟抓到的真缺陷)**: jieba analyzer **把空白也切成 token**——实测 20 个子块 2540 个 token 里, 空格 `" "` 占 **1109(44%)**、表格竖线 `"|"` 331、换行 `"\n"` 62, 而它们几乎存在于每一块。后果: 任何**带空格**的 query 都会在 sparse 路命中一大批无关子块(分数还几乎相同, 实测 5 条都是 0.139), 而 RRF 只吃名次 → 噪声票与真信号票等值, 融合被噪声左右(修复前 `HNSW 索引的参数怎么定` 的第一名是靠 `dense#1 + sparse#8(噪声)` 胜出的)。**修法**: `text` 字段的 `analyzer_params` 加 `stop` filter, 停用词 = 空白 + 标点(ASCII 用 `string.punctuation` 展开, 全角另列) + 单字母 + 中文单字虚词, 共 138 个(`store.py::_ANALYZER_STOP_WORDS`)。**只有 `stop` 可用**: 这个 tokenizer 实测对 `removepunct` / `length` / `alphanumonly` / `lowercase` 一律报 `unsupport filter type`, 没有"按长度丢 token""丢标点"这类通用规则。修完 sparse 路从"匹配一切"变成精确关键词匹配(同批 query: 旧 5 条噪声 → 新 1 条精确命中)。**副作用**: 服务端分词结果不再逐 token 等于裸 `jieba.lcut_for_search`, 差集恰好是停用词表(验收口径见 §五)。
2. **`sparse` 字段不要手填**: 它是 Function 的输出字段, 入库 payload 里带 `sparse` 会报错或被忽略(以实测为准); 入库只给 `text`。**实测(2026-09-21)**: 不带 `sparse` 的 4 行 upsert 成功, 返回 `{'upsert_count': 4, 'ids': [...]}`, 且随后 sparse 检索能命中 —— 服务端确实自动生成了。
3. **版本要求**: BM25 Function 与 collection 级 analyzer 都是 Milvus 2.5+ / pymilvus 2.5+ 的能力(01 篇已锁 pymilvus>=2.5 + 服务端 2.6.x)。**上线前必须先跑"建表 → 插中文 → 稀疏检索能命中"这一步**再灌全量语料——这正是全局定稿风险注记要求实测的动作; 若失败, 切备胎(§2.5)并在 [./05-混合检索与RRF融合.md](./05-混合检索与RRF融合.md) 的检索层替换 sparse 数据来源。**实测通过(2026-09-21, 服务端 2.6.0 / pymilvus 2.6.17)**: 建表时 `text_bm25` Function 正常挂上, sparse 检索命中目标 chunk(BM25 分 4.8875, 其余 3 条 0.09~0.12 量级), **不需要切 rank_bm25 备胎**; 备胎路径保留为"换到低版本 Milvus 时"的退路。
4. **Windows / milvus-lite 限制(定稿)**: milvus-lite 官方不支持 Windows, 教程里 `MilvusClient("./milvus.db")` 的写法在本机直接跑不了; 统一用 docker compose 起 standalone, **部署在 WSL 内的 docker、19530 暴露给宿主**(compose 片段与 WSL 要点见 [./01-环境与依赖.md](./01-环境与依赖.md) §2.4)。**运行注意(2026-09-21)**: WSL 空转会被回收, 之后连 19530 报 `Connection refused`; 首次访问会唤醒 distro、容器按 restart 策略自动拉起, **等 `milvus-standalone` 变 `healthy` 再跑**(`wsl -e docker inspect -f "{{.State.Health.Status}}" milvus-standalone`), 否则脚本会连不上。
5. **维度硬校验**: schema 的 `dim=1024` 与 embedding 输出必须一致, 且**两处都查**——embed 层自检(§4.1)负责快速失败, `ensure_collections()` 负责对比已存在 collection 的 dim(读 schema)并给出清晰报错; 只靠 Milvus 的报错定位会很绕。
6. **换模型 ≠ 换维度**: qwen3-vl-embedding 与 bge-m3 同为 1024 维, 但**向量空间不同**(多模态模型的文本子空间与纯文本模型也不通用), 旧向量与新查询向量不可混用——换 `RAG2_EMBED_MODEL`(或改 `dimension`/`RAG2_EMBED_DIM`)的同时必须整库重嵌(drop collection 或全量 re-ingest)。这是本模块最容易踩的"看起来兼容"陷阱。
7. **VARCHAR 长度与 `page_no` 空值**: (a) 官方 `max_length` 口径按字节计、上限 65535(中文 UTF-8 约 3 字节/字, 约 2 万汉字), 本文按此设计 parents.text=65535; 落地时造一条 3 万汉字文本实测确认, 并注意 **api 模式的 OpenAI 兼容服务对超长 input 也有自身限制**; (b) 超大表格父块可能逼近上限——ingest 时做前置长度检查并计入报告 warning, **不静默截断**; **落地口径(2026-09-21)**: 子块 `text` 超 8192 字节(约 2700 汉字)**跳过该条 + 记 warnings**, 父块超 65535 字节的极端情况暂未遇到(pdf 实测再处理); (c) `page_no` 的未知值本方案用 `-1` 哨兵; (d) **实测(2026-09-21)**: 标量字段 `nullable=True` **可用**(建表与写入都通过), 但**向量字段不支持 nullable**(`1100 vector type not support null`); 本方案仍用 `-1` 哨兵 —— 换成 nullable 会让"无页码"在查询结果里变成 `None`, 多一个分支要处理, 收益不抵扰动。
8. **刚灌完搜不到(可见性)**: 默认一致性级别下, 写入后立刻检索可能看不到新数据。处置组合拳: ingest 收尾 `flush()` + 检索/查询调用统一传 `consistency_level="Strong"`(单用户本地, 这点开销无感)。验证办法: ingest 后**立刻** search 一次, 断言能命中刚灌的 doc_id; 若命中不到, 先确认 flush 与一致性参数是否生效。**实测(2026-09-21)通过**: `flush()` 之后立刻 `search` 与 `query` 都能看见(且是带 `consistency_level="Strong"` 调的); 落地时把 Strong 写死在 `dense_search` / `sparse_search` / `get_parents` 三处, 不留给调用方配。
9. **parents 无向量索引的取数方式** —— **实测结论(2026-09-21)已推翻原设想**: 只含标量字段的 collection **建不出来**(`1100 schema does not contain vector field`), 加 INVERTED 标量索引也一样; 补一个向量字段后若不建索引, 则 `load_collection` 报 `65535 there is no vector index on field: [dense], please create index firstly`; 向量字段 `nullable=True` 被拒。**落地方案**: parents = `parent_id`/`text`/`doc_id`/`metadata` + 占位 `dense`(FLOAT_VECTOR **dim=2**, 恒零向量, dim 下限实测为 2, 1 报 `invalid dimension`) + `FLAT` 索引(从不检索) + `doc_id` **INVERTED** 标量索引; 取数用 `client.query(filter='parent_id in [...]')` 或 `client.get(ids=[...])`, 两者均实测可用。
10. **批量与 RPC 上限**: 单次 upsert 按 500~1000 行分批; embed 侧按 `RAG2_EMBED_BATCH` 分批(API 模式服务端 input 数组有上限, 旧系统踩过智谱的错误码 1214, 见 `src/tools/rag/embed.py:38`); 分批后注意**保持 vectors 与 rows 的顺序对齐**, 错位会把向量写到错误的 chunk 上——单测里用能区分的内容造数据断言映射。
11. **`filter` 字符串拼接**: `delete_doc` 的 `doc_id` 是 SHA-256 前 16 位 hex(03 篇), 无注入面; 但 05 篇 `get_parents` 拼 `parent_id in [...]` 时仍要引号转义规范(统一走一个小工具函数), 不要把任意文本直接塞进 filter。
12. **百炼多模态接口的三个约定(定稿路径)**: (a) 单请求内容元素总数 ≤20、**图片 ≤5**(文本/图片分别按 `RAG2_EMBED_BATCH` / `RAG2_EMBED_IMG_BATCH` 分批, 超限直接报错); (b) 维度经 `dimension=1024(SDK 顶层参数)` 指定(参数名 `dimension`, 默认 2560), 并保留 embed 层维度自检; (c) 限流 RPM 2400 / TPM 120 万(2026-09 官方口径), 单机 ingest 远达不到, 但 ingest.py 仍应把"服务端 4xx/5xx 的有限重试"做成可开关(默认开), 免费额度(100 万 token / 90 天)与计费以控制台为准。实测(dashscope 1.27.6): 维度必须用顶层参数 `dimension=1024`,写成 `parameters={"dimension":1024}` 会被静默忽略、返回默认 2560。**另实测**: `client.run_analyzer(...)` 返回的是 `list[AnalyzeResult]`, token 在 `.tokens` 里 —— 它不是 list, 直接和 `jieba.lcut_for_search(...)` 比会得到 False(两边打印一模一样却"不相等"), 这是写验收脚本时最容易踩的一脚。
13. **图片块的稀疏路基本失效**: image 块的 `text` 是占位文本(02 篇), BM25 只能匹配占位词/标题路径; 要图片能被关键词命中需开 `RAG2_IMAGE_DESC`(VLM 描述写入 text)。dense 路不受影响——图片有真向量, 文本 query 可跨模态命中。

## 五、验收标准(自测清单)

- [x] `ensure_collections()` 幂等: 连跑两次不报错, 第二次不重建 —— **实测通过**(2026-09-21: 两次调用后 `created_timestamp` 不变);
- [x] **建表自检三查**: 已存在的 children 要过 `_check_children` 三道 —— dense 维度、BM25 Function、**analyzer 停用词**(第三道是 2026-09-21 模块 05 抓到的空格噪声缺陷后补的)。真机验证: 加停用词之前建的表被拦下并给出可读错误 `children.text 的 analyzer 缺少停用词 [' ', '\n', '的'] —— ... 必须 drop 后按当前 schema 重建并重新 ingest`; 重建后同一调用通过; 单测覆盖"旧格式 / 坏 JSON / 缺字段"三种情况;
- [x] schema 自检: 可见 `dense` dim=1024、HNSW(M=16, efConstruction=200, COSINE)、`sparse` 索引 metric=BM25、functions 里有 `text_bm25` —— **实测通过**, 但读法要改: `describe_collection` 的 `indexes` 键是空的, 索引用 `client.list_indexes(coll)` + `client.describe_index(coll, name)`, 字段维度用 `describe_collection()["fields"][i]["params"]["dim"]`, Function 用 `describe_collection()["functions"]`, analyzer 用同处 `["params"]["analyzer_params"]`(JSON 字符串);
- [x] 分词实测: `client.run_analyzer([...], analyzer_params)` 的分词与本地 jieba 口径一致(中文按 `lcut_for_search` 切、多字词带 2-gram) —— **实测通过**; 但**加了停用词过滤后差集恰好是停用词表**(如 `'HNSW 索引的参数怎么定'` → `['HNSW', '索引', '参数', '怎么', '定']`, 空格与「的」被去掉), 验收时按"差集 ⊆ 停用词表"判, 别再要求逐 token 全等; 注意返回的是 `AnalyzeResult` 对象, token 在 `.tokens` 里(不是 list, 直接比较会误判 False);
- [x] **风险注记专项实测**: 手造 4 条中文子块(正文/表格/图片/降级图片) upsert 后, dense 路与 sparse 路**各能命中**正确 chunk_id —— **实测通过, 不触发备胎**: dense 路 0.9093 命中目标子块, sparse 路 BM25 4.8875 命中目标子块(§2.5 的 rank_bm25 备胎路径**不需要**);
- [x] 幂等: 同一批 upsert 两次, 按 `chunk_id` 计数的行数不变 —— **实测通过**(子块数 2 → 2);
- [x] 删除: `delete_doc(doc_id)` 后两个 collection 的 `query` 都不再返回该 doc_id 的行 —— **实测通过**(children / parents 各剩 0 行); 清理后 live rows 均为 0, 但 `total_rows` 仍有残留计数: 删除是逻辑删除, 物理回收等 compaction(§4.5);
- [x] 维度自检(两处): (a) 假 embedder 返回 768 维时 embed 层抛可读 `ValueError` —— 单测覆盖; (b) 已建 children(dim=1024)时跑 `ensure_collections(dim=768)`, 得到可读报错 —— **实测报错文案**: `children.dense dim=1024, 与 RAG2_EMBED_DIM=768 不一致; 改维度只能重建 collection(...)`;
- [x] 归一化断言: 单测任取一条向量 `abs(‖v‖ - 1) < 1e-5` —— 通过; 真机 4 条子块向量同样全部单位化;
- [x] embed 单测(不联网): local 模式注入假 `SentenceTransformer`, api 模式 monkeypatch `MultiModalEmbedding.call` —— 32 项覆盖分批次数(文本 ≤20 / 图片 ≤5)、返回顺序与入参一致、`index` 乱序归位、维度自检触发、图片以 Base64 形态发出、`embed_chunks` 按 `chunk_type` 分组拼回与无本地图降级;
- [x] 跨模态抽查: 渲染一张写着 "HNSW vector index diagram" 的 PNG, 测 `cos(图, 相关文)=0.7444` vs `cos(图, 无关文)=0.2282` —— **同一向量空间成立**; 另: 文本 query "HNSW 索引示意图" 在含图片块的 4 条子块里把图片块排到第 2(0.6845), 跨模态召回可用;
- [x] store 单测(不连真实 Milvus): client 作为可注入依赖 —— 29 项覆盖 upsert payload **不含 sparse**、`dense` 顺序与入参逐条对应、`page_no` 哨兵、metadata 键集、超长跳过并告警、数量不一致抛错、`UPSERT_BATCH` 分批、`delete_doc` filter 字符串、三查询原语的 `search_params`/`consistency_level`、`_hit` 缺字段兜底;
- [ ] FLAT 对照(可选, 评估期做): 同一批 query 分别用 HNSW(ef=64/96/128)与 `FLAT` 跑 dense 路量召回差距 —— **未做**, 留到 06 篇评估期;
- [x] `uv run ruff check src/rag_v01` 通过; `uv run pytest src/rag_v01/tests -q` → 103 passed。

**执行记录(2026-09-21, 开发机)**: 栈 = WSL docker(服务端 2.6.0) + pymilvus 2.6.17 + dashscope 1.27.6, embedding = 百炼多模态 `qwen3-vl-embedding`(api 模式, 1024 维, 顶层 `dimension=1024` 生效)。冒烟脚本按 §3.1 链路走通"建表 → embed_chunks → upsert → flush → dense/sparse 召回 → get_parents → delete_doc", 全部符合预期; 两个 collection 保留(空), 冒烟数据已按 `doc_id` 清除, 临时图片文件已删。**注意**: WSL 空转会被回收, 隔一段时间再连 19530 会 `Connection refused`, 首次访问会自动唤醒(容器 restart 策略), 等 `milvus-standalone` 变 `healthy` 再跑即可。

## 六、与其它模块的依赖关系

```
01 篇(部署/依赖/RAG2_ 变量)           02 篇(text 内容、表格 Markdown、图片路径)
        │                                      │
        v                                      v
   embed.py + store.py(本篇) <──── 03 篇(ParentChunk / ChildChunk、chunk_id/parent_id/doc_id 规则)
        │
        ├─> 05 篇: dense_search / sparse_search / get_parents 三原语 + ef 参数
        ├─> 06 篇: 检索上下文质量评估(ef 与索引参数的调参依据回灌本篇)
        └─> 07 篇: ingest.py 流水线编排(parse -> clean -> chunk -> embed -> store)与 facade 的落地
```

- **上游**: 01 篇给部署与依赖(19530 的 WSL docker、pymilvus、多模态 embedding(百炼 qwen3-vl-embedding, DashScope SDK)与 local 纯文本备选、备胎包); 02 篇给文本与元信息来源; 03 篇给确定性的 `chunk_id` / `parent_id` 与九字段元信息——**本篇不重新定义任何一个字段名**;
- **下游**: 05 篇只调用 §3.2 的三个查询原语(不直接碰 pymilvus); 06 篇用评估结果反推 `ef` / `M` / 候选深度; 07 篇的 `ingest()` facade 串起本篇的写入路径;
- **横向**: 无。本篇不 import 本项目任何包, 配置全部走 `RAG2_` 前缀, 移植时随目录整体带走(前提是目标环境有 19530 的 Milvus 与对应环境变量)。
