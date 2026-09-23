# 06-ragas评估

> 对应实现: `src/rag_v01/evaluation.py`(**不叫 `evaluate.py`**: 与包级 facade 的公开函数
> `evaluate` 同名会被 import 覆盖, 见 §4.9)、`cli.py` 的 `evaluate` 子命令、`contracts.py` 的 `EvalReport`;
> 检索链路见 [./05-混合检索与RRF融合.md](./05-混合检索与RRF融合.md), 评估 embedding 与 [./04-嵌入与Milvus索引.md](./04-嵌入与Milvus索引.md)
> 的 embed 双模式同源, facade 签名与契约清单以 [./07-接口设计与移植方案.md](./07-接口设计与移植方案.md) 为准。本篇只产出文档。

## 一、模块目标与边界

**目标**:

1. **可复现的量化基线**: 固定测试集(30~50 条人工校对 QA) + 固定 judge 配置 + 固定四项指标 + 固定依赖版本,
   之后任何参数改动(03 的切块目标、04 的 HNSW/analyzer、05 的候选深度与 k=60)重跑一次即可对比;
2. **支撑新旧系统同集对比**: 同一份测试集、同一个最简回答生成器, 分别喂 rag_0.1 与旧 `src/tools/rag/` 的检索,
   产出并排对比报告, 作为"是否移植替换旧 rag"的决策依据。

**边界**: 评估对象是**检索 + 最简生成**两端, 不是端到端 agent 体验(不含 supervisor 路由、多轮、memory);
最简生成只有一个固定 prompt, **不做提示词工程**(理由见二.5); 评估跑真实豆包 + 真实 Milvus, 属离线流程,
`tests/` 只测数据组装与报告函数(fake judge、预录 samples、`pytest.importorskip("ragas")`)。

**本篇不新增 `RAG2_` 环境变量**: judge 三件套(`RAG2_JUDGE_API_BASE` / `RAG2_JUDGE_API_KEY` /
`RAG2_JUDGE_MODEL`)已在 [01](./01-环境与依赖.md) 定稿; 测试集路径与报告目录走函数参数(07 篇签名)。
日后若想环境变量化, 先在 01 篇变量表补一行再实现, 不绕过 01 篇偷偷读 env。

## 二、设计决策与理由(含备选对比)

### 2.1 为什么用 ragas

| 备选 | 结论 | 一句话理由 |
|---|---|---|
| 人工标注打分 | 弃 | 金标准但不可复现: 每改一次参数重标一遍, 30~50 条 × 4 维度撑不住迭代 |
| 自写规则指标(词面重叠/MRR) | 弃 | 只能量检索端, 量不了"回答是否忠实于上下文" |
| DeepEval / TruLens | 弃 | 指标思想相近, 但 ragas 四指标已成事实标准、文档社区最全, 还自带出题器 |
| ragas | **选定** | 四指标口径与链路两端一一对应, TestsetGenerator 解决冷启动无标注数据 |

代价要认下: ragas **迭代极快、大版本间有破坏性变更**(见四.8 坑位 1), 必须锁版本, 版本号进报告。

### 2.2 四个指标分别考核哪一段

四指标恰好把链路切成"检索端"与"生成端", 坏了能直接定位是 03/05 的锅还是生成的锅:

| 口径名(固定) | ragas 0.4.x 类 | 需要的输入 | 主要考核 |
|---|---|---|---|
| context_precision | `LLMContextPrecisionWithReference` | user_input / retrieved_contexts / reference | 05 的排序质量 |
| context_recall | `LLMContextRecall` | user_input / retrieved_contexts / reference | 03 的切块 + 05 的召回覆盖 |
| faithfulness | `Faithfulness` | retrieved_contexts / response | 生成侧(是否编造) |
| answer_relevancy | `ResponseRelevancy` | user_input / response(+ embedding) | 生成侧(是否答非所问) |

报告与文档统一用左列口径名; ragas 结果里的列名 / 指标 `name`(如 `llm_context_precision_with_reference`)
写进报告 meta 的对照表, 不硬编码在业务代码里, 免受版本改名影响。

### 2.3 四指标逐个讲透

以下算法按 ragas 0.4.x 口径描述, 细节以所锁版本源码为准。

**context_precision —— 检索结果的排序精度。** 量的是"命中上下文里, 相关内容是否排在前面"。算法: judge 对每条 retrieved context 出二值 verdict("这条对推出 reference 有没有用"), 再按平均精度(思想同 average precision)汇总——对每个判相关的条目算它位置上的 precision, 取平均; 相关块越靠前分越高, 一条不相关为 0。
掉分的坏法: ① 相关父块被 RRF 挤到 top-5 之外; ② top-5 全是"词面/向量相似但不答问题"的块(典型: 问具体数值, 召回一堆同主题背景段); ③ 子块切得太碎, 单条上下文信息量不足被判否。
它直接考核 05 的双路召回与 RRF 排序; 排序敏感, 组 contexts 时必须**保持检索返回顺序**(见二.4)。

**context_recall —— 检索的覆盖度。** 量的是"参考答案的每个要点, 是否都能在上下文里找到出处"。算法: 把 ground_truth 拆成句/断言, 逐句问 judge"能否归属到某条 context", 分数 = 可归属数 / 总数。
掉分的坏法: ① 该进的父块没进 top-5; ② 答案恰落在另一个子块边界而那个子块没被召回(03 的边界问题, 父块回溯缓解但不消除); ③ 表格被切坏导致要点残缺。
它同时压着 03 与 05, 是对"分块策略改没改好"最敏感的一项。

**faithfulness —— 回答对上下文的忠实度。** 量的是"回答里每条事实陈述, 有多少被上下文支撑", 幻觉率 ≈ 1 - faithfulness。算法两步: ① judge 把 response 拆成原子陈述; ② 逐条判是否被 contexts 支持, 分数 = 支持条数 / 总条数。
掉分的坏法: ① 生成侧编造; ② 检索喂的上下文不相关, 模型只能"自由发挥"补全。
所以它名义上考核生成, 检索太差也会被连带拉低——解读必须与 context_recall 联判: **recall 低 + faithfulness 低 = 检索缺料诱发生成硬编; recall 高 + faithfulness 低 = 生成侧自身问题。**

**answer_relevancy —— 回答与问题的相关度。** 量的是"是否答非所问、是否冗长离题"。算法是反向生成: 用 response 让 judge 生成若干个(数量由该指标的 strictness 参数控制)"这个回答可能对应的问题", 再算它们与原 user_input 的 embedding 余弦相似度均值。
掉分的坏法: ① 答非所问(检索偏 + 生成不纠偏); ② 兜底话术——"根据提供的资料无法回答"这类回答会大幅掉分, **即使检索确实没找到**, 要结合 context_recall 一起读, 不能单看它判死刑; ③ 回答冗长夹带大段无关内容。
它吃评估 embedding 的相似度, **换 embedding 型号会改变读数**, 跨基线对比必须同 embedding。

### 2.4 contexts 的口径: 只取父块, 不混子块

`search()` 的每个 `RetrievedChunk` 同时带 `parent_text`(父块)与 `hits`(命中子块明细, 每条含 `text`), 三种口径对比:

| 口径 | 结论 | 理由 |
|---|---|---|
| 只取父块文本, 按返回顺序 | **选定** | 与最终喂给模型/生成的载荷一致; 条数=父块数(≤5), judge 调用量最小 |
| 父块 + 子块混排 | 弃 | 子块是父块子串, 同一信息占两个名额, 人为抬高相关率、稀释排序信号 |
| 只取子块文本 | 备选(诊断用) | 对 05 排序更敏感, 但断章取义会让 recall 偏低; 只作第二口径排查用, 不与主口径混比 |

选定口径写进报告 meta(`contexts_source: parent_text`); 命中子块明细(chunk_id / dense_rank / sparse_rank /
rrf_score)留在中间产物 jsonl 的 `debug` 字段, 归因"某题为什么没答对"时用, **不进 contexts**。

### 2.5 为什么 evaluation.py 要内置「最简回答生成」

faithfulness / answer_relevancy 需要 `response`, 而 rag_0.1 只是检索库。两个选择:

- **a. 评估时拼上下文直接调豆包(固定 prompt, temperature=0)——采用**: 生成端变量压到最小, 指标变化主要
  反映检索质量; 新旧对比时两系统共用同一个生成器, 检索差异才是唯一变量;
- b. 接入主图 answer 节点跑完整链路——弃: 引入路由/memory 等无关变量, 且违反"新模块不 import agent/"纪律。

已知偏差: judge 与生成器同为豆包, 可能有轻微自评偏好(self-preference); 固定不变时对同集对比无害, 若追求
更严可换非豆包生成器, 但**两个系统必须同时换**, 否则对比失效。

### 2.6 judge 与评估 embedding, 以及版本策略

- **judge 复用豆包**: 火山方舟 OpenAI 兼容(`base_url` 填 `RAG2_JUDGE_API_BASE`), judge 是大量小步调用, 用
  便宜快速的模型合适。照 `src/agent/build.py:44` 的做法传 `extra_body={"thinking": {"type": "disabled"}}`
  关思考降首 token 延迟(对几十次小调用是数量级差别), `temperature=0` 保证可复现;
- **评估 embedding 走同一 embedding API(文本路)**: 与入库同配置(百炼 `qwen3-vl-embedding`, DashScope SDK, 1024 维, 见 04 篇)。评估只用**文本** embedding——
  图片块不单独送图, 仅以 `parent_text` / 占位文本进入 contexts(图片的真向量由 05 篇检索链路承担, 评估不重复吃)。answer_relevancy 算余弦相似度, 换模型
  直接改读数; "评估模型 == 检索模型"才让指标反映真实链路;
- **版本策略**: 本篇撰写时 PyPI 最新为 `ragas 0.4.3`(2026-01-13 发布, PyPI 核实)。01 篇 extra 里的
  `ragas>=0.2` 是宽口径, 落地时建议收紧为 `ragas==0.4.3`(或落地当日最新), 实际版本写进每份报告。本篇
  import 断言按 v0.4.3 源码核对, **以所锁版本实测为准**(验证命令见四.8 坑位 1)。

### 2.7 独立评估环境 `eval/`(定稿: 2026-09-22 落地)

ragas 与主项目的依赖**互斥**(实测): `ragas>=0.4` → `instructor` → 要 `rich<15`, 而主项目声明 `rich>=15.0.0`。
两种出路里选了**独立环境**, 主环境一个字节都不动:

| 项 | 落地值 |
|---|---|
| 位置 | `eval/pyproject.toml` + `eval/uv.lock` + `eval/.venv`(`.venv/` 已在 .gitignore 里) |
| 依赖 | `ragas 0.4.3`(PyPI 最新) + `langchain-community **0.3.31**` + `taskforce[rag2]`(路径依赖 `../`, 拿主项目的运行期依赖) |
| 隔离手段 | `[tool.uv] override-dependencies = ["rich>=14,<15"]` —— **只在这个环境里**放宽 rich |
| 加依赖 | `uv add --directory eval <包>` |
| 跑法 | `uv run --directory eval --env-file "<repo>/.env" python -m rag_v01.cli evaluate [--testset <绝径>] [--report-dir <绝径>]` |

三条实测约束, 少一条就跑不起来:

1. **`langchain-community` 必须钉 `<0.4`**: ragas 0.4.3 的 `ragas/llms/base.py` 会 `from langchain_community.chat_models.vertexai import ChatVertexAI`, 而该模块在 langchain-community 0.4 里已被移除 —— 不钉就 `import ragas` 即崩(与旧记录里 0.3.1 栽在同一处是同一个病);
2. **`--env-file` / `--testset` / `--report-dir` 要给绝对路径**: `uv run --directory eval` 会把工作目录切到 `eval/`, 相对路径会找不到 `.env` 与测试集;
3. **`PYTHONPATH` 必须显式给**: 本包靠 `pythonpath` 暴露(没进 hatch packages), 评估环境里 `import rag_v01` 靠它; 顺带也把机器级的全局站点(`PYTHONPATH=D:\python\site-packages`, 实测会遮蔽 venv 里的包)**挤出 sys.path**。

分工与注意: **入库(ingest)用主环境跑, 评估(evaluate)用评估环境跑** —— 评估环境里的 torch 是 CUDA 版、DLL 加载失败, 而 docling 的解析链会拉 torch; `cli.py` 已改成**按子命令懒加载**, 所以 `evaluate` 子命令不会再拖 docling 进来(这也是"无 import 副作用"在 CLI 上的落地)。`openai` 在评估环境里被 ragas 连锁降到 3.3.0(经 `jiter`), 只影响评估环境。

## 三、数据流与接口契约

### 3.1 评估全链路

```
data/eval_qa_v1.jsonl(人工校对后的静态测试集, 30~50 条)
   │ 逐条取 question
   ▼
search(question, top_k=5)          ← 05 篇链路, 经包级 facade, 不直摸 Milvus
   │ contexts = [h.parent_text ...], 保持返回顺序(父块口径)
   ▼
simple_answer(question, contexts)  ← 豆包, temperature=0, 关思考; 固定 prompt
   │ 落盘 ragas_samples_{system_label}.jsonl(即"仅评估"入口与新旧的交换格式)
   ▼
EvaluationDataset(samples=[SingleTurnSample(...)])
   ▼  ragas.evaluate(metrics=四指标, llm=judge, embeddings=评估 embedding(百炼 qwen3-vl-embedding · 文本路), run_config=RunConfig)
EvalReport(metrics / question_count / report_dir) + reports/ 下 json + markdown
```

### 3.2 静态测试集 `eval_qa_v1.jsonl`(每行一个对象, 字段固定)

| 字段 | 类型 | 说明 |
|---|---|---|
| `question_id` | str | 稳定主键(如 `q001`), 报告明细与中间产物都用它回链 |
| `question` | str | 问题原文 |
| `ground_truth` | str | 参考答案(人工校对后的定稿) |
| `question_type` | str | 出题器给的查询类型(simple / reasoning / multi-hop 等), 校对时可改 |
| `source_doc` | str | 出题处来源(doc_id 或文件名), 用于排查召回失败 |
| `reviewed` | bool | 人工校对通过标记; 基线集只收 `true` |
| `notes` | str | 校对备注(如"改写了问法") |

**这批基线题的来源(2026-09-22 落地)**: 用 **DuReader-Robust** 的公开镜像抽样, 而不是让 ragas 的出题器现生成
—— 后者生成出来的答案与出处都要人工校对, 前者的答案、出处、以及在段落里的**位置偏移**都是现成的:

| 项 | 值 |
|---|---|
| 来源 | `https://huggingface.co/datasets/dirtycomputer/dureader_robust-data`(单 JSON, 18.5 MB, 无加载脚本、无需令牌) |
| 规模 | 源 14,520 段(每段配 1 问, 全部有答案) → **抽 500 段当语料** → 从语料里选 **40 条**当测试题 |
| 语料形态 | **一段一个 `.md`**(`data/eval_dureader/corpus/pNNNNN.md`), 段落中位 215 字 —— 与 DuReader 的标注粒度一致; 代价是父子分块在这份语料上退化成 1 文档 = 1 父块, 多子块 / 跨页 / 同父去重这些能力**没被这套基线覆盖**(换长文档语料是后续增强) |
| 字段映射 | `question` → `question`; `answers[0].text` → `ground_truth`; 段落文件名 → `source_doc`; `question_type` 源数据没有 → 统一 `simple`, 来源记在 `notes` |
| 自动校验(替代人工校对第一关) | 只收「1 段配 1 问、答案非空、且 `context[answer_start:answer_start+len(text)] == text`」的样本 —— 「参考答案确实落在源段落里」这条机器就能证; `notes` 写明问法未人工改写 |
| 许可 | 学术研究用途; 原始数据与派生语料**都不进 git**, 报告 meta 里的 `testset_sha256_16` 保证可比性 |
| 词表 | 参考答案是**短语**(「300」「方太」「15个」)而非整句: 上下文召回的分母小、读数偏乐观, 报告解读时必须一并看 |

规模 30~50 条: 少于 30 均值抖动太大, 多于 50 judge 成本线性上涨而信息增量有限。测试集建议入库版本控制,
修改要**新起 v2 文件名**, 不原地改——基线报告的对比对象不能漂移。

### 3.3 中间产物 `ragas_samples_{system_label}.jsonl`

| 字段 | 说明 |
|---|---|
| `question_id` | 回链静态集 |
| `user_input` / `retrieved_contexts` / `response` / `reference` | **字段名对齐 ragas 样本**, 使"仅评估"入口能直接构造 `SingleTurnSample` |
| `debug` | 命中子块明细(chunk_id / parent_id / chunk_type / dense_rank / sparse_rank / rrf_score)与参数快照, ragas 不读, 报告与排查用 |

contexts 与 response 是**跑评估时动态生成**的, 不回写静态集——否则换系统对比时会带上旧系统运行的结果。
`system_label` 只体现在文件名与报告内容里(`rag_0.1` / `tools_rag_v1`), 不进契约。

### 3.4 输出契约与依赖

```python
# contracts.py 中的 EvalReport, 与 07 篇 §3.2 逐字一致
@dataclass
class EvalReport:
    metrics: dict[str, float]   # 四指标均值, 键用 2.2 的口径名
    question_count: int
    report_dir: str             # json + markdown 报告的落盘目录
```

逐题明细、system_label、judge 型号、ragas 版本、检索参数快照都放**报告文件**的 meta 与 `per_question` 数组,
不进契约——契约只保留程序要用的最小面, 报告是给人读的产物。

| 依赖 | 接口/类型 | 用途 |
|---|---|---|
| contracts.py | `EvalReport` | 本篇输出 |
| contracts.py | `RetrievedChunk` | 组 contexts(取 `parent_text`, 保序) |
| 05 篇 facade | `search(query, top_k)` | 逐题检索(落路口径: `top_k=None` 走 `RAG2_RETRIEVE_TOP_K`) |
| 04 篇 embed.py | `embed_documents` / `embed_query`(双模式) | 评估 embedding 适配器 |
| config.py | `RAG2_JUDGE_*` 三件套; 参数快照用 `RAG2_RETRIEVE_TOP_K` / `RAG2_CANDIDATE_TOP_N` / `RAG2_RRF_K` | judge 装配与报告 meta |

## 四、实现要点(API + 骨架 + 坑位)

### 4.1 judge 装配: 走 ragas 新接口 `llm_factory`(落地定稿)

```python
# 落地版(2026-09-22 实测可用); ragas / openai 的 import 都在函数体内 —— 主环境不装评估库也能导入本模块
def build_judge(cfg):
    import openai
    from ragas.llms import llm_factory
    client = openai.OpenAI(base_url=cfg.judge_api_base, api_key=cfg.judge_api_key,
                           timeout=120, max_retries=3)
    return llm_factory(cfg.judge_model, client=client, temperature=0)   # -> InstructorLLM

def build_metrics() -> list:                    # 单点适配: 升级 ragas 只改这一个函数
    return [LLMContextPrecisionWithReference(), # -> context_precision
            LLMContextRecall(),                 # -> context_recall
            Faithfulness(),                     # -> faithfulness
            ResponseRelevancy()]                # -> answer_relevancy
```

落地时改掉的两处口径:

1. **不用 `LangchainLLMWrapper`(原定稿)**: 它把 langchain 拉进评估链路, 而 ragas 0.4.3 与
   langchain-community 0.4 本就不兼容(要钉 `<0.4`, 见 §2.7); 少一层依赖少一个雷, `llm_factory` 是
   ragas 自己给的升级方案, 试跑与全量都通过;
2. **judge 实际是 DeepSeek 不是豆包**: `.env` 里 `RAG2_JUDGE_MODEL=deepseek-flash` /
   `RAG2_JUDGE_API_BASE=https://api.deepseek.com`。因此**不要**照 `src/agent/build.py:44` 传
   豆包专用的 `extra_body={"thinking": {"type": "disabled"}}`(那是方舟的参数); 用 `temperature=0` 保证
   可复现就够了。judge 型号写进每份报告的 meta, 换型号 = 换基线。

### 4.2 评估 embedding: 直接继承 ragas 的嵌入基类(落地定稿)

```python
def build_embeddings():
    from ragas.embeddings import BaseRagasEmbedding

    class _Rag2Embeddings(BaseRagasEmbedding):     # 文本路复用 embed.py: 与入库同模型同维度
        def embed_text(self, text, **kwargs):
            from .embed import embed_query
            return embed_query(text)

        def embed_texts(self, texts, **kwargs):
            from .embed import embed_documents
            return embed_documents(texts)

        async def aembed_text(self, text, **kwargs):          # 库的抽象基类要求(见下)
            return await asyncio.to_thread(self.embed_text, text, **kwargs)

        async def aembed_texts(self, texts, **kwargs):
            return await asyncio.to_thread(self.embed_texts, texts, **kwargs)

    return _Rag2Embeddings()
```

两件落地时才弄清的事:

1. **ragas 自带的 provider 用不了**: 百炼的 OpenAI 兼容端点**不支持** `qwen3-vl-embedding`
   (实测返回 404 `model_not_supported`), 而 answer_relevancy 直接吃这个向量的余弦相似度 —— 必须与入库
   同一模型同一模式(§2.6), 所以只能自己适配, 不能换模型绕过;
2. **基类强制要求一对 async 方法**: `BaseRagasEmbedding`(以及它的子类 `RagasBaseEmbedding`)把
   `embed_text` 与 `aembed_text` **都**声明为抽象方法, 没有纯同步的基类可继承。于是本包唯一一处
   `async def` 出现在这里 —— 两个薄壳只做"同步 HTTP 调用丢进线程"(`asyncio.to_thread`), 不建事件循环、
   不 await 业务代码, 不改变项目自己的同步执行模型。这条已记进坑位 9, 也是"少写一层 langchain 包装"的代价。

### 4.3 测试集生成 + 人工校对(一次性脚本, 不进包)

```python
# 一次性脚本(仓库根或 scripts/, 不进 src/rag_v01/): 出题 -> 人工审 -> eval_qa_v1.jsonl
from langchain_core.documents import Document
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution   # 注意: 不在 ragas.testset 顶层

docs = [Document(page_content=t, metadata={"source": s}) for s, t in load_corpus()]
dist = default_query_distribution(llm=judge)          # 三种题型: 单跳具体 / 多跳抽象 / 多跳具体
generator = TestsetGenerator(llm=judge, embedding_model=eval_embeddings)
testset = generator.generate_with_langchain_docs(docs, testset_size=40, query_distribution=dist)
testset.to_pandas().to_json("testset_raw.jsonl", orient="records", lines=True, force_ascii=False)
```

人工校对五查(每条必过): ① ground_truth 能否在语料里找到出处, 找不到删; ② 是否要语料外常识才能答(超纲删); ③ multi-hop 题人为是否可答; ④ 问法改成贴近真实使用的口吻; ⑤ 与已有题去重/合并。
校对通过的打 `reviewed=true`, 另存为 `eval_qa_v1.jsonl`(原始生成件保留不动)。**未校对的生成题直接进基线会系统性拉低 context_recall 并污染基线**, 这是硬性步骤。

### 4.4 evaluation.py 主流程(两段式)

先落中间产物再喂 ragas, 让评估器"只认 jsonl 不认检索器"——旧系统因此不必进 rag_0.1 的代码:

```python
RAGAS_FIELDS = ("user_input", "retrieved_contexts", "response", "reference")

def run_eval(testset_path=None, report_dir="./eval_reports", samples_path=None, system_label="rag_0.1", limit=None):
    if samples_path:                                   # 仅评估入口: 直接吃既有中间产物
        samples = load_jsonl(samples_path)
    else:                                              # 完整跑: 检索 + 最简生成
        samples = [make_sample(qa) for qa in load_jsonl(testset_path or DEFAULT_TESTSET)[:limit]]
    dump_jsonl(samples_path or f"ragas_samples_{system_label}.jsonl", samples)
    return build_report(ragas_evaluate(samples), samples, report_dir=report_dir, system_label=system_label)

def make_sample(qa) -> dict:
    hits = search(qa["question"], top_k=5)             # facade, 05 篇链路
    contexts = [h.parent_text for h in hits]           # 父块口径, 保持 RRF 名次顺序
    return {"question_id": qa["question_id"], "user_input": qa["question"], "retrieved_contexts": contexts,
            "response": simple_answer(qa["question"], contexts),
            "reference": qa["ground_truth"], "debug": debug_of(hits)}

def ragas_evaluate(samples):
    dataset = EvaluationDataset(samples=[SingleTurnSample(**{k: s[k] for k in RAGAS_FIELDS})
                                        for s in samples])
    return evaluate(dataset=dataset, metrics=build_metrics(),
                    llm=judge, embeddings=eval_embeddings, run_config=run_config,
                    raise_exceptions=False)            # 失败样本留 NaN, 不炸整批
```

要点: `simple_answer` 用固定 prompt("只依据资料回答; 资料没有就说不知道; 不要展开"), `temperature=0`, 限
`max_tokens`——它变了就是换基线, 改动进报告 meta; 结果映射从 `result.to_pandas()` 的列名(或各 metric 的
`name`)到四口径名, 映射表写进报告 meta, **不硬编码**版本相关键名; `cli.py` 的 `evaluate` 参数面以 07 篇为准
(`--testset` / `--report-dir`), `limit` / `samples_path` / `system_label` 是内部函数的调试与对比参数, 由包外
一次性脚本直接调 `run_eval(...)` 使用(若日后要暴露, 需同步回填 07 篇的 cli 表)。

### 4.5 报告: json + markdown

- **json**: `meta`(system_label, 测试集路径+条数(+可选 sha256), ragas 版本, judge 型号, embedding 型号与模式,
  检索参数快照 top-20 / top-5 / k=60, contexts 口径, 生成 prompt 版本) + `metrics`(四口径名 → 均值) +
  `per_question`(每题四分数、NaN 标记、contexts 条数、answer 与命中明细);
- **markdown**: 表 1 四指标均值 + NaN 占比; 表 2 逐题明细; 若 `report_dir` 下存在其他 system_label 的同数据集
  报告, 追加**并排对比表**(行=指标, 列=系统, 另加 delta 列), 这就是"新旧同集对比"的呈现形式;
- 文件名带 system_label 与时间戳(`eval_rag0.1_20260919-1200.json`), 不覆盖历史。

### 4.6 成本与限流控制

- 每题 judge 调用是"个位数到十几次"量级(取决于所锁版本的批处理粒度: precision 逐条 context 判、faithfulness 拆陈述 + 逐条验证、recall 逐参考句归属、answer_relevancy 反向生成), 40 题即**几百次**量级; 不硬编码预算, 用 `limit=5` 试跑实测 token 账单后再放全量;
- `RunConfig(max_workers=4)` 起步, 遇 429 先降并发(4→2), 再考虑加大 `max_retries`;
- 生成端与 judge 共用同一 key 与限流池: **先生成完所有 answer(单线程), 再跑 ragas 并发**, 避免两股流量叠加;
- 父块口径天然比"父块+子块"省 judge 调用, 这是二.4 口径选择的附带收益。

### 4.7 与旧系统同集对比的跑法

旧系统入口: `tools.rag.store.RAGStore.search(query, top_k) -> list[dict]`(`src/tools/rag/store.py:212`, dict 含 `content` / `filename` / `doc_id` / `score`); 高层 `kb_search` 只是它的 JSON 字符串封装(`src/tools/rag/kb_search.py:34`), 不适合做评估取数。跑法:

1. 包外一次性脚本里 `RAGStore().search(question, top_k=5)`, 取 `contexts = [h["content"] ...]`, 复用 rag_0.1 的 `simple_answer`(同一个函数, 保证 prompt 完全一致), 产出 `ragas_samples_tools_rag_v1.jsonl`;
2. 两边都走"仅评估"入口 `run_eval(samples_path=..., system_label="tools_rag_v1")` —— **命令行同理**:
   `cli evaluate --samples <旧系统的中间产物> --label tools_rag_v1`, 标签不同, 报告里就自动多一张并排对比表;
   中间产物默认落在报告目录下(`<report_dir>/ragas_samples_<label>.jsonl`), 便于整目录归档与复评;
3. 公平性清单(逐项核对): 同测试集、同 top_k=5、同生成 prompt/模型/温度、同 judge、同四指标、同 contexts 口径; 旧系统自身参数(**如其 `rrf_fuse` 的 k 默认值**, 见 `src/tools/rag/bm25.py:86`)按其现状记录进 meta, 不对齐——对比的就是"旧系统现状 vs 新系统"。

### 4.8 坑位与待验证点(本篇重点)

1. **ragas 大版本 API 变动(最大坑)**。0.1.x 指标是小写模块级常量(`from ragas.metrics import context_precision`), 0.2 起改为类, `answer_relevancy` 的类名是 `ResponseRelevancy`, 数据集从 dict/HF Dataset 演进为 `EvaluationDataset`/`SingleTurnSample`。
   本篇按 **v0.4.3 源码**核对的可用路径(撰写时快照, 以所锁版本实测为准): `from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig`; 四指标类在 `ragas.metrics`; `TestsetGenerator` 在 `ragas.testset`; `default_query_distribution` 在 `ragas.testset.synthesizers`(**不在** `ragas.testset` 顶层)。
   对策: 版本钉死 + 指标构造集中在 `build_metrics()` 一处 + 报告记录版本。验证命令(以输出为准): `uv run python -X utf8 -c "import ragas; from ragas import SingleTurnSample; from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall; print(ragas.__version__, SingleTurnSample.model_fields)"`; 个别字段对不上时, `evaluate()` 仍有 `column_map` 可做列名映射兜底。
2. **依赖链冲突(2026-09-22 已解决: 独立评估环境)**。原记录: `ragas>=0.4` 依赖 instructor, instructor 钉 `rich<15`, 而主项目是 `rich>=15.0.0` → uv 直接报 unsatisfiable; 降到 `ragas==0.3.1` 则 `import ragas` 即崩(引 `langchain_community.chat_models.vertexai`, 该模块已从 langchain-community 0.4 移除)。
   **落地口径**: 走独立环境 `eval/`(配方与三条实测约束见 §2.7), 主环境零改动。注意 **`ragas 0.4.3` 也栽在同一个 `vertexai` 模块上** —— 不是 0.3.1 独有的问题; 修法是在评估环境里钉 `langchain-community<0.4`(实测 0.3.31 与 langchain-core 1.6.3 共存无冲突)。
3. **两个 Wrapper 已被标记弃用(0.4.3)** —— **落地时干脆没用它们**: judge 走 `llm_factory`(§4.1), embedding 直接继承 `BaseRagasEmbedding`(§4.2), 于是评估链路里没有 langchain 的位置, 弃用告警与「升级要切两处」的问题一起消失。
4. **judge 对中文指令的遵循性**: ragas 内置 prompt 是英文指令 + 中文样本, 豆包一般能处理, 但 faithfulness 的中文长句陈述拆分可能过碎(分数虚高)或漏拆。验证: 挑 3 条把逐题明细与人工判断对照, 偏差大就换 judge 型号做一次漂移对照, 并把 judge 型号写进报告。
5. **无参考答案场景的指标选择**: 缺 `reference` 时 context_recall 与带参考的 context_precision 都不可跑, 只能跑 `LLMContextPrecisionWithoutReference` + faithfulness + answer_relevancy(三指标分支)。静态集默认要求全部带 ground_truth; 实在写不出参考答案的题单独打标走三指标分支, **不混入四指标基线**。
6. **`raise_exceptions` 与 NaN**: 默认 False 时失败样本得 NaN 而非整批炸掉; 报告必须统计 NaN 条数与占比并列出 question_id, 不允许静默丢弃。常见原因是 judge 输出解析失败(重试耗尽)与超长样本。
7. **contexts 的顺序与去重**: `search` 已按父去重且顺序即 RRF 名次, 组 contexts 时**不要**再排序或用 set 去重(set 打乱顺序, context_precision 对顺序敏感); 空结果题(contexts=[])照常进评估并标记, 它们正是 recall 掉分的证据。
8. **prompt/口径变更 = 换基线**: 最简生成 prompt、contexts 口径、judge 型号、ragas 版本任一变化, 旧报告不可直接比; 报告 meta 就是为了让这种"不可比"显式化。

9. **嵌入基类强制一对 async 方法(本项目唯一一处 `async def`)**: `BaseRagasEmbedding` 把 `embed_text` 与 `aembed_text` **都**声明为抽象方法, 没有纯同步基类可继承 —— 想给 ragas 一个自定义嵌入(我们必须给, 因为它自带 provider 不支持 `qwen3-vl-embedding`, 见 §4.2), 就得实现那对 async 方法。落地写成「同步调用丢进 `asyncio.to_thread`」的薄壳: 不建事件循环、不 await 业务代码、项目自己的执行模型仍是全同步; 已加注释说明这不是架构选型, 而是第三方基类的硬要求。

10.1 **费用/用量统计未接(如实留痕)**: 报告 meta 里的 `total_tokens` / `total_cost` 现在是 `<不可用: ValueError>` / `None` —— `total_tokens()` 在没有 `token_usage_parser` 时也会抛错。ragas 自带的解析器(`ragas.cost.get_token_usage_for_openai` 等)签名收的是 **langchain** 的 `LLMResult`, 而我们走 `llm_factory`(instructor 客户端), 因此要接费用统计得自己写一个 parser。本期不做: 单机跑一次 40 题的规模, 用量上限从平台控制台看就够; 但**报告里保留了这两个字段并写明不可用**, 免得以后有人以为"没花钱"。
10. **ragas 的几个统计量是方法而不是属性**: `EvaluationResult.total_tokens` / `total_cost` 实测是**方法**(直接 `getattr` 拿到 bound method, 写报告时 `json.dumps` 报 `Object of type method is not JSON serializable`)。落地做两件事: 取值时「是方法就调用」(`evaluation._value_of`), 写报告时 `json.dumps(..., default=str)` 兜底 —— 报告是跑了几分钟 judge 之后才写的, 不该因为一个字段类型整批白跑。

11. **doc_id 是内容寻址 → 同内容的两个文件会合成一份文档**: 500 篇评估语料里有 2 篇文本完全相同, 于是 `list_docs()` 只列出 499 份(doc_id 相同 → chunk_id 相同 → 后写的那份覆盖前一份, `source` 取后者)。这是内容寻址的**预期行为**(同内容即同一文档), 不是丢数据; 但排查「文件数与文档数不一致」时要想到它。

12. **LLM 两端带来不可复现的噪声(实测, 最影响"对着比"的一条)**: 同配置重跑, **检索侧确定性**(dense/BM25/RRF 逐题一致, 两轮 context_precision / recall 都是 1.0), 但生成与 judge 是随机的 —— 5 题重复实验里 faithfulness Δ0.50、answer_relevancy Δ0.34, 而根源是 **3/5 题的生成答案都不一样**(`temperature=0` 在 deepseek-flash 上并不等于逐字确定)。三条影响与对策:
    (a) **基线对比要同口径**: 比"检索参数改动"的效果时, 最稳的做法是先固定一份中间产物, 用「仅评估」入口(`--samples`)两系统/两参数共用同一批 `response`, 否则生成噪声会淹没参数差异;
    (b) **更进一步的做法(未实现)**: 按 `(question, contexts)` 的哈希缓存生成结果 —— 检索参数变了就重生成, 没变就复用, 这样"参数改动"的对比里生成端是常量;
    (c) **判读阈值(实测)**: 全量 40 题重跑一遍, **检索侧漂移 ≤0.003(context_precision Δ0.0026 / context_recall Δ0.0000)**, 而**生成+判分侧漂移 0.028~0.048**(faithfulness Δ0.0477 / answer_relevancy Δ0.0280, NaN 题数还会从 7 变成 10)。把输入完全固定下来再评两遍: 用「仅评估」入口把**同一份中间产物**(同 contexts + 同 response)评两遍 —— context_precision / context_recall 两轮**完全一致(Δ=0)**, 但 **faithfulness Δ0.20、answer_relevancy Δ0.09**(5 题规模; 逐题里 q002 的 0.79→0.66、q003 的 0.74→0.87, 连「某题是 NaN 还是 0」都会变)。
        结论: **检索侧可复现, 生成侧与 judge 侧不可复现**; 因此 **指标差异小于 ~0.1 时不要声称「改进了」**, 要判参数好坏就得加大样本或多次取均值。

### 4.9 落地记录(2026-09-22)

1. **模块名是 `evaluation.py`, 不叫 `evaluate.py`**: 后者与包级 facade 的公开函数 `evaluate` 同名, Python 在
   import 子模块时会把父包上的同名属性**换成模块对象** —— 「先用没问题、某次 import 之后 `rag_v01.evaluate(...)`
   就变成调用模块」。同名冲突只能靠改名消灭(与流水线模块叫 `pipeline.py` 同一条规则); 公开 API 名不变。
   单测里留了回归用例(import 子模块之后再断言 facade 的 `evaluate` 仍是函数)。
2. **评估库与 openai 的 import 全在函数体内**(`_ragas_api()` 单点负责并给出缺依赖指引): 主环境没装 ragas
   也能导入本模块、跑单测; facade 的 `evaluate()` 在主环境调用时抛一条带跑法的 `ImportError`。
3. **中间产物落进报告目录**(`report_dir/ragas_samples_{label}.jsonl`): 评估环境是用 `uv run --directory eval`
   跑的, cwd 在 `eval/` 下, 用相对路径会把产物写到奇怪的地方; 落进报告目录同时方便整目录归档与「仅评估」复评。
4. **CLI 改成按子命令懒加载**: 原来顶层就 import `pipeline`(→ docling → torch), 而评估环境里的 torch 是 CUDA 版、
   DLL 加载失败 —— 于是 `evaluate` 子命令在评估环境里根本起不来。现在每个 `_cmd_*` 里各自 import, 跑评估不再拖
   docling, 这也是 07 篇「无 import 副作用」在 CLI 上的落地(单测的 patch 点相应改到源模块, 如 `rag_v01.pipeline.run_ingest`)。
5. **`--limit` / `--samples` 从内部参数升成 CLI 选项**: 06 篇原文说这两个是包外脚本用的调试参数, 但基线试跑
   (先 5 题看账单)本身就要求它能从命令行触发, 所以暴露出来并同步回填 07 篇的 cli 参数表。
6. **markdown 逐题表补了「参考答案 / 生成回答」两列**(各截 24 字): 归因「某题为什么没答对」时, 光看四个分数
   不够, 得能一眼看到模型答了什么、期望是什么。
7. **facade 三入口齐了**: `evaluate(testset_path, report_dir, *, samples_path, system_label, limit)` 与
   `rag_v01.EvalReport` 一并导出。

## 五、验收标准(自测清单)

- [x] `eval_qa_v1.jsonl` 40 条, 全部 `reviewed=true`, 每条 ground_truth 都能在语料里找到出处 —— **来源与自动校验见 §3.2**(DuReader-Robust 抽样, 「参考答案在源段落内且与 `answer_start` 一致」由脚本强校验);
- [x] 试跑通过: 先 5 题(3m09s, 验链路与账单量级)后 2 题(1m33s, 验报告落盘), 四指标均有数值; 中间产物 `ragas_samples_rag_v01.jsonl` 落在报告目录且字段齐全(40 行);
- [x] 全量跑通: `eval_reports/` 下产出 `eval_rag_v01_20260922-170437.json` + `.md`, meta 含 ragas 版本 / judge 型号 / embedding 型号与模式 / 检索参数快照 / contexts 口径 / 生成 prompt 版本 / 语料快照 / NaN 名单; **未接费用统计**(坑位 10.1), 字段留痕为「不可用」;
- [x] 复现性(同配置重跑 ±0.02 量级) —— **全量实测: 检索侧达标、生成与判分侧不达标**(2026-09-22, 40 题跑两遍, 语料/测试集/参数/prompt/judge 全部一致): **context_precision 0.8958 → 0.8932(Δ0.0026)✓**、**context_recall 0.9500 → 0.9500(Δ0.0000)✓**, 而 **faithfulness 0.9595 → 0.9118(Δ0.0477)✗**、**answer_relevancy 0.5015 → 0.4735(Δ0.0280)✗**, NaN 题数 7 → 10。结论与对策见坑位 12: **评估的噪声来自 LLM 两端(生成 + judge), 不是检索**; 把输入固定下来再评两遍(「仅评估」入口), judge 自身的抖动是 faithfulness Δ0.20、answer_relevancy Δ0.09(检索侧 Δ=0), 所以**指标差异小于 ~0.1 不足以判定改进了**(坑位 12c); 想比"参数改动的效果", 要么固定同一份中间产物走「仅评估」入口, 要么把生成结果按 (question, contexts) 缓存起来复用;
- [x] 仅评估入口: `cli evaluate --samples <既有中间产物>` 只评不检索, 单测覆盖该分支(`run_eval` 的 samples-only 路径断言"不再检索、不重落产物"); 真机用旧系统对比时会走它(里程碑 B);
- [x] 新旧对比(2026-09-22 完成): 属里程碑 B 步骤 4 —— 旧 `tools_rag_v1` 的中间产物由一次性脚本产出后走 `--samples` 入口, 与新系统同集同 judge 对比: 旧 精度 0.6896 / 召回 0.9000 / 忠实度 0.8681 / 相关度 0.4774 → 新 **0.8958 / 0.9500 / 0.9595 / 0.5015**; 四项全升但只有精度(+0.206)超出噪声带(坑位 12c 的阈值), 结论与完整对照条件见 07 篇 §4.6.2。报告 `eval_reports/eval_tools_rag_v1_20260922-203552.md` 末尾自动追加了并排对比表(本项的验收点);
- [x] `tests/` 单测(不连真 LLM / Milvus): `tests/test_evaluate.py` **24 项** —— jsonl 读写与行号报错、测试集字段校验、`make_sample` 保序与 debug、固定 prompt 与 `temperature=0`、`summarize` 的 NaN 语义、指标名映射、报告 json/markdown/对比表、Milvus 缺失容错、`run_eval` 两条入口、facade 回归、`cli evaluate` 参数与输出、`_value_of` 对"抛错的统计量"的容错;
- [x] `uv run ruff check src/rag_v01` 通过; `uv run pytest src/rag_v01/tests -q` → **201 passed**(模块 06 贡献 23 项)。

### 执行记录: 基线(2026-09-22, 真 Milvus + 真 embedding + 真 judge)

**跑法**: `uv run --directory eval --env-file "<repo>/.env" python -m rag_v01.cli evaluate --testset "<repo>/data/eval_qa_v1.jsonl" --report-dir "<repo>/eval_reports"`(评估环境配方与三条约束见 §2.7)。

**语料**: 500 篇 DuReader-Robust 段落入库耗时 **1m59s**, 库里 499 份文档 / 597 个子块(2 篇同内容被内容寻址合并, 见坑位 11)。

**四指标(40 题)**:

| 指标 | 均值 | 有效题数 | 读法 |
|---|---|---|---|
| context_precision | **0.8958** | 40/40 | 检索排序质量: 五条上下文里偶有"同主题但不答问题"的段落 |
| context_recall | **0.9500** | 40/40 | 覆盖度好 —— 但也受"参考答案是短语、分母小"影响, 偏乐观 |
| faithfulness | **0.9595** | 37/40 | 生成基本不编造; 3 题 NaN 来自 judge 输出被截断 |
| answer_relevancy | **0.5015** | 36/40 | **最弱的一项**, 主要不是检索的锅: 见下 |

**answer_relevancy 为什么只有 0.50(三条实证, 都值得记住)**:
1. **兜底话术拉低**: 4 条最低分里有 2 条是模型答「根据资料无法回答。」(q004「暗黑2 存档位置」、q008「低保家庭人均年收入」)—— 上下文没喂到答案时它不编造, 忠实度满分但相关度 0; 这正是 06 篇 §2.3 反复提醒的"别单看这一项判死刑";
2. **judge 只给了 1 个反向问题**: 全量日志里 40 次 `LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.` —— ResponseRelevancy 默认要生成 3 个"这段回答可能对应的问题"(strictness=3)再与原始问题算相似度均值, 而 deepseek-flash 每次只返 1 个, 分数因此**方差更大、系统性偏低**(q005 期望「今日は何日ですか」而模型答「何日」, 相关度直接 0);
3. **参考答案是短语**: `ground_truth` 如「300」「方太」, 最简生成器往往答得极短, 反向生成的问题与原始问题的语义重叠小。

**未验证/已知短板**: 复现性未做全量重跑(见清单); 父子分块的多子块/跨页/同父去重未被这份语料覆盖(§3.2); 费用统计未接(坑位 10.1)。

**过程中的两次事故(都记下来, 因为都不是代码逻辑错)**:
1. 第一次全量跑到最后一刻才崩在 `total_cost()` 上(40 题 judge 已花完) → 修 `_value_of` 容错 + `json` 兜底, 并补回归用例;
2. 第二次全量的 judge 账户**余额不足**(`Error code: 402 Insufficient Balance`), 40 题里 39 题拿不到分数, 那份报告已作废; 充值后重跑即得到上表。

## 六、与其它模块的依赖关系

```
05 篇 search ───────────────┐
04 篇 embed(双模式, 默认百炼 API) ─┼──> evaluation.py(本篇) ──> EvalReport / reports
01 篇 RAG2_JUDGE_* 三件套 ──┘        │
                                     ├──> 07 篇 facade: evaluate(testset_path, report_dir)
                                     └──> 一次性对比脚本(包外): 旧 tools/rag + 仅评估入口
```

- **上游**: 05 的 `search` 提供 contexts(父块口径); 04 的 embed 提供评估 embedding; 01 的 judge 三件套与依赖
  清单(ragas / langchain-openai)是运行前提; 03 的切块质量会直接体现在 context_recall 上;
- **下游**: 07 篇 facade 的 `evaluate(...)` 就是本篇 `run_eval` 的包级暴露; 基线报告是"是否移植替换旧 rag"的
  决策依据, 也是 03/05 参数调优的回归参照;
- **纪律**: 与旧 `tools/rag` 的对比只经中间产物 jsonl 交换, 不产生任何 import; 对比脚本放 `src/rag_v01/`
  之外; 本模块继续零项目依赖、全同步(ragas 的 `evaluate` 本身就是同步 API)。
