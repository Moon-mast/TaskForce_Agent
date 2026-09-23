"""evaluation.py —— ragas 四指标评估(06 篇)。

**模块名为什么不叫 `evaluate.py`**: 它跟包级 facade 的公开函数 `evaluate` 同名, 而 Python 在 import
子模块时会把父包上的同名属性换成模块对象 —— 于是"先用没问题、某次 import 之后
`rag_v01.evaluate(...)` 就变成调用模块"。同名冲突只能靠改名消灭(与 `pipeline.py` 同一条规则)。

**两段式**(06 篇 §4.4): 先把"检索 + 最简生成"的结果落成中间产物 jsonl, 再喂给 ragas —— 评估器
只认 jsonl 不认检索器, 于是旧系统的对比脚本不必进本包(里程碑 B 用 `samples_path` 入口跑)。

**依赖隔离**: ragas 装在独立的评估环境 `eval/` 里(ragas 与主项目依赖互斥, 见 06 篇 §2.7), 所以
ragas / openai 的 import **全部在函数体内** —— 本模块在主环境能导入、能单测; 缺依赖时的报错直接给出
跑法。**唯一例外**是嵌入适配器 `build_embeddings()`: ragas 的嵌入基类把 `embed_text` 与
`aembed_text` 都声明为抽象方法(库的硬要求, 没有纯同步的基类可继承), 所以那里有一对 `async` 方法 ——
它们只做"同步 HTTP 调用丢进线程", 不建事件循环、不改变本项目自己的同步执行模型, 全包仅此一处。

本文件只 import 同级模块, 不 import 项目其它包, 整目录可搬走。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .config import Config, load_config
from .contracts import EvalReport

DEFAULT_TESTSET = "data/eval_qa_v1.jsonl"
DEFAULT_REPORT_DIR = "eval_reports"
DEFAULT_SYSTEM_LABEL = "rag_v01"

# 中间产物/数据集里给 ragas 的四个字段(名字必须与 ragas 的样本字段一致)
RAGAS_FIELDS = ("user_input", "retrieved_contexts", "response", "reference")
TESTSET_FIELDS = (
    "question_id",
    "question",
    "ground_truth",
    "question_type",
    "source_doc",
    "reviewed",
    "notes",
)

# ragas 结果列名 → 06 篇 §2.2 的口径名。只在这一处映射: 升级 ragas 改名时业务代码与报告格式都不动
# (06 篇坑位 1 的对策)。
METRIC_ALIASES = {
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
}

# 最简生成: 固定 prompt + temperature=0 —— 生成端变量压到最小, 指标变化主要反映检索质量;
# 新旧对比时两系统共用这一个生成器, 检索差异才是唯一变量(06 篇 §2.5)。改它就是换基线, 版本号进报告。
ANSWER_PROMPT_VERSION = "v1"
_ANSWER_PROMPT = (
    "只依据下面的资料回答问题。资料里没有的信息不要编造; 资料不足时直接回答: 根据资料无法回答。\n"
    "不要展开, 不要复述资料原文, 回答尽量简短。\n\n资料:\n{contexts}\n\n问题: {question}\n回答:"
)

_RAGAS_HINT = (
    "评估需要 ragas, 它装在独立的评估环境里(ragas 与主项目依赖互斥, 见 06 篇 §2.7)。跑法:\n"
    "  uv run --directory eval --env-file .env python -m rag_v01.cli evaluate [--testset 路径]\n"
    "主环境不装 ragas 也能导入本模块与跑单测。"
)


def _cfg() -> Config:
    """配置懒加载(与 embed.py / store.py 同款)。"""
    return load_config()


# ---------------------------------------------------------------- jsonl 读写


def load_jsonl(path: str | Path) -> list[dict]:
    """读 jsonl: 每行一个对象; 空行跳过。"""
    rows: list[dict] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON: {exc}") from exc
    return rows


def dump_jsonl(path: str | Path, rows: list[dict]) -> None:
    """写 jsonl: 覆盖写, 保证父目录存在(中间产物要能落盘才算"可拆流程")。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_testset(path: str | Path) -> list[dict]:
    """读静态测试集并校验字段(06 篇 §3.2): 缺字段/空值立刻报错并点出行号。

    为什么校验得严: 基线是"以后所有参数改动都要对着比"的参照物, 一条脏数据进来会安静地拉低读数,
    比直接报错难查得多。
    """
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"测试集是空的: {path}")
    for index, row in enumerate(rows, start=1):
        missing = [field for field in TESTSET_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{path} 第 {index} 条缺字段 {missing}")
        if not str(row["question"]).strip() or not str(row["ground_truth"]).strip():
            raise ValueError(f"{path} 第 {index} 条的 question / ground_truth 为空")
    return rows


# ---------------------------------------------------------------- 单条样本


def debug_of(chunks: list[Any]) -> list[dict]:
    """命中子块明细(06 篇 §3.3 的 debug 字段): ragas 不读, 排查"某题为什么没答对"时用。

    入参是**检索结果**(`RetrievedChunk`)而不是 `ChildHit` —— `parent_id` 挂在父块条目上, 子块明细里
    没有(契约如此)。顺序即 RRF 名次: 归因时"相关块排第几"与"有没有被召回"是两件事。
    """
    return [
        {
            "chunk_id": hit.chunk_id,
            "parent_id": chunk.parent_id,
            "chunk_type": hit.chunk_type,
            "dense_rank": hit.dense_rank,
            "sparse_rank": hit.sparse_rank,
            "rrf_score": hit.rrf_score,
        }
        for chunk in chunks
        for hit in chunk.hits
    ]


def simple_answer(question: str, contexts: list[str], cfg: Config | None = None) -> str:
    """最简回答生成(06 篇 §2.5): 固定 prompt + temperature=0, 只依据给到的资料。

    生成端与 judge 共用同一套 `RAG2_JUDGE_*` 配置(同一个 key 与限流池, 见 06 篇 §4.6)。
    """
    cfg = cfg or _cfg()
    import openai  # 懒 import: 主环境不跑评估时不必装

    client = openai.OpenAI(
        base_url=cfg.judge_api_base, api_key=cfg.judge_api_key, timeout=120, max_retries=3
    )
    blocks = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))
    resp = client.chat.completions.create(
        model=cfg.judge_model,
        messages=[
            {
                "role": "user",
                "content": _ANSWER_PROMPT.format(contexts=blocks or "(无)", question=question),
            }
        ],
        temperature=0,
        max_tokens=512,
    )
    return (resp.choices[0].message.content or "").strip()


def make_sample(qa: dict, *, top_k: int | None = None, cfg: Config | None = None) -> dict:
    """一条测试题 → 一条 ragas 样本(检索 + 最简生成 + 命中明细)。"""
    from .retrieve import search  # 包内检索链路的唯一入口

    hits = search(qa["question"], top_k=top_k)
    # 父块口径(06 篇 §2.4): 与最终喂给生成器的载荷一致; **保持 RRF 名次顺序**, 不排序不去重
    contexts = [chunk.parent_text for chunk in hits]
    return {
        "question_id": qa["question_id"],
        "user_input": qa["question"],
        "retrieved_contexts": contexts,
        "response": simple_answer(qa["question"], contexts, cfg),
        "reference": qa["ground_truth"],
        "debug": debug_of(hits),
    }


# ---------------------------------------------------------------- ragas 面


def _ragas_api() -> SimpleNamespace:
    """一次性导入评估库的公开面, 缺依赖时给可操作的指引。

    集中在一处是有意的: 升级 ragas 时改这里 + `METRIC_ALIASES` 两处就够(06 篇坑位 1 的对策)。
    """
    try:
        from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
        from ragas.embeddings import BaseRagasEmbedding
        from ragas.llms import llm_factory
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError as exc:
        raise ImportError(_RAGAS_HINT) from exc
    return SimpleNamespace(
        EvaluationDataset=EvaluationDataset,
        RunConfig=RunConfig,
        SingleTurnSample=SingleTurnSample,
        evaluate=evaluate,
        BaseRagasEmbedding=BaseRagasEmbedding,
        llm_factory=llm_factory,
        Faithfulness=Faithfulness,
        LLMContextPrecisionWithReference=LLMContextPrecisionWithReference,
        LLMContextRecall=LLMContextRecall,
        ResponseRelevancy=ResponseRelevancy,
    )


def build_judge(cfg: Config | None = None):
    """judge: 走 ragas 现代接口 `llm_factory` + OpenAI 兼容客户端(06 篇 §4.1 的升级方案)。

    为什么不用已弃用的 `LangchainLLMWrapper`: 它把 langchain 拉进评估链路, 而 ragas 0.4.3 与
    langchain-community 0.4 本就不兼容(要钉 <0.4); 少一层依赖少一个雷。`temperature=0` 保证可复现。
    """
    cfg = cfg or _cfg()
    import openai  # 懒 import: 见模块 docstring

    ragas = _ragas_api()
    client = openai.OpenAI(
        base_url=cfg.judge_api_base, api_key=cfg.judge_api_key, timeout=120, max_retries=3
    )
    # max_tokens 给足: ResponseRelevancy 一次要生成 3 个问题, 实测默认预算会被截断
    # (IncompleteOutputException: The output is incomplete due to a max_tokens length limit)
    return ragas.llm_factory(cfg.judge_model, client=client, temperature=0, max_tokens=2048)


def build_metrics() -> list:
    """四项指标, 顺序与 06 篇 §2.2 的表格一致; 升级 ragas 只改这个函数。"""
    ragas = _ragas_api()
    return [
        ragas.LLMContextPrecisionWithReference(),  # -> context_precision
        ragas.LLMContextRecall(),  # -> context_recall
        ragas.Faithfulness(),  # -> faithfulness
        ragas.ResponseRelevancy(),  # -> answer_relevancy
    ]


def build_embeddings():
    """把本包的 embed 双模式包成 ragas 的嵌入接口: 评估 embedding == 入库 embedding。

    为什么不用 ragas 自带的 provider: 会引入第二份模型配置(百炼的 OpenAI 兼容端点实测**不支持**
    `qwen3-vl-embedding`, 返回 404 model_not_supported), 而 answer_relevancy 直接吃这个向量的
    余弦相似度 —— 换模型就换读数, 必须与入库同一模型同一模式(06 篇 §2.6)。

    **关于那对 async 方法**: ragas 的嵌入基类把同步与异步都声明为抽象方法(库的硬要求), 我们照它
    实现一对薄壳 —— 只把同步的 HTTP 调用丢进线程, 不建事件循环、不 await 业务代码。这是全包唯一的
    async 出现处, 也是"能少写一层 langchain 包装"的代价。
    """
    ragas = _ragas_api()

    class _Rag2Embeddings(ragas.BaseRagasEmbedding):
        """适配器: 文本路复用 `embed.py`, 与入库同模型同维度。

        四组方法都要有: ragas 0.4.3 里**新旧两套接口并存** —— 基类要 `embed_text(s)`, 而
        `ResponseRelevancy` 实测调的是旧名 `embed_query`(漏了它整道题会 AttributeError);
        异步那四个是基类的抽象要求(见 §4.2 与 06 篇坑位 9), 只做同步转线程。
        """

        def embed_text(self, text: str, **kwargs) -> list[float]:
            from .embed import embed_query

            return embed_query(text)

        def embed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
            from .embed import embed_documents

            return embed_documents(texts)

        # —— 旧接口名(ragas 内部仍在用) ——
        def embed_query(self, text: str, **kwargs) -> list[float]:
            return self.embed_text(text, **kwargs)

        def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
            return self.embed_texts(texts, **kwargs)

        async def aembed_text(self, text: str, **kwargs) -> list[float]:
            return await asyncio.to_thread(self.embed_text, text, **kwargs)

        async def aembed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_texts, texts, **kwargs)

        async def aembed_query(self, text: str, **kwargs) -> list[float]:
            return await asyncio.to_thread(self.embed_query, text, **kwargs)

        async def aembed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_documents, texts, **kwargs)

    return _Rag2Embeddings()


def ragas_evaluate(samples: list[dict], cfg: Config | None = None):
    """喂 ragas 并把结果化成 DataFrame(逐题分数 + 指标列)。失败样本留 NaN。"""
    cfg = cfg or _cfg()
    ragas = _ragas_api()
    dataset = ragas.EvaluationDataset(
        samples=[
            ragas.SingleTurnSample(**{field: sample[field] for field in RAGAS_FIELDS})
            for sample in samples
        ]
    )
    result = ragas.evaluate(
        dataset=dataset,
        metrics=build_metrics(),
        llm=build_judge(cfg),
        embeddings=build_embeddings(),
        run_config=ragas.RunConfig(max_workers=4, timeout=120, max_retries=2, seed=42),
        raise_exceptions=False,  # 单题失败留 NaN, 不炸整批(坑位 6)
        show_progress=False,
    )
    import ragas as ragas_pkg

    df = result.to_pandas()
    df.attrs["ragas_version"] = getattr(ragas_pkg, "__version__", "unknown")
    # 这几个在 ragas 里是**方法**(实测)而不是属性: 写成 bound method 会让报告 json 崩, 所以统一取值
    df.attrs["total_tokens"] = _value_of(result, "total_tokens")
    df.attrs["total_cost"] = _value_of(result, "total_cost")
    df.attrs["run_id"] = str(_value_of(result, "run_id") or "")
    return df


def _value_of(obj, name: str):
    """取统计量: 是方法就调用它(ragas 的几个是方法)。

    调用失败**绝不能**把整次评估带崩: `total_cost()` 在没配 `token_usage_parser` 时会抛错, 而那时
    四十题的 judge 调用早已花完 —— 报告就差这一步落盘。取不到就写一个可读的占位串。
    """
    value = getattr(obj, name, None)
    if not callable(value):
        return value
    try:
        return value()
    except Exception as exc:  # noqa: BLE001  统计量拿不到不该影响结果
        return f"<不可用: {type(exc).__name__}>"


# ---------------------------------------------------------------- 报告


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _mean(series) -> float | None:
    """均值: 全 NaN 时返回 None(不是 0 —— 0 会被当成"这项很差")。"""
    value = series.dropna()
    return round(float(value.mean()), 4) if len(value) else None


def summarize(df, samples: list[dict]) -> tuple[dict[str, float | None], list[dict], list[str]]:
    """DataFrame → (四指标均值, 逐题明细, NaN 的 question_id 列表)。"""
    metrics: dict[str, float | None] = {}
    per_question: list[dict] = []
    nan_ids: list[str] = []
    for position, sample in enumerate(samples):
        row = df.iloc[position]
        item = {
            "question_id": sample["question_id"],
            "user_input": sample["user_input"],
            "reference": sample["reference"],
            "response": sample["response"],
            "contexts_count": len(sample["retrieved_contexts"]),
            "debug": sample.get("debug", []),
        }
        missing = False
        for column, name in METRIC_ALIASES.items():
            value = row.get(column) if column in df.columns else None
            item[name] = None if value is None or value != value else round(float(value), 4)
            if item[name] is None:
                missing = True
        if missing:
            nan_ids.append(sample["question_id"])
        per_question.append(item)
    for column, name in METRIC_ALIASES.items():
        metrics[name] = _mean(df[column]) if column in df.columns else None
    return metrics, per_question, nan_ids


def _corpus_snapshot() -> dict | None:
    """库里现在有哪些文档(基线可复现的关键: 同一份测试集对着不同语料跑, 结果不可比)。"""
    try:
        from .store import list_docs

        docs = list_docs()
        return {"doc_count": len(docs), "children": sum(d["children"] for d in docs)}
    except Exception as exc:  # noqa: BLE001  评估不因取不到语料快照而失败, 但要留痕
        return {"error": f"{type(exc).__name__}: {exc}"}


def build_report(
    *,
    metrics: dict,
    per_question: list[dict],
    nan_ids: list[str],
    report_dir: str | Path,
    system_label: str,
    testset_path: str | Path,
    question_count: int,
    cfg: Config,
    meta_extra: dict | None = None,
) -> EvalReport:
    """落盘 json + markdown 报告, 返回契约对象(06 篇 §4.5)。"""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).astimezone().strftime("%Y%m%d-%H%M%S")
    meta = {
        "system_label": system_label,
        "testset": str(testset_path),
        "testset_sha256_16": _sha256(testset_path),
        "question_count": question_count,
        "ragas_version": (meta_extra or {}).pop("ragas_version", "unknown"),
        "judge_model": cfg.judge_model,
        "judge_api_base": cfg.judge_api_base,
        "embedding_model": cfg.embed_model,
        "embedding_mode": cfg.embed_mode,
        "retrieve": {
            "top_k": cfg.retrieve_top_k,
            "candidates": cfg.candidate_top_n,
            "rrf_k": cfg.rrf_k,
            "hnsw_ef": cfg.hnsw_ef,
        },
        "contexts_source": "parent_text",
        "answer_prompt_version": ANSWER_PROMPT_VERSION,
        "corpus": _corpus_snapshot(),
        "nan_question_ids": nan_ids,
        **(meta_extra or {}),
    }
    json_path = report_dir / f"eval_{system_label}_{stamp}.json"
    json_path.write_text(
        # default=str 兜底: 报告是跑了几分钟 judge 之后才写的, 不该因为某个字段不是原生类型整批白跑
        json.dumps({"meta": meta, "metrics": metrics, "per_question": per_question},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    md_path = report_dir / f"eval_{system_label}_{stamp}.md"
    md_path.write_text(
        _render_markdown(
            meta, metrics, per_question, report_dir=report_dir, system_label=system_label
        ),
        encoding="utf-8",
    )
    return EvalReport(metrics={k: v for k, v in metrics.items() if v is not None},
                      question_count=question_count, report_dir=str(report_dir))


def _fmt(value: float | None) -> str:
    return "NaN" if value is None else f"{value:.4f}"


def _cell(text: str, limit: int = 24) -> str:
    """表格单元: 竖线转义 + 截断(报告给人看, 一行太长就没法读)。"""
    flat = " ".join(str(text).split()).replace("|", r"\|")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _render_markdown(meta: dict, metrics: dict, per_question: list[dict], *,
                     report_dir: Path, system_label: str) -> str:
    """markdown 报告: 指标表 + 逐题表 + (有别的系统报告时)并排对比表。"""
    lines = [
        f"# 评估报告 · {system_label}",
        "",
        f"- 测试集: `{meta['testset']}` (sha256[:16] = `{meta['testset_sha256_16']}`, "
        f"{meta['question_count']} 条)",
        f"- ragas: `{meta['ragas_version']}` | judge: `{meta['judge_model']}` | "
        f"embedding: `{meta['embedding_model']}` ({meta['embedding_mode']})",
        f"- 检索参数: {meta['retrieve']} | contexts 口径: `{meta['contexts_source']}` | "
        f"生成 prompt: {meta['answer_prompt_version']}",
        f"- 语料: {meta['corpus']}",
        "",
        "## 四指标",
        "",
        "| 指标 | 均值 |",
        "|---|---|",
    ]
    for name in METRIC_ALIASES.values():
        lines.append(f"| {name} | {_fmt(metrics.get(name))} |")
    nan_count = len(meta["nan_question_ids"])
    lines += [
        "",
        f"NaN 题数: {nan_count}/{meta['question_count']}"
        + (f" → {', '.join(meta['nan_question_ids'])}" if nan_count else ""),
        "",
        "## 逐题明细",
        "",
        "| 题号 | 问题 | 参考答案 | 生成回答 | 上下文数 | context_precision |"
        " context_recall | faithfulness | answer_relevancy |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in per_question:
        question = item["user_input"].replace("|", "\\|")
        lines.append(
            f"| {item['question_id']} | {question} | {_cell(item['reference'])} | "
            f"{_cell(item['response'])} | {item['contexts_count']} | "
            f"{_fmt(item['context_precision'])} | {_fmt(item['context_recall'])} | "
            f"{_fmt(item['faithfulness'])} | {_fmt(item['answer_relevancy'])} |"
        )
    compare = _compare_table(report_dir, system_label, metrics)
    if compare:
        lines += ["", "## 与其它系统的同集对比", ""] + compare
    lines.append("")
    return "\n".join(lines)


def _compare_table(report_dir: Path, system_label: str, metrics: dict) -> list[str]:
    """若目录下已有别的系统(label 不同)的报告, 附一张并排对比表(06 篇 §4.5)。"""
    others: list[tuple[str, dict]] = []
    for path in sorted(report_dir.glob("eval_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        label = payload.get("meta", {}).get("system_label")
        if label and label != system_label and payload.get("metrics"):
            others.append((label, payload["metrics"]))
    if not others:
        return []
    labels = [label for label, _ in others]
    header = [system_label, *labels]
    lines = ["| 指标 | " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    for name in METRIC_ALIASES.values():
        cells = [_fmt(metrics.get(name))] + [_fmt(m.get(name)) for _l, m in others]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


# ---------------------------------------------------------------- 入口


def run_eval(
    testset_path: str | Path | None = None,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    *,
    samples_path: str | Path | None = None,
    system_label: str = DEFAULT_SYSTEM_LABEL,
    limit: int | None = None,
    top_k: int | None = None,
    cfg: Config | None = None,
) -> EvalReport:
    """跑一次评估。两种用法(06 篇 §4.4):

    - **完整跑**: 给 `testset_path`(默认 `data/eval_qa_v1.jsonl`), 逐题"检索 + 最简生成", 再评;
    - **仅评估**: 给 `samples_path`(既有中间产物), 只评不检索 —— 旧系统对比走这条, 两系统共用同一套
      四指标与 judge。

    `limit` 是试跑用的(06 篇 §4.6: 先 limit=5 看账单再放全量); `top_k` 覆盖 `RAG2_RETRIEVE_TOP_K`。
    """
    cfg = cfg or _cfg()
    if samples_path:
        samples = load_jsonl(samples_path)
        testset_ref = samples_path
    else:
        testset_ref = Path(testset_path or DEFAULT_TESTSET)
        qa_rows = load_testset(testset_ref)
        if limit is not None:
            qa_rows = qa_rows[:limit]
        samples = [make_sample(qa, top_k=top_k, cfg=cfg) for qa in qa_rows]
        # 中间产物落进报告目录: 路径不受当前工作目录影响(评估环境是 `uv run --directory eval` 跑的,
        # cwd 在 eval/ 下), 也方便整目录归档/交给"仅评估"入口复评
        dump_jsonl(Path(report_dir) / f"ragas_samples_{system_label}.jsonl", samples)
    df = ragas_evaluate(samples, cfg)
    metrics, per_question, nan_ids = summarize(df, samples)
    return build_report(
        metrics=metrics,
        per_question=per_question,
        nan_ids=nan_ids,
        report_dir=report_dir,
        system_label=system_label,
        testset_path=testset_ref,
        question_count=len(samples),
        cfg=cfg,
        meta_extra={"ragas_version": df.attrs.get("ragas_version", "unknown"),
                    "total_tokens": df.attrs.get("total_tokens")},
    )
