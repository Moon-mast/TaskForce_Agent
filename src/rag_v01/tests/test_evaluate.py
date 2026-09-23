"""模块 06 评估单测: 数据面(读写/组样本/生成) + 报告面 + 接线, **不连真库、不连真模型**。

评估库 ragas 只装在独立评估环境(`eval/`)里, 所以本文件不 import 它 —— 需要评估库的路径这里一律
跳过或只断言"缺依赖时给出可读指引"; 真机跑评估是 `cli evaluate` 的活(验收记录见 06 篇 §五)。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_v01 import cli, evaluation
from rag_v01.config import Config
from rag_v01.contracts import EvalReport, RetrievedChunk

pd = pytest.importorskip("pandas")
RAGAS_INSTALLED = importlib.util.find_spec("ragas") is not None


def qa_row(question_id: str = "q001", **overrides) -> dict:
    row = {
        "question_id": question_id,
        "question": "石英表多少钱",
        "ground_truth": "300",
        "question_type": "simple",
        "source_doc": "p00035.md",
        "reviewed": True,
        "notes": "",
    }
    row.update(overrides)
    return row


def write_testset(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "ts.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


def chunk(parent_id: str, parent_text: str, *, chunk_id: str = "c1") -> RetrievedChunk:
    from rag_v01.contracts import ChildHit

    return RetrievedChunk(
        parent_id=parent_id,
        parent_text=parent_text,
        rrf_rank=1,
        hits=[
            ChildHit(
                chunk_id=f"{parent_id}:{chunk_id}",
                text="命中",
                chunk_type="text",
                dense_rank=1,
                sparse_rank=2,
                rrf_score=0.031,
            )
        ],
        doc_id="doc",
        source="p00035.md",
    )


# ---------------------------------------------------------------- 数据面: jsonl 读写


def test_jsonl_roundtrip_skips_blank_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")

    assert evaluation.load_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_load_jsonl_points_at_the_bad_line(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n不是 json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="第 2 行"):
        evaluation.load_jsonl(path)


def test_dump_jsonl_makes_parent_dirs(tmp_path):
    target = tmp_path / "sub" / "samples.jsonl"

    evaluation.dump_jsonl(target, [{"a": 1}, {"b": 2}])

    assert [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()] == [
        {"a": 1},
        {"b": 2},
    ]


def test_load_testset_validates_fields(tmp_path):
    bad = write_testset(tmp_path, [{"question_id": "q1", "question": "x"}])

    with pytest.raises(ValueError, match="缺字段"):
        evaluation.load_testset(bad)


def test_load_testset_rejects_blank_question_or_answer(tmp_path):
    for overrides in ({"question": "  "}, {"ground_truth": ""}):
        path = write_testset(tmp_path, [qa_row(**overrides)])
        with pytest.raises(ValueError, match="为空"):
            evaluation.load_testset(path)


def test_load_testset_rejects_empty_file(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="空的"):
        evaluation.load_testset(empty)


def test_real_testset_on_disk_is_valid():
    """仓库里那份基线测试集必须过校验(它是以后所有对比的参照物)。"""
    rows = evaluation.load_testset(evaluation.DEFAULT_TESTSET)

    assert len(rows) >= 30
    assert all(row["reviewed"] is True for row in rows)
    assert len({row["question_id"] for row in rows}) == len(rows)
    assert len({row["question"] for row in rows}) == len(rows)


# ---------------------------------------------------------------- 数据面: 样本组装


def test_debug_of_keeps_both_ranks():
    """入参是检索结果(父块条目): parent_id 挂在父块上, 子块明细里没有。"""
    debug = evaluation.debug_of([chunk("p1", "父块")])

    assert debug == [
        {
            "chunk_id": "p1:c1",
            "parent_id": "p1",
            "chunk_type": "text",
            "dense_rank": 1,
            "sparse_rank": 2,
            "rrf_score": 0.031,
        }
    ]


def test_make_sample_keeps_parent_order_and_flattens_debug(monkeypatch):
    """contexts 取父块文本且**保持检索顺序**(06 篇坑位 7: 排序会让 context_precision 失真)。"""
    seen: list[str] = []

    def fake_search(query, top_k=None):
        seen.append(query)
        return [chunk("p2", "第二段"), chunk("p1", "第一段")]

    monkeypatch.setattr("rag_v01.retrieve.search", fake_search)
    monkeypatch.setattr(evaluation, "simple_answer", lambda q, contexts, cfg=None: "答案")

    sample = evaluation.make_sample(qa_row(), top_k=3, cfg=Config())

    assert seen == ["石英表多少钱"]
    assert sample["retrieved_contexts"] == ["第二段", "第一段"]  # 顺序原样, 不排序不去重
    assert sample["user_input"] == "石英表多少钱"
    assert sample["reference"] == "300"
    assert sample["response"] == "答案"
    assert [item["parent_id"] for item in sample["debug"]] == ["p2", "p1"]


def test_simple_answer_uses_fixed_prompt_and_zero_temperature(monkeypatch):
    captured: list[dict] = []

    class FakeClient:
        def __init__(self, **kwargs):
            captured.append({"client": kwargs})
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            captured.append({"create": kwargs})
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="  根据资料无法回答。  "))]
            )

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    cfg = Config(judge_model="m", judge_api_base="http://judge", judge_api_key="k")

    answer = evaluation.simple_answer("首都在哪?", ["北京是首都"], cfg)

    assert answer == "根据资料无法回答。"  # 两端空白被 strip
    call = captured[1]["create"]
    assert call["model"] == "m"
    assert call["temperature"] == 0
    assert "[1] 北京是首都" in call["messages"][0]["content"]
    assert "首都在哪?" in call["messages"][0]["content"]
    assert captured[0]["client"]["base_url"] == "http://judge"


# ---------------------------------------------------------------- 报告面: 汇总


def frame(rows: list[dict]):
    """造一个形如 ragas 逐题结果的 DataFrame(列名就是 ragas 的指标 name)。"""
    return pd.DataFrame(rows)


def test_summarize_maps_names_and_skips_nan():
    df = frame(
        [
            {"llm_context_precision_with_reference": 1.0, "context_recall": 0.5,
             "faithfulness": 1.0, "answer_relevancy": 0.8},
            {"llm_context_precision_with_reference": 0.0, "context_recall": 0.0,
             "faithfulness": 1.0, "answer_relevancy": float("nan")},
            {"llm_context_precision_with_reference": float("nan"), "context_recall": float("nan"),
             "faithfulness": float("nan"), "answer_relevancy": float("nan")},
        ]
    )
    samples = [
        {"question_id": "q001", "user_input": "a", "reference": "x", "response": "r",
         "retrieved_contexts": ["c"], "debug": []},
        {"question_id": "q002", "user_input": "b", "reference": "y", "response": "r",
         "retrieved_contexts": [], "debug": []},
        {"question_id": "q003", "user_input": "c", "reference": "z", "response": "r",
         "retrieved_contexts": [], "debug": []},
    ]

    metrics, per_question, nan_ids = evaluation.summarize(df, samples)

    assert metrics == {
        "context_precision": 0.5,  # NaN 不计入均值(1.0 与 0.0 平均)
        "context_recall": 0.25,
        "faithfulness": 1.0,
        "answer_relevancy": 0.8,
    }
    assert nan_ids == ["q002", "q003"]  # q002 缺一项, q003 全缺
    assert per_question[1]["answer_relevancy"] is None
    assert per_question[0]["contexts_count"] == 1


def test_summarize_all_nan_gives_none_not_zero():
    df = frame([{"faithfulness": float("nan")}])
    samples = [{"question_id": "q1", "user_input": "a", "reference": "x", "response": "r",
                "retrieved_contexts": [], "debug": []}]

    metrics, _per_question, nan_ids = evaluation.summarize(df, samples)

    assert metrics["faithfulness"] is None  # 0.0 会被读成"这项很差", None 才是"没测出来"
    assert nan_ids == ["q1"]


def test_metric_aliases_use_the_four_documented_gauges():
    assert sorted(evaluation.METRIC_ALIASES.values()) == [
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "faithfulness",
    ]
    # ragas 侧的键名是版本相关的, 但改名的对策是"只改这一处映射"
    assert evaluation.METRIC_ALIASES["llm_context_precision_with_reference"] == "context_precision"


# ---------------------------------------------------------------- 报告面: 落盘


def build(tmp_path: Path, **overrides):
    testset = overrides.pop("testset", None) or write_testset(tmp_path, [qa_row()])
    metrics = overrides.pop("metrics", {"context_precision": 1.0, "context_recall": 0.5,
                                        "faithfulness": 0.75, "answer_relevancy": None})
    per_question = overrides.pop("per_question", [
        {"question_id": "q001", "user_input": "石英表多少钱", "reference": "300",
         "response": "300", "contexts_count": 2, "debug": [], "context_precision": 1.0,
         "context_recall": 0.5, "faithfulness": 0.75, "answer_relevancy": None},
    ])
    return evaluation.build_report(
        metrics=metrics,
        per_question=per_question,
        nan_ids=overrides.pop("nan_ids", ["q001"]),
        report_dir=overrides.pop("report_dir", tmp_path / "reports"),
        system_label=overrides.pop("system_label", "rag_v01"),
        testset_path=testset,
        question_count=overrides.pop("question_count", 1),
        cfg=Config(judge_model="deepseek-flash", embed_model="qwen3-vl-embedding"),
        meta_extra=overrides.pop("meta_extra", {"ragas_version": "0.4.3"}),
    )


def test_build_report_writes_json_and_markdown(tmp_path):
    report = build(tmp_path)

    assert isinstance(report, EvalReport)
    assert report.question_count == 1
    assert report.metrics["context_precision"] == 1.0
    assert "answer_relevancy" not in report.metrics  # NaN 不冒充 0.0

    report_dir = Path(report.report_dir)
    (json_path,) = report_dir.glob("eval_*_*.json")
    (md_path,) = report_dir.glob("eval_*_*.md")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["meta"]["ragas_version"] == "0.4.3"
    assert payload["meta"]["judge_model"] == "deepseek-flash"
    assert payload["meta"]["contexts_source"] == "parent_text"
    assert payload["meta"]["testset_sha256_16"]
    assert payload["meta"]["retrieve"]["top_k"] and payload["meta"]["retrieve"]["rrf_k"]
    assert payload["meta"]["nan_question_ids"] == ["q001"]
    assert "corpus" in payload["meta"]

    text = md_path.read_text(encoding="utf-8")
    assert "quartz" not in text.lower()
    assert "context_precision" in text and "answer_relevancy" in text
    assert "NaN 题数: 1/1" in text
    assert "石英表多少钱" in text and "300" in text


def test_markdown_compare_table_appears_when_another_system_report_exists(tmp_path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "eval_tools_rag_v1_20260101-000000.json").write_text(
        json.dumps({"meta": {"system_label": "tools_rag_v1"},
                    "metrics": {"context_precision": 0.25, "context_recall": 0.1}}),
        encoding="utf-8",
    )

    report = build(tmp_path, report_dir=report_dir)

    text = (Path(report.report_dir) / next(
        p.name for p in report_dir.glob("eval_rag_v01_*.md")
    )).read_text(encoding="utf-8")
    assert "与其它系统的同集对比" in text
    assert "tools_rag_v1" in text
    assert "0.2500" in text  # 对方系统的分数进了对比表


def test_build_report_tolerates_missing_milvus(tmp_path, monkeypatch):
    """取不到语料快照不该让整个评估失败, 但要在报告里留痕。"""
    def boom():
        raise RuntimeError("Milvus 连不上")

    monkeypatch.setattr("rag_v01.store.list_docs", boom)

    report = build(tmp_path)

    payload = json.loads(
        (Path(report.report_dir) / next(Path(report.report_dir).glob("eval_*.json")).name)
        .read_text(encoding="utf-8")
    )
    assert "error" in payload["meta"]["corpus"]


# ---------------------------------------------------------------- 入口编排


def fake_evaluate_frame(n: int):
    return frame([{
        "llm_context_precision_with_reference": 1.0,
        "context_recall": 1.0,
        "faithfulness": 1.0,
        "answer_relevancy": 1.0,
    }] * n)


def test_run_eval_full_path_makes_samples_and_dumps_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evaluation, "make_sample",
                        lambda qa, top_k=None, cfg=None: {
                            "question_id": qa["question_id"], "user_input": qa["question"],
                            "retrieved_contexts": ["c"], "response": "r",
                            "reference": qa["ground_truth"], "debug": []})
    monkeypatch.setattr(
        evaluation, "ragas_evaluate", lambda samples, cfg=None: fake_evaluate_frame(len(samples))
    )
    testset = write_testset(
        tmp_path, [qa_row("q001"), qa_row("q002", question="欧莱雅精华液哪个好")]
    )

    report = evaluation.run_eval(testset, tmp_path / "reports", limit=1, cfg=Config())

    assert report.question_count == 1  # limit 生效(试跑用)
    assert report.metrics["context_precision"] == 1.0
    samples_file = tmp_path / "reports" / "ragas_samples_rag_v01.jsonl"
    assert [json.loads(line)["question_id"] for line in
            samples_file.read_text(encoding="utf-8").splitlines()] == ["q001"]


def test_run_eval_samples_only_path_does_not_search(tmp_path, monkeypatch):
    """仅评估入口: 吃既有中间产物, 不检索、不落新的中间产物(旧系统对比走这条)。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evaluation, "make_sample",
                        lambda *a, **kw: pytest.fail("仅评估入口不该再检索"))
    monkeypatch.setattr(
        evaluation, "ragas_evaluate", lambda samples, cfg=None: fake_evaluate_frame(len(samples))
    )
    samples_path = tmp_path / "samples.jsonl"
    evaluation.dump_jsonl(samples_path, [{"question_id": "q001", "user_input": "a",
                                          "retrieved_contexts": ["c"], "response": "r",
                                          "reference": "x", "debug": []}])

    report = evaluation.run_eval(
        None, tmp_path / "reports", samples_path=samples_path, cfg=Config()
    )

    assert report.question_count == 1
    assert not (tmp_path / "reports" / "ragas_samples_rag_v01.jsonl").exists()


def test_value_of_survives_a_raising_stat():
    """回归用例: ragas 的 `total_cost()` 在没配 token_usage_parser 时会抛错(真踩过)。

    那次是四十题 judge 全跑完、报告就差落盘时整批崩掉 —— 统计量取不到只能留痕, 不能中断评估。
    """

    class FakeResult:
        tokens = 123

        def total_tokens(self):
            return self.tokens

        def total_cost(self):
            raise ValueError("The evaluate() run was not configured for computing cost.")

    marker = evaluation._value_of(FakeResult(), "total_cost")

    assert marker == "<不可用: ValueError>"
    assert evaluation._value_of(FakeResult(), "total_tokens") == 123
    assert evaluation._value_of(FakeResult(), "run_id") is None


def test_missing_ragas_gives_actionable_hint():
    """主环境没装评估库: 报错必须给出跑法, 而不是一句 ModuleNotFoundError。"""
    if RAGAS_INSTALLED:
        pytest.skip("当前环境装了 ragas(评估环境), 这条在主环境才有意义")

    with pytest.raises(ImportError, match="评估环境"):
        evaluation._ragas_api()
    with pytest.raises(ImportError, match="评估环境"):
        evaluation.build_metrics()


# ---------------------------------------------------------------- facade 与 cli


def test_facade_evaluate_is_a_function_even_after_importing_the_module():
    """回归用例(这个坑真踩过两次): 子模块若与 facade 函数同名, import 会把函数换成 module。"""
    import rag_v01
    import rag_v01.evaluation  # noqa: F401  (import 子模块正是会触发覆盖的动作)

    assert callable(rag_v01.evaluate)
    assert "evaluate" in rag_v01.__all__


def test_cli_evaluate_prints_metrics_and_passes_options(monkeypatch, capsys):
    seen: list[dict] = []

    def fake_run_eval(testset=None, report_dir="eval_reports", **kwargs):
        seen.append({"testset": testset, "report_dir": report_dir, **kwargs})
        return EvalReport(metrics={"context_precision": 0.5, "context_recall": 0.25},
                          question_count=5, report_dir=str(report_dir))

    monkeypatch.setattr("rag_v01.evaluation.run_eval", fake_run_eval)
    monkeypatch.setattr(sys, "argv", ["rag_v01.cli", "evaluate", "--limit", "5",
                                      "--testset", "data/ts.jsonl", "--report-dir", "out"])

    cli.main()

    out = capsys.readouterr().out
    assert seen == [
        {
            "testset": "data/ts.jsonl",
            "report_dir": "out",
            "samples_path": None,
            "limit": 5,
            "system_label": "rag_v01",
        }
    ]
    assert "评估完成: 5 条题" in out
    assert "context_precision: 0.5000" in out


def test_cli_evaluate_accepts_a_system_label(monkeypatch):
    """新旧对比靠它: 旧系统走「仅评估」入口时标成另一个名字, 报告里才会出现并排对比表。"""
    seen: list[dict] = []

    def fake_run_eval(*args, **kwargs):
        seen.append(kwargs)
        return EvalReport(metrics={}, question_count=0, report_dir="x")

    monkeypatch.setattr("rag_v01.evaluation.run_eval", fake_run_eval)
    monkeypatch.setattr(
        sys,
        "argv",
        ["rag_v01.cli", "evaluate", "--samples", "old.jsonl", "--label", "tools_rag_v1"],
    )

    cli.main()

    assert seen[0]["system_label"] == "tools_rag_v1"


def test_cli_evaluate_rejects_bad_arguments(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rag_v01.cli", "evaluate", "--nope", "1"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert "不认识参数" in capsys.readouterr().out
