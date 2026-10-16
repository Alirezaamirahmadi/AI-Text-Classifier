"""Run reproducible local RAG evaluation without calling an LLM provider.

Retrieval metrics are measured against the committed human-authored gold set.
Generation metrics intentionally remain null: this script does not generate model
answers or treat synthetic/mock answers as real evidence.
"""
from __future__ import annotations

import json
import platform
import time
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
INDEX = ROOT / "data/rag_index/strategy_500_100"


# معیارهای رتبه‌بندی منابع مرتبط را برای یک سؤال محاسبه می‌کند.
def retrieval_metrics(ranked_sources: list[str], expected_sources: list[str]) -> dict[str, float | None]:
    """Calculate source-level Recall@k and reciprocal rank for one query."""
    expected = set(expected_sources)
    if not expected:
        return {"recall_at_1": None, "recall_at_3": None, "recall_at_5": None, "mrr": None}
    out: dict[str, float | None] = {}
    for k in (1, 3, 5):
        out[f"recall_at_{k}"] = len(expected.intersection(ranked_sources[:k])) / len(expected)
    first_rank = next((i for i, source in enumerate(ranked_sources, 1) if source in expected), None)
    out["mrr"] = 1 / first_rank if first_rank else 0.0
    return out


# میانگین مقادیر اندازه‌گیری‌شده را حساب می‌کند؛ برای فهرست خالی مقدار نامشخص برمی‌گرداند.
def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


# اجرای محلی ارزیابی را انجام می‌دهد و نتیجه‌های اندازه‌گیری‌نشده را صریحاً مشخص می‌کند.
def main() -> int:
    OUT.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    from src.rag.evaluation_dataset import DATASET as dataset
    cases = dataset["cases"]
    started = time.perf_counter()
    results: dict[str, Any] = {
        "dataset_version": dataset["version"], "dataset_path": "src/rag/evaluation_dataset.py",
        "total_cases": len(cases), "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_mode": "local retrieval only; no OpenRouter/LLM calls; no mocks used for metrics",
        "retrieval": {"status": "not_run", "index": str(INDEX.relative_to(ROOT)), "metrics": {}, "by_category": {}},
        "generation": {"status": "not_measured", "answer_correctness": None, "faithfulness": None,
                       "citation_validity": None, "citation_support": None, "abstention_quality": None,
                       "reason": "Generation evaluation requires recorded real model outputs; this offline runner does not call a provider."},
        "runtime_seconds": None, "provider_cost_usd": 0.0,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    # نتایج هر سؤال جداگانه نگه داشته می‌شوند تا تحلیل دسته‌ای امکان‌پذیر باشد.
    retrieval_rows = []
    # خطاهای محیطی یا ایندکس به‌عنوان وضعیت مسدودشده ثبت می‌شوند، نه نتیجه موفق.
    try:
        from src.rag.retrieval import Retriever
        retriever = Retriever(index_dir=INDEX)
        for case in cases:
            retrieved = retriever.retrieve(case["question"], top_k=5)
            ranked = [Path(str(r.get("source", ""))).name for r in retrieved]
            expected = case.get("expected_sources", []) if case.get("answerable") else []
            metrics = retrieval_metrics(ranked, expected)
            retrieval_rows.append({"id": case["id"], "question": case["question"], "category": case["category"],
                                   "answerable": case["answerable"], "expected_sources": expected,
                                   "retrieved_sources": ranked,
                                   "retrieval_scores": [r.get("score") for r in retrieved], "metrics": metrics,
                                   "abstention_retrieval_diagnostic": "retrieved results alone do not establish answerability" if not case["answerable"] else None})
        results["retrieval"]["status"] = "measured"
        results["retrieval"]["metrics"] = {
            key: mean([row["metrics"][key] for row in retrieval_rows if row["metrics"][key] is not None])
            for key in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr")
        }
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in retrieval_rows:
            groups[row["category"]].append(row)
        results["retrieval"]["by_category"] = {
            category: {key: mean([r["metrics"][key] for r in rows if r["metrics"][key] is not None])
                       for key in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr")}
            for category, rows in sorted(groups.items())
        }
    except Exception as exc:  # preserve failure as an explicit execution result
        results["retrieval"]["status"] = "blocked"
        results["retrieval"]["error_type"] = type(exc).__name__
        results["retrieval"]["error"] = str(exc)
    results["runtime_seconds"] = round(time.perf_counter() - started, 4)
    results["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    results["retrieval"]["cases"] = retrieval_rows
    (OUT / "rag_evaluation_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "dataset_cases": len(cases), "answerable_cases": sum(c["answerable"] for c in cases),
        "unanswerable_cases": sum(not c["answerable"] for c in cases),
        "categories": {cat: sum(c["category"] == cat for c in cases) for cat in sorted({c["category"] for c in cases})},
        "retrieval_status": results["retrieval"]["status"], "retrieval_metrics": results["retrieval"]["metrics"],
        "retrieval_by_category": results["retrieval"]["by_category"],
        "generation_status": "not_measured", "generation_metrics": results["generation"],
        "runtime_seconds": results["runtime_seconds"], "provider_cost_usd": 0.0,
        "interpretation": "No generation quality claim is made. Retrieval metrics exclude unanswerable cases because no expected source exists; those cases are used for later abstention evaluation.",
    }
    (OUT / "rag_evaluation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    failure = {"status": "not_measured", "real_failures_analyzed": 0,
               "reason": "A failure requires an actual execution with retrieved results, final context, generated answer and citations. No generated model answers were collected in this offline run.",
               "cases": [], "template_fields": ["question", "reference_answer", "retrieval_results_and_scores", "final_context", "generated_answer", "citations", "root_cause", "pipeline_stage", "proposed_fix"]}
    (OUT / "rag_failure_analysis.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")
    report = f'''# RAG Evaluation & Reliability Report\n\n- Dataset: `src/rag/evaluation_dataset.py` ({len(cases)} human-authored cases)\n- Run: {results['started_at_utc']}\n- Python: {platform.python_version()}\n- Mode: local retrieval only; no OpenRouter requests and no mock answers used as results.\n- Runtime: {results['runtime_seconds']} seconds; provider cost: $0.00.\n\n## Dataset coverage\n\n| Category | Cases |\n|---|---:|\n'''
    report += ''.join(f"| {cat} | {count} |\n" for cat, count in summary['categories'].items())
    report += f"\nAnswerable: {summary['answerable_cases']}; unanswerable/abstention cases: {summary['unanswerable_cases']}.\n\n## Retrieval\n\nStatus: **{results['retrieval']['status']}**.\n\n"
    if results['retrieval']['status'] == 'measured':
        report += "Metrics are source-level recall and reciprocal rank over answerable cases.\n\n"
        report += ''.join(f"- {k}: {v:.4f}\n" if v is not None else f"- {k}: not available\n" for k,v in results['retrieval']['metrics'].items())
        report += "\nCategory breakdown:\n\n| Category | Recall@1 | Recall@3 | Recall@5 | MRR |\n|---|---:|---:|---:|---:|\n"
        for cat, vals in results['retrieval']['by_category'].items():
            report += f"| {cat} | " + " | ".join('N/A' if vals[k] is None else f"{vals[k]:.4f}" for k in ('recall_at_1','recall_at_3','recall_at_5','mrr')) + " |\n"
    else:
        report += f"Blocked: `{results['retrieval'].get('error_type')}: {results['retrieval'].get('error')}`. Metrics are not fabricated.\n"
    report += '''\n## Generation, citations, and abstention\n\nAnswer Correctness, Faithfulness, Citation Validity, Citation Support, and Abstention Quality are **not measured in this run**. No real model outputs were collected. See the results JSON for explicit `null` values and reason. The citation validator added in this stage separates reference validity from lexical overlap and never labels lexical overlap as semantic entailment.\n\n## Limitations and reproducibility\n\nRun `python scripts/run_rag_evaluation.py` from the repository root. The runner does not call OpenRouter. It loads the configured local FAISS index and embedding model; if dependencies or model files are unavailable, the blocked status and error are saved. The current before/after comparison is unavailable because no comparable pre-change result artifact was provided. The dataset is human-authored and independent of generated answers; expected source labels are gold annotations and should be reviewed when corpus documents change.\n\n## Failure analysis\n\nNo real generation failures are claimed in this offline run. Populate `outputs/rag_failure_analysis.json` only from actual runs containing the retrieved list, scores, final context, generated answer, and citations.\n'''
    (REPORTS / "RAG_EVALUATION_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"retrieval_status": results["retrieval"]["status"], "retrieval_metrics": results["retrieval"]["metrics"], "generation_status": "not_measured", "runtime_seconds": results["runtime_seconds"], "error": results["retrieval"].get("error")}, indent=2))
    return 0 if results["retrieval"]["status"] == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
