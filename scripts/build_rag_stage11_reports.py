#!/usr/bin/env python3
"""Reconcile Stage 11 RAG reports from archived, provenance-bearing runs.

This script deliberately does not call an LLM or claim a fresh end-to-end run.
It computes retrieval metrics from the saved Stage 8 retrieval run and
summarizes generation/performance from their separately saved runs.
"""
from __future__ import annotations
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

def load(name):
    with (OUT / name).open(encoding="utf-8") as f:
        return json.load(f)

def dump(name, value):
    with (OUT / name).open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")

def main():
    retrieval = load("rag_evaluation_results.json")
    generation = load("rag_run_qwen3b.json")
    performance = load("rag_performance_final.json")
    meta = load("rag_run_metadata.json")
    answerable = [r for r in retrieval if r.get("answerable")]
    metrics = {}
    for k in (1, 3, 5):
        hits, precision = [], []
        for row in answerable:
            expected = set(row.get("expected_sources") or [])
            ranked = [x.get("source") for x in (row.get("retrieved") or [])[:k]]
            relevant = sum(1 for source in ranked if source in expected)
            hits.append(bool(expected.intersection(ranked)))
            precision.append(relevant / k)
        metrics[f"recall_at_{k}"] = round(sum(hits) / len(hits), 4) if hits else None
        metrics[f"precision_at_{k}"] = round(sum(precision) / len(precision), 4) if precision else None
    reciprocal_ranks = []
    for row in answerable:
        expected = set(row.get("expected_sources") or [])
        rank = next((i for i, item in enumerate(row.get("retrieved") or [], 1)
                     if item.get("source") in expected), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    metrics["mrr"] = round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else None

    gen_counts = dict(Counter(row.get("status", "missing_status") for row in generation))
    gen_answerable = [row for row in generation if row.get("answerable") is True]
    gen_unanswerable = [row for row in generation if row.get("answerable") is False]
    gen_answerable_counts = dict(Counter(row.get("status", "missing_status") for row in gen_answerable))
    gen_unanswerable_counts = dict(Counter(row.get("status", "missing_status") for row in gen_unanswerable))
    perf_counts = performance.get("status_counts", {})
    evidence_path = OUT / "rag_external_validation.json"
    external_validation = load("rag_external_validation.json") if evidence_path.exists() else None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    benchmark = {
        "status": "historical_runs_reconciled_not_fresh_end_to_end_run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head_at_report_generation": commit,
        "dataset_cases": len(retrieval),
        "retrieval": {
            "status": "measured_from_saved_execution",
            "source_file": "outputs/rag_evaluation_results.json",
            "source_run_metadata": meta,
            "answerable_cases": len(answerable),
            "metrics": metrics,
            "metric_definition": "Recall@K is query hit-rate: at least one expected source appears in top K. Precision@K is relevant retrieved sources divided by K, averaged over answerable questions. MRR uses the first expected source rank.",
        },
        "generation": {
            "status": "measured_from_saved_execution",
            "source_file": "outputs/rag_run_qwen3b.json",
            "cases": len(generation),
            "answerable_cases": len(gen_answerable),
            "unanswerable_cases": len(gen_unanswerable),
            "model": "Qwen2.5-3B-Instruct (saved run; see source metadata)",
            "status_counts": gen_counts,
            "answerable_status_counts": gen_answerable_counts,
            "unanswerable_status_counts": gen_unanswerable_counts,
            "semantic_metrics": {key: "not_measured" for key in
                ("answer_correctness", "faithfulness", "citation_support", "answer_relevancy", "abstention_quality")},
        },
        "external_validation": external_validation or "not_provided",
        "performance": {
            "status": "measured_from_separate_saved_run",
            "source_file": "outputs/rag_performance_final.json",
            "cases": performance.get("cases"),
            "average_latency_s": performance.get("average_latency_s"),
            "p95_latency_s": performance.get("p95_latency_s"),
            "prompt_tokens_total": performance.get("prompt_tokens_total", "not_measured"),
            "completion_tokens_total": performance.get("completion_tokens_total", "not_measured"),
            "provider_errors": performance.get("provider_errors", "not_recorded_in_source"),
            "status_counts": perf_counts,
            "warning": "Performance is a separate historical run; do not combine its status counts with the generation run.",
        },
        "limitations": [
            "No fresh end-to-end execution was possible in this environment: faiss and sentence-transformers are unavailable and package installation is blocked by network/DNS.",
            "Semantic generation metrics remain not_measured; lexical citation overlap is not semantic entailment.",
            "The archived retrieval, generation, and performance results come from different runs and are reported separately.",
            "Provider live-failure and concurrency tests require the configured local LLM/embedding services and a fully provisioned environment.",
        ],
    }
    dump("rag_final_benchmark.json", benchmark)
    dump("rag_final_evaluation.json", {
        "retrieval": {"status": "measured_from_saved_execution", "source": "outputs/rag_evaluation_results.json", "answerable_cases": len(answerable), **metrics},
        "generation": {"status": "measured_from_saved_execution", "source": "outputs/rag_run_qwen3b.json", "cases": len(generation),
            "answerable_cases": len(gen_answerable), "unanswerable_cases": len(gen_unanswerable),
            "status_counts": gen_counts, "answerable_status_counts": gen_answerable_counts,
            "unanswerable_status_counts": gen_unanswerable_counts,
            "answer_correctness": "not_measured", "faithfulness": "not_measured", "citation_support": "not_measured", "answer_relevancy": "not_measured", "abstention_quality": "not_measured"},
    })
    dump("rag_final_performance.json", {
        "status": "measured_from_separate_saved_run", "source": "outputs/rag_performance_final.json",
        "model": performance.get("model"), "base_url": performance.get("base_url"), "cases": performance.get("cases"),
        "average_latency_s": performance.get("average_latency_s"), "p95_latency_s": performance.get("p95_latency_s"),
        "prompt_tokens_total": performance.get("prompt_tokens_total", "not_measured"),
        "completion_tokens_total": performance.get("completion_tokens_total", "not_measured"),
        "provider_errors": performance.get("provider_errors", "not_recorded_in_source"),
        "status_counts": perf_counts,
        "note": "Historical performance execution; distinct from the generation result file. No new provider call was made.",
    })
    if external_validation and external_validation.get("pytest", {}).get("result") == "passed":
        pytest_evidence = external_validation["pytest"]
        smoke_evidence = external_validation.get("citation_smoke_test", {})
        regression = {
            "status": "passed_in_user_environment",
            "evidence_source": external_validation.get("evidence_source"),
            "verification_note": external_validation.get("verification_note"),
            "environment": external_validation.get("environment"),
            "pytest": pytest_evidence,
            "citation_smoke_test": smoke_evidence,
            "limitations": [
                "A passing pytest suite does not establish a fresh end-to-end RAG benchmark.",
                "Live provider, semantic answer-quality, and concurrency acceptance remain separate checks unless covered by explicit tests and evidence."
            ],
        }
    else:
        regression = {
            "status": "blocked_in_report_generation_environment",
            "command_attempted": "PYTHONPATH=. pytest -q",
            "result": "collection_error",
            "error": "ModuleNotFoundError: No module named 'sentence_transformers' while importing tests/conftest.py",
            "dependency_install_attempt": "failed: package index unreachable (DNS/network unavailable)",
            "pytest_version": "9.0.2",
            "passed": "not_measured", "failed": "not_measured", "not_run": "entire suite; collection stopped",
            "required_to_unblock": ["install pinned requirements in a clean environment", "run full pytest suite", "run provider-backed and concurrency tests with actual services"],
        }
    dump("rag_final_regression.json", regression)
    print(json.dumps({"retrieval": metrics, "generation_counts": gen_counts, "performance": {k: performance.get(k) for k in ("average_latency_s", "p95_latency_s", "prompt_tokens_total", "completion_tokens_total")}}, indent=2))

if __name__ == "__main__":
    main()
