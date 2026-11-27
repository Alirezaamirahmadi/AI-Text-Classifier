"""Run Stage 12 checks and write outputs/rag_stage12_*.json from REAL results only.

    python scripts/build_rag_stage12_reports.py

1. Runs the pytest suite (real run) and records it.
2. With --reuse, the saved E2E file is re-read instead of re-running the questions.
   Otherwise, if a model is configured (LLM_BASE_URL or OPENROUTER_API_KEY), runs all dataset
   questions; otherwise the E2E part is written as BLOCKED.
3. Semantic metrics stay "not_measured" (they need human or independent-judge labels).
Only rag_stage12_* files are written; Stage 9/11 files are never touched or merged.
"""
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag.evaluation_dataset import DATASET  # noqa: E402

OUT = ROOT / "outputs"
INDEX_DIR = "data/rag_index/strategy_800_150"
TOP_K, MAX_CONTEXT_CHARS = 5, 10000
NOW = datetime.now(timezone.utc).isoformat()
CASES = DATASET["cases"]


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def clean(message):
    """Error text for the report: no API key, short."""
    text = str(message)
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        text = text.replace(key, "***")
    return text[:300]


def run_pytest():
    xml = OUT / "_stage12_junit.xml"
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "--junitxml", str(xml)],
                          cwd=ROOT, capture_output=True, text=True)
    tests = []
    for tc in ET.parse(xml).getroot().iter("testcase"):
        bad = tc.find("failure") is not None or tc.find("error") is not None
        skipped = tc.find("skipped") is not None
        tests.append({"module": tc.get("classname"), "test": tc.get("name"),
                      "result": "failed" if bad else "skipped" if skipped else "passed"})
    xml.unlink()
    return proc.returncode, tests, proc.stdout.strip().splitlines()[-1:]


def run_e2e():
    from src.rag.answering import answer_question
    from src.rag.retrieval import Retriever

    retriever = Retriever(index_dir=INDEX_DIR)
    records = []
    for case in CASES:
        print("running", case["id"], flush=True)
        trace, rec = {}, {"question_id": case["id"], "category": case["category"],
                          "answerable": case["answerable"], "expected_sources": case["expected_sources"]}
        start = time.perf_counter()
        try:
            res = answer_question(case["question"], retriever, top_k=TOP_K,
                                  max_context_chars=MAX_CONTEXT_CHARS, trace=trace)
            rec.update(
                retrieved_sources=[Path(str(c.get("source", ""))).name for c in res["retrieved_chunks"]],
                final_context=trace.get("final_context"), raw_generation=trace.get("raw_generation"),
                final_answer=res["answer"], citation_validation=res.get("citation_quality"),
                final_status=res["status"], token_usage=trace.get("token_usage"), error_details=None)
        except Exception as exc:  # one broken question must not stop the run
            rec.update(final_status="runner_error", error_details=clean(exc))
        rec["execution_time_s"] = round(time.perf_counter() - start, 3)
        records.append(rec)
    return records


def abstention(records):
    """Label-based abstention quality (labels = 'answerable' flag in the dataset)."""
    no_answer = [r for r in records if not r["answerable"]]
    has_answer = [r for r in records if r["answerable"]]
    abstained = lambda rs: sum(r["final_status"] == "insufficient_context" for r in rs)
    return {
        "method": "correct_abstention_rate = unanswerable cases whose final_status is insufficient_context / all "
                  "unanswerable cases; wrong_abstention_rate = answerable cases with insufficient_context / all "
                  "answerable cases. Labels come from the dataset. Measures system behaviour, not wording quality.",
        "unanswerable_total": len(no_answer), "correct_abstentions": abstained(no_answer),
        "correct_abstention_rate": round(abstained(no_answer) / len(no_answer), 4) if no_answer else "not_measured",
        "unanswerable_not_abstained_statuses": dict(Counter(r["final_status"] for r in no_answer
                                                            if r["final_status"] != "insufficient_context")),
        "answerable_total": len(has_answer), "wrong_abstentions": abstained(has_answer),
        "wrong_abstention_rate": round(abstained(has_answer) / len(has_answer), 4) if has_answer else "not_measured"}


def write_semantic(records):
    path = OUT / "rag_stage12_semantic_evaluation.json"
    judged = {}  # judge results are added to this file separately and must survive a rebuild
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            judged = {"llm_judge": old["llm_judge"]} if "llm_judge" in old else {}
            judged_metrics = {m: v for m, v in old.get("metrics", {}).items() if isinstance(v, dict) and m != "abstention_quality"}
        except ValueError:
            judged_metrics = {}
    else:
        judged_metrics = {}
    metrics = {m: "not_measured" for m in ["answer_correctness", "faithfulness", "citation_support",
                                            "answer_relevancy"]}
    metrics.update(judged_metrics)
    metrics["abstention_quality"] = abstention(records) if records else "not_measured"
    dump("rag_stage12_semantic_evaluation.json", {
        "status": "partially_measured" if records else "not_measured",
        "reason": "abstention_quality is label-based. Other metrics are not_measured unless filled by the "
                  "separately recorded llm_judge block (judge model is never the answering model).",
        "metrics": metrics, **judged,
        "methodology_to_use": {
            "answer_correctness": "human label correct/partial/wrong vs reference_answer + required_facts",
            "faithfulness": "human: every claim entailed by its cited chunk (yes/no per claim)",
            "citation_support": "human: cited chunk really supports its claim (per claim)",
            "answer_relevancy": "human 0/1/2: answers the asked question",
            "judge_separation": "record judge model apart from the answering model; never judge with the answering model"}})


def blocked(reason):
    dump("rag_stage12_e2e_results.json", {"status": "BLOCKED", "reason": reason, "generated_at_utc": NOW, "results": []})
    dump("rag_stage12_performance.json", {"status": "BLOCKED", "reason": reason})
    write_semantic(None)
    print(reason)


def main():
    OUT.mkdir(exist_ok=True)
    code, tests, tail = run_pytest()
    passed = sum(t["result"] == "passed" for t in tests)
    stage12 = [t for t in tests if "test_rag_stage12" in (t["module"] or "")]
    dump("rag_stage12_regression.json", {
        "generated_at_utc": NOW, "git_head": git("rev-parse", "HEAD"), "python": sys.version.split()[0],
        "full_suite": {"total": len(tests), "passed": passed, "failed": len(tests) - passed, "pytest_exit_code": code, "last_line": tail},
        "stage12_tests": stage12, "ci": "not_run_locally (check GitHub Actions after push)"})
    dump("rag_stage12_citation_validation.json", {
        "generated_at_utc": NOW, "method": "Deterministic lexical checks; NOT semantic entailment.",
        "scenario_tests": stage12,
        "known_limitation": "A wrong claim that reuses words from the cited chunk still passes (test_known_limit_*)."})

    saved_path = OUT / "rag_stage12_e2e_results.json"
    model = {"LLM_BASE_URL": os.getenv("LLM_BASE_URL"), "LLM_MODEL": os.getenv("LLM_MODEL")}
    if "--reuse" in sys.argv:
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        records, model = saved.get("results"), saved.get("model")
        if not records or not model:
            # nothing is overwritten here, so a bad file cannot destroy more data
            return print("--reuse needs a completed E2E file (status COMPLETED). "
                         "Run once without --reuse, or restore your backup of rag_stage12_e2e_results.json.")
    elif not (os.getenv("LLM_BASE_URL") or os.getenv("OPENROUTER_API_KEY")):
        return blocked("BLOCKED: no model configured (set LLM_BASE_URL or OPENROUTER_API_KEY)")
    else:
        try:
            records = run_e2e()
        except Exception as exc:  # e.g. missing index or embedding model
            return blocked("BLOCKED: " + clean(exc))

    times = sorted(r["execution_time_s"] for r in records)
    usages = [r["token_usage"] for r in records if r.get("token_usage")]
    errors = sum(r["final_status"] in ("runner_error", "generation_unavailable", "retrieval_unavailable") for r in records)
    counts = dict(Counter(r["final_status"] for r in records))
    dump("rag_stage12_e2e_results.json", {
        "status": "COMPLETED" if not errors else "COMPLETED_WITH_ERRORS", "generated_at_utc": NOW, "model": model,
        "settings": {"index_dir": INDEX_DIR, "top_k": TOP_K, "max_context_chars": MAX_CONTEXT_CHARS},
        "total": len(records), "status_counts": counts,
        "note": "Separate run; not merged with Stage 9/11 history. retrieved_sources are after de-duplication.",
        "results": records})
    dump("rag_stage12_performance.json", {
        "status": "measured", "model": model, "total": len(records),
        "avg_latency_s": round(sum(times) / len(times), 3), "p95_latency_s": times[int(0.95 * (len(times) - 1))],
        "max_latency_s": times[-1], "questions_with_token_usage": len(usages),
        "total_tokens": sum(u.get("total_tokens", 0) for u in usages) if usages else "not_reported_by_server",
        "concurrency": "sequential run; concurrency covered by unit tests with mocks only"})
    write_semantic(records)
    print(json.dumps(counts, indent=1))
    print(json.dumps(abstention(records), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
