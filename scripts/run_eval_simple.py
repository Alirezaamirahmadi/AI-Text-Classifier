"""ساده‌ترین نسخه‌ی ارزیابی: هر سؤال را اجرا می‌کند و همه‌ی مراحل را ذخیره می‌کند.

اجرا (از پوشه‌ی پروژه، با venv فعال):
    python scripts/run_eval_simple.py
"""
import json
import os
import subprocess
import time
from datetime import datetime, timezone

from src.rag.answering import answer_question
from src.rag.context_builder import build_context
from src.rag.evaluation_dataset import DATASET
from src.rag.retrieval import Retriever

INDEX_DIR = "data/rag_index/strategy_800_150"
OUTPUT_FILE = "outputs/rag_evaluation_results.json"
TOP_K = 5
MAX_CONTEXT_CHARS = 10000
SLEEP_SECONDS = 1  # برای مدل محلی کافی است؛ برای مدل ابری بزرگ‌تر کن

retriever = Retriever(index_dir=INDEX_DIR)
results = []

for case in DATASET["cases"]:
    question = case["question"]
    print("running", case["id"], flush=True)

    # ۱. بازیابی: سندها و امتیازها
    retrieved = retriever.retrieve(question, top_k=TOP_K)
    retrieved_info = [
        {"source": item.get("source"), "chunk_id": item.get("chunk_id"), "score": item.get("score")}
        for item in retrieved
    ]

    # ۲. پاسخ کامل (شامل Context و اعتبارسنجی)
    result = answer_question(question, retriever, top_k=TOP_K, max_context_chars=MAX_CONTEXT_CHARS)

    # ۳. Context نهایی را هم جدا می‌سازیم تا ببینیم چه چیزی به مدل رسید
    final_context = build_context(retrieved, max_chars=MAX_CONTEXT_CHARS)

    results.append({
        "id": case["id"],
        "category": case["category"],
        "answerable": case["answerable"],
        "question": question,
        "reference_answer": case["reference_answer"],
        "expected_sources": case["expected_sources"],
        "retrieved": retrieved_info,
        "final_context": final_context,
        "status": result["status"],
        "answer": result["answer"],
        "citations": [c.get("source") for c in result.get("citations", [])],
        "citation_quality": result.get("citation_quality"),
    })

    time.sleep(SLEEP_SECONDS)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# متادیتای اجرا: با این فایل مشخص می‌شود نتیجه از کدام کامیت و تنظیمات آمده است
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
metadata = {
    "run_time_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": commit,
    "index_dir": INDEX_DIR,
    "top_k": TOP_K,
    "max_context_chars": MAX_CONTEXT_CHARS,
    "max_tokens": 2048,
    "llm_base_url": os.getenv("LLM_BASE_URL", "openrouter (default)"),
    "llm_model": os.getenv("LLM_MODEL", "default"),
    "dataset_cases": len(DATASET["cases"]),
    "dataset_version": DATASET.get("version"),
}
with open("outputs/rag_run_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

# خلاصه و فهرست شکست‌ها (از همین نتایج، بدون اجرای مدل)
counts = {}
for item in results:
    key = item["status"]
    counts[key] = counts.get(key, 0) + 1

failures = []
for item in results:
    failed = item["status"] == "generation_unavailable"
    if item["answerable"] and item["status"] != "answered":
        failed = True
    if not item["answerable"] and item["status"] == "answered":
        failed = True
    if failed:
        failures.append({
            "id": item["id"],
            "answerable": item["answerable"],
            "status": item["status"],
            "expected_sources": item["expected_sources"],
            "retrieved_sources": [r["source"] for r in item["retrieved"]],
            "answer": item["answer"],
        })

with open("outputs/rag_evaluation_summary.json", "w", encoding="utf-8") as f:
    json.dump({"metadata": metadata, "total": len(results), "status_counts": counts,
               "failures_count": len(failures)}, f, ensure_ascii=False, indent=2)
with open("outputs/rag_failure_analysis.json", "w", encoding="utf-8") as f:
    json.dump({"failures": failures}, f, ensure_ascii=False, indent=2)

print("done", len(results), "saved to", OUTPUT_FILE)
