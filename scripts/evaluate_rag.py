import json
import sys
from pathlib import Path

# ریشه پروژه به مسیر جست‌وجوی پایتون اضافه می‌شود تا اجرای مستقیم فایل ممکن باشد.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.embeddings import get_embedding_model
from src.rag.vector_store import FAISSVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERY_FILE = PROJECT_ROOT / "data" / "rag_evaluation" / "queries.json"


def evaluate_index(index_dir: Path, queries: list[dict], top_k: int = 5) -> dict:
    store = FAISSVectorStore.load(index_dir)
    embedding_model = get_embedding_model()

    recall_hits = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []
    rows = []

    for item in queries:
        query = item["query"]
        expected_source = item["expected_source"]
        query_vector = embedding_model.encode([query])[0]
        results = store.search(query_vector, top_k=top_k)
        retrieved_sources = [result["source"] for result in results]

        for k in recall_hits:
            if expected_source in retrieved_sources[:k]:
                recall_hits[k] += 1

        rank = next(
            (i for i, source in enumerate(retrieved_sources, start=1)
             if source == expected_source),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        rows.append(
            {
                "query": query,
                "expected_source": expected_source,
                "retrieved_sources": retrieved_sources,
                "scores": [round(result["score"], 4) for result in results],
                "rank": rank,
            }
        )

    total = len(queries)
    return {
        "index_dir": str(index_dir),
        "query_count": total,
        "recall_at_1": recall_hits[1] / total if total else 0.0,
        "recall_at_3": recall_hits[3] / total if total else 0.0,
        "recall_at_5": recall_hits[5] / total if total else 0.0,
        "mrr": sum(reciprocal_ranks) / total if total else 0.0,
        "rows": rows,
    }


def main() -> None:
    if not QUERY_FILE.exists():
        raise FileNotFoundError(f"Evaluation query file not found: {QUERY_FILE}")

    with QUERY_FILE.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    strategies = [
        PROJECT_ROOT / "data" / "rag_index" / "strategy_500_100",
        PROJECT_ROOT / "data" / "rag_index" / "strategy_800_150",
    ]

    for index_dir in strategies:
        result = evaluate_index(index_dir, queries)
        print(f"\nStrategy: {result['index_dir']}")
        print(f"Queries: {result['query_count']}")
        print(f"Recall@1: {result['recall_at_1']:.4f}")
        print(f"Recall@3: {result['recall_at_3']:.4f}")
        print(f"Recall@5: {result['recall_at_5']:.4f}")
        print(f"MRR: {result['mrr']:.4f}")

        failures = [
            row for row in result["rows"]
            if row["rank"] is None or row["rank"] > 3
        ]
        print(f"Queries not found in top 3: {len(failures)}")
        for row in failures:
            print(f"- Query: {row['query']}")
            print(f"  Expected: {row['expected_source']}")
            print(f"  Retrieved: {row['retrieved_sources']}")
            print(f"  Scores: {row['scores']}")


if __name__ == "__main__":
    main()
