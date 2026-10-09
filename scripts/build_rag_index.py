# این اسکریپت ایندکس برداری RAG را با تنظیمات انتخاب‌شده می‌سازد.

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.chunking import chunk_documents
from src.rag.embeddings import get_embedding_model
from src.rag.ingestion import load_documents
from src.rag.vector_store import FAISSVectorStore


# ساخت و ذخیره‌سازی ایندکس برداری از اسناد ورودی.
def build_index(
    chunk_size: int,
    chunk_overlap: int,
    output_dir: str | Path,
) -> None:
    documents = load_documents()
    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    embedding_model = get_embedding_model()
    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_model.encode(texts)

    store = FAISSVectorStore(embedding_model.dimension)
    store.add(embeddings, chunks)
    store.save(output_dir)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding dimension: {embedding_model.dimension}")
    print(f"Index directory: {output_dir}")


# دریافت تنظیمات ساخت ایندکس از خط فرمان.
def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS RAG index.")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        default="data/rag_index/strategy_500_100",
    )
    args = parser.parse_args()

    build_index(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
