# این ماژول ایندکس FAISS و فراداده قطعات را مدیریت می‌کند.

import json
from pathlib import Path

import faiss
import numpy as np

from src.rag.chunking import Chunk


# مدیریت ایندکس FAISS و فراداده متناظر با هر بردار.
class FAISSVectorStore:
    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: list[dict] = []

# افزودن بردارها و اطلاعات قطعات به ایندکس.
    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")

        if len(embeddings) == 0:
            return

        vectors = np.asarray(embeddings, dtype="float32")

        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("Embedding dimension does not match vector store")

        self.index.add(vectors)

        self.metadata.extend(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source": chunk.source,
                "title": chunk.title,
                "text": chunk.text,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }
            for chunk in chunks
        )

# جست‌وجو بر اساس شباهت ضرب داخلی و مرتب‌سازی نتایج.
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if self.index.ntotal == 0:
            return []

        vector = np.asarray(query_embedding, dtype="float32")

        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        if vector.shape != (1, self.dimension):
            raise ValueError("Query embedding dimension does not match vector store")

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vector, k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            result = dict(self.metadata[int(index)])
            result["score"] = float(score)
            results.append(result)

        return results

# ذخیره ایندکس، فراداده و پیکربندی روی دیسک.
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "index.faiss"))

        with (directory / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(self.metadata, file, ensure_ascii=False, indent=2)

        config = {"dimension": self.dimension}

        with (directory / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

# بارگذاری دوباره ایندکس و بررسی سازگاری فراداده‌ها.
    @classmethod
    def load(cls, directory: str | Path) -> "FAISSVectorStore":
        directory = Path(directory)

        index_path = directory / "index.faiss"
        metadata_path = directory / "metadata.json"
        config_path = directory / "config.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        store = cls(dimension=int(config["dimension"]))
        store.index = faiss.read_index(str(index_path))

        with metadata_path.open("r", encoding="utf-8") as file:
            store.metadata = json.load(file)

        if store.index.ntotal != len(store.metadata):
            raise ValueError("FAISS index and metadata size do not match")

        return store