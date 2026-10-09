# این ماژول جست‌وجوی برداری را روی ایندکس ذخیره‌شده انجام می‌دهد.

from pathlib import Path

from src.rag.embeddings import EmbeddingModel, get_embedding_model
from src.rag.vector_store import FAISSVectorStore


# مسیر پیش‌فرض ایندکس برداری ذخیره‌شده.
DEFAULT_INDEX_DIR = "data/rag_index/strategy_500_100"


# بارگذاری ایندکس و اجرای جست‌وجوی برداری برای پرسش‌ها.
class Retriever:
    def __init__(
        self,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        embedding_model: EmbeddingModel | None = None,
    ):
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model or get_embedding_model()
        self.vector_store = FAISSVectorStore.load(self.index_dir)

    # تابع کمکی برای بازیابی نتایج با تنظیمات پیش‌فرض.
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = self.embedding_model.encode([query.strip()])

        return self.vector_store.search(
            query_embedding[0],
            top_k=top_k,
        )


def retrieve(
    query: str,
    top_k: int = 5,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> list[dict]:
    retriever = Retriever(index_dir=index_dir)
    return retriever.retrieve(query=query, top_k=top_k)