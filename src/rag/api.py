# این ماژول نقاط پایانی API مربوط به ورود اسناد و بازیابی را تعریف می‌کند.

from pathlib import Path
from threading import Lock

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.rag.chunking import chunk_documents
from src.rag.embeddings import get_embedding_model
from src.rag.ingestion import load_documents
from src.rag.retrieval import DEFAULT_INDEX_DIR, Retriever
from src.rag.vector_store import FAISSVectorStore

from src.rag.answering import answer_question
from src.rag.generation import GenerationError


# روتر مستقل برای نقاط پایانی ورود اسناد و بازیابی قطعات.
router = APIRouter()
_retrievers: dict[str, Retriever] = {}
_retrievers_lock = Lock()


# پارامترهای درخواست برای پردازش اسناد و ساخت ایندکس.
class IngestRequest(BaseModel):
    input_dir: str = "data/rag_documents"
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    output_dir: str = DEFAULT_INDEX_DIR


# اطلاعات خلاصه‌ای که پس از ساخت ایندکس برگردانده می‌شود.
class IngestResponse(BaseModel):
    document_count: int
    chunk_count: int
    embedding_dimension: int
    index_dir: str


# پارامترهای جست‌وجو در ایندکس ذخیره‌شده.
class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    index_dir: str = DEFAULT_INDEX_DIR

# مدل ورودی برای پاسخ‌گویی مبتنی بر اسناد بازیابی‌شده
class AnswerRequest(BaseModel):
    """"RAG accountability endpoint input validation."""

    # سؤال کاربر باید حداقل یک نویسه داشته باشد؛ خالی بودنِ فاصله‌ها نیز در مسیر بررسی می‌شود.
    question: str = Field(min_length=1)
    # تعداد قطعات بازیابی‌شده باید در بازه مجاز باقی بماند.
    top_k: int = Field(default=5, ge=1, le=20)
    # مسیر ایندکسی که برای بازیابی شواهد استفاده می‌شود.
    index_dir: str = DEFAULT_INDEX_DIR
    # سقف طول زمینه برای کنترل اندازه ورودی مدل زبانی.
    max_context_chars: int = Field(default=10000, gt=0)


# یکسان‌سازی مسیر ایندکس برای استفاده به‌عنوان کلید کش.
def _normalise_index_dir(index_dir: str) -> str:
    """Return an absolute, normalized path suitable for use as a cache key."""
    return str(Path(index_dir).resolve())


# دریافت اسناد، قطعه‌بندی، تولید بردارها و ذخیره ایندکس.
@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(request: IngestRequest) -> IngestResponse:
    """Load documents, split them into chunks, create embeddings, and save the index."""
    if request.chunk_overlap >= request.chunk_size:
        raise HTTPException(
            status_code=422,
            detail="chunk_overlap must be smaller than chunk_size",
        )

    try:
        documents = load_documents(request.input_dir)
        chunks = chunk_documents(
            documents,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        if not chunks:
            raise ValueError("No text chunks were produced")

        embedding_model = get_embedding_model()
        embeddings = embedding_model.encode([chunk.text for chunk in chunks])
        store = FAISSVectorStore(embedding_model.dimension)
        store.add(embeddings, chunks)
        store.save(request.output_dir)

        index_key = _normalise_index_dir(request.output_dir)
        retriever = Retriever(
            index_dir=index_key,
            embedding_model=embedding_model,
        )
        with _retrievers_lock:
            _retrievers[index_key] = retriever

        return IngestResponse(
            document_count=len(documents),
            chunk_count=len(chunks),
            embedding_dimension=embedding_model.dimension,
            index_dir=index_key,
        )
    except HTTPException:
        raise
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed",
        ) from exc


# بازیابی نزدیک‌ترین قطعات و برگرداندن امتیاز و فراداده.
@router.post("/retrieve")
def retrieve_documents(request: RetrieveRequest) -> dict:
    """Search the stored vector index and return matching chunks with metadata."""
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")

    index_key = _normalise_index_dir(request.index_dir)

    try:
        with _retrievers_lock:
            retriever = _retrievers.get(index_key)

        if retriever is None:
            if not (Path(index_key) / "index.faiss").exists():
                raise HTTPException(
                    status_code=503,
                    detail="RAG index is not available; call POST /ingest first",
                )
            retriever = Retriever(index_dir=index_key)
            with _retrievers_lock:
                _retrievers[index_key] = retriever

        results = retriever.retrieve(request.query, top_k=request.top_k)
        return {
            "query": request.query,
            "count": len(results),
            "results": [
                {
                    "chunk_id": result["chunk_id"],
                    "score": result["score"],
                    "text": result["text"],
                    "metadata": {
                        "document_id": result["document_id"],
                        "source": result["source"],
                        "title": result["title"],
                        "start_char": result["start_char"],
                        "end_char": result["end_char"],
                    },
                }
                for result in results
            ],
        }
    except HTTPException:
        raise
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Retrieval failed",
        ) from exc


# پاک‌کردن کش بازیاب برای ایزوله‌سازی تست‌ها.
def clear_retriever_cache() -> None:
    """Clear cached retrievers, primarily to isolate tests."""
    with _retrievers_lock:
        _retrievers.clear()


# پاسخ‌گویی با اتکا به اسناد بازیابی‌شده و ارجاع به منابع
@router.post("/answer")
def answer_documents(request: AnswerRequest) -> dict:
    """Retrieve evidence and generate an answer with validated source citations."""

    # جلوگیری از پذیرش سؤال خالی یا فقط شامل فاصله
    if not request.question.strip():
        raise HTTPException(
            status_code=422,
            detail="question must not be empty",
        )

    # استفاده از مسیر استاندارد ایندکس و کش Retriever موجود
    index_key = _normalise_index_dir(request.index_dir)

    try:
        with _retrievers_lock:
            retriever = _retrievers.get(index_key)

        # اگر Retriever در کش نیست، بررسی و بارگذاری ایندکس
        if retriever is None:
            if not (Path(index_key) / "index.faiss").exists():
                raise HTTPException(
                    status_code=503,
                    detail="RAG index is not available; call POST /ingest first",
                )

            retriever = Retriever(index_dir=index_key)

            with _retrievers_lock:
                _retrievers[index_key] = retriever

        # بازیابی قطعه‌ها، تولید پاسخ و بررسی ارجاع‌ها
        return answer_question(
            question=request.question.strip(),
            retriever=retriever,
            top_k=request.top_k,
            max_context_chars=request.max_context_chars,
        )

    except HTTPException:
        raise
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # جزئیات داخلی خطا به پاسخ عمومی API نشت نمی‌کند
        raise HTTPException(
            status_code=500,
            detail="Answer generation failed",
        ) from exc