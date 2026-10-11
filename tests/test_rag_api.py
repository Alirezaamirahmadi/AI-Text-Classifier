"""Test request validation and error handling for the RAG API endpoints."""

# این آزمون‌ها اعتبارسنجی و پاسخ‌های نقاط پایانی RAG را بررسی می‌کنند.

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.rag import api as rag_api


app = FastAPI()
app.include_router(rag_api.router)
client = TestClient(app)


# بررسی ردشدن هم‌پوشانی نامعتبر هنگام ورود اسناد.
def test_ingest_rejects_overlap_equal_to_chunk_size():
    response = client.post(
        "/ingest",
        json={
            "chunk_size": 100,
            "chunk_overlap": 100,
            "input_dir": "data/rag_documents",
            "output_dir": "data/rag_index/test_invalid",
        },
    )

    assert response.status_code == 422
    assert "chunk_overlap" in response.json()["detail"]


def test_ingest_rejects_nonpositive_chunk_size():
    response = client.post(
        "/ingest",
        json={"chunk_size": 0, "chunk_overlap": 0},
    )

    assert response.status_code == 422


# بررسی اعتبارسنجی ورودی‌های نامعتبر برای بازیابی.
def test_retrieve_rejects_blank_query():
    response = client.post("/retrieve", json={"query": "   "})

    assert response.status_code == 422


def test_retrieve_rejects_empty_query():
    response = client.post("/retrieve", json={"query": ""})

    assert response.status_code == 422


def test_retrieve_rejects_zero_top_k():
    response = client.post(
        "/retrieve",
        json={"query": "path parameter", "top_k": 0},
    )

    assert response.status_code == 422


def test_retrieve_rejects_top_k_over_limit():
    response = client.post(
        "/retrieve",
        json={"query": "path parameter", "top_k": 21},
    )

    assert response.status_code == 422


# بررسی پاسخ مناسب وقتی ایندکس هنوز ساخته نشده است.
def test_retrieve_returns_service_unavailable_for_missing_index(tmp_path):
    response = client.post(
        "/retrieve",
        json={
            "query": "path parameter",
            "index_dir": str(tmp_path / "missing_index"),
        },
    )

    assert response.status_code == 503
    assert "POST /ingest" in response.json()["detail"]


def test_ingest_returns_bad_request_for_missing_documents(tmp_path):
    response = client.post(
        "/ingest",
        json={
            "input_dir": str(tmp_path / "missing_documents"),
            "output_dir": str(tmp_path / "index"),
        },
    )

    assert response.status_code == 400
    assert "Document directory not found" in response.json()["detail"]


def test_ingest_returns_bad_request_for_overlap_greater_than_chunk_size():
    response = client.post(
        "/ingest",
        json={"chunk_size": 100, "chunk_overlap": 101},
    )

    assert response.status_code == 422


def test_retrieve_response_schema_rejects_invalid_top_k_type():
    response = client.post(
        "/retrieve",
        json={"query": "path parameter", "top_k": "many"},
    )

    assert response.status_code == 422


def test_ingest_uses_default_chunk_parameters():
    payload = rag_api.IngestRequest()
    assert payload.chunk_size == 800
    assert payload.chunk_overlap == 150


def test_retrieve_uses_default_top_k():
    payload = rag_api.RetrieveRequest(query="path parameter")
    assert payload.top_k == 5


def test_clear_retriever_cache_runs():
    rag_api.clear_retriever_cache()
    assert isinstance(rag_api._retrievers, dict)
    assert len(rag_api._retrievers) == 0

def test_answer_rejects_empty_question():
    response = client.post(
        "/answer",
        json={"question": "   "},
    )
    assert response.status_code == 422


def test_answer_rejects_top_k_out_of_range():
    for top_k in (0, 21):
        response = client.post(
            "/answer",
            json={"question": "What is RAG?", "top_k": top_k},
        )
        assert response.status_code == 422


def test_answer_rejects_non_positive_context_limit():
    response = client.post(
        "/answer",
        json={
            "question": "What is RAG?",
            "max_context_chars": 0,
        },
    )
    assert response.status_code == 422


def test_answer_returns_503_when_index_is_missing(monkeypatch, tmp_path):
    index_dir = tmp_path / "missing_index"

    monkeypatch.setattr(
        rag_api,
        "_normalise_index_dir",
        lambda _: str(index_dir),
    )

    response = client.post(
        "/answer",
        json={
            "question": "What is RAG?",
            "index_dir": str(index_dir),
        },
    )

    assert response.status_code == 503
    assert "call POST /ingest first" in response.json()["detail"]


def test_answer_uses_cached_retriever(monkeypatch, tmp_path):
    index_dir = str(tmp_path / "index")
    fake_retriever = object()
    expected = {
        "answer": "RAG uses retrieved context. [Source 1]",
        "status": "answered",
        "citations": [{"source_number": 1}],
        "retrieved_chunks": [],
    }

    monkeypatch.setattr(rag_api, "_normalise_index_dir", lambda _: index_dir)
    rag_api.clear_retriever_cache()
    rag_api._retrievers[index_dir] = fake_retriever

    def fake_answer_question(**kwargs):
        assert kwargs["question"] == "What is RAG?"
        assert kwargs["retriever"] is fake_retriever
        assert kwargs["top_k"] == 3
        assert kwargs["max_context_chars"] == 500
        return expected

    monkeypatch.setattr(rag_api, "answer_question", fake_answer_question)

    try:
        response = client.post(
            "/answer",
            json={
                "question": "  What is RAG?  ",
                "top_k": 3,
                "max_context_chars": 500,
                "index_dir": index_dir,
            },
        )
        assert response.status_code == 200
        assert response.json() == expected
    finally:
        rag_api.clear_retriever_cache()

# خطاهای قابل انتظار در منطق پاسخ‌گویی باید به پاسخ HTTP 400 تبدیل شوند.
def test_answer_returns_400_when_answer_question_raises_value_error(
    monkeypatch, tmp_path
):
    index_dir = str(tmp_path / "index")
    fake_retriever = object()

    monkeypatch.setattr(rag_api, "_normalise_index_dir", lambda _: index_dir)
    rag_api.clear_retriever_cache()
    rag_api._retrievers[index_dir] = fake_retriever
    monkeypatch.setattr(
        rag_api,
        "answer_question",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("invalid context")),
    )

    try:
        response = client.post(
            "/answer",
            json={"question": "What is RAG?", "index_dir": index_dir},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "invalid context"
    finally:
        rag_api.clear_retriever_cache()


# خطاهای پیش‌بینی‌نشده باید پاسخ عمومی بدهند و جزئیات داخلی را افشا نکنند.
def test_answer_returns_500_for_unexpected_error(monkeypatch, tmp_path):
    index_dir = str(tmp_path / "index")
    fake_retriever = object()

    monkeypatch.setattr(rag_api, "_normalise_index_dir", lambda _: index_dir)
    rag_api.clear_retriever_cache()
    rag_api._retrievers[index_dir] = fake_retriever
    monkeypatch.setattr(
        rag_api,
        "answer_question",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("internal detail")),
    )

    try:
        response = client.post(
            "/answer",
            json={"question": "What is RAG?", "index_dir": index_dir},
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "Answer generation failed"
        assert "internal detail" not in response.text
    finally:
        rag_api.clear_retriever_cache()