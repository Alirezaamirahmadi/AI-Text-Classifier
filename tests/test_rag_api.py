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