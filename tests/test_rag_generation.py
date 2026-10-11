"""Test answer generation and its integration with the RAG answering flow."""

# این آزمون‌ها فراخوانی تولید پاسخ و مدیریت خطاهای وابسته را بررسی می‌کنند.
from unittest.mock import Mock, patch

import pytest

from src.rag.generation import GenerationError, generate_answer
from src.rag.answering import answer_question


# نبود کلید API باید بدون ارسال درخواست شبکه مدیریت شود.
def test_generate_answer_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(GenerationError):
        generate_answer("What is AI?", "[Source 1] AI means artificial intelligence.")


# سؤال خالی ورودی معتبری برای تولید پاسخ نیست.
def test_generate_answer_rejects_empty_question(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    with pytest.raises(ValueError):
        generate_answer("", "Some context")


# تولید پاسخ بدون زمینه مستند مجاز نیست.
def test_generate_answer_rejects_empty_context(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    with pytest.raises(ValueError):
        generate_answer("What is AI?", "")


@patch("src.rag.generation.httpx.post")
# پاسخ معتبر شبیه‌سازی می‌شود تا آزمون به سرویس خارجی وابسته نباشد.
def test_generate_answer_success(mock_post, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "AI means artificial intelligence. [Source 1]"
                }
            }
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = generate_answer(
        "What is AI?",
        "[Source 1] AI means artificial intelligence.",
    )

    assert result == "AI means artificial intelligence. [Source 1]"
    mock_post.assert_called_once()


def test_answer_question_empty_question():
    retriever = Mock()

    with pytest.raises(ValueError):
        answer_question("", retriever)


def test_answer_question_invalid_top_k():
    retriever = Mock()

    with pytest.raises(ValueError):
        answer_question("What is AI?", retriever, top_k=0)


# نبود نتایج بازیابی باید به وضعیت insufficient_context منجر شود.
def test_answer_question_no_results():
    retriever = Mock()
    retriever.retrieve.return_value = []

    result = answer_question("What is AI?", retriever)

    assert result["status"] == "insufficient_context"
    assert result["citations"] == []

# پاسخ دارای ارجاع معتبر باید پذیرفته شود.
@patch("src.rag.answering.generate_answer")
def test_answer_question_valid_citation(mock_generate):
    retriever = Mock()
    retriever.retrieve.return_value = [
        {
            "text": "AI means artificial intelligence.",
            "source": "ai.txt",
            "title": "AI introduction",
            "document_id": "doc-1",
            "chunk_id": "chunk-1",
        }
    ]
    mock_generate.return_value = (
        "AI means artificial intelligence. [Source 1]"
    )

    result = answer_question("What is AI?", retriever)

    assert result["status"] == "answered"
    assert len(result["citations"]) == 1


# ارجاع به منبع ناموجود باید رد شود.
@patch("src.rag.answering.generate_answer")
def test_answer_question_invalid_citation(mock_generate):
    retriever = Mock()
    retriever.retrieve.return_value = [
        {
            "text": "AI means artificial intelligence.",
            "source": "ai.txt",
            "title": "AI introduction",
            "document_id": "doc-1",
            "chunk_id": "chunk-1",
        }
    ]
    mock_generate.return_value = "AI is useful. [Source 9]"

    result = answer_question("What is AI?", retriever)

    assert result["status"] == "invalid_citations"
    assert result["citations"] == []


# پاسخ بدون ارجاع معتبر باید با وضعیت uncited مشخص شود.
@patch("src.rag.answering.generate_answer")
def test_answer_question_uncited_answer(mock_generate):
    retriever = Mock()
    retriever.retrieve.return_value = [
        {
            "text": "AI means artificial intelligence.",
            "source": "ai.txt",
            "title": "AI introduction",
            "document_id": "doc-1",
            "chunk_id": "chunk-1",
        }
    ]
    mock_generate.return_value = "AI is useful."

    result = answer_question("What is AI?", retriever)

    assert result["status"] == "uncited"
    assert result["citations"] == []

    @patch("src.rag.answering.generate_answer")
    def test_answer_question_generation_error(mock_generate):
        retriever = Mock()
        retriever.retrieve.return_value = [
            {
                "text": "AI means artificial intelligence.",
                "source": "ai.txt",
                "title": "AI introduction",
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
            }
        ]
        mock_generate.side_effect = GenerationError("Provider unavailable")

        result = answer_question("What is AI?", retriever)

        assert result["status"] == "generation_unavailable"
        assert result["citations"] == []