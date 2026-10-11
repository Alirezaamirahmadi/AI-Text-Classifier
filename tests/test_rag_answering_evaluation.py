"""Test answer status handling, abstention, and citation validation."""

# این آزمون‌ها رفتار پاسخ‌گویی را در برابر شواهد معتبر، نامعتبر یا ناکافی می‌سنجند.
import pytest

import src.rag.answering as answering


# بازیاب ساختگی، نتایج کنترل‌شده‌ای برای آزمون منطق پاسخ‌گویی فراهم می‌کند.
class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, question, top_k=5):
        return self.results


# یک قطعه نمونه با فراداده لازم برای ساخت زمینه و ارجاع‌ها.
def sample_result():
    return {
        "chunk_id": "chunk_001",
        "document_id": "doc_001",
        "source": "fastapi.md",
        "title": "FastAPI",
        "start_char": 0,
        "end_char": 50,
        "text": "FastAPI uses Pydantic for data validation.",
    }


# ارجاع معتبر باید وضعیت پاسخ را به answered تغییر دهد.
def test_valid_citation_returns_answered(monkeypatch):
    monkeypatch.setattr(
        answering,
        "generate_answer",
        lambda **kwargs: "FastAPI uses Pydantic. [Source 1]",
    )
    result = answering.answer_question(
        "What does FastAPI use?",
        FakeRetriever([sample_result()]),
    )
    assert result["status"] == "answered"
    assert len(result["citations"]) == 1
    assert result["citations"][0]["source_number"] == 1


# ارجاع به منبعی خارج از نتایج بازیابی‌شده باید رد شود.
def test_invalid_citation_is_rejected(monkeypatch):
    monkeypatch.setattr(
        answering,
        "generate_answer",
        lambda **kwargs: "The answer is supported. [Source 99]",
    )
    result = answering.answer_question(
        "What does FastAPI use?",
        FakeRetriever([sample_result()]),
    )
    assert result["status"] == "invalid_citations"
    assert result["citations"] == []


# پاسخ بدون ارجاع نباید به‌عنوان پاسخ تأییدشده به کاربر تحویل داده شود.
def test_uncited_answer_is_withheld(monkeypatch):
    monkeypatch.setattr(
        answering,
        "generate_answer",
        lambda **kwargs: "FastAPI uses Pydantic.",
    )
    result = answering.answer_question(
        "What does FastAPI use?",
        FakeRetriever([sample_result()]),
    )
    assert result["status"] == "uncited"
    assert result["citations"] == []
    assert "withheld" in result["answer"]


# اگر مدل کمبود شواهد را اعلام کند، وضعیت باید همین موضوع را نشان دهد.
def test_abstention_returns_insufficient_context(monkeypatch):
    monkeypatch.setattr(
        answering,
        "generate_answer",
        lambda **kwargs: (
            "The provided documents do not contain enough information "
            "to answer this question."
        ),
    )
    result = answering.answer_question(
        "What is the capital of Japan?",
        FakeRetriever([sample_result()]),
    )
    assert result["status"] == "insufficient_context"
    assert result["citations"] == []


# متن شبیه تحلیل داخلی به‌عنوان پاسخ نهایی پذیرفته نمی‌شود.
def test_suspicious_generation_is_rejected(monkeypatch):
    monkeypatch.setattr(
        answering,
        "generate_answer",
        lambda **kwargs: (
            "Here's a thinking process: let me examine the excerpts."
        ),
    )
    result = answering.answer_question(
        "What does FastAPI use?",
        FakeRetriever([sample_result()]),
    )
    assert result["status"] == "invalid_generation"
    assert result["citations"] == []


# بدون نتیجه قابل استفاده، مدل نباید فراخوانی شود.
def test_empty_retrieval_returns_insufficient_context(monkeypatch):
    def unexpected_generation(**kwargs):
        pytest.fail("Generation must not run without usable results")

    monkeypatch.setattr(
        answering,
        "generate_answer",
        unexpected_generation,
    )
    result = answering.answer_question(
        "What is FastAPI?",
        FakeRetriever([]),
    )
    assert result["status"] == "insufficient_context"
    assert result["citations"] == []
    assert result["retrieved_chunks"] == []