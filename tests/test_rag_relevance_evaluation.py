"""Test that context construction remains grounded in retrieved evidence."""

# این آزمون بررسی می‌کند که ساخت زمینه فقط به شواهد بازیابی‌شده متکی باشد.
from src.rag.context_builder import build_context


# پرسش نامرتبط نباید اطلاعات تازه‌ای وارد زمینه مبتنی بر شواهد کند.
def test_irrelevant_question_does_not_change_evidence():
    evidence = [
        {
            "chunk_id": "chunk_001",
            "document_id": "doc_001",
            "source": "fastapi.md",
            "title": "FastAPI",
            "text": "FastAPI uses Pydantic for data validation.",
        }
    ]

    context = build_context(evidence, max_chars=10000)

    # متن زمینه باید فقط از شواهد بازیابی‌شده ساخته شود.
    assert "FastAPI uses Pydantic" in context

    # اطلاعاتی درباره ژاپن در شواهد وجود ندارد.
    assert "Tokyo" not in context