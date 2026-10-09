# این آزمون‌ها رفتار ساخت زمینه و حفظ فراداده را بررسی می‌کنند.

import pytest

from src.rag.context_builder import build_context


# اطمینان از حفظ منبع، شناسه‌ها و امتیاز در متن زمینه.
def test_build_context_preserves_source_metadata():
    results = [
        {
            "chunk_id": "doc_001_chunk_0000",
            "document_id": "doc_001",
            "source": "tutorial_path_params.md",
            "title": "Path Parameters",
            "text": "FastAPI supports path parameters.",
            "score": 0.91,
        }
    ]

    context = build_context(results)

    assert "tutorial_path_params.md" in context
    assert "Path Parameters" in context
    assert "doc_001" in context
    assert "doc_001_chunk_0000" in context
    assert "0.9100" in context
    assert "FastAPI supports path parameters." in context


# بررسی حفظ ترتیب نتایج بازیابی در زمینه نهایی.
def test_build_context_handles_multiple_results_in_order():
    results = [
        {"source": "first.md", "title": "First", "text": "First text", "score": 0.9},
        {"source": "second.md", "title": "Second", "text": "Second text", "score": 0.8},
    ]

    context = build_context(results)

    assert context.index("First text") < context.index("Second text")


def test_build_context_skips_empty_text():
    results = [
        {"source": "empty.md", "title": "Empty", "text": "   "},
        {"source": "valid.md", "title": "Valid", "text": "Useful content"},
    ]

    context = build_context(results)

    assert "empty.md" not in context
    assert "Useful content" in context


# بررسی رعایت محدودیت طول متن زمینه.
def test_build_context_respects_max_chars():
    results = [
        {"source": "doc.md", "title": "Document", "text": "A" * 500, "score": 0.8}
    ]

    context = build_context(results, max_chars=150)

    assert len(context) <= 150


def test_build_context_rejects_nonpositive_max_chars():
    with pytest.raises(ValueError, match="max_chars must be greater than zero"):
        build_context([], max_chars=0)
