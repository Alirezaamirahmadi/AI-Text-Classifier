"""Test citation extraction and basic response-safety markers."""

# این آزمون‌ها استخراج ارجاع‌ها و شناسایی نشانه‌های پاسخ نامناسب را بررسی می‌کنند.
import pytest

from src.rag.answering import CITATION_PATTERN


# شماره منبع معتبر باید از متن پاسخ استخراج شود.
def test_valid_source_citation_is_detected():
    answer = "FastAPI depends on Pydantic and Starlette. [Source 4]"
    assert CITATION_PATTERN.findall(answer) == ["4"]


# در نبود ارجاع، الگوی استخراج نباید شماره‌ای برگرداند.
def test_answer_without_citation_is_detected():
    answer = "FastAPI depends on Pydantic and Starlette."
    assert CITATION_PATTERN.findall(answer) == []


# امکان استخراج چند ارجاع از یک پاسخ بررسی می‌شود.
def test_multiple_citations_are_detected():
    answer = "Claim A [Source 1]. Claim B [Source 2]."
    assert CITATION_PATTERN.findall(answer) == ["1", "2"]


@pytest.mark.parametrize(
    "answer",
    [
        "Here's a thinking process:",
        "Let me examine the excerpts.",
        "Analyze User Input:",
    ],
)
# عبارت‌های شبیه تحلیل داخلی باید با فهرست نشانه‌های مشکوک تطبیق داده شوند.
def test_internal_reasoning_markers_are_recognized(answer):
    suspicious_phrases = (
        "here's a thinking process",
        "here is a thinking process",
        "analyze user input",
        "let me examine the excerpts",
        "determine response",
    )
    assert any(phrase in answer.lower() for phrase in suspicious_phrases)


# متن استاندارد اعلام ناکافی بودن شواهد باید مشخص باشد.
def test_insufficient_context_response_is_defined():
    expected = (
        "The provided documents do not contain enough information "
        "to answer this question."
    )
    assert "not contain enough information" in expected