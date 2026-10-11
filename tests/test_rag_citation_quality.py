"""Test whether generated citations refer to available sources."""

# این آزمون‌ها وجود ارجاع و معتبر بودن شماره منبع را بررسی می‌کنند.
from src.rag.answering import CITATION_PATTERN


# ارجاع به شماره‌ای که در فهرست منابع وجود دارد باید معتبر شناخته شود.
def test_answer_citation_points_to_available_source():
    answer = "FastAPI uses Pydantic. [Source 1]"
    available_source_numbers = {1, 2, 3}

    cited_numbers = {
        int(number) for number in CITATION_PATTERN.findall(answer)
    }

    assert cited_numbers
    assert cited_numbers.issubset(available_source_numbers)


# شماره منبع خارج از محدوده باید به‌عنوان ارجاع نامعتبر شناسایی شود.
def test_answer_with_nonexistent_source_is_detected():
    answer = "Tokyo is the capital of Japan. [Source 99]"
    available_source_numbers = {1, 2, 3}

    cited_numbers = {
        int(number) for number in CITATION_PATTERN.findall(answer)
    }

    assert not cited_numbers.issubset(available_source_numbers)


# پاسخ فاقد الگوی ارجاع باید بدون citation تشخیص داده شود.
def test_answer_without_citation_is_detected():
    answer = "FastAPI uses Pydantic."
    cited_numbers = CITATION_PATTERN.findall(answer)

    assert cited_numbers == []