"""Coordinate retrieval, grounded answer generation, and citation validation."""

import re
from typing import Any

from src.rag.context_builder import build_context
from src.rag.generation import GenerationError, generate_answer
from src.rag.retrieval import Retriever


# الگوی تشخیص ارجاع به منابع در پاسخ، مانند [Source 1]
CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


def answer_question(
    question: str,
    retriever: Retriever,
    top_k: int = 5,
    max_context_chars: int = 10000,
) -> dict[str, Any]:
    """Retrieve evidence, generate an answer, and validate its source citations."""

    # بررسی خالی نبودن سؤال
    if not question.strip():
        raise ValueError("question must not be empty")

    # بررسی معتبر بودن تعداد نتایج بازیابی
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")

    # بررسی مثبت بودن حداکثر طول متن زمینه
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be greater than zero")

    # بازیابی بخش‌های مرتبط با سؤال از اسناد
    results = retriever.retrieve(question, top_k=top_k)

    # حذف نتایجی که متن ندارند یا متن آن‌ها خالی است
    usable_results = [
        item for item in results
        if isinstance(item.get("text"), str) and item["text"].strip()
    ]

    # اگر هیچ نتیجه قابل استفاده‌ای پیدا نشد، پاسخ ناکافی بودن اطلاعات برگردانده می‌شود
    if not usable_results:
        return {
            "answer": "The available documents do not contain enough information to answer this question.",
            "status": "insufficient_context",
            "citations": [],
            "retrieved_chunks": [],
        }

    # ساخت متن زمینه از نتایج بازیابی‌شده با رعایت محدودیت طول
    context = build_context(usable_results, max_chars=max_context_chars)

    # اگر متن زمینه خالی باشد، اطلاعات کافی برای پاسخ وجود ندارد
    if not context.strip():
        return {
            "answer": "The available documents do not contain enough information to answer this question.",
            "status": "insufficient_context",
            "citations": [],
            "retrieved_chunks": [],
        }

    try:
        # تولید پاسخ با استفاده از سؤال و متن زمینه
        generated = generate_answer(question=question, context=context)

        # پاسخ خام برای اشکال‌زدایی ثبت می‌شود؛ این خروجی بخشی از پاسخ API نیست.
        print(f"RAW GENERATED ANSWER: {generated!r}", flush=True)

    except GenerationError as exc:
        # ثبت خطای تولید پاسخ
        print(f"Generation failed: {exc}")

        # برگرداندن پیام مناسب در صورت در دسترس نبودن سرویس تولید پاسخ
        return {
            "answer": "Answer generation is temporarily unavailable. Please try again later.",
            "status": "generation_unavailable",
            "citations": [],
            "retrieved_chunks": usable_results,
        }

    # این نشانه‌ها برای جلوگیری از بازگرداندن متن تحلیل داخلی به کاربر بررسی می‌شوند.
    suspicious_phrases = (
        "here's a thinking process",
        "here is a thinking process",
        "analyze user input",
        "let me examine the excerpts",
        "determine response",
    )

    # رد کردن پاسخ‌هایی که شامل عبارت‌های مشکوک هستند
    if any(phrase in generated.lower() for phrase in suspicious_phrases):
        return {
            "answer": (
                "The generated response could not be verified. "
                "Please try again."
            ),
            "status": "invalid_generation",
            "citations": [],
            "retrieved_chunks": usable_results,
        }

    # تعریف پاسخ استاندارد برای زمانی که اسناد اطلاعات کافی ندارند
    abstention = (
        "The provided documents do not contain enough information "
        "to answer this question."
    )

    # بررسی اینکه آیا مدل پاسخ استانداردِ ناکافی بودن اطلاعات را داده است
    if generated.strip().strip('"') == abstention:
        return {
            "answer": abstention,
            "status": "insufficient_context",
            "citations": [],
            "retrieved_chunks": usable_results,
        }

    # استخراج شماره منابع ذکرشده در پاسخ
    source_numbers = {
        int(number) for number in CITATION_PATTERN.findall(generated)
    }

    # فقط شماره‌هایی معتبرند که با موقعیت منابع موجود در زمینه تطبیق داشته باشند.
    valid_numbers = {
        number for number in source_numbers
        if 1 <= number <= len(usable_results)
    }

    # رد کردن پاسخ در صورت وجود ارجاع به منابع نامعتبر یا ناموجود
    if source_numbers != valid_numbers:
        return {
            "answer": "The generated answer contained invalid citations and was withheld.",
            "status": "invalid_citations",
            "citations": [],
            "retrieved_chunks": usable_results,
        }

    # آماده‌سازی فهرست ارجاع‌های معتبر
    citations = []

    # جمع‌آوری اطلاعات هر منبعی که در پاسخ به آن ارجاع شده است
    for number in sorted(valid_numbers):
        # تطبیق شماره منبع با نتیجه بازیابی‌شده
        result = usable_results[number - 1]

        citations.append({
            # شماره منبع در پاسخ
            "source_number": number,

            # شناسه بخش بازیابی‌شده
            "chunk_id": result.get("chunk_id"),

            # شناسه سند اصلی
            "document_id": result.get("document_id"),

            # نام یا مسیر فایل منبع
            "source": result.get("source"),

            # عنوان سند
            "title": result.get("title"),

            # موقعیت شروع متن در سند اصلی
            "start_char": result.get("start_char"),

            # موقعیت پایان متن در سند اصلی
            "end_char": result.get("end_char"),

            # متن شواهد مربوط به این منبع
            "text": result.get("text", ""),
        })

    # تعیین وضعیت پاسخ بر اساس وجود ارجاع معتبر
    status = "answered" if citations else "uncited"

    # اگر پاسخ هیچ ارجاع معتبری نداشته باشد، متن آن با پیام ایمن جایگزین می‌شود
    if not citations:
        generated = (
            "The answer could not be verified against a cited source, "
            "so it has been withheld."
        )

    # برگرداندن پاسخ، وضعیت، ارجاع‌ها و نتایج بازیابی‌شده
    return {
        "answer": generated,
        "status": status,
        "citations": citations,
        "retrieved_chunks": usable_results,
    }