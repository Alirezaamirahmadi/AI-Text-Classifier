"""Coordinate retrieval, grounded answer generation, and citation validation."""

import re
from typing import Any

from src.rag.context_builder import build_context_with_sources
from src.rag.citation_validation import CITATION_PATTERN, validate_citations  # noqa: F401 (CITATION_PATTERN is imported by older tests)
from src.rag.generation import GenerationError, generate_answer, get_last_usage
from src.rag.retrieval import Retriever


ABSTENTION = (
    "The provided documents do not contain enough information "
    "to answer this question."
)
NO_CONTEXT = (
    "The available documents do not contain enough information to answer this question."
)
UNVERIFIED_FALLBACK = (
    "The cited sources do not appear to support the answer, "
    "so it has been withheld."
)
UNCITED_FALLBACK = (
    "The answer could not be verified against a cited source, "
    "so it has been withheld."
)


def _failure(answer: str, status: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "answer": answer,
        "status": status,
        "citations": [],
        "retrieved_chunks": retrieved_chunks,
    }





def looks_like_context_dump(text):
    """True if the answer is copied context or only citation markers."""
    lowered = text.lower()
    if "chunk_id=" in lowered or "source=" in lowered or "title=" in lowered:
        return True
    words_only = re.sub(r"\[Source\s+\d+\]", "", text, flags=re.IGNORECASE)
    return len(words_only.strip()) < 5


def answer_question(
    question: str,
    retriever: Retriever,
    top_k: int = 5,
    max_context_chars: int = 10000,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrieve evidence, generate an answer, and validate its citations.

    If ``trace`` (a dict) is given, it is filled with ``final_context``,
    ``raw_generation`` and ``token_usage`` for evaluation. The returned
    result is not changed by it.
    """

    if not question.strip():
        raise ValueError("question must not be empty")
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be greater than zero")

    # اگر بازیابی (مثلاً Embedding) خطا داد، برنامه کرش نمی‌کند.
    try:
        results = retriever.retrieve(question, top_k=top_k)
    except Exception:
        return _failure(
            "Retrieval is temporarily unavailable. Please try again later.",
            "retrieval_unavailable",
            [],
        )

    # حذف قطعه‌های تکراری (متن یکسان) تا یک سند دوبار وارد Context نشود.
    usable_results = []
    seen_texts = set()
    for item in results:
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        if text in seen_texts:
            continue
        seen_texts.add(text)
        usable_results.append(item)
    if not usable_results:
        return _failure(NO_CONTEXT, "insufficient_context", [])

    # فهرست منابعی که واقعاً وارد Context شده‌اند مستقیماً از سازنده Context می‌آید.
    # شماره [Source N] یعنی N-امین عضو همین فهرست.
    context, context_results = build_context_with_sources(
        usable_results, max_chars=max_context_chars
    )
    if not context.strip() or not context_results:
        return _failure(NO_CONTEXT, "insufficient_context", [])

    if trace is not None:
        trace["final_context"] = context
    try:
        generated = generate_answer(question=question, context=context)
        if trace is not None:
            trace["raw_generation"] = generated
            trace["token_usage"] = get_last_usage()
    except GenerationError:
        return _failure(
            "Answer generation is temporarily unavailable. Please try again later.",
            "generation_unavailable",
            usable_results,
        )

    if not isinstance(generated, str) or not generated.strip():
        return _failure(
            "The model returned an empty answer. Please try again.",
            "invalid_generation",
            usable_results,
        )

    suspicious_phrases = (
        "here's a thinking process",
        "here is a thinking process",
        "analyze user input",
        "let me examine the excerpts",
        "determine response",
    )
    if any(phrase in generated.lower() for phrase in suspicious_phrases):
        return _failure(
            "The generated response could not be verified. Please try again.",
            "invalid_generation",
            usable_results,
        )

    # جواب باید متن واقعی باشد؛ کپی Context یا فقط ارجاع خالی قبول نیست.
    if looks_like_context_dump(generated):
        return _failure(
            "The generated response could not be verified. Please try again.",
            "invalid_generation",
            usable_results,
        )

    # اگر مدل جمله‌ی استاندارد را نوشته، حتی با متن اضافه، یعنی اطلاعات کافی نیست.
    if "do not contain enough information" in generated.lower():
        return _failure(ABSTENTION, "insufficient_context", usable_results)

    # تصمیم نهایی بر اساس خروجی validate_citations و به‌صورت جداگانه برای هر Claim است.
    quality = validate_citations(generated, context_results)

    def _withhold(message: str, status: str) -> dict[str, Any]:
        result = _failure(message, status, usable_results)
        result["citation_quality"] = quality
        return result

    if not quality["cited_source_numbers"]:
        return _withhold(UNCITED_FALLBACK, "uncited")
    if quality["invalid_references"]:
        return _withhold(
            "The generated answer contained invalid citations and was withheld.",
            "invalid_citations",
        )
    if quality["uncited_claims"]:
        return _withhold(UNCITED_FALLBACK, "uncited")
    if quality["claims_with_unverified_support"]:
        return _withhold(UNVERIFIED_FALLBACK, "unverified_support")

    citations = []
    for number in quality["cited_source_numbers"]:
        item = context_results[number - 1]
        citations.append({
            "source_number": number,
            "chunk_id": item.get("chunk_id"),
            "document_id": item.get("document_id"),
            "source": item.get("source"),
            "title": item.get("title"),
            "start_char": item.get("start_char"),
            "end_char": item.get("end_char"),
            "text": item.get("text", ""),
        })

    return {
        "answer": generated.strip(),
        "status": "answered",
        "citations": citations,
        "citation_quality": quality,
        "retrieved_chunks": usable_results,
    }
