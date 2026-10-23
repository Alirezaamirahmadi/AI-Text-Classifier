"""Coordinate retrieval, grounded answer generation, and citation validation."""

import re
from typing import Any

from src.rag.context_builder import build_context
from src.rag.citation_validation import CITATION_PATTERN, extract_source_numbers, validate_citations
from src.rag.generation import GenerationError, generate_answer
from src.rag.retrieval import Retriever


ABSTENTION = (
    "The provided documents do not contain enough information "
    "to answer this question."
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
) -> dict[str, Any]:
    """Retrieve evidence, generate an answer, and validate its citations."""

    if not question.strip():
        raise ValueError("question must not be empty")
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be greater than zero")

    results = retriever.retrieve(question, top_k=top_k)
    usable_results = [
        item for item in results
        if isinstance(item.get("text"), str) and item["text"].strip()
    ]
    if not usable_results:
        return _failure(
            "The available documents do not contain enough information to answer this question.",
            "insufficient_context",
            [],
        )

    context = build_context(usable_results, max_chars=max_context_chars)
    context_source_count = len(
        re.findall(r"^\[Source\s+\d+\]", context, re.IGNORECASE | re.MULTILINE)
    )
    context_results = usable_results[:context_source_count]
    if not context.strip() or not context_results:
        return _failure(
            "The available documents do not contain enough information to answer this question.",
            "insufficient_context",
            [],
        )

    try:
        generated = generate_answer(question=question, context=context)
    except GenerationError as exc:
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

    if generated.strip().strip('"') == ABSTENTION:
        return _failure(ABSTENTION, "insufficient_context", usable_results)

    source_numbers = set(extract_source_numbers(generated))
    valid_numbers = {
        n for n in source_numbers if 1 <= n <= len(context_results)
    }
    if not source_numbers:
        result = _failure(UNCITED_FALLBACK, "uncited", usable_results)
        result["citation_quality"] = validate_citations(generated, context_results)
        return result

    if source_numbers != valid_numbers:
        result = _failure(
            "The generated answer contained invalid citations and was withheld.",
            "invalid_citations",
            usable_results,
        )
        result["citation_quality"] = validate_citations(generated, context_results)
        return result

    citations = []
    for number in sorted(valid_numbers):
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

    citation_quality = validate_citations(generated, context_results)
    return {
        "answer": generated.strip(),
        "status": "answered",
        "citations": citations,
        "citation_quality": citation_quality,
        "retrieved_chunks": usable_results,
    }