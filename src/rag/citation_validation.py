"""Conservative, deterministic citation checks for RAG answers.

Lexical overlap is a diagnostic heuristic, not proof of semantic entailment. A
claim is only marked ``supported_lexically`` when it overlaps its cited context;
all other claims remain unverified rather than being treated as false or true.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

# الگوی استخراج شماره منبع از متن پاسخ را تعریف می‌کند.
# الگوی یک منبع (برای سازگاری با تست‌های قبلی)
CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)
# الگوی براکت؛ می‌تواند چند منبع داشته باشد، مثل [Source 2, Source 4]
LIST_CITATION_PATTERN = re.compile(r"\[((?:\s*Source\s+\d+\s*,?)+)\]", re.IGNORECASE)


def extract_source_numbers(text):
    """شماره‌ی همه‌ی منابع ارجاع‌شده در متن را برمی‌گرداند (مثلاً [Source 2, Source 4] → 2 و 4)."""
    numbers = []
    for group in LIST_CITATION_PATTERN.findall(text or ""):
        numbers.extend(int(n) for n in re.findall(r"\d+", group))
    return numbers

_WORDS = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{2,}")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


# پاسخ را به ادعاهای جداگانه تقسیم می‌کند تا هر ادعا مستقل بررسی شود.
def split_claims(answer: str) -> list[str]:
    """Split a response into sentence-like claims without claiming perfect NLP."""
    if not isinstance(answer, str) or not answer.strip():
        return []
    pieces = _SENTENCE_SPLIT.split(answer.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


# واژه‌های قابل مقایسه را برای سنجش هم‌پوشانی متنی استخراج می‌کند.
def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _WORDS.findall(text)}


# نسبت واژه‌های مشترک ادعا و شاهد را محاسبه می‌کند؛ این معیار اثبات معنایی نیست.
def lexical_support_ratio(claim: str, evidence: str) -> float:
    """Return token-overlap ratio for diagnostics; this is not entailment."""
    claim_tokens = _tokens(LIST_CITATION_PATTERN.sub("", claim))
    evidence_tokens = _tokens(evidence)
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


# ارجاع‌ها را با منابع موجود در زمینه نهایی تطبیق می‌دهد و پشتیبانی را محافظه‌کارانه گزارش می‌کند.
def validate_citations(
    answer: str,
    retrieved_chunks: Sequence[dict[str, Any]],
    *,
    lexical_threshold: float = 0.35,
) -> dict[str, Any]:
    """Validate reference existence, context membership and conservative support.

    Source numbers are positional labels in the *final context* (1-based). A
    chunk's presence in retrieval alone is not sufficient unless it is in the
    context sequence passed here. The lexical score cannot verify semantic support.
    """
    if not 0.0 <= lexical_threshold <= 1.0:
        raise ValueError("lexical_threshold must be between 0 and 1")
    cited = sorted(set(extract_source_numbers(answer)))
    available = set(range(1, len(retrieved_chunks) + 1))
    missing = [n for n in cited if n not in available]
    claims = split_claims(answer or "")
    claim_results = []
    for claim in claims:
        refs = sorted(set(extract_source_numbers(claim)))
        if not refs:
            claim_results.append({"claim": claim, "citations": [], "status": "uncited", "support_scores": {}})
            continue
        scores = {}
        for number in refs:
            if number in available:
                scores[str(number)] = round(
                    lexical_support_ratio(claim, str(retrieved_chunks[number - 1].get("text", ""))), 4
                )
        if any(n not in available for n in refs):
            status = "invalid_reference"
        elif any(score >= lexical_threshold for score in scores.values()):
            status = "supported_lexically"
        else:
            status = "unverified_support"
        claim_results.append({"claim": claim, "citations": refs, "status": status, "support_scores": scores})
    return {
        "cited_source_numbers": cited,
        "available_context_source_numbers": sorted(available),
        "invalid_references": missing,
        "all_citations_exist_in_context": not missing,
        "claim_results": claim_results,
        "uncited_claims": sum(c["status"] == "uncited" for c in claim_results),
        "claims_with_unverified_support": sum(c["status"] in {"unverified_support", "invalid_reference"} for c in claim_results),
        "semantic_entailment_verified": False,
        "support_method": "lexical token overlap heuristic; not semantic entailment",
        "lexical_threshold": lexical_threshold,
    }
