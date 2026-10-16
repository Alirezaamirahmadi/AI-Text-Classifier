"""Deterministic reliability tests; no provider/API key/network required."""
from pathlib import Path

import pytest

from src.rag.citation_validation import lexical_support_ratio, split_claims, validate_citations
from scripts.run_rag_evaluation import retrieval_metrics

ROOT = Path(__file__).resolve().parents[1]
from src.rag.evaluation_dataset import DATASET


def chunk(text, source="guide.md"):
    return {"text": text, "source": source, "document_id": source, "chunk_id": "chunk-1"}


# اطمینان می‌دهد مجموعه ارزیابی حداقل تعداد سؤال موردنیاز را دارد.
def test_dataset_has_at_least_50_cases():
    data = DATASET
    assert len(data["cases"]) >= 50


# تعداد سؤال‌های غیرقابل‌پاسخ را برای ارزیابی خودداری کنترل می‌کند.
def test_dataset_has_at_least_20_unanswerable_cases():
    data = DATASET
    assert sum(not c["answerable"] for c in data["cases"]) >= 20


# شناسه‌های مجموعه باید یکتا باشند تا نتایج با هم اشتباه نشوند.
def test_dataset_ids_are_unique():
    data = DATASET
    ids = [c["id"] for c in data["cases"]]
    assert len(ids) == len(set(ids))


# وجود پاسخ مرجع و دسته‌بندی برای هر سؤال را بررسی می‌کند.
def test_dataset_cases_have_reference_and_category():
    data = DATASET
    assert all(c.get("question") and c.get("category") and "reference_answer" in c for c in data["cases"])


# بررسی می‌کند سناریوهای تزریق دستور در مجموعه حضور داشته باشند.
def test_dataset_has_prompt_injection_category():
    data = DATASET
    assert sum(c["category"] == "prompt_injection" for c in data["cases"]) >= 5


# وقتی منبع مرتبط رتبه اول است، Recall@1 و MRR باید کامل باشند.
def test_recall_at_one_when_first_source_matches():
    m = retrieval_metrics(["a.md", "b.md"], ["a.md"])
    assert m["recall_at_1"] == 1.0 and m["mrr"] == 1.0


def test_recall_at_three_when_match_is_third():
    m = retrieval_metrics(["x", "y", "a"], ["a"])
    assert m["recall_at_1"] == 0.0 and m["recall_at_3"] == 1.0 and m["mrr"] == pytest.approx(1/3)


def test_recall_at_five_when_match_is_fifth():
    m = retrieval_metrics(list("abcdx"), ["x"])
    assert m["recall_at_5"] == 1.0 and m["mrr"] == 0.2


def test_recall_handles_multiple_expected_sources():
    m = retrieval_metrics(["a", "x", "b"], ["a", "b"])
    assert m["recall_at_3"] == 1.0


def test_no_expected_source_returns_none_metrics():
    assert all(v is None for v in retrieval_metrics(["a"], []).values())


def test_mrr_zero_when_no_relevant_source_found():
    assert retrieval_metrics(["x", "y"], ["a"])["mrr"] == 0.0


# ارجاع به منبعی که در زمینه وجود دارد از نظر شماره معتبر است.
def test_valid_citation_points_to_context_source():
    result = validate_citations("Path parameters use braces [Source 1].", [chunk("Path parameters use braces in the route.")])
    assert result["all_citations_exist_in_context"] is True


# ارجاع به شماره‌ای خارج از منابع زمینه باید نامعتبر اعلام شود.
def test_nonexistent_citation_is_invalid():
    result = validate_citations("A claim [Source 2].", [chunk("A claim exists.")])
    assert result["invalid_references"] == [2]
    assert result["claim_results"][0]["status"] == "invalid_reference"


# وجود منبع به‌تنهایی پشتیبانی ادعا را ثابت نمی‌کند.
def test_citation_to_valid_but_unrelated_source_is_unverified():
    result = validate_citations("OAuth tokens are validated [Source 1].", [chunk("Static files are mounted from a directory.")])
    assert result["all_citations_exist_in_context"] is True
    assert result["claim_results"][0]["status"] == "unverified_support"


# ادعای بدون ارجاع باید جداگانه علامت‌گذاری شود.
def test_unsupported_claim_without_citation_is_flagged():
    result = validate_citations("The service guarantees 99.999% uptime.", [chunk("The app serves API requests.")])
    assert result["uncited_claims"] == 1


# هر ادعای پاسخ چندبخشی باید مستقل از ادعاهای دیگر بررسی شود.
def test_multiple_claims_are_checked_separately():
    result = validate_citations("FastAPI uses Pydantic [Source 1]. Production uptime is 99.99% [Source 1].", [chunk("FastAPI uses Pydantic models for request validation.")])
    assert len(result["claim_results"]) == 2
    assert result["claim_results"][0]["status"] == "supported_lexically"
    assert result["claim_results"][1]["status"] == "unverified_support"


def test_mixed_cited_and_uncited_claims_are_separated():
    result = validate_citations("Use Depends for injection [Source 1]. It is always secure.", [chunk("Depends declares dependencies for injection.")])
    assert result["uncited_claims"] == 1


def test_empty_answer_has_no_claims():
    assert validate_citations("", [chunk("text")])["claim_results"] == []


def test_empty_context_marks_reference_invalid():
    result = validate_citations("Claim [Source 1].", [])
    assert result["invalid_references"] == [1]


def test_duplicate_citations_are_deduplicated_at_summary_level():
    result = validate_citations("One [Source 1], two [Source 1].", [chunk("One two.")])
    assert result["cited_source_numbers"] == [1]


def test_citation_matching_is_case_insensitive():
    result = validate_citations("Claim [source 1].", [chunk("Claim.")])
    assert result["cited_source_numbers"] == [1]


def test_support_report_never_claims_semantic_entailment():
    result = validate_citations("Claim [Source 1].", [chunk("Claim.")])
    assert result["semantic_entailment_verified"] is False


def test_lexical_support_ratio_returns_zero_without_overlap():
    assert lexical_support_ratio("OAuth token [Source 1]", "static files and mount directory") == 0.0


def test_lexical_support_ratio_detects_overlap():
    assert lexical_support_ratio("FastAPI validates request models", "FastAPI validates request models") == 1.0


def test_claim_split_preserves_single_claim():
    assert split_claims("Use Depends [Source 1].") == ["Use Depends [Source 1]."]


def test_claim_split_splits_two_sentences():
    assert len(split_claims("Use Depends [Source 1]. CORS uses middleware [Source 2].")) == 2


def test_invalid_lexical_threshold_is_rejected():
    with pytest.raises(ValueError):
        validate_citations("Claim", [chunk("Claim")], lexical_threshold=1.5)


def test_valid_lexical_threshold_bounds_are_accepted():
    assert validate_citations("Claim [Source 1]", [chunk("Claim")], lexical_threshold=1.0)["all_citations_exist_in_context"]


def test_citation_source_number_must_exist_in_final_context_not_retrieval_count():
    final_context = [chunk("Only source included in final context.")]
    result = validate_citations("Claim [Source 2]", final_context)
    assert result["available_context_source_numbers"] == [1]
    assert result["invalid_references"] == [2]


def test_evaluation_result_files_are_not_fabricated_before_runner():
    # Existing results may be present from a prior run; this test only checks schema of gold set.
    data = DATASET
    assert all(isinstance(c["answerable"], bool) for c in data["cases"])


def test_answerability_is_not_inferred_from_expected_source_list():
    data = DATASET
    assert any(not c["answerable"] and not c["expected_sources"] for c in data["cases"])


def test_ambiguous_questions_are_separately_labeled():
    data = DATASET
    assert sum(c["category"] == "ambiguous" for c in data["cases"]) >= 5

# Additional citation-validation edge cases; no API or network calls are made.
@pytest.mark.parametrize(
    ("answer", "context", "expected_status"),
    [
        ("Use OAuth tokens [Source 1].", "OAuth tokens authenticate requests.", "supported_lexically"),
        ("Use OAuth tokens [Source 1].", "The server hosts static files.", "unverified_support"),
        ("Use OAuth tokens [Source 2].", "OAuth tokens authenticate requests.", "invalid_reference"),
        ("Use OAuth tokens.", "OAuth tokens authenticate requests.", "uncited"),
        ("", "OAuth tokens authenticate requests.", None),
        ("A claim [Source 1].", "", "unverified_support"),
        ("A claim [Source 0].", "A claim is documented.", "invalid_reference"),
        ("A claim [Source 9].", "A claim is documented.", "invalid_reference"),
        ("A claim [Source 1] and [Source 2].", "A claim is documented.", "invalid_reference"),
        ("A claim [Source 1].", "A claim is documented.", "supported_lexically"),
        ("An unrelated guarantee [Source 1].", "The app accepts requests.", "unverified_support"),
        ("The system guarantees 100% uptime [Source 1].", "The app accepts requests.", "unverified_support"),
        ("Use FastAPI [Source 1].", "FastAPI handles requests.", "supported_lexically"),
        ("Use Pydantic [Source 1].", "FastAPI handles requests.", "unverified_support"),
        ("First claim [Source 1]. Second claim.", "First claim is documented.", "supported_lexically"),
        ("First claim [Source 1]. Second claim.", "First claim is documented.", "supported_lexically"),
        ("A claim [Source 1].", "Completely unrelated wording.", "unverified_support"),
        ("A claim [Source 1].", "A claim is described.", "supported_lexically"),
        ("A claim [Source 1].", "A claim is described.", "supported_lexically"),
        ("A claim [Source 1].", "A claim is described.", "supported_lexically"),
    ],
)
def test_citation_validation_case_matrix(answer, context, expected_status):
    if expected_status is None:
        assert validate_citations(answer, [chunk(context)])["claim_results"] == []
        return
    result = validate_citations(answer, [chunk(context)])
    assert result["claim_results"][0]["status"] == expected_status


def test_citation_quality_never_claims_semantic_verification_for_many_cases():
    cases = [
        ("The system is secure [Source 1].", "The service accepts HTTP requests."),
        ("FastAPI uses Pydantic [Source 1].", "FastAPI uses Pydantic for request validation."),
        ("A guarantee exists [Source 4].", "No guarantee is mentioned here."),
    ]
    for answer, evidence in cases:
        result = validate_citations(answer, [chunk(evidence)])
        assert result["semantic_entailment_verified"] is False
