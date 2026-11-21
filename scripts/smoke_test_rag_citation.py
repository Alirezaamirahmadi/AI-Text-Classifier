"""Dependency-light contract checks for citation validation (no LLM/FAISS)."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "src/rag/citation_validation.py"
spec = importlib.util.spec_from_file_location("citation_validation_standalone", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

checks = []
def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

chunks = [{"text": "FastAPI validates request bodies using Pydantic models."}]
valid = module.validate_citations("FastAPI validates request bodies using Pydantic models. [Source 1]", chunks)
check("valid citation is present in final context", valid["all_citations_exist_in_context"])
check("lexical overlap is exposed", valid["claim_results"][0]["status"] == "supported_lexically")
check("semantic entailment is never falsely claimed", valid["semantic_entailment_verified"] is False)
invalid = module.validate_citations("Tokyo is the capital of Japan. [Source 9]", chunks)
check("out-of-range source is invalid", invalid["invalid_references"] == [9])
uncited = module.validate_citations("A claim without any reference.", chunks)
check("uncited claim is reported", uncited["uncited_claims"] == 1)
empty = module.validate_citations("", chunks)
check("empty answer has no claims", empty["claim_results"] == [])
try:
    module.validate_citations("x [Source 1]", chunks, lexical_threshold=1.5)
except ValueError:
    check("invalid threshold rejected", True)
else:
    check("invalid threshold rejected", False)
print(f"PASS: {len(checks)} citation contract checks")
for item in checks:
    print(f" - {item}")
