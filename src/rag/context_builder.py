# این ماژول زمینه متنی را از قطعات بازیابی‌شده آماده می‌کند.

from collections.abc import Sequence


# ساخت زمینه متنی از نتایج بازیابی با حفظ منبع و امتیاز.
def build_context(results: Sequence[dict], max_chars: int | None = None) -> str:
    """Generating context text while preserving source information and metadata."""
    if max_chars is not None and max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    sections = []
    used_chars = 0

    for rank, result in enumerate(results, start=1):
        text = str(result.get("text", "")).strip()
        if not text:
            continue

        source = str(result.get("source", "unknown"))
        title = str(result.get("title", "Untitled"))
        document_id = str(result.get("document_id", "unknown"))
        chunk_id = str(result.get("chunk_id", "unknown"))
        score = result.get("score")

        header = (
            f"[Source {rank}] title={title} | source={source} | "
            f"document_id={document_id} | chunk_id={chunk_id}"
        )
        if score is not None:
            header += f" | score={float(score):.4f}"

        section = f"{header}\n{text}"

        if max_chars is not None:
            remaining = max_chars - used_chars
            if remaining <= 0:
                break
            if len(section) > remaining:
                if remaining <= len(header) + 1:
                    break
                section = f"{header}\n{text[:remaining - len(header) - 1].rstrip()}"
                sections.append(section)
                break

        sections.append(section)
        used_chars += len(section) + 2

    return "\n\n".join(sections)