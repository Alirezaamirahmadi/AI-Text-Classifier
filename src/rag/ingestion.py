from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class Document:
    document_id: str
    source: str
    title: str
    text: str


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()

    return Path(fallback).stem.replace("_", " ").replace("-", " ").title()


def load_documents(directory: str | Path = "data/rag_documents") -> list[Document]:
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Document directory not found: {directory}")

    paths = sorted(directory.glob("*.md"))

    if not paths:
        raise ValueError(f"No Markdown documents found in: {directory}")

    documents = []

    for index, path in enumerate(paths, start=1):
        raw_text = path.read_text(encoding="utf-8")

        if not raw_text.strip():
            raise ValueError(f"Empty document: {path.name}")

        text = clean_text(raw_text)
        title = extract_title(text, path.name)

        documents.append(
            Document(
                document_id=f"doc_{index:03d}",
                source=path.name,
                title=title,
                text=text,
            )
        )

    return documents