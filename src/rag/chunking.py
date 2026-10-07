from dataclasses import dataclass

from src.rag.ingestion import Document


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    source: str
    title: str
    text: str
    start_char: int
    end_char: int


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = document.text
    chunks = []

    start = 0
    chunk_index = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}_chunk_{chunk_index:04d}",
                    document_id=document.document_id,
                    source=document.source,
                    title=document.title,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                )
            )

        chunk_index += 1
        start += step

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    chunks = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return chunks