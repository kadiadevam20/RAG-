"""
Document processing utilities.

Provides text extraction for supported document formats and a
recursive character-based chunking function used to prepare content
for embedding and retrieval.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List

import pypdf
from docx import Document as DocxDocument

SUPPORTED_EXTENSIONS = ("pdf", "docx", "txt")


@dataclass
class DocumentChunk:
    """A single chunk of text extracted from a source document."""

    text: str
    source: str
    chunk_index: int


class DocumentProcessingError(Exception):
    """Raised when a document cannot be read or parsed."""


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> str:
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception as exc:
        raise DocumentProcessingError(
            f"Unable to extract text from PDF file: {filename}"
        ) from exc


def extract_text_from_docx(file_bytes: bytes, filename: str) -> str:
    try:
        document = DocxDocument(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in document.paragraphs]
        return "\n".join(paragraphs).strip()
    except Exception as exc:
        raise DocumentProcessingError(
            f"Unable to extract text from Word document: {filename}"
        ) from exc


def extract_text_from_txt(file_bytes: bytes, filename: str) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore").strip()
    except Exception as exc:
        raise DocumentProcessingError(
            f"Unable to read text file: {filename}"
        ) from exc


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract text from a supported document based on its extension."""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension == "pdf":
        return extract_text_from_pdf(file_bytes, filename)
    if extension == "docx":
        return extract_text_from_docx(file_bytes, filename)
    if extension == "txt":
        return extract_text_from_txt(file_bytes, filename)

    raise DocumentProcessingError(
        f"Unsupported file type: {filename}. Supported types are: "
        f"{', '.join(SUPPORTED_EXTENSIONS)}."
    )


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[DocumentChunk]:
    """
    Split text into overlapping chunks using a recursive character
    strategy. Splitting is attempted on paragraph boundaries first,
    then sentence boundaries, then falls back to a fixed character
    window.
    """
    if not text:
        return []

    separators = ["\n\n", "\n", ". ", " "]
    segments = _recursive_split(text, separators, chunk_size)

    chunks: List[DocumentChunk] = []
    current = ""
    index = 0

    for segment in segments:
        if len(current) + len(segment) <= chunk_size:
            current += segment
        else:
            if current.strip():
                chunks.append(DocumentChunk(current.strip(), source, index))
                index += 1
            overlap_text = current[-chunk_overlap:] if chunk_overlap else ""
            current = overlap_text + segment

    if current.strip():
        chunks.append(DocumentChunk(current.strip(), source, index))

    return chunks


def _recursive_split(text: str, separators: List[str], chunk_size: int) -> List[str]:
    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator = separators[0]
    parts = text.split(separator)

    result: List[str] = []
    for part in parts:
        piece = part + separator
        if len(piece) > chunk_size:
            result.extend(_recursive_split(piece, separators[1:], chunk_size))
        else:
            result.append(piece)

    return result
