from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_EXPERT_WORK_TYPE,
    EXPERT_WORK_TYPE_FOLDERS,
    KnowledgeScope,
    CorpusRef,
    resolve_corpus,
    resolve_paths,
)

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - optional dependency in minimal test envs.
    RecursiveCharacterTextSplitter = None


SUPPORTED_MARKDOWN_SUFFIXES = {".md", ".markdown"}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
SVG_RE = re.compile(r"<svg\b.*?</svg>", re.IGNORECASE | re.DOTALL)
IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"</?[^>]+>")


@dataclass(frozen=True)
class RagChunk:
    page_content: str
    metadata: dict[str, Any]
    score: float = 0.0


def _iter_markdown_files(
    corpus: CorpusRef,
    root_dir: Path | str = ".",
    *,
    source_kinds: set[str] | None = None,
) -> list[Path]:
    paths = resolve_paths(root_dir)
    corpus_dir = paths.corpus_knowledge_dir(corpus)
    files: list[Path] = []
    if corpus_dir.exists():
        for path in corpus_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_MARKDOWN_SUFFIXES:
                continue
            if source_kinds and corpus.knowledge_scope == "people":
                kind = _source_kind_for_person_file(corpus_dir, path)
                if kind not in source_kinds:
                    continue
            files.append(path)
    return sorted(dict.fromkeys(files))


def _work_type_for_expert_file(expert_dir: Path, file_path: Path) -> str:
    try:
        relative = file_path.relative_to(expert_dir)
        if len(relative.parts) > 1:
            folder = relative.parts[0].lower()
            return EXPERT_WORK_TYPE_FOLDERS.get(folder, DEFAULT_EXPERT_WORK_TYPE)
    except ValueError:
        pass
    return DEFAULT_EXPERT_WORK_TYPE


def _source_file(path: Path, root_dir: Path) -> str:
    try:
        # T24: store source_file with POSIX separators so metadata/references are
        # stable across platforms (Windows native str() uses backslashes).
        return path.resolve().relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _source_kind_for_person_file(person_dir: Path, file_path: Path) -> str:
    try:
        relative = file_path.relative_to(person_dir)
        if relative.parts:
            return relative.parts[0]
    except ValueError:
        pass
    return "unknown"


def _extract_title(text: str, fallback: str) -> str:
    for match in HEADING_RE.finditer(text):
        if len(match.group(1)) == 1:
            return match.group(2).strip()
    return fallback


def _heading_index(text: str) -> list[tuple[int, int, str]]:
    headings: list[tuple[int, int, str]] = []
    for match in HEADING_RE.finditer(text):
        headings.append((match.start(), len(match.group(1)), match.group(2).strip()))
    return headings


def _chapter_for_chunk(
    *,
    headings: list[tuple[int, int, str]],
    chunk_text: str,
    start_index: int,
    title: str,
) -> str:
    chunk_heading = HEADING_RE.search(chunk_text)
    if chunk_heading and len(chunk_heading.group(1)) >= 2:
        return chunk_heading.group(2).strip()

    # ``headings`` is computed once per document.  The previous implementation
    # rescanned the entire book for every chunk, which made large Markdown books
    # effectively O(document_size * chunk_count).
    position = bisect_right(headings, start_index, key=lambda heading: heading[0])
    for index in range(position - 1, -1, -1):
        _heading_start, level, heading_title = headings[index]
        if level >= 2:
            return heading_title
    return title


def _fallback_split_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int]]:
    if not text.strip():
        return []

    chunks: list[tuple[str, int]] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(text_length, start + chunk_size)
        if end < text_length:
            boundary = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start))
        if end >= text_length:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def _split_text_with_offsets(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int]]:
    if RecursiveCharacterTextSplitter is None:
        return _fallback_split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", ". ", " ", ""],
    )
    documents = splitter.create_documents([text])
    return [
        (
            document.page_content.strip(),
            int(document.metadata.get("start_index", 0)),
        )
        for document in documents
        if document.page_content.strip()
    ]


def clean_markdown_text(text: str) -> str:
    text = MARKDOWN_IMAGE_RE.sub("", text)
    text = SVG_RE.sub("", text)
    text = IMG_RE.sub("", text)
    text = HTML_TAG_RE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_markdown_file(
    path: Path,
    *,
    corpus: CorpusRef,
    root_dir: Path | str = ".",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[RagChunk]:
    root = Path(root_dir).expanduser().resolve()
    paths = resolve_paths(root)
    corpus_dir = paths.corpus_knowledge_dir(corpus)
    text = clean_markdown_text(path.read_text(encoding="utf-8", errors="ignore"))
    title = _extract_title(text, path.stem.replace("_", " ").title())
    headings = _heading_index(text)
    source_file = _source_file(path, root)

    metadata_base: dict[str, Any] = {
        "corpus_id": corpus.corpus_id,
        "knowledge_scope": corpus.knowledge_scope,
        "expert_name": corpus.corpus_id,
        "source_file": source_file,
        "title": title,
    }
    if corpus.knowledge_scope == "people":
        metadata_base["source_kind"] = _source_kind_for_person_file(corpus_dir, path)
    else:
        metadata_base["work_type"] = _work_type_for_expert_file(corpus_dir, path)

    chunks: list[RagChunk] = []
    for chunk_index, (chunk_text, start_index) in enumerate(
        _split_text_with_offsets(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ):
        chunks.append(
            RagChunk(
                page_content=chunk_text,
                metadata={
                    **metadata_base,
                    "chapter": _chapter_for_chunk(
                        headings=headings,
                        chunk_text=chunk_text,
                        start_index=start_index,
                        title=title,
                    ),
                    "chunk_index": chunk_index,
                },
            )
        )
    return chunks


def chunk_corpus_markdown(
    corpus_id: str,
    *,
    knowledge_scope: KnowledgeScope = "experts",
    root_dir: Path | str = ".",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    source_kinds: set[str] | None = None,
) -> list[RagChunk]:
    corpus = resolve_corpus(corpus_id, knowledge_scope=knowledge_scope)
    chunks: list[RagChunk] = []
    for path in _iter_markdown_files(corpus, root_dir=root_dir, source_kinds=source_kinds):
        chunks.extend(
            chunk_markdown_file(
                path,
                corpus=corpus,
                root_dir=root_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks


def chunk_expert_markdown(
    expert_name: str,
    *,
    root_dir: Path | str = ".",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[RagChunk]:
    return chunk_corpus_markdown(
        expert_name,
        knowledge_scope="experts",
        root_dir=root_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
