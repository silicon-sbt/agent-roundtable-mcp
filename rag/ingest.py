from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.chunker import RagChunk, chunk_corpus_markdown
from rag.config import (
    CHROMA_COLLECTION_NAME,
    DEFAULT_EMBEDDING_PROVIDER,
    INDEX_MANIFEST_FILE,
    INDEX_SCHEMA_VERSION,
    KEYWORD_INDEX_FILE,
    CorpusRef,
    KnowledgeScope,
    create_embedding_function,
    discover_corpora,
    resolve_corpus,
    resolve_paths,
)


def _write_text_atomic(path: Path, text: str) -> None:
    """Replace a generated index artifact without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            temp_path = Path(handle.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _chunk_id(chunk: RagChunk) -> str:
    raw = "|".join(
        [
            str(chunk.metadata.get("knowledge_scope", "")),
            str(chunk.metadata.get("corpus_id", "")),
            str(chunk.metadata.get("source_file", "")),
            str(chunk.metadata.get("chunk_index", "")),
            chunk.page_content,
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _serialize_chunk(chunk: RagChunk) -> dict[str, Any]:
    return {
        "id": _chunk_id(chunk),
        "page_content": chunk.page_content,
        "metadata": dict(chunk.metadata),
    }


def _read_keyword_chunks(persist_dir: Path) -> list[RagChunk]:
    index_path = persist_dir / KEYWORD_INDEX_FILE
    if not index_path.exists():
        return []
    chunks: list[RagChunk] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        chunks.append(
            RagChunk(
                page_content=str(item.get("page_content", "")),
                metadata=dict(item.get("metadata", {})),
            )
        )
    return chunks


def _write_keyword_index(chunks: list[RagChunk], persist_dir: Path) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    index_path = persist_dir / KEYWORD_INDEX_FILE
    lines = [json.dumps(_serialize_chunk(chunk), ensure_ascii=False) for chunk in chunks]
    _write_text_atomic(index_path, "\n".join(lines) + ("\n" if lines else ""))


def _write_index_manifest(
    *,
    corpus: CorpusRef,
    chunks: list[RagChunk],
    persist_dir: Path,
    requested_embedding_provider: str,
    backend: str,
) -> Path:
    source_files = sorted(
        {
            str(chunk.metadata.get("source_file", ""))
            for chunk in chunks
            if chunk.metadata.get("source_file")
        }
    )
    fingerprint = hashlib.sha256()
    for chunk in chunks:
        fingerprint.update(_chunk_id(chunk).encode("ascii"))
        fingerprint.update(b"\n")
    path = persist_dir / INDEX_MANIFEST_FILE
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "corpus_id": corpus.corpus_id,
        "knowledge_scope": corpus.knowledge_scope,
        "backend": backend,
        "requested_embedding_provider": requested_embedding_provider,
        "chunks_indexed": len(chunks),
        "files_indexed": len(source_files),
        "source_files": source_files,
        "content_fingerprint": fingerprint.hexdigest(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


def _write_chroma(
    *,
    chunks: list[RagChunk],
    persist_dir: Path,
    embedding_provider: str,
    reset: bool,
) -> bool:
    if not chunks:
        return False

    try:
        import chromadb
    except ImportError:
        return False

    embedding_function = create_embedding_function(embedding_provider)
    if embedding_function is None:
        return False

    client = chromadb.PersistentClient(path=str(persist_dir))
    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(
        ids=[_chunk_id(chunk) for chunk in chunks],
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[dict(chunk.metadata) for chunk in chunks],
    )
    return True


def _merge_chunks(
    *,
    existing: list[RagChunk],
    incoming: list[RagChunk],
    source_kinds: set[str] | None,
    preserve_existing: bool,
) -> list[RagChunk]:
    if source_kinds:
        retained = [
            chunk
            for chunk in existing
            if str(chunk.metadata.get("source_kind", "")) not in source_kinds
        ]
        return retained + incoming
    if not preserve_existing:
        return incoming

    # ``--no-reset`` keeps sources that are not part of the current crawl, but
    # replaces every old chunk belonging to a source that was freshly parsed.
    # Matching by content hash would retain both the old and edited versions.
    incoming_sources = {
        str(chunk.metadata.get("source_file", ""))
        for chunk in incoming
        if chunk.metadata.get("source_file")
    }
    retained = [
        chunk
        for chunk in existing
        if str(chunk.metadata.get("source_file", "")) not in incoming_sources
    ]
    return retained + incoming


def ingest_corpus(
    corpus_id: str,
    *,
    knowledge_scope: KnowledgeScope = "experts",
    root_dir: Path | str = ".",
    embedding_provider: str | None = None,
    reset: bool = True,
    source_kinds: set[str] | None = None,
) -> dict[str, Any]:
    corpus = resolve_corpus(corpus_id, knowledge_scope=knowledge_scope)
    paths = resolve_paths(root_dir)
    persist_dir = paths.corpus_vector_dir(corpus)

    merge_mode = bool(source_kinds)
    if reset and not merge_mode and persist_dir.exists():
        shutil.rmtree(persist_dir)

    existing_chunks = [] if reset and not merge_mode else _read_keyword_chunks(persist_dir)
    new_chunks = chunk_corpus_markdown(
        corpus_id,
        knowledge_scope=knowledge_scope,
        root_dir=paths.root_dir,
        source_kinds=source_kinds,
    )
    chunks = _merge_chunks(
        existing=existing_chunks,
        incoming=new_chunks,
        source_kinds=source_kinds,
        preserve_existing=not reset,
    )

    # ``chunks`` is now the complete desired snapshot, including retained
    # sources for incremental ingests. Recreate generated storage so edited
    # chunks cannot leave stale Chroma ids or files from a previous backend.
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    _write_keyword_index(chunks, persist_dir)

    provider = (embedding_provider or DEFAULT_EMBEDDING_PROVIDER).lower()
    chroma_written = False
    if provider != "keyword":
        chroma_written = _write_chroma(
            chunks=chunks,
            persist_dir=persist_dir,
            embedding_provider=provider,
            reset=reset or merge_mode,
        )

    backend = "chroma" if chroma_written else "keyword"
    manifest_path = _write_index_manifest(
        corpus=corpus,
        chunks=chunks,
        persist_dir=persist_dir,
        requested_embedding_provider=provider,
        backend=backend,
    )

    return {
        "corpus_id": corpus_id,
        "knowledge_scope": knowledge_scope,
        "expert_name": corpus_id,
        "files_indexed": len({chunk.metadata.get("source_file") for chunk in chunks}),
        "chunks_indexed": len(chunks),
        "persist_dir": str(persist_dir),
        "backend": backend,
        "embedding_provider": provider,
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "manifest_path": str(manifest_path),
        "source_kinds": sorted(source_kinds) if source_kinds else [],
        "merge_mode": merge_mode,
    }


def ingest_expert(
    expert_name: str,
    *,
    root_dir: Path | str = ".",
    embedding_provider: str | None = None,
    reset: bool = True,
) -> dict[str, Any]:
    return ingest_corpus(
        expert_name,
        knowledge_scope="experts",
        root_dir=root_dir,
        embedding_provider=embedding_provider,
        reset=reset,
    )


def ingest_person(
    person_id: str,
    *,
    root_dir: Path | str = ".",
    embedding_provider: str | None = None,
    reset: bool = True,
    source_kinds: set[str] | None = None,
) -> dict[str, Any]:
    return ingest_corpus(
        person_id,
        knowledge_scope="people",
        root_dir=root_dir,
        embedding_provider=embedding_provider,
        reset=reset,
        source_kinds=source_kinds,
    )


def prune_obsolete_indexes(root_dir: Path | str = ".") -> list[str]:
    """Delete generated corpus directories that no configured knowledge corpus can use."""
    paths = resolve_paths(root_dir)
    expected = {corpus.vector_key for corpus in discover_corpora(paths.root_dir)}
    if not expected or not paths.vector_db_dir.exists():
        return []

    removed: list[str] = []
    for path in sorted(paths.vector_db_dir.iterdir()):
        if not path.is_dir() or path.name in expected:
            continue
        generated_markers = (
            path / KEYWORD_INDEX_FILE,
            path / INDEX_MANIFEST_FILE,
            path / "chroma.sqlite3",
        )
        if not any(marker.exists() for marker in generated_markers):
            continue
        shutil.rmtree(path)
        removed.append(path.name)
    return removed


def _parse_corpus_target(value: str) -> CorpusRef:
    if "/" in value:
        scope_name, corpus_id = value.split("/", 1)
        if scope_name not in {"experts", "people"}:
            raise ValueError(f"Unsupported knowledge scope in '{value}'. Use experts/ or people/.")
        return resolve_corpus(corpus_id, knowledge_scope=scope_name)  # type: ignore[arg-type]
    return resolve_corpus(value, knowledge_scope="experts")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Markdown knowledge into local RAG storage.")
    parser.add_argument(
        "--corpus",
        action="append",
        help="Corpus target, e.g. experts/macroeconomics or people/desmond_shum.",
    )
    parser.add_argument("--expert-name", action="append", help="Expert folder under knowledge/experts/.")
    parser.add_argument("--person-name", action="append", help="Person folder under knowledge/people/.")
    parser.add_argument(
        "--source-kind",
        action="append",
        choices=["book", "x", "news", "report"],
        help="For people corpora, ingest only these source kinds and merge into the existing index.",
    )
    parser.add_argument("--root-dir", default=".", help="Project root directory.")
    parser.add_argument(
        "--embedding-provider",
        default=None,
        choices=["keyword", "mock", "openai", "gemini", "openrouter"],
        help="Embedding backend. keyword skips embeddings; mock uses deterministic local vectors.",
    )
    parser.add_argument("--no-reset", action="store_true", help="Keep existing vector DB contents.")
    parser.add_argument(
        "--prune-obsolete",
        action="store_true",
        help="Remove generated index directories whose corpus ids no longer exist.",
    )
    return parser.parse_args(argv)


def _targets_from_args(args: argparse.Namespace) -> list[CorpusRef]:
    targets: list[CorpusRef] = []
    for value in args.corpus or []:
        targets.append(_parse_corpus_target(value))
    for expert_name in args.expert_name or []:
        targets.append(resolve_corpus(expert_name, knowledge_scope="experts"))
    for person_name in args.person_name or []:
        targets.append(resolve_corpus(person_name, knowledge_scope="people"))
    if targets:
        return targets
    return discover_corpora(args.root_dir)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    targets = _targets_from_args(args)
    if not targets:
        raise SystemExit("No corpora found under knowledge/experts/ or knowledge/people/.")

    source_kinds = set(args.source_kind) if args.source_kind else None
    for corpus in targets:
        if source_kinds and corpus.knowledge_scope != "people":
            raise SystemExit("--source-kind is only supported for people corpora.")
        stats = ingest_corpus(
            corpus.corpus_id,
            knowledge_scope=corpus.knowledge_scope,
            root_dir=args.root_dir,
            embedding_provider=args.embedding_provider,
            reset=not args.no_reset and not source_kinds,
            source_kinds=source_kinds,
        )
        merge_note = " merged" if stats["merge_mode"] else ""
        kinds_note = f", kinds={','.join(stats['source_kinds'])}" if stats["source_kinds"] else ""
        print(
            f"{stats['knowledge_scope']}/{stats['corpus_id']}: indexed {stats['files_indexed']} files, "
            f"{stats['chunks_indexed']} chunks -> {stats['persist_dir']} "
            f"({stats['backend']}, provider={stats['embedding_provider']}{kinds_note}{merge_note})"
        )

    if args.prune_obsolete:
        removed = prune_obsolete_indexes(args.root_dir)
        print(
            "Pruned obsolete index directories: "
            + (", ".join(removed) if removed else "none")
        )


if __name__ == "__main__":
    main()
