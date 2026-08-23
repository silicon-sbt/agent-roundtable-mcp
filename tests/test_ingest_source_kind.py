from pathlib import Path

from rag.ingest import ingest_expert, ingest_person


def test_ingest_person_source_kind_merges_without_dropping_other_kinds(tmp_path: Path):
    book = tmp_path / "knowledge" / "people" / "desmond_shum" / "book" / "book.md"
    x_post = tmp_path / "knowledge" / "people" / "desmond_shum" / "x" / "post.md"
    book.parent.mkdir(parents=True)
    x_post.parent.mkdir(parents=True)
    book.write_text(
        "# Red Roulette\n\n" + ("Book content about political access. " * 40),
        encoding="utf-8",
    )
    x_post.write_text(
        "# Balance Sheet\n\n" + ("Household debt is rising quickly. " * 40),
        encoding="utf-8",
    )

    ingest_person("desmond_shum", root_dir=tmp_path, embedding_provider="keyword", reset=True)
    updated = tmp_path / "knowledge" / "people" / "desmond_shum" / "x" / "post.md"
    updated.write_text(
        "# Balance Sheet\n\n" + ("Household debt keeps compressing consumption. " * 40),
        encoding="utf-8",
    )
    stats = ingest_person(
        "desmond_shum",
        root_dir=tmp_path,
        embedding_provider="keyword",
        source_kinds={"x"},
    )

    assert stats["merge_mode"] is True
    assert stats["chunks_indexed"] >= 2
    index_path = (
        tmp_path / "vector_db" / "chroma" / "people__desmond_shum" / "chunks.jsonl"
    )
    text = index_path.read_text(encoding="utf-8")
    assert "book/book.md" in text
    assert "x/post.md" in text
    assert "compressing consumption" in text


def test_no_reset_preserves_missing_sources_and_replaces_edited_sources(tmp_path: Path):
    corpus_dir = tmp_path / "knowledge" / "experts" / "history"
    old_source = corpus_dir / "old.md"
    edited_source = corpus_dir / "edited.md"
    corpus_dir.mkdir(parents=True)
    old_source.write_text("# Old Source\n\nMaterial retained from an earlier crawl.", encoding="utf-8")
    edited_source.write_text("# Edited\n\nOriginal wording.", encoding="utf-8")

    ingest_expert("history", root_dir=tmp_path, embedding_provider="keyword", reset=True)
    old_source.unlink()
    edited_source.write_text("# Edited\n\nFresh replacement wording.", encoding="utf-8")

    ingest_expert("history", root_dir=tmp_path, embedding_provider="keyword", reset=False)

    index_path = tmp_path / "vector_db" / "chroma" / "experts__history" / "chunks.jsonl"
    text = index_path.read_text(encoding="utf-8")
    assert "Material retained from an earlier crawl" in text
    assert "Fresh replacement wording" in text
    assert "Original wording" not in text


def test_no_reset_removes_stale_backend_artifacts(tmp_path: Path):
    source = tmp_path / "knowledge" / "people" / "person" / "book" / "book.md"
    source.parent.mkdir(parents=True)
    source.write_text("# Book\n\nOriginal material.", encoding="utf-8")
    ingest_person("person", root_dir=tmp_path, embedding_provider="keyword")
    persist_dir = tmp_path / "vector_db" / "chroma" / "people__person"
    stale = persist_dir / "chroma.sqlite3"
    stale.write_text("stale", encoding="utf-8")

    source.write_text("# Book\n\nUpdated material.", encoding="utf-8")
    ingest_person(
        "person",
        root_dir=tmp_path,
        embedding_provider="keyword",
        reset=False,
    )

    assert not stale.exists()
    chunks = (persist_dir / "chunks.jsonl").read_text(encoding="utf-8")
    assert "Updated material" in chunks
    assert "Original material" not in chunks
