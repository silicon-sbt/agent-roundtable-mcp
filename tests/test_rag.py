import json
from pathlib import Path

from rag import chunker as chunker_module
from rag.chunker import chunk_corpus_markdown, chunk_expert_markdown
from rag.ingest import ingest_corpus, ingest_expert, prune_obsolete_indexes
from rag.retriever import get_retriever


def test_chunk_expert_markdown_keeps_required_metadata(tmp_path: Path):
    source = tmp_path / "knowledge" / "experts" / "macroeconomics" / "keynes" / "demand.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# Keynesian Demand\n\n"
        "Intro paragraph about aggregate demand.\n\n"
        "## Fiscal Policy\n\n"
        + ("Multiplier policy can stabilize employment and demand. " * 40),
        encoding="utf-8",
    )

    chunks = chunk_expert_markdown("macroeconomics", root_dir=tmp_path)

    assert chunks
    assert chunks[0].metadata["corpus_id"] == "macroeconomics"
    assert chunks[0].metadata["knowledge_scope"] == "experts"
    assert chunks[0].metadata["source_file"] == "knowledge/experts/macroeconomics/keynes/demand.md"
    assert chunks[0].metadata["title"] == "Keynesian Demand"
    assert any(chunk.metadata["chapter"] == "Fiscal Policy" for chunk in chunks)
    assert all(len(chunk.page_content) <= 900 for chunk in chunks)


def test_chunk_expert_markdown_tracks_work_type(tmp_path: Path):
    source = tmp_path / "knowledge" / "experts" / "macroeconomics" / "papers" / "essay.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# Essays in Persuasion\n\n" + ("Fiscal stimulus can stabilize demand. " * 40),
        encoding="utf-8",
    )

    chunks = chunk_expert_markdown("macroeconomics", root_dir=tmp_path)

    assert chunks
    assert chunks[0].metadata["work_type"] == "paper"


def test_chunk_person_markdown_tracks_source_kind(tmp_path: Path):
    source = tmp_path / "knowledge" / "people" / "desmond_shum" / "x" / "post.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# Balance Sheet Recession\n\n"
        "## Four Sectors\n\n"
        + ("Debt pressure spreads across households, firms, banks, and local governments. " * 30),
        encoding="utf-8",
    )

    chunks = chunk_corpus_markdown("desmond_shum", knowledge_scope="people", root_dir=tmp_path)

    assert chunks
    assert chunks[0].metadata["source_kind"] == "x"
    assert chunks[0].metadata["source_file"] == "knowledge/people/desmond_shum/x/post.md"


def test_keyword_mock_retriever_works_without_embedding_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    source = tmp_path / "knowledge" / "experts" / "investing" / "buffett" / "moat.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# Durable Moats\n\n"
        "## Pricing Power\n\n"
        "A strong business has durable pricing power, high return on capital, "
        "and managers who avoid leverage.",
        encoding="utf-8",
    )

    stats = ingest_expert("investing", root_dir=tmp_path, embedding_provider="keyword")
    retriever = get_retriever("investing", top_k=2, root_dir=tmp_path, backend="keyword")
    results = retriever.invoke("pricing power leverage capital")

    assert stats["chunks_indexed"] >= 1
    assert stats["backend"] == "keyword"
    persist_dir = tmp_path / "vector_db" / "chroma" / "experts__investing"
    assert (persist_dir / "chunks.jsonl").exists()
    manifest = json.loads((persist_dir / "index_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["corpus_id"] == "investing"
    assert manifest["backend"] == "keyword"
    assert manifest["chunks_indexed"] == stats["chunks_indexed"]
    assert results
    assert results[0].metadata["source_file"] == "knowledge/experts/investing/buffett/moat.md"
    assert "pricing power" in results[0].page_content


def test_ingest_person_corpus_uses_people_vector_dir(tmp_path: Path):
    source = tmp_path / "knowledge" / "people" / "desmond_shum" / "book" / "memoir.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# Red Roulette\n\n"
        "## Chapter One\n\n"
        + ("Political access shaped every major real estate deal in Beijing. " * 40),
        encoding="utf-8",
    )

    stats = ingest_corpus(
        "desmond_shum",
        knowledge_scope="people",
        root_dir=tmp_path,
        embedding_provider="keyword",
    )

    assert stats["chunks_indexed"] >= 1
    assert (tmp_path / "vector_db" / "chroma" / "people__desmond_shum" / "chunks.jsonl").exists()


def test_chunk_expert_markdown_ignores_external_sources_yaml(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    external_dir = tmp_path / "doc2md" / "宏观经济专家"
    external_dir.mkdir(parents=True)
    external_file = external_dir / "general_theory.md"
    external_file.write_text(
        "# General Theory\n\n"
        "## Employment\n\n"
        "Effective demand is central to employment analysis.",
        encoding="utf-8",
    )
    (project_root / "rag").mkdir()
    (project_root / "rag" / "sources.yaml").write_text(
        "macroeconomics:\n"
        f"  - {external_dir}\n",
        encoding="utf-8",
    )

    chunks = chunk_expert_markdown("macroeconomics", root_dir=project_root)

    assert chunks == []


def test_keyword_retriever_prefers_higher_weight_source_kind(tmp_path: Path):
    book = tmp_path / "knowledge" / "people" / "desmond_shum" / "book" / "book.md"
    x_post = tmp_path / "knowledge" / "people" / "desmond_shum" / "x" / "post.md"
    book.parent.mkdir(parents=True)
    x_post.parent.mkdir(parents=True)
    book.write_text("# Book\n\nHousehold balance sheet repair takes years.", encoding="utf-8")
    x_post.write_text("# Post\n\nHousehold balance sheet repair takes years.", encoding="utf-8")

    ingest_corpus("desmond_shum", knowledge_scope="people", root_dir=tmp_path, embedding_provider="keyword")
    retriever = get_retriever(
        "desmond_shum",
        top_k=1,
        knowledge_scope="people",
        root_dir=tmp_path,
        backend="keyword",
    )
    results = retriever.invoke("household balance sheet repair")

    assert results
    assert results[0].metadata["source_kind"] == "book"


def test_chunker_strips_doc2md_image_and_html_noise(tmp_path: Path):
    source = tmp_path / "knowledge" / "experts" / "computing" / "turing.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "![](/tmp/cover.jpg)\n\n"
        "<div><svg><image href=\"cover.jpg\"></image></svg></div>\n\n"
        "# Computing Machinery\n\n"
        "<span id=\"p1\"></span> The imitation game asks whether machines can think.",
        encoding="utf-8",
    )

    chunks = chunk_expert_markdown("computing", root_dir=tmp_path)
    text = "\n".join(chunk.page_content for chunk in chunks)

    assert "cover.jpg" not in text
    assert "<svg" not in text
    assert "imitation game" in text


def test_chunker_builds_heading_index_once_per_document(tmp_path: Path, monkeypatch):
    source = tmp_path / "knowledge" / "experts" / "computing" / "long.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# Long Book\n\n## First\n\n"
        + ("A long section about systems and computation. " * 300)
        + "\n\n## Second\n\n"
        + ("A second long section about verification. " * 300),
        encoding="utf-8",
    )
    original = chunker_module._heading_index
    calls = 0

    def counting_heading_index(text: str):
        nonlocal calls
        calls += 1
        return original(text)

    monkeypatch.setattr(chunker_module, "_heading_index", counting_heading_index)

    chunks = chunk_expert_markdown("computing", root_dir=tmp_path)

    assert len(chunks) > 10
    assert calls == 1


def test_prune_obsolete_indexes_keeps_only_discovered_corpora(tmp_path: Path):
    (tmp_path / "knowledge" / "experts" / "history").mkdir(parents=True)
    vector_root = tmp_path / "vector_db" / "chroma"
    expected = vector_root / "experts__history"
    obsolete = vector_root / "history_strategist"
    unknown = vector_root / "manually_managed"
    expected.mkdir(parents=True)
    obsolete.mkdir()
    unknown.mkdir()
    (obsolete / "chunks.jsonl").write_text("generated", encoding="utf-8")

    removed = prune_obsolete_indexes(tmp_path)

    assert removed == ["history_strategist"]
    assert expected.exists()
    assert not obsolete.exists()
    assert unknown.exists()
