from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
KNOWLEDGE_DIR = Path("knowledge")
EXPERTS_SUBDIR = "experts"
PEOPLE_SUBDIR = "people"
PERSON_SOURCE_KINDS = frozenset({"book", "x", "weibo", "news", "report"})
SOURCE_KIND_WEIGHTS: dict[str, float] = {
    "book": 4.0,
    "report": 3.0,
    "x": 2.0,
    "weibo": 2.0,
    "news": 1.0,
}
EXPERT_WORK_TYPE_FOLDERS: dict[str, str] = {
    "papers": "paper",
    "paper": "paper",
    "letters": "letter",
    "letter": "letter",
    "reports": "report",
    "report": "report",
}
DEFAULT_EXPERT_WORK_TYPE = "book"
KnowledgeScope = Literal["experts", "people"]
VECTOR_DB_DIR = Path("vector_db") / "chroma"
KEYWORD_INDEX_FILE = "chunks.jsonl"
INDEX_MANIFEST_FILE = "index_manifest.json"
INDEX_SCHEMA_VERSION = 1
CHROMA_COLLECTION_NAME = "rag_chunks"
DEFAULT_TOP_K = 5
DEFAULT_EMBEDDING_PROVIDER = os.getenv("RAG_EMBEDDING_PROVIDER", "mock").lower()
MOCK_EMBEDDING_DIMENSIONS = 384

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


@dataclass(frozen=True)
class CorpusRef:
    corpus_id: str
    knowledge_scope: KnowledgeScope

    @property
    def vector_key(self) -> str:
        return corpus_vector_key(self.knowledge_scope, self.corpus_id)

    @property
    def knowledge_dir(self) -> Path:
        return corpus_knowledge_dir(self.knowledge_scope, self.corpus_id)


@dataclass(frozen=True)
class RagPaths:
    root_dir: Path
    knowledge_dir: Path
    vector_db_dir: Path

    def corpus_knowledge_dir(self, corpus: CorpusRef) -> Path:
        return self.knowledge_dir / corpus.knowledge_scope / corpus.corpus_id

    def corpus_vector_dir(self, corpus: CorpusRef) -> Path:
        return self.vector_db_dir / corpus.vector_key

    def expert_knowledge_dir(self, expert_name: str) -> Path:
        return corpus_knowledge_dir("experts", expert_name, root_dir=self.root_dir)

    def expert_vector_dir(self, expert_name: str) -> Path:
        return self.vector_db_dir / corpus_vector_key("experts", expert_name)

    def person_knowledge_dir(self, person_id: str) -> Path:
        return corpus_knowledge_dir("people", person_id, root_dir=self.root_dir)

    def person_vector_dir(self, person_id: str) -> Path:
        return self.vector_db_dir / corpus_vector_key("people", person_id)


def corpus_knowledge_dir(
    knowledge_scope: KnowledgeScope,
    corpus_id: str,
    *,
    root_dir: Path | str = ".",
) -> Path:
    root = Path(root_dir).expanduser().resolve()
    return root / KNOWLEDGE_DIR / knowledge_scope / corpus_id


def corpus_vector_key(knowledge_scope: KnowledgeScope, corpus_id: str) -> str:
    return f"{knowledge_scope}__{corpus_id}"


def resolve_corpus(
    corpus_id: str,
    *,
    knowledge_scope: KnowledgeScope | None = None,
    agent_type: str | None = None,
) -> CorpusRef:
    if knowledge_scope is None:
        if agent_type == "domain_expert":
            knowledge_scope = "experts"
        elif agent_type == "persona_inspired":
            knowledge_scope = "people"
        else:
            knowledge_scope = "experts"
    return CorpusRef(corpus_id=corpus_id, knowledge_scope=knowledge_scope)


def infer_knowledge_scope(
    *,
    agent_type: str | None,
    explicit_scope: str | None = None,
) -> KnowledgeScope | None:
    if explicit_scope in {"experts", "people"}:
        return explicit_scope  # type: ignore[return-value]
    if agent_type == "domain_expert":
        return "experts"
    if agent_type == "persona_inspired":
        return "people"
    return None


def resolve_paths(root_dir: Path | str = ".") -> RagPaths:
    root = Path(root_dir).expanduser().resolve()
    return RagPaths(
        root_dir=root,
        knowledge_dir=root / KNOWLEDGE_DIR,
        vector_db_dir=root / VECTOR_DB_DIR,
    )


def discover_corpora(root_dir: Path | str = ".") -> list[CorpusRef]:
    paths = resolve_paths(root_dir)
    corpora: list[CorpusRef] = []
    for scope in ("experts", "people"):
        scope_dir = paths.knowledge_dir / scope
        if not scope_dir.exists():
            continue
        for path in sorted(scope_dir.iterdir()):
            if path.is_dir():
                corpora.append(CorpusRef(corpus_id=path.name, knowledge_scope=scope))  # type: ignore[arg-type]
    return corpora


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class MockEmbeddingFunction:
    """Small deterministic embedding function accepted by Chroma."""

    def __init__(self, dimensions: int = MOCK_EMBEDDING_DIMENSIONS) -> None:
        self.dimensions = dimensions

    def __call__(self, input: list[str]) -> list[list[float]]:  # Chroma expects this signature.
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = tokenize(text) or [text.lower()]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class OpenAICompatibleEmbeddingFunction:
    """Placeholder for OpenAI/OpenRouter-compatible embeddings."""

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url

    def __call__(self, input: list[str]) -> list[list[float]]:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is required for {self.model} embeddings.")

        provider = "openrouter" if "openrouter.ai" in (self.base_url or "") else "openai"
        overrides = {self.api_key_env: api_key}
        if self.base_url:
            overrides[f"{provider.upper()}_BASE_URL"] = self.base_url
        return _embed_direct(provider, input, self.model, env_override=overrides)


class GeminiEmbeddingFunction:
    """Gemini embedding placeholder.

    The project can run without this path by using the mock embedding function or
    the keyword retriever. Fill this adapter in when a production Gemini embedding
    model is chosen.
    """

    def __init__(self, *, model: str, api_key_env: str = "GEMINI_API_KEY") -> None:
        self.model = model
        self.api_key_env = api_key_env

    def __call__(self, input: list[str]) -> list[list[float]]:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is required for Gemini embeddings.")
        return _embed_direct(
            "gemini",
            input,
            self.model,
            env_override={"GEMINI_API_KEY": api_key},
        )


def _embed_direct(
    provider: str,
    input: list[str],
    model: str,
    *,
    env_override: dict[str, str],
) -> list[list[float]]:
    """Call llm_gateway.direct.embed_direct, loaded lazily (gateway is optional)."""
    try:
        from llm_gateway.direct import embed_direct
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "This embedding provider requires the optional 'llm_gateway' package. "
            "Install it or set RAG_EMBEDDING_PROVIDER=mock/keyword."
        ) from exc
    return embed_direct(provider, input, model, env_override=env_override)


def create_embedding_function(provider: str | None = None):
    selected = (provider or DEFAULT_EMBEDDING_PROVIDER).lower()
    if selected == "keyword":
        return None
    if selected == "mock":
        return MockEmbeddingFunction()
    if selected == "openai":
        return OpenAICompatibleEmbeddingFunction(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key_env=os.getenv("OPENAI_EMBEDDING_API_KEY_ENV", "OPENAI_API_KEY"),
        )
    if selected == "openrouter":
        return OpenAICompatibleEmbeddingFunction(
            model=os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
            api_key_env=os.getenv("OPENROUTER_EMBEDDING_API_KEY_ENV", "OPENROUTER_API_KEY_1"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
    if selected == "gemini":
        return GeminiEmbeddingFunction(
            model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
            api_key_env=os.getenv("GEMINI_EMBEDDING_API_KEY_ENV", "GEMINI_API_KEY"),
        )
    raise ValueError(f"Unsupported RAG embedding provider: {provider}")
