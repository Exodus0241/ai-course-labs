"""Semantic memory backed by ChromaDB."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import chromadb

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional heavy dependency
    SentenceTransformer = None


class SemanticMemory:
    """Stores and retrieves semantically close user facts and messages."""

    def __init__(
        self,
        persist_directory: str = "./chroma_store",
        collection_name: str = "agent_memory",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.model = None

        # Real multilingual embeddings are preferred, but tests should also work
        # offline. If the model is unavailable, we use a deterministic fallback.
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name, local_files_only=True)
            except Exception:
                self.model = None

    def _fallback_embedding(self, text: str) -> list[float]:
        """Builds a small deterministic embedding without external downloads."""
        buckets = [0.0] * 16
        for index, char in enumerate(text.lower()):
            buckets[index % 16] += (ord(char) % 31) / 31.0

        norm = math.sqrt(sum(value * value for value in buckets)) or 1.0
        return [value / norm for value in buckets]

    def _embed(self, texts: Iterable[str]) -> list[list[float]]:
        """Embeds one or many strings for ChromaDB storage and search."""
        text_list = list(texts)
        if self.model is not None:
            return self.model.encode(text_list).tolist()
        return [self._fallback_embedding(text) for text in text_list]

    def add_memory(self, memory_id: str, text: str, metadata: dict | None = None) -> None:
        """Saves a new semantic memory record."""
        normalized_metadata = metadata or {"source": "unspecified"}
        self.collection.upsert(
            ids=[memory_id],
            documents=[text],
            metadatas=[normalized_metadata],
            embeddings=self._embed([text]),
        )

    def search(self, query: str, n_results: int = 3) -> list[str]:
        """Retrieves the closest stored memories for the current query."""
        results = self.collection.query(
            query_embeddings=self._embed([query]),
            n_results=n_results,
        )
        return results.get("documents", [[]])[0]
