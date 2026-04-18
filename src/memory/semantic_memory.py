"""Semantic memory backed by ChromaDB with an in-memory fallback."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import chromadb

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional heavy dependency
    SentenceTransformer = None


class _InMemoryCollection:
    def __init__(self) -> None:
        self._items: list[dict[str, object]] = []

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        for item_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            self._items = [item for item in self._items if item["id"] != item_id]
            self._items.append(
                {
                    "id": item_id,
                    "document": document,
                    "metadata": metadata,
                    "embedding": embedding,
                }
            )

    def query(self, query_embeddings: list[list[float]], n_results: int = 3) -> dict[str, list[list[str]]]:
        if not self._items:
            return {"documents": [[]]}
        query_embedding = query_embeddings[0]
        scored = sorted(
            self._items,
            key=lambda item: -sum(a * b for a, b in zip(query_embedding, item["embedding"])),
        )
        documents = [str(item["document"]) for item in scored[:n_results]]
        return {"documents": [documents]}


class SemanticMemory:
    """Stores and retrieves semantically close user facts and messages."""

    def __init__(
        self,
        persist_directory: str = "./chroma_store",
        collection_name: str = "week3_memory",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = None
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except BaseException:
            self.collection = _InMemoryCollection()
        self.model = None
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name, local_files_only=True)
            except Exception:
                self.model = None

    def _fallback_embedding(self, text: str) -> list[float]:
        buckets = [0.0] * 16
        for index, char in enumerate(text.lower()):
            buckets[index % 16] += (ord(char) % 31) / 31.0
        norm = math.sqrt(sum(value * value for value in buckets)) or 1.0
        return [value / norm for value in buckets]

    def _embed(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        if self.model is not None:
            return self.model.encode(text_list).tolist()
        return [self._fallback_embedding(text) for text in text_list]

    def add_memory(self, memory_id: str, text: str, metadata: dict | None = None) -> None:
        normalized_metadata = metadata or {"source": "unspecified"}
        self.collection.upsert(
            ids=[memory_id],
            documents=[text],
            metadatas=[normalized_metadata],
            embeddings=self._embed([text]),
        )

    def search(self, query: str, n_results: int = 3) -> list[str]:
        results = self.collection.query(
            query_embeddings=self._embed([query]),
            n_results=n_results,
        )
        return results.get("documents", [[]])[0]
