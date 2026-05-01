"""Wrapper around ChromaDB used by the RAG pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.rag.embeddings import EmbeddingFactory

LOGGER = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages a Chroma collection with pluggable embeddings."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "rag_documents",
        embedding_model: Optional[str] = None,
        embeddings=None,
    ) -> None:
        self.persist_directory = str(Path(persist_directory))
        self.collection_name = collection_name
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.embeddings = embeddings or EmbeddingFactory.create(
            model_name=embedding_model,
        )
        self.embedding_model_name = (
            embedding_model or getattr(self.embeddings, "model_name", EmbeddingFactory.DEFAULT_MODEL)
        )

        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def add_documents(self, documents: List[Document], batch_size: int = 100) -> Dict[str, Any]:
        if not documents:
            return {
                "documents_added": 0,
                "collection_name": self.collection_name,
                "total_documents": self._count_documents(),
            }

        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            self.vectorstore.add_documents(batch)

        return {
            "documents_added": len(documents),
            "collection_name": self.collection_name,
            "total_documents": self._count_documents(),
        }

    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        results = self.vectorstore.similarity_search(query=query, k=k, filter=filter_metadata)
        return [
            {
                "rank": index,
                "content": document.page_content,
                "metadata": document.metadata,
                "similarity_score": None,
            }
            for index, document in enumerate(results, start=1)
        ]

    def search_with_scores(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        docs_and_scores = self.vectorstore.similarity_search_with_score(query=query, k=k)
        ranked_results: List[Dict[str, Any]] = []
        for index, (document, score) in enumerate(docs_and_scores, start=1):
            ranked_results.append(
                {
                    "rank": index,
                    "content": document.page_content,
                    "metadata": document.metadata,
                    "similarity_score": round(1 / (1 + float(score)), 4),
                }
            )
        return ranked_results

    def clear(self) -> bool:
        try:
            ids = self.vectorstore.get()["ids"]
            if ids:
                self.vectorstore.delete(ids=ids)
            return True
        except Exception as exc:  # pragma: no cover - chroma edge cases
            LOGGER.error("Ошибка очистки коллекции %s: %s", self.collection_name, exc)
            return False

    def delete_collection(self) -> bool:
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            client.delete_collection(self.collection_name)
            return True
        except Exception as exc:  # pragma: no cover - chroma edge cases
            LOGGER.error("Ошибка удаления коллекции %s: %s", self.collection_name, exc)
            return False

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": EmbeddingFactory.estimate_dimension(self.embeddings),
            "total_documents": self._count_documents(),
        }

    def _count_documents(self) -> int:
        collection = getattr(self.vectorstore, "_collection", None)
        if collection is None:
            return 0
        return int(collection.count())
