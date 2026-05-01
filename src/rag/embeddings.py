"""Embedding providers used by the RAG pipeline."""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

LOGGER = logging.getLogger(__name__)


class EmbeddingFactory:
    """Creates embedding models for Chroma."""

    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    @classmethod
    def create(
        cls,
        model_name: Optional[str] = None,
        device: str = "cpu",
    ) -> Embeddings:
        selected_model = model_name or cls.DEFAULT_MODEL
        LOGGER.info("Загрузка модели embeddings: %s", selected_model)
        return HuggingFaceEmbeddings(
            model_name=selected_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    @staticmethod
    def estimate_dimension(embeddings: Embeddings) -> int:
        sample = embeddings.embed_query("test")
        return len(sample)


def iter_preview(texts: Iterable[str], limit: int = 2) -> List[str]:
    """Return a tiny preview of source texts for logging or debugging."""
    preview: List[str] = []
    for index, item in enumerate(texts):
        if index >= limit:
            break
        preview.append(item[:120])
    return preview
