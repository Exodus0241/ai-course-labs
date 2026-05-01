"""Chunking strategies for splitting source documents."""

from __future__ import annotations

import logging
from typing import Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

LOGGER = logging.getLogger(__name__)


class ChunkingStrategy:
    """Factory for text splitting strategies."""

    @staticmethod
    def get_splitter(
        strategy: str = "recursive",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap должен быть меньше chunk_size")

        if strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
        if strategy == "character":
            return CharacterTextSplitter(
                separator="\n",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
        if strategy == "token":
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        raise ValueError(f"Неизвестная стратегия разбиения: {strategy}")

    @classmethod
    def split_documents(
        cls,
        documents: List[Document],
        strategy: str = "recursive",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> List[Document]:
        splitter = cls.get_splitter(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        LOGGER.info("Разбиение %s документов стратегией %s", len(documents), strategy)
        chunks = splitter.split_documents(documents)
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks, start=1):
            chunk.metadata["chunk_id"] = index
            chunk.metadata["total_chunks"] = total_chunks
        return chunks

    @staticmethod
    def get_statistics(chunks: List[Document]) -> Dict[str, float]:
        if not chunks:
            return {
                "count": 0,
                "min_size": 0,
                "max_size": 0,
                "avg_size": 0,
                "total_characters": 0,
            }

        sizes = [len(chunk.page_content) for chunk in chunks]
        return {
            "count": len(chunks),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "avg_size": round(sum(sizes) / len(sizes), 2),
            "total_characters": sum(sizes),
        }
