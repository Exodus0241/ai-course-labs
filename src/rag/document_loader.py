"""Utilities for loading source documents for the RAG pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)

LOGGER = logging.getLogger(__name__)


class DocumentLoader:
    """Loads documents from a directory and normalizes metadata."""

    def __init__(self, source_directory: str) -> None:
        self.source_directory = Path(source_directory)
        self.supported_extensions = {".pdf", ".txt", ".md", ".docx"}
        self.source_directory.mkdir(parents=True, exist_ok=True)

    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document and enrich its metadata."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Неподдерживаемый формат: {path.suffix}")
        if path.stat().st_size == 0:
            LOGGER.warning("Пропущен пустой файл: %s", path)
            return []

        documents = self._select_loader(path)
        normalized_documents: List[Document] = []
        for index, document in enumerate(documents, start=1):
            content = document.page_content.strip()
            if not content:
                continue
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "source": str(path),
                    "file_name": path.name,
                    "file_type": path.suffix.lower(),
                    "relative_source": str(path.relative_to(self.source_directory)),
                    "document_position": index,
                }
            )
            normalized_documents.append(Document(page_content=content, metadata=metadata))

        return normalized_documents

    def load_directory(self, pattern: str = "**/*") -> List[Document]:
        """Load every supported document from the directory."""
        all_documents: List[Document] = []
        for path in sorted(self.source_directory.glob(pattern)):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.supported_extensions:
                continue
            try:
                all_documents.extend(self.load_document(str(path)))
            except Exception as exc:  # pragma: no cover - defensive for user files
                LOGGER.warning("Пропущен файл %s: %s", path, exc)
        return all_documents

    def get_statistics(self) -> Dict[str, Any]:
        """Return a small summary of the source directory."""
        files_by_extension = {ext: 0 for ext in sorted(self.supported_extensions)}
        total_size_bytes = 0

        for path in self.source_directory.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.supported_extensions:
                continue
            files_by_extension[path.suffix.lower()] += 1
            total_size_bytes += path.stat().st_size

        return {
            "directory": str(self.source_directory),
            "files": files_by_extension,
            "total_files": sum(files_by_extension.values()),
            "total_size_bytes": total_size_bytes,
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 4),
        }

    def _select_loader(self, path: Path) -> List[Document]:
        extension = path.suffix.lower()
        if extension == ".txt":
            return TextLoader(str(path), encoding="utf-8").load()

        if extension == ".md":
            return TextLoader(str(path), encoding="utf-8").load()

        if extension == ".pdf":
            return PyPDFLoader(str(path)).load()

        if extension == ".docx":
            return Docx2txtLoader(str(path)).load()

        raise ValueError(f"Неподдерживаемый формат: {extension}")
