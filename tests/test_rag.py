from pathlib import Path

from langchain_core.documents import Document

from src.rag.chunking import ChunkingStrategy
from src.rag.document_loader import DocumentLoader
from src.rag.rag_pipeline import RAGPipeline
from src.rag.speciality_rag_pipeline import DairyPackagingRAGPipeline


class StubVectorStore:
    def __init__(self, results):
        self.results = results

    def search_with_scores(self, query: str, k: int = 4):
        return self.results[:k]

    def get_statistics(self):
        return {"collection_name": "stub", "total_documents": len(self.results)}


class StubLLM:
    def invoke(self, prompt: str):
        return "Сгенерированный ответ по контексту."


def test_document_loader_skips_empty_files_and_loads_text(tmp_path: Path):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "empty.md").write_text("", encoding="utf-8")
    (docs_dir / "process.txt").write_text(
        "Дефект закупорки возникает при смещении крышки и нестабильной подаче тары.",
        encoding="utf-8",
    )
    (docs_dir / "notes.md").write_text(
        "# Наблюдение\n\nСистема фиксирует инциденты по данным датчиков и оператора.",
        encoding="utf-8",
    )

    loader = DocumentLoader(str(docs_dir))
    documents = loader.load_directory()

    assert len(documents) == 2
    assert {doc.metadata["file_name"] for doc in documents} == {"process.txt", "notes.md"}


def test_chunking_adds_chunk_metadata():
    documents = [
        Document(
            page_content="Контроль дефектов закупорки " * 80,
            metadata={"source": "test.txt", "file_name": "test.txt"},
        )
    ]

    chunks = ChunkingStrategy.split_documents(documents, chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert chunks[0].metadata["chunk_id"] == 1
    assert chunks[-1].metadata["total_chunks"] == len(chunks)


def test_rag_pipeline_requires_llm():
    vectorstore = StubVectorStore([])
    try:
        RAGPipeline(vectorstore=vectorstore, llm=None, top_k=1)
    except ValueError as exc:
        assert "LLM" in str(exc)
    else:
        raise AssertionError("RAGPipeline должен требовать LLM")


def test_rag_pipeline_returns_llm_answer():
    vectorstore = StubVectorStore(
        [
            {
                "rank": 1,
                "content": "Система фиксирует дефекты закупорки по данным датчиков линии.",
                "metadata": {"file_name": "manual.md", "chunk_id": 1},
                "similarity_score": 0.91,
            }
        ]
    )
    pipeline = RAGPipeline(vectorstore=vectorstore, llm=StubLLM(), top_k=1)

    result = pipeline.query("Как фиксируются дефекты?")

    assert result["success"] is True
    assert result["answer"] == "Сгенерированный ответ по контексту."
    assert result["sources_count"] == 1


def test_speciality_pipeline_uses_llm_when_available():
    vectorstore = StubVectorStore(
        [
            {
                "rank": 1,
                "content": "Инцидент связывается с партией, линией и укупорочным узлом.",
                "metadata": {"file_name": "domain.md", "chunk_id": 2},
                "similarity_score": 0.88,
            }
        ]
    )
    pipeline = DairyPackagingRAGPipeline(vectorstore=vectorstore, llm=StubLLM(), top_k=1)

    result = pipeline.query("Какие сущности использует система?")

    assert result["success"] is True
    assert result["answer"] == "Сгенерированный ответ по контексту."
