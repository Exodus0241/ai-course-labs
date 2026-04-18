"""LLM assembly and runtime diagnostics for the Week 3 lab."""

from __future__ import annotations

import os
import socket

from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.memory.semantic_memory import SemanticMemory

try:
    from langchain_community.chat_models import ChatYandexGPT
except Exception:  # pragma: no cover - optional dependency
    ChatYandexGPT = None


class Week3AgentRuntime:
    """Builds the shared LLM runtime used by the Week 3 crew."""

    def __init__(self, llm: BaseLanguageModel | None = None) -> None:
        load_dotenv()
        load_dotenv(".env")
        self.semantic_memory = SemanticMemory(
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_store"),
            collection_name="week3_research_memory",
        )
        self.llm = llm or self._build_llm()

    def _build_llm(self) -> BaseLanguageModel:
        raw_api_key = os.getenv("YC_API_KEY")
        iam_token = os.getenv("YC_IAM_TOKEN") or os.getenv("YANDEX_IAM_TOKEN")
        folder_id = os.getenv("YC_FOLDER_ID") or os.getenv("YANDEX_FOLDER_ID")
        model_name = os.getenv("YC_MODEL_NAME", "yandexgpt-lite")
        model_version = os.getenv("YC_MODEL_VERSION", "latest")
        disable_logging = os.getenv("YC_DISABLE_LOGGING", "false").lower() == "true"
        api_key = raw_api_key

        if raw_api_key and raw_api_key.startswith("t1."):
            iam_token = raw_api_key
            api_key = None

        if folder_id and ChatYandexGPT is not None and (api_key or iam_token):
            llm_kwargs = {
                "folder_id": folder_id,
                "model_name": model_name,
                "model_version": model_version,
                "temperature": 0,
                "disable_request_logging": disable_logging,
            }
            if api_key:
                llm_kwargs["api_key"] = api_key
            else:
                llm_kwargs["iam_token"] = iam_token
            return ChatYandexGPT(**llm_kwargs)

        return FakeListChatModel(
            responses=[
                "1. Краткое описание дефекта.\n"
                "2. Вероятные причины.\n"
                "3. Какие параметры и датчики нужно контролировать.\n"
                "4. Какие данные должна хранить информационная система.\n"
                "5. Какие корректирующие действия рекомендовать оператору.\n"
                "6. Какие метрики и визуализации показать в дипломе и в интерфейсе системы."
            ]
        )

    def get_runtime_info(self) -> dict[str, str]:
        model_class = type(self.llm).__name__
        dns_status = "ok"
        try:
            socket.gethostbyname("llm.api.cloud.yandex.net")
        except Exception:
            dns_status = "dns_unavailable"

        return {
            "model_class": model_class,
            "uses_fallback": str(model_class == "FakeListChatModel").lower(),
            "dns_status": dns_status,
            "auth_mode": "iam_token"
            if (
                os.getenv("YC_IAM_TOKEN")
                or os.getenv("YANDEX_IAM_TOKEN")
                or (os.getenv("YC_API_KEY") or "").startswith("t1.")
            )
            else "api_key",
        }
