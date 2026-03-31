"""Core agent assembly for the Week 2 lab."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from uuid import uuid4

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, AgentType, initialize_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel

from week2.src.guardrails.input_validator import InputValidator
from week2.src.memory.semantic_memory import SemanticMemory
from week2.src.memory.working_memory import WorkingMemory
from week2.src.tools.calc_tool import CalculateTool
from week2.src.tools.custom_tool import MilkClosureDefectTool
from week2.src.tools.date_tool import MonthsSinceDateTool
from week2.src.tools.search_tool import SearchWebTool

try:
    from langchain_community.chat_models import ChatYandexGPT
except Exception:  # pragma: no cover - optional dependency
    ChatYandexGPT = None


@dataclass
class AgentResponse:
    """Response envelope returned by the application layer."""

    answer: str
    used_context: str
    recalled_memories: list[str]


class LabAgent:
    """MVP AI agent with tools, memory and input guardrails."""

    def __init__(
        self,
        llm=None,
        semantic_memory: SemanticMemory | None = None,
        working_memory: WorkingMemory | None = None,
        validator: InputValidator | None = None,
    ) -> None:
        load_dotenv()
        self.validator = validator or InputValidator()
        self.working_memory = working_memory or WorkingMemory()
        self.semantic_memory = semantic_memory or SemanticMemory(
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
        )
        self.llm = llm or self._build_llm()
        self.tools = [
            SearchWebTool(base_url=os.getenv("SEARCH_API_URL", "https://duckduckgo.com/html/")),
            CalculateTool(),
            MonthsSinceDateTool(),
            MilkClosureDefectTool(llm=self.llm),
        ]
        self.agent_executor: AgentExecutor | None = None

    def _build_llm(self) -> BaseLanguageModel:
        """Creates YandexGPT when credentials are available or a local fallback for demos."""
        raw_api_key = os.getenv("YC_API_KEY")
        iam_token = os.getenv("YC_IAM_TOKEN") or os.getenv("YANDEX_IAM_TOKEN")
        folder_id = os.getenv("YC_FOLDER_ID") or os.getenv("YANDEX_FOLDER_ID")
        model_name = os.getenv("YC_MODEL_NAME", "yandexgpt-lite")
        model_version = os.getenv("YC_MODEL_VERSION", "latest")
        disable_logging = os.getenv("YC_DISABLE_LOGGING", "false").lower() == "true"
        api_key = raw_api_key

        # In student setups IAM tokens are often pasted into the API key field.
        # Tokens that start with "t1." should be passed as iam_token instead.
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

        # Offline fallback keeps the lab runnable even without external services.
        return FakeListChatModel(
            responses=[
                "Thought: I should answer directly.\nFinal Answer: "
                "Я готов помочь с лабораторной работой и использовать подключённые инструменты."
            ]
        )

    def _build_executor(self) -> AgentExecutor:
        """Initialises a structured agent that can work with multi-argument tools."""
        agent_type = (
            AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
            if isinstance(self.llm, BaseChatModel)
            else AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=agent_type,
            verbose=True,
            handle_parsing_errors=True,
        )

    def ask(self, user_input: str) -> AgentResponse:
        """Validates input, restores context, queries the agent and persists memory."""
        validation = self.validator.validate(user_input)
        if not validation.is_valid:
            return AgentResponse(
                answer=validation.reason,
                used_context=self.working_memory.get_context(),
                recalled_memories=[],
            )

        recalled = self.semantic_memory.search(validation.cleaned_text, n_results=3)
        context = self.working_memory.get_context()
        prompt = self._compose_prompt(validation.cleaned_text, context, recalled)
        if self.agent_executor is None:
            self.agent_executor = self._build_executor()
        try:
            result = self.agent_executor.invoke({"input": prompt})
            answer = result["output"] if isinstance(result, dict) else str(result)
        except Exception as error:
            if self._looks_like_dns_error(error):
                answer = (
                    "YandexGPT подключён, но выполнить запрос не удалось из-за сетевой ошибки DNS. "
                    "Среда не может разрешить адрес `llm.api.cloud.yandex.net`. "
                    "Проверьте интернет, DNS или ограничения сети в Jupyter."
                )
            else:
                raise

        self.working_memory.add(f"Пользователь: {validation.cleaned_text}")
        self.working_memory.add(f"Агент: {answer}")
        self.semantic_memory.add_memory(str(uuid4()), validation.cleaned_text, {"source": "user"})
        self.semantic_memory.add_memory(str(uuid4()), answer, {"source": "agent"})

        return AgentResponse(answer=answer, used_context=context, recalled_memories=recalled)

    def _compose_prompt(self, question: str, context: str, recalled: list[str]) -> str:
        """Merges the current question with working and semantic memory."""
        context_block = context or "История диалога пока пуста."
        memories_block = "\n".join(recalled) if recalled else "Релевантных воспоминаний нет."
        return (
            "Ты AI-агент для поддержки учебных, инженерных и дипломных задач.\n"
            "Специализация агента: информационная система контроля дефектов закупорки "
            "молочной тары на производстве.\n"
            "Если запрос требует актуальных данных, сначала используй search_web.\n"
            "Если нужно найти, сколько месяцев прошло с даты релиза или обновления, "
            "сначала найди точную дату, а затем используй months_since_date.\n"
            "Calculate используй только для чистой арифметики и формул. "
            "Никогда не передавай в calculate текст вроде 'months since last update'.\n"
            "Для анализа производственных дефектов используй custom_tool.\n\n"
            f"Краткосрочная память:\n{context_block}\n\n"
            f"Семантическая память:\n{memories_block}\n\n"
            f"Текущий запрос пользователя:\n{question}"
        )

    def _looks_like_dns_error(self, error: Exception) -> bool:
        """Detects common Yandex Cloud DNS and transport failures."""
        error_text = str(error).lower()
        markers = [
            "could not contact dns servers",
            "address lookup failed",
            "errors resolving llm.api.cloud.yandex.net",
            "statuscode.unavailable",
            "hostname lookup error",
        ]
        return any(marker in error_text for marker in markers)

    def get_runtime_info(self) -> dict[str, str]:
        """Returns a short runtime diagnostics snapshot for notebook debugging."""
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
            if (os.getenv("YC_IAM_TOKEN") or os.getenv("YANDEX_IAM_TOKEN") or (os.getenv("YC_API_KEY") or "").startswith("t1."))
            else "api_key",
        }


def build_demo_agent() -> LabAgent:
    """Factory for notebook and CLI demos."""
    return LabAgent()
