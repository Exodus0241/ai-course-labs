"""Custom tool for milk packaging blockage defect analysis."""

from __future__ import annotations

import re

from langchain.tools import BaseTool
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, PrivateAttr


class PackagingDefectInput(BaseModel):
    defect_type: str = Field(..., description="Type of blockage or capping defect")
    production_stage: str = Field(..., description="Production stage where the defect is observed")
    symptoms: str = Field(..., description="Observed symptoms, signals or consequences")


class MilkClosureDefectTool(BaseTool):
    name: str = "custom_tool"
    description: str = (
        "Анализирует дефекты закупорки молочной тары на производстве: формирует вероятные причины, "
        "точки контроля, корректирующие действия и KPI. Использует подключённую LLM для профильного анализа."
    )
    args_schema: type[BaseModel] = PackagingDefectInput
    _llm: BaseLanguageModel | None = PrivateAttr(default=None)

    def __init__(self, llm: BaseLanguageModel | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._llm = llm

    def _build_prompt(self, defect_type: str, production_stage: str, symptoms: str) -> str:
        return (
            "Ты технологический помощник для дипломной работы по теме "
            "'Информационная система контроля дефектов закупорки молочной тары на производстве'.\n"
            "Подготовь практический анализ дефекта для будущей информационной системы.\n"
            "Ответ дай на русском языке и строго в 6 блоках:\n"
            "1. Краткое описание дефекта.\n"
            "2. Вероятные причины.\n"
            "3. Какие параметры и датчики нужно контролировать.\n"
            "4. Какие данные должна хранить информационная система.\n"
            "5. Какие корректирующие действия рекомендовать оператору.\n"
            "6. Какие метрики и визуализации показать в дипломе и в интерфейсе системы.\n\n"
            f"Тип дефекта: {defect_type}\n"
            f"Этап производства: {production_stage}\n"
            f"Симптомы: {symptoms}\n\n"
            "Сделай ответ прикладным, связанным с молочной линией розлива, контролем закупорки, качеством продукции и снижением брака."
        )

    def _normalize_llm_output(self, response: str | BaseMessage) -> str:
        raw_text = str(response.content) if isinstance(response, BaseMessage) else str(response)
        normalized = raw_text.replace("Final Answer:", "").strip()
        normalized = re.sub(r"Thought:.*?(?=\n|$)", "", normalized, flags=re.IGNORECASE).strip()
        return normalized

    def _is_structured_response(self, text: str) -> bool:
        lowered = text.lower()
        return all(marker in lowered for marker in ["1.", "2.", "причин", "контрол"])

    def _fallback_response(self, defect_type: str, production_stage: str, symptoms: str, error: str | None = None) -> str:
        prefix = "LLM для custom_tool недоступна, поэтому показан резервный профильный ответ."
        if error:
            prefix += f"\nПричина: {error}"
        return (
            f"{prefix}\n\n"
            f"Тип дефекта: {defect_type}\n"
            f"Этап производства: {production_stage}\n"
            f"Наблюдаемые признаки: {symptoms}\n\n"
            "1. Вероятные причины: некорректная настройка укупорочного узла, износ механики, нестабильная подача крышек или тары.\n"
            "2. Контрольные параметры: момент затяжки, положение крышки, герметичность, скорость линии, частота повторения брака.\n"
            "3. Данные в системе: партия, смена, линия, фото дефекта, параметры оборудования, действия оператора.\n"
            "4. Корректирующие действия: остановка участка, перенастройка, повторный контроль партии.\n"
            "5. KPI: процент дефекта, MTTR, повторяемость причины, динамика по сменам.\n"
            "6. Для диплома: карточка инцидента, дашборд брака и схема потока данных."
        )

    def _run(self, defect_type: str, production_stage: str, symptoms: str) -> str:
        if self._llm is None:
            return self._fallback_response(defect_type, production_stage, symptoms)
        prompt = self._build_prompt(defect_type, production_stage, symptoms)
        try:
            response = self._llm.invoke(prompt)
            normalized = self._normalize_llm_output(response)
            if self._is_structured_response(normalized):
                return normalized
            return self._fallback_response(
                defect_type,
                production_stage,
                symptoms,
                error="модель вернула ответ вне ожидаемой структуры",
            )
        except Exception as error:
            return self._fallback_response(defect_type, production_stage, symptoms, error=str(error))

    async def _arun(self, defect_type: str, production_stage: str, symptoms: str) -> str:
        return self._run(
            defect_type=defect_type,
            production_stage=production_stage,
            symptoms=symptoms,
        )
