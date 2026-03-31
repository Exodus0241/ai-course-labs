"""Custom tool for milk packaging blockage defect analysis."""

from __future__ import annotations

from langchain.tools import BaseTool
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from pydantic import PrivateAttr


class PackagingDefectInput(BaseModel):
    """Expected parameters for the production defect helper tool."""

    defect_type: str = Field(..., description="Type of blockage or capping defect")
    production_stage: str = Field(..., description="Production stage where the defect is observed")
    symptoms: str = Field(..., description="Observed symptoms, signals or consequences")


class MilkClosureDefectTool(BaseTool):
    """Supports diploma work about milk packaging closure defect control."""

    name: str = "custom_tool"
    description: str = (
        "Анализирует дефекты закупорки молочной тары на производстве: "
        "формирует вероятные причины, точки контроля, корректирующие действия и KPI. "
        "Использует подключённую LLM для профильного анализа."
    )
    args_schema: type[BaseModel] = PackagingDefectInput
    _llm: BaseLanguageModel | None = PrivateAttr(default=None)

    def __init__(self, llm: BaseLanguageModel | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._llm = llm

    def _build_prompt(self, defect_type: str, production_stage: str, symptoms: str) -> str:
        """Builds a domain-specific prompt for the LLM."""
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
            "Сделай ответ прикладным, связанным с молочной линией розлива, контролем закупорки, "
            "качеством продукции и снижением брака."
        )

    def _normalize_llm_output(self, response: str | BaseMessage) -> str:
        """Converts LLM output to plain text."""
        if isinstance(response, BaseMessage):
            return str(response.content)
        return str(response)

    def _fallback_response(self, defect_type: str, production_stage: str, symptoms: str) -> str:
        """Returns a deterministic answer when the LLM is unavailable."""
        return (
            "LLM для custom_tool недоступна, поэтому показан резервный профильный ответ.\n\n"
            f"Тип дефекта: {defect_type}\n"
            f"Этап производства: {production_stage}\n"
            f"Наблюдаемые признаки: {symptoms}\n\n"
            "1. Вероятные причины: некорректная настройка укупорочного узла, износ механики, "
            "нестабильная подача крышек или тары.\n"
            "2. Контрольные параметры: момент затяжки, положение крышки, герметичность, "
            "скорость линии, частота повторения брака.\n"
            "3. Данные в системе: партия, смена, линия, фото дефекта, параметры оборудования, "
            "действия оператора.\n"
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
            return self._normalize_llm_output(response)
        except Exception:
            return self._fallback_response(defect_type, production_stage, symptoms)

    async def _arun(
        self,
        defect_type: str,
        production_stage: str,
        symptoms: str,
    ) -> str:
        return self._run(
            defect_type=defect_type,
            production_stage=production_stage,
            symptoms=symptoms,
        )
