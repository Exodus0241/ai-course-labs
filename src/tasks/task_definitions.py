"""Task descriptions used by the crew and the lab report."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaskDefinition:
    """Portable representation of a crew task."""

    name: str
    description: str
    expected_output: str


def build_task_definitions(payload: dict[str, str]) -> list[TaskDefinition]:
    topic = payload["topic"]
    defect_type = payload["defect_type"]
    production_stage = payload["production_stage"]
    return [
        TaskDefinition(
            name="research_task",
            description=(
                f"Изучи тему '{topic}', собери сведения по дефекту '{defect_type}' "
                f"на этапе '{production_stage}', а затем передай выводы аналитику."
            ),
            expected_output="Краткий research brief с источниками, наблюдениями и гипотезами.",
        ),
        TaskDefinition(
            name="analysis_task",
            description=(
                "Проанализируй материалы исследователя, выдели причины дефекта, "
                "точки контроля и требования к информационной системе."
            ),
            expected_output="Структурированный аналитический отчёт с причинами и рекомендациями.",
        ),
        TaskDefinition(
            name="writing_task",
            description=(
                "Собери результаты предыдущих агентов и оформи финальный отчёт "
                "для демонстрации лабораторной работы."
            ),
            expected_output="Итоговая связная сводка с ролями агентов и рекомендациями.",
        ),
    ]
