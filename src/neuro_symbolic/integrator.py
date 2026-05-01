"""Integration logic between neural and symbolic outputs."""

from __future__ import annotations

from typing import Any, Dict

from src.symbolic.rule_engine import InferenceResult


class NeuroSymbolicIntegrator:
    """Fuse LLM conclusions with rule-based inference."""

    def __init__(self, neural_weight: float = 0.55, symbolic_weight: float = 0.45):
        total = neural_weight + symbolic_weight
        if total <= 0:
            raise ValueError("Weights must have a positive sum.")
        self.neural_weight = neural_weight / total
        self.symbolic_weight = symbolic_weight / total

    def integrate(
        self,
        neural_output: Dict[str, Any],
        symbolic_output: InferenceResult,
    ) -> Dict[str, Any]:
        neural_text = neural_output.get("analysis", "").strip()
        neural_label = neural_output.get("classification", {}).get(
            "predicted_category", ""
        )
        symbolic_confidence = 1.0 if symbolic_output.success else 0.2
        neural_confidence = neural_output.get("confidence", 0.75)
        final_confidence = round(
            neural_confidence * self.neural_weight
            + symbolic_confidence * self.symbolic_weight,
            3,
        )

        fragments = []
        if neural_label:
            fragments.append(f"Категория LLM: {neural_label}.")
        if neural_text:
            fragments.append(f"Нейронный анализ: {neural_text}")
        if symbolic_output.conclusions:
            fragments.append(
                "Символьные выводы: " + "; ".join(symbolic_output.conclusions)
            )
        else:
            fragments.append("Символьные выводы отсутствуют.")

        explanation = (
            "Интеграция основана на комбинации нейронного текста и сработавших правил. "
            f"Вес LLM={self.neural_weight:.2f}, вес правил={self.symbolic_weight:.2f}.\n"
            f"{symbolic_output.explanation}"
        )

        return {
            "decision": " ".join(fragments).strip(),
            "confidence": final_confidence,
            "explanation": explanation,
        }
