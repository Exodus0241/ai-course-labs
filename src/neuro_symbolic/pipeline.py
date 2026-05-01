"""Main neuro-symbolic pipeline."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Optional

from src.neural.llm_client import LLMClient
from src.neuro_symbolic.integrator import NeuroSymbolicIntegrator
from src.symbolic.knowledge_base import KnowledgeBase
from src.symbolic.rule_engine import RuleEngine
from src.symbolic.rules import build_default_rules


class NeuroSymbolicPipeline:
    """Run neural analysis, symbolic inference and integration."""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        rule_engine: Optional[RuleEngine] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
        integrator: Optional[NeuroSymbolicIntegrator] = None,
        domain: str = "milk_packaging_quality",
    ) -> None:
        self.domain = domain
        self.llm = llm or LLMClient()
        self.rule_engine = rule_engine or RuleEngine(build_default_rules(domain=domain))
        self.knowledge_base = knowledge_base or KnowledgeBase()
        if not self.knowledge_base.facts:
            self.knowledge_base.seed_default_facts(domain=domain)
        self.integrator = integrator or NeuroSymbolicIntegrator()

    def process(
        self,
        input_data: Dict[str, Any],
        include_explanation: bool = True,
    ) -> Dict[str, Any]:
        started = time.time()
        facts = input_data.get("facts", {})
        query = input_data.get("query", "")
        categories = input_data.get(
            "categories",
            ["норма", "предупреждение", "критично"],
        )

        neural_output = self._run_neural_stage(query=query, facts=facts, categories=categories)
        symbolic_output = self.rule_engine.infer(facts)
        integrated = self.integrator.integrate(neural_output, symbolic_output)

        return {
            "success": True,
            "domain": self.domain,
            "input": input_data,
            "knowledge_context": self.knowledge_base.get_threshold_map(self.domain),
            "neural_output": neural_output,
            "symbolic_output": symbolic_output.to_dict(),
            "final_decision": integrated["decision"],
            "confidence": integrated["confidence"],
            "explanation": integrated["explanation"] if include_explanation else "",
            "timestamp": datetime.now().isoformat(),
            "execution_time": round(time.time() - started, 3),
        }

    def _run_neural_stage(
        self,
        query: str,
        facts: Dict[str, Any],
        categories: list[str],
    ) -> Dict[str, Any]:
        knowledge_context = self.knowledge_base.get_threshold_map(self.domain)
        classification = self.llm.classify(query or str(facts), categories)
        prompt = (
            "Ты анализируешь данные системы контроля дефектов закупорки молочной тары.\n"
            f"Предметная область: {self.domain}\n"
            f"Запрос: {query}\n"
            f"Факты: {facts}\n"
            f"Пороговые знания: {knowledge_context}\n"
            "Сделай краткий аналитический вывод о состоянии линии, качестве закупорки и риске появления дефектов."
        )
        analysis = self.llm.generate(
            prompt=prompt,
            system_prompt=(
                "Ты эксперт по компьютерному зрению и контролю качества на молочном производстве. "
                "Пиши кратко, по делу и на русском языке."
            ),
            temperature=0.2,
            max_tokens=250,
        )
        return {
            "classification": classification,
            "analysis": analysis["text"],
            "confidence": 0.85,
            "tokens_input": analysis["tokens_input"],
            "tokens_output": analysis["tokens_output"],
        }

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "rules": self.rule_engine.get_rule_statistics(),
            "knowledge_base": self.knowledge_base.get_statistics(),
            "weights": {
                "neural": self.integrator.neural_weight,
                "symbolic": self.integrator.symbolic_weight,
            },
        }
