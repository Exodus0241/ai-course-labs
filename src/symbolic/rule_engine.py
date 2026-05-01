"""Rule engine for the symbolic part of the neuro-symbolic system."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class RulePriority(IntEnum):
    """Priority for rule execution and explanation ordering."""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class Rule:
    """Declarative IF-THEN rule."""

    rule_id: str
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    conclusion: str
    priority: RulePriority = RulePriority.MEDIUM
    description: str = ""
    domain: str = "general"
    recommendation: str = ""

    def evaluate(self, facts: Dict[str, Any]) -> bool:
        """Return True when the rule condition is satisfied."""
        try:
            return bool(self.condition(facts))
        except Exception as exc:
            logger.exception("Failed to evaluate rule %s: %s", self.rule_id, exc)
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize rule metadata."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "conclusion": self.conclusion,
            "priority": self.priority.name,
            "description": self.description,
            "domain": self.domain,
            "recommendation": self.recommendation,
        }


@dataclass
class InferenceResult:
    """Result of symbolic inference."""

    success: bool
    conclusions: List[str]
    triggered_rules: List[Rule]
    explanation: str
    facts_used: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        return {
            "success": self.success,
            "conclusions": self.conclusions,
            "triggered_rules": [rule.to_dict() for rule in self.triggered_rules],
            "explanation": self.explanation,
            "facts_used": self.facts_used,
            "timestamp": self.timestamp,
        }


class RuleEngine:
    """Evaluate a set of symbolic rules against input facts."""

    def __init__(self, rules: Optional[List[Rule]] = None) -> None:
        self.rules: List[Rule] = []
        self.facts: Dict[str, Any] = {}
        self.inference_history: List[InferenceResult] = []
        if rules:
            self.add_rules(rules)

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)
        self.rules.sort(key=lambda item: item.priority)

    def add_rules(self, rules: List[Rule]) -> None:
        for rule in rules:
            self.add_rule(rule)

    def set_facts(self, facts: Dict[str, Any]) -> None:
        self.facts = dict(facts)

    def clear_facts(self) -> None:
        self.facts = {}

    def infer(self, facts: Optional[Dict[str, Any]] = None) -> InferenceResult:
        """Run forward-chaining style evaluation over all rules."""
        if facts is not None:
            self.set_facts(facts)

        triggered_rules: List[Rule] = []
        conclusions: List[str] = []
        explanation_lines: List[str] = []

        for rule in self.rules:
            if rule.evaluate(self.facts):
                triggered_rules.append(rule)
                conclusions.append(rule.conclusion)
                line = (
                    f"[{rule.priority.name}] {rule.name}: {rule.description} "
                    f"=> {rule.conclusion}"
                )
                if rule.recommendation:
                    line += f" | Рекомендация: {rule.recommendation}"
                explanation_lines.append(line)

        explanation = (
            "\n".join(explanation_lines)
            if explanation_lines
            else "Ни одно символьное правило не сработало."
        )
        result = InferenceResult(
            success=bool(triggered_rules),
            conclusions=conclusions,
            triggered_rules=triggered_rules,
            explanation=explanation,
            facts_used=dict(self.facts),
        )
        self.inference_history.append(result)
        return result

    def export_rules(self) -> List[Dict[str, Any]]:
        return [rule.to_dict() for rule in self.rules]

    def get_rule_statistics(self) -> Dict[str, Any]:
        by_priority = {
            priority.name: sum(1 for rule in self.rules if rule.priority == priority)
            for priority in RulePriority
        }
        by_domain: Dict[str, int] = {}
        for rule in self.rules:
            by_domain[rule.domain] = by_domain.get(rule.domain, 0) + 1
        return {
            "total_rules": len(self.rules),
            "rules_by_priority": by_priority,
            "rules_by_domain": by_domain,
            "total_inferences": len(self.inference_history),
        }
