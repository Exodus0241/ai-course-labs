"""Knowledge base for symbolic facts and thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class KnowledgeFact:
    """Atomic fact stored in the knowledge base."""

    fact_id: str
    subject: str
    predicate: str
    object: Any
    confidence: float = 1.0
    source: str = "manual"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at,
        }


class KnowledgeBase:
    """Simple in-memory knowledge base indexed by subject and predicate."""

    def __init__(self) -> None:
        self.facts: Dict[str, KnowledgeFact] = {}

    def add_fact(self, fact: KnowledgeFact) -> str:
        self.facts[fact.fact_id] = fact
        return fact.fact_id

    def seed_default_facts(self, domain: str = "milk_packaging_quality") -> None:
        defaults = [
            KnowledgeFact(
                fact_id="KB001",
                subject=domain,
                predicate="max_cap_alignment_error_mm",
                object=1.0,
                source="lab6_policy",
            ),
            KnowledgeFact(
                fact_id="KB002",
                subject=domain,
                predicate="min_seal_integrity_score",
                object=0.95,
                source="lab6_policy",
            ),
            KnowledgeFact(
                fact_id="KB003",
                subject=domain,
                predicate="cap_torque_range_nm",
                object="0.9-1.4",
                source="lab6_policy",
            ),
            KnowledgeFact(
                fact_id="KB004",
                subject=domain,
                predicate="min_vision_confidence",
                object=0.90,
                source="lab6_policy",
            ),
            KnowledgeFact(
                fact_id="KB005",
                subject=domain,
                predicate="max_defect_rate_percent",
                object=1.0,
                source="lab6_policy",
            ),
            KnowledgeFact(
                fact_id="KB006",
                subject=domain,
                predicate="max_conveyor_speed_bpm",
                object=180,
                source="lab6_policy",
            ),
        ]
        for fact in defaults:
            self.add_fact(fact)

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
    ) -> List[KnowledgeFact]:
        results: List[KnowledgeFact] = []
        for fact in self.facts.values():
            if subject is not None and fact.subject != subject:
                continue
            if predicate is not None and fact.predicate != predicate:
                continue
            results.append(fact)
        return results

    def get_threshold_map(self, subject: str) -> Dict[str, Any]:
        return {fact.predicate: fact.object for fact in self.query(subject=subject)}

    def get_statistics(self) -> Dict[str, Any]:
        avg_confidence = (
            sum(fact.confidence for fact in self.facts.values()) / len(self.facts)
            if self.facts
            else 0.0
        )
        return {
            "total_facts": len(self.facts),
            "avg_confidence": round(avg_confidence, 3),
            "subjects": len({fact.subject for fact in self.facts.values()}),
        }
