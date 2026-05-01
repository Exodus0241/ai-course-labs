"""Domain rules for the symbolic subsystem."""

from __future__ import annotations

from typing import List

from .rule_engine import Rule, RulePriority


def build_default_rules(domain: str = "milk_packaging_quality") -> List[Rule]:
    """Create a reusable rule set for milk packaging clogging-defect control."""
    return [
        Rule(
            rule_id="CAP_ALIGNMENT_CRITICAL",
            name="Критическое смещение крышки",
            condition=lambda facts: facts.get("cap_alignment_error_mm", 0.0) > 2.0,
            conclusion="Обнаружен критический дефект позиционирования крышки.",
            priority=RulePriority.HIGH,
            description="Смещение крышки выше 2 мм может приводить к негерметичной закупорке.",
            domain=domain,
            recommendation="Проверить механизм позиционирования и перенастроить укупорочный узел.",
        ),
        Rule(
            rule_id="SEAL_INTEGRITY_LOW",
            name="Нарушение герметичности",
            condition=lambda facts: facts.get("seal_integrity_score", 1.0) < 0.8,
            conclusion="Есть риск нарушения герметичности молочной тары.",
            priority=RulePriority.HIGH,
            description="Низкая оценка герметичности указывает на вероятность дефекта закупорки.",
            domain=domain,
            recommendation="Проверить параметры прижима и состояние уплотняющих элементов.",
        ),
        Rule(
            rule_id="TORQUE_OUT_OF_RANGE",
            name="Момент закрутки вне допуска",
            condition=lambda facts: (
                facts.get("cap_torque_nm", 0.0) < 0.9
                or facts.get("cap_torque_nm", 0.0) > 1.4
            ),
            conclusion="Момент закрутки выходит за технологический диапазон.",
            priority=RulePriority.MEDIUM,
            description="Недостаточный или избыточный момент закрутки ухудшает качество закупорки.",
            domain=domain,
            recommendation="Скорректировать усилие закрутки и проверить исполнительный механизм.",
        ),
        Rule(
            rule_id="VISION_CONFIDENCE_LOW",
            name="Недостаточная уверенность машинного зрения",
            condition=lambda facts: facts.get("vision_confidence", 1.0) < 0.75,
            conclusion="Нейронный анализ изображения недостаточно надёжен для автоматического решения.",
            priority=RulePriority.CRITICAL,
            description="Низкая уверенность модели требует дополнительной проверки оператором или повторного снимка.",
            domain=domain,
            recommendation="Проверить освещение, фокус камеры и качество поступающих кадров.",
        ),
        Rule(
            rule_id="DEFECT_RATE_GROWTH",
            name="Рост процента дефектов",
            condition=lambda facts: facts.get("defect_rate_percent", 0.0) >= 3.0,
            conclusion="Процент дефектов закупорки превышает допустимый уровень.",
            priority=RulePriority.HIGH,
            description="Увеличение доли дефектной тары указывает на нестабильность производственной линии.",
            domain=domain,
            recommendation="Остановить линию для диагностики узла укупорки и подачи крышек.",
        ),
        Rule(
            rule_id="LINE_READY",
            name="Линия работает в штатном режиме",
            condition=lambda facts: (
                facts.get("cap_alignment_error_mm", 99.0) <= 1.0
                and facts.get("seal_integrity_score", 0.0) >= 0.95
                and 0.9 <= facts.get("cap_torque_nm", 0.0) <= 1.4
                and facts.get("vision_confidence", 0.0) >= 0.9
                and facts.get("defect_rate_percent", 100.0) < 1.0
            ),
            conclusion="Признаков дефектов закупорки не обнаружено, линия работает стабильно.",
            priority=RulePriority.LOW,
            description="Все ключевые показатели системы контроля находятся в допустимых пределах.",
            domain=domain,
            recommendation="Можно продолжать автоматический контроль в штатном режиме.",
        ),
        Rule(
            rule_id="CONVEYOR_SPEED_HIGH",
            name="Слишком высокая скорость конвейера",
            condition=lambda facts: facts.get("conveyor_speed_bpm", 0) > 180,
            conclusion="Скорость линии может провоцировать рост дефектов закупорки.",
            priority=RulePriority.MEDIUM,
            description="При превышении рекомендуемой скорости возрастает вероятность пропуска и перекоса крышки.",
            domain=domain,
            recommendation="Снизить скорость конвейера и повторить контроль качества.",
        ),
    ]
