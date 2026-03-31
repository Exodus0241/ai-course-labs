"""Safe calculator tool."""

from __future__ import annotations

import ast
import operator
from typing import Any

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    """Expected calculator arguments."""

    expression: str = Field(..., description="Arithmetic expression to evaluate")
    precision: int = Field(4, description="Number of decimal digits")


class SafeCalculator:
    """Evaluates arithmetic expressions using a restricted AST."""

    allowed_operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.Mod: operator.mod,
    }

    def evaluate(self, expression: str) -> float:
        """Parses and computes a safe arithmetic expression."""
        node = ast.parse(expression, mode="eval").body
        return float(self._eval_node(node))

    def _eval_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in self.allowed_operations:
            return self.allowed_operations[type(node.op)](self._eval_node(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in self.allowed_operations:
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.allowed_operations[type(node.op)](left, right)
        raise ValueError("Допустимы только безопасные арифметические выражения.")


class CalculateTool(BaseTool):
    """LangChain tool wrapper for the safe calculator."""

    name: str = "calculate"
    description: str = (
        "Выполняет только арифметические вычисления. "
        "Передавай только формулу или выражение, например: 12/3, (2+5)*4, 100*0.18. "
        "Не передавай текстовые запросы, даты или описания на естественном языке."
    )
    args_schema: type[BaseModel] = CalculatorInput

    def _run(self, expression: str, precision: int = 4) -> str:
        calculator = SafeCalculator()
        try:
            result = calculator.evaluate(expression)
            return str(round(result, precision))
        except Exception as error:
            return (
                "Ошибка калькулятора: требуется арифметическое выражение. "
                f"Детали: {error}"
            )

    async def _arun(self, expression: str, precision: int = 4) -> str:
        return self._run(expression=expression, precision=precision)
