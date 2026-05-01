"""YandexGPT client for the neural component."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Thin wrapper around YandexGPT completion API."""

    def __init__(
        self,
        iam_token: Optional[str] = None,
        folder_id: Optional[str] = None,
        api_url: Optional[str] = None,
        model_uri: Optional[str] = None,
    ) -> None:
        self.iam_token = iam_token or os.getenv("YANDEX_IAM_TOKEN")
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        if not self.iam_token or not self.folder_id:
            raise ValueError(
                "YANDEX_IAM_TOKEN and YANDEX_FOLDER_ID must be configured."
            )
        self.api_url = (
            api_url
            or "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        )
        self.model_uri = (
            model_uri or f"gpt://{self.folder_id}/yandexgpt/latest"
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.iam_token}",
            "Content-Type": "application/json",
            "x-folder-id": self.folder_id,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: str = "Ты аналитический помощник.",
        temperature: float = 0.2,
        max_tokens: int = 400,
    ) -> Dict[str, Any]:
        payload = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens),
            },
            "messages": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": prompt},
            ],
        }
        response = requests.post(
            self.api_url,
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        raw = response.json()
        result = raw["result"]
        alternative = result["alternatives"][0]["message"]["text"]
        usage = result.get("usage", {})
        return {
            "success": True,
            "text": alternative.strip(),
            "tokens_input": usage.get("inputTextTokens", 0),
            "tokens_output": usage.get("completionTokens", 0),
            "model_uri": self.model_uri,
            "timestamp": datetime.now().isoformat(),
            "raw": raw,
        }

    def classify(
        self,
        text: str,
        categories: List[str],
        prompt_template: Optional[str] = None,
    ) -> Dict[str, Any]:
        template = prompt_template or (
            "Классифицируй текст в одну категорию из списка: {categories}. "
            "Ответь только названием категории.\n"
            "Текст: {text}\n"
            "Категория:"
        )
        prompt = template.format(categories=", ".join(categories), text=text)
        response = self.generate(
            prompt=prompt,
            system_prompt="Ты работаешь как точный классификатор.",
            temperature=0.0,
            max_tokens=50,
        )
        return {
            "success": True,
            "predicted_category": response["text"],
            "llm_response": response,
        }
