"""Minimal YandexGPT client for synchronous completion requests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class YandexGPTLLM:
    """Small adapter with ``invoke`` method compatible with the RAG pipeline."""

    api_key: Optional[str] = None
    iam_token: Optional[str] = None
    folder_id: Optional[str] = None
    model_uri: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 800
    endpoint: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    timeout: int = 60

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("YANDEX_API_KEY")
        self.iam_token = self.iam_token or os.getenv("YANDEX_IAM_TOKEN")
        self.folder_id = self.folder_id or os.getenv("YANDEX_FOLDER_ID")

        if not self.folder_id:
            raise ValueError("Не задан YANDEX_FOLDER_ID.")
        if not (self.api_key or self.iam_token):
            raise ValueError("Нужно задать YANDEX_API_KEY или YANDEX_IAM_TOKEN.")

        self.model_uri = self.model_uri or f"gpt://{self.folder_id}/yandexgpt/latest"

    def invoke(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "x-folder-id": self.folder_id,
        }
        if self.api_key:
            headers["Authorization"] = f"Api-Key {self.api_key}"
        else:
            headers["Authorization"] = f"Bearer {self.iam_token}"

        payload = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": self.temperature,
                "maxTokens": str(self.max_tokens),
            },
            "messages": [
                {"role": "system", "text": "Ты полезный ассистент по производственной документации."},
                {"role": "user", "text": prompt},
            ],
        }

        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        result = data.get("result", data)
        alternatives = result.get("alternatives", [])
        if not alternatives:
            raise RuntimeError(f"YandexGPT вернул пустой ответ: {data}")

        message = alternatives[0].get("message", {})
        text = message.get("text", "").strip()
        if not text:
            raise RuntimeError(f"YandexGPT вернул ответ без текста: {data}")
        return text
