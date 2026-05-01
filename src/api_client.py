"""Helpers for building YandexGPT requests and loading workflow settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv


@dataclass
class WorkflowSettings:
    """Configuration required by local scripts and tests."""

    base_url: str
    application_webhook_path: str
    defect_webhook_path: str
    webhook_secret: str
    folder_id: str
    iam_token: str
    email_to: str


@dataclass
class YandexRequestParts:
    """Prepared request pieces for direct HTTP calls to YandexGPT."""

    url: str
    headers: dict[str, str]
    body: dict[str, Any]


YDEX_COMPLETION_URL = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion'


def load_workflow_settings() -> WorkflowSettings:
    """Loads local configuration from .env files when they exist."""
    load_dotenv()
    load_dotenv('.env')
    load_dotenv('docker/.env')
    return WorkflowSettings(
        base_url=os.getenv('WEBHOOK_URL', 'http://localhost:5678/').rstrip('/'),
        application_webhook_path=os.getenv('APPLICATION_WEBHOOK_PATH', 'application'),
        defect_webhook_path=os.getenv('DEFECT_WEBHOOK_PATH', 'milk-defect'),
        webhook_secret=os.getenv('WEBHOOK_SECRET', ''),
        folder_id=os.getenv('YANDEX_FOLDER_ID', ''),
        iam_token=os.getenv('YANDEX_IAM_TOKEN', ''),
        email_to=os.getenv('EMAIL_TO', ''),
    )


def build_yandex_headers(iam_token: str, folder_id: str) -> dict[str, str]:
    """Builds HTTP headers for YandexGPT REST calls."""
    return {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {iam_token}',
        'x-folder-id': folder_id,
    }


def build_yandex_classification_request(
    message: str,
    *,
    folder_id: str,
    iam_token: str,
    system_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 100,
) -> YandexRequestParts:
    """Builds a ready-to-send YandexGPT classification request."""
    body = {
        'modelUri': f'gpt://{folder_id}/yandexGPT/latest',
        'completionOptions': {
            'temperature': temperature,
            'maxTokens': max_tokens,
        },
        'messages': [
            {
                'role': 'system',
                'text': system_prompt,
            },
            {
                'role': 'user',
                'text': message,
            },
        ],
    }
    return YandexRequestParts(
        url=YDEX_COMPLETION_URL,
        headers=build_yandex_headers(iam_token=iam_token, folder_id=folder_id),
        body=body,
    )
