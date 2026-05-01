# -*- coding: utf-8 -*-
"""Обработчик webhook для тестирования workflow.

Лабораторная работа №4
Дисциплина: Искусственный интеллект
Автор: [ФИО]
Группа: [НОМЕР ГРУППЫ]
Дата: 2026
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

import requests

from src.api_client import load_workflow_settings


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class WorkflowClient:
    """Клиент для взаимодействия с n8n workflow."""

    def __init__(
        self,
        base_url: str = 'http://localhost:5678',
        webhook_path: str = 'application',
        secret: Optional[str] = None,
    ) -> None:
        settings = load_workflow_settings()
        resolved_base_url = (base_url or settings.base_url).rstrip('/')
        resolved_secret = secret if secret is not None else settings.webhook_secret
        self.base_url = resolved_base_url
        self.webhook_path = webhook_path
        self.secret = resolved_secret
        self.webhook_url = f'{self.base_url}/webhook/{self.webhook_path}'
        logger.info('WorkflowClient инициализирован: %s', self.webhook_url)

    def _headers(self) -> dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if self.secret:
            headers['X-Webhook-Secret'] = self.secret
        return headers

    def send_application(
        self,
        message: str,
        contact: str,
        priority: str = 'normal',
    ) -> dict[str, Any]:
        """Отправка заявки в workflow."""
        payload = {
            'message': message,
            'contact': contact,
            'priority': priority,
            'timestamp': datetime.now().isoformat(),
        }
        logger.info('Отправка заявки: %s', message[:50])
        try:
            response = requests.post(
                self.webhook_url,
                headers=self._headers(),
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            return {
                'success': True,
                'response': self._safe_json(response),
                'status_code': response.status_code,
            }
        except requests.exceptions.RequestException as error:
            logger.error('Ошибка отправки: %s', error)
            return {
                'success': False,
                'error': str(error),
                'status_code': None,
            }

    def send_defect_report(
        self,
        defect_type: str,
        production_stage: str,
        symptoms: str,
        line_id: str,
    ) -> dict[str, Any]:
        """Отправка сообщения в специализированный workflow по дефектам закупорки."""
        defect_path = load_workflow_settings().defect_webhook_path
        webhook_url = f'{self.base_url}/webhook/{defect_path}'
        payload = {
            'defect_type': defect_type,
            'production_stage': production_stage,
            'symptoms': symptoms,
            'line_id': line_id,
            'timestamp': datetime.now().isoformat(),
        }
        logger.info('Отправка сообщения о дефекте: %s', defect_type)
        try:
            response = requests.post(
                webhook_url,
                headers=self._headers(),
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            return {
                'success': True,
                'response': self._safe_json(response),
                'status_code': response.status_code,
            }
        except requests.exceptions.RequestException as error:
            logger.error('Ошибка отправки дефекта: %s', error)
            return {
                'success': False,
                'error': str(error),
                'status_code': None,
            }

    def check_workflow_status(self) -> dict[str, Any]:
        """Проверка доступности n8n."""
        try:
            response = requests.get(f'{self.base_url}/healthz', timeout=5)
            return {
                'available': response.status_code == 200,
                'status_code': response.status_code,
            }
        except Exception as error:  # pragma: no cover - network dependent
            return {
                'available': False,
                'error': str(error),
            }

    def _safe_json(self, response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return response.text


if __name__ == '__main__':
    print('=' * 80)
    print('ЛАБОРАТОРНАЯ РАБОТА №4')
    print('Тестирование workflow automation')
    print('=' * 80)

    settings = load_workflow_settings()
    client = WorkflowClient(
        base_url=settings.base_url,
        webhook_path=settings.application_webhook_path,
    )

    print('\nПроверка доступности n8n...')
    status = client.check_workflow_status()
    if status.get('available'):
        print('✅ n8n доступен')
    else:
        print(f"❌ n8n недоступен: {status.get('error')}")
        raise SystemExit(1)

    print('\n' + '=' * 80)
    print('ТЕСТОВАЯ ЗАЯВКА')
    print('=' * 80)
    result = client.send_application(
        message='Не работает вход в систему, ошибка 403',
        contact='user@example.com',
        priority='high',
    )
    if result['success']:
        print('✅ Заявка отправлена успешно')
        print(f"Статус код: {result['status_code']}")
        print(f"Ответ: {result['response']}")
    else:
        print(f"❌ Ошибка: {result['error']}")
    print('=' * 80)
