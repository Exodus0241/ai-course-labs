"""Tests for Week 4 workflow helpers and exported workflows."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.api_client import build_yandex_classification_request, load_workflow_settings
from src.webhook_handler import WorkflowClient


ROOT = Path(__file__).resolve().parents[1]


def test_workflow_exports_are_valid_json() -> None:
    for name in ['basic_workflow.json', 'ai_classifier.json', 'specialty_workflow.json']:
        path = ROOT / 'workflows' / name
        data = json.loads(path.read_text())
        assert 'nodes' in data
        assert data['nodes']


def test_load_workflow_settings_has_expected_fields() -> None:
    settings = load_workflow_settings()
    assert hasattr(settings, 'base_url')
    assert hasattr(settings, 'application_webhook_path')
    assert hasattr(settings, 'defect_webhook_path')


def test_build_yandex_classification_request_contains_message() -> None:
    request = build_yandex_classification_request(
        'Тестовая заявка',
        folder_id='folder-id',
        iam_token='token',
        system_prompt='Классифицируй сообщение',
    )
    assert request.headers['Authorization'] == 'Bearer token'
    assert request.body['messages'][1]['text'] == 'Тестовая заявка'


@patch('src.webhook_handler.requests.post')
def test_send_application_success(mock_post: Mock) -> None:
    response = Mock()
    response.status_code = 200
    response.json.return_value = {'status': 'ok'}
    response.raise_for_status.return_value = None
    mock_post.return_value = response

    client = WorkflowClient(base_url='http://localhost:5678', webhook_path='application', secret='secret')
    result = client.send_application('Ошибка 403', 'user@example.com', 'high')

    assert result['success'] is True
    assert result['status_code'] == 200
    mock_post.assert_called_once()


@patch('src.webhook_handler.requests.post')
def test_send_defect_report_success(mock_post: Mock) -> None:
    response = Mock()
    response.status_code = 200
    response.json.return_value = {'status': 'received'}
    response.raise_for_status.return_value = None
    mock_post.return_value = response

    client = WorkflowClient(base_url='http://localhost:5678', webhook_path='application', secret='secret')
    result = client.send_defect_report(
        defect_type='неполная закупорка',
        production_stage='укупорочный модуль',
        symptoms='смещение крышки и протечка',
        line_id='L-01',
    )

    assert result['success'] is True
    assert result['response']['status'] == 'received'
