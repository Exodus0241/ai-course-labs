from src.neural.llm_client import LLMClient
from src.neuro_symbolic.pipeline import NeuroSymbolicPipeline


def _build_response(text, input_tokens=42, output_tokens=18):
    return {
        "result": {
            "alternatives": [{"message": {"text": text}}],
            "usage": {
                "inputTextTokens": input_tokens,
                "completionTokens": output_tokens,
            },
        }
    }


def test_pipeline_returns_combined_result(monkeypatch):
    class DummyResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_post(url, headers, json, timeout):
        user_text = json["messages"][1]["text"]
        if "Категория:" in user_text:
            return DummyResponse(_build_response("критично"))
        return DummyResponse(
            _build_response(
                "По совокупности признаков линия демонстрирует высокий риск дефектов закупорки и требует немедленной проверки."
            )
        )

    monkeypatch.setattr("src.neural.llm_client.requests.post", fake_post)
    pipeline = NeuroSymbolicPipeline(llm=LLMClient(iam_token="token", folder_id="folder"))
    result = pipeline.process(
        {
            "query": "Критично: вырос процент дефектов закупорки и есть перекос крышки",
            "facts": {
                "cap_alignment_error_mm": 2.4,
                "seal_integrity_score": 0.72,
                "cap_torque_nm": 1.55,
                "vision_confidence": 0.68,
                "defect_rate_percent": 4.2,
                "conveyor_speed_bpm": 196,
            },
            "categories": ["норма", "предупреждение", "критично"],
        }
    )
    assert result["success"] is True
    assert result["confidence"] > 0
    assert "Символьные выводы" in result["final_decision"]
    assert result["symbolic_output"]["success"] is True
    assert len(result["symbolic_output"]["triggered_rules"]) >= 4
    assert (
        "Процент дефектов закупорки превышает допустимый уровень."
        in result["symbolic_output"]["conclusions"]
    )


def test_pipeline_can_report_positive_case(monkeypatch):
    class DummyResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_post(url, headers, json, timeout):
        user_text = json["messages"][1]["text"]
        if "Категория:" in user_text:
            return DummyResponse(_build_response("норма"))
        return DummyResponse(
            _build_response(
                "Параметры закупорки находятся в допустимых пределах, линия работает стабильно."
            )
        )

    monkeypatch.setattr("src.neural.llm_client.requests.post", fake_post)
    pipeline = NeuroSymbolicPipeline(llm=LLMClient(iam_token="token", folder_id="folder"))
    result = pipeline.process(
        {
            "query": "Система контроля закупорки работает стабильно",
            "facts": {
                "cap_alignment_error_mm": 0.4,
                "seal_integrity_score": 0.98,
                "cap_torque_nm": 1.1,
                "vision_confidence": 0.96,
                "defect_rate_percent": 0.4,
                "conveyor_speed_bpm": 150,
            },
            "categories": ["норма", "предупреждение", "критично"],
        }
    )
    conclusions = result["symbolic_output"]["conclusions"]
    assert "Признаков дефектов закупорки не обнаружено, линия работает стабильно." in conclusions


def test_llm_client_builds_request_and_parses_response(monkeypatch):
    captured = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "result": {
                    "alternatives": [
                        {"message": {"text": "Дефектов закупорки не обнаружено."}}
                    ],
                    "usage": {
                        "inputTextTokens": 21,
                        "completionTokens": 9,
                    },
                }
            }

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr("src.neural.llm_client.requests.post", fake_post)
    client = LLMClient(iam_token="token", folder_id="folder")
    result = client.generate("Проверь качество закупорки")

    assert captured["url"].endswith("/completion")
    assert captured["headers"]["Authorization"] == "Bearer token"
    assert captured["json"]["messages"][1]["text"] == "Проверь качество закупорки"
    assert result["text"] == "Дефектов закупорки не обнаружено."
    assert result["tokens_input"] == 21
    assert result["tokens_output"] == 9
