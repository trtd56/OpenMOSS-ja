import json

import pytest

from moss_ja_eval.local_llm import (
    DEFAULT_OLLAMA_MODEL,
    LocalLLMError,
    correct_transcript,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_correct_transcript_calls_ollama_and_preserves_order() -> None:
    captured = {}

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        content = {
            "corrections": [
                {"index": 1, "text": "二つ目。"},
                {"index": 0, "text": "今日は晴れです。"},
            ]
        }
        return FakeResponse({"message": {"content": json.dumps(content)}})

    result = correct_transcript(
        ["今日わ晴れです。", "二つめ。"],
        model="qwen3:8b",
        context="天気の会話",
        opener=opener,
    )

    assert result == ["今日は晴れです。", "二つ目。"]
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["body"]["model"] == "qwen3:8b"
    assert "天気の会話" in captured["body"]["messages"][0]["content"]


def test_correct_transcript_rejects_missing_utterance() -> None:
    def opener(_request, timeout):
        del timeout
        content = {"corrections": [{"index": 0, "text": "訂正後"}]}
        return FakeResponse({"message": {"content": json.dumps(content)}})

    with pytest.raises(LocalLLMError, match="欠落または追加"):
        correct_transcript(["一", "二"], model="test", opener=opener)


def test_correct_transcript_requires_model() -> None:
    with pytest.raises(LocalLLMError, match="モデル名"):
        correct_transcript(["本文"], model="  ")


def test_default_model_is_installed_qwen36_variant() -> None:
    assert DEFAULT_OLLAMA_MODEL == "qwen3.6:27b-mtp-q4_K_M"
