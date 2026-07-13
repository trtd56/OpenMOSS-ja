from __future__ import annotations

import json
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_MODEL = "qwen3.6:27b-mtp-q4_K_M"

DEFAULT_CORRECTION_PROMPT = """あなたは日本語の音声認識結果を校正する編集者です。
音声認識で生じた同音異義語、助詞、漢字、句読点、明らかな脱字だけを訂正してください。
意味の追加、要約、言い換え、フィラーの削除はしないでください。
各発話は独立した番号で識別されています。入力と同じ件数・indexで、本文だけを返してください。
出力は説明やMarkdownを含めず、必ず次の形のJSONにしてください:
{"corrections":[{"index":0,"text":"訂正後の本文"}]}"""


class LocalLLMError(RuntimeError):
    """Raised when correction through the local LLM cannot be completed safely."""


def _extract_corrections(payload: Any, expected_count: int) -> list[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("corrections"), list):
        raise LocalLLMError("Local LLMの応答に corrections 配列がありません。")

    corrections = payload["corrections"]
    by_index: dict[int, str] = {}
    for item in corrections:
        if not isinstance(item, dict):
            raise LocalLLMError("Local LLMの訂正結果が不正な形式です。")
        index, text = item.get("index"), item.get("text")
        if not isinstance(index, int) or isinstance(index, bool) or not isinstance(text, str):
            raise LocalLLMError("Local LLMの訂正結果に不正なindexまたはtextがあります。")
        if index in by_index:
            raise LocalLLMError(f"Local LLMの応答でindex {index}が重複しています。")
        by_index[index] = text.strip()

    expected = set(range(expected_count))
    if set(by_index) != expected:
        raise LocalLLMError("Local LLMが一部の発話を欠落または追加しました。訂正を適用しません。")
    if any(not by_index[index] for index in range(expected_count)):
        raise LocalLLMError("Local LLMが空の発話を返しました。訂正を適用しません。")
    return [by_index[index] for index in range(expected_count)]


def correct_transcript(
    texts: Sequence[str],
    *,
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    context: str = "",
    timeout: float = 120.0,
    opener: Callable[..., Any] = urlopen,
) -> list[str]:
    """Correct utterance texts with Ollama while preserving their exact indexes."""
    original = [str(text).strip() for text in texts]
    if not original:
        return []
    if not model.strip():
        raise LocalLLMError("Local LLMのモデル名を入力してください。")

    instructions = DEFAULT_CORRECTION_PROMPT
    if context.strip():
        instructions += f"\n\n校正時に考慮する用語・文脈:\n{context.strip()}"
    user_payload = {
        "utterances": [
            {"index": index, "text": text} for index, text in enumerate(original)
        ]
    }
    body = {
        "model": model.strip(),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ],
    }
    endpoint = f"{base_url.strip().rstrip('/')}/api/chat"
    request = Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with opener(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise LocalLLMError(f"Ollama APIがHTTP {exc.code}を返しました: {detail}") from exc
    except URLError as exc:
        raise LocalLLMError(
            f"Ollamaに接続できません（{endpoint}）。Ollamaが起動しているか確認してください。"
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalLLMError(f"Ollama APIの応答を読み取れませんでした: {exc}") from exc

    try:
        content = response_payload["message"]["content"]
        correction_payload = json.loads(content)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise LocalLLMError("Ollamaが有効なJSON訂正結果を返しませんでした。") from exc
    return _extract_corrections(correction_payload, len(original))
