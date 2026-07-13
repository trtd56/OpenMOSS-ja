from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable

from .infer import JAPANESE_PROMPT, _load_runtime

MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"

_runtime: tuple[Any, Any, Any, Any] | None = None
_runtime_lock = threading.Lock()


def format_timestamp(seconds: float) -> str:
    """Convert seconds to a stable HH:MM:SS.mmm timestamp."""
    total_ms = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def format_transcript(segments: Iterable[Any]) -> str:
    """Render parsed MOSS segments in the app's downloadable text format."""
    lines: list[str] = []
    for segment in segments:
        speaker = str(segment.speaker).strip().strip("[]") or "不明"
        text = str(segment.text).strip()
        if text:
            lines.append(f"[{format_timestamp(float(segment.start))}] {speaker}: {text}")
    return "\n".join(lines)


def _get_runtime() -> tuple[Any, Any, Any, Any]:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _, model, processor, device, dtype = _load_runtime(
                    MODEL_ID, os.getenv("MOSS_DEVICE", "auto")
                )
                _runtime = model, processor, device, dtype
    return _runtime


def transcribe(audio_path: str | None, max_new_tokens: int) -> tuple[str, str]:
    """Transcribe one uploaded file and return display text plus a .txt path."""
    if not audio_path:
        raise ValueError("音声ファイルを選択してください。")

    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        generate_transcription,
    )

    model, processor, device, dtype = _get_runtime()
    messages = build_transcription_messages(audio_path, prompt=JAPANESE_PROMPT)

    # generate() and the shared model are serialized to avoid concurrent GPU OOMs.
    with _runtime_lock:
        result = generate_transcription(
            model,
            processor,
            messages,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            device=device,
            dtype=dtype,
        )

    output = format_transcript(parse_transcript(result["text"]))
    if not output:
        raise RuntimeError(
            "発話区間を取得できませんでした。音声を確認するか、最大出力トークン数を増やしてください。"
        )

    output_dir = Path(tempfile.mkdtemp(prefix="moss-transcript-"))
    output_path = output_dir / "transcription.txt"
    output_path.write_text(output + "\n", encoding="utf-8")
    return output, str(output_path)


def build_demo():
    import gradio as gr

    def run(audio_path: str | None, max_new_tokens: int):
        try:
            return transcribe(audio_path, max_new_tokens)
        except (ValueError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc
        except Exception as exc:
            raise gr.Error(f"処理に失敗しました: {exc}") from exc

    with gr.Blocks(title="MOSS 音声文字起こし・話者分離") as demo:
        gr.Markdown(
            "# MOSS 音声文字起こし・話者分離\n"
            "音声ファイルをアップロードすると、時刻・話者ラベル付きで文字起こしします。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(
                    label="音声ファイル",
                    sources=["upload"],
                    type="filepath",
                )
                with gr.Accordion("詳細設定", open=False):
                    max_tokens = gr.Slider(
                        minimum=1024,
                        maximum=65536,
                        value=8192,
                        step=1024,
                        label="最大出力トークン数",
                        info="長い音声で結果が途中までの場合は増やしてください。",
                    )
                submit = gr.Button("文字起こしを開始", variant="primary")
            with gr.Column(scale=2):
                transcript = gr.Textbox(
                    label="文字起こし結果",
                    lines=18,
                    placeholder="[00:00:00.000] S01: 発話内容",
                )
                download = gr.File(label="テキストをダウンロード", interactive=False)

        submit.click(
            fn=run,
            inputs=[audio, max_tokens],
            outputs=[transcript, download],
            concurrency_limit=1,
        )
        gr.Markdown(
            "話者ラベル（S01、S02…）は音声内だけで有効な匿名ラベルです。"
            "初回実行時はモデルのダウンロードと読み込みに時間がかかります。"
        )
    return demo
