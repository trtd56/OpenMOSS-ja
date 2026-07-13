from __future__ import annotations

import argparse
import time
from pathlib import Path

from .io import Recording, Segment, read_jsonl, write_jsonl

JAPANESE_PROMPT = (
    "音声を日本語で正確に文字起こししてください。各区間は開始時刻と話者番号"
    "（[S01]、[S02]、[S03]…）で始め、本文の後に終了時刻を付けてください。"
    "句読点を補い、聞こえない内容を推測しないでください。"
)


def _load_runtime(model_id: str, device_name: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from moss_transcribe_diarize.inference_utils import resolve_device
    except ImportError as exc:
        raise SystemExit('Inference dependencies are missing. Run: uv sync --extra inference') from exc

    if device_name == "auto" and torch.backends.mps.is_available():
        device_name = "mps"
    device = resolve_device(device_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, dtype="auto"
    ).to(dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return torch, model, processor, device, dtype


def run(args: argparse.Namespace) -> None:
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        generate_transcription,
    )

    torch, model, processor, device, dtype = _load_runtime(args.model, args.device)
    del torch
    manifest_path = Path(args.manifest).resolve()
    records = read_jsonl(manifest_path)
    predictions: list[Recording] = []
    for index, record in enumerate(records, 1):
        if not record.audio:
            raise ValueError(f"Recording {record.id!r} has no audio path")
        audio_path = Path(record.audio)
        if not audio_path.is_absolute():
            manifest_relative = manifest_path.parent / audio_path
            cwd_relative = Path.cwd() / audio_path
            audio_path = manifest_relative if manifest_relative.exists() else cwd_relative
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio for {record.id!r} was not found: {audio_path}. "
                "Relative paths are resolved against the manifest directory, then the current directory."
            )
        messages = build_transcription_messages(audio_path, prompt=args.prompt)
        started = time.perf_counter()
        result = generate_transcription(
            model,
            processor,
            messages,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            device=device,
            dtype=dtype,
        )
        elapsed = time.perf_counter() - started
        parsed = parse_transcript(result["text"])
        predictions.append(
            Recording(
                id=record.id,
                audio=record.audio,
                segments=tuple(
                    Segment(float(s.start), float(s.end), str(s.speaker), str(s.text))
                    for s in parsed
                ),
                inference_seconds=elapsed,
            )
        )
        write_jsonl(args.output, predictions)
        print(f"[{index}/{len(records)}] {record.id}: {len(parsed)} segments, {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic MOSS inference over a JSONL manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="OpenMOSS-Team/MOSS-Transcribe-Diarize")
    parser.add_argument("--device", default="auto", help="auto, mps, cpu, or cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--prompt", default=JAPANESE_PROMPT)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
