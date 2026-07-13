from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .io import read_jsonl
from .metrics import AsrCounts, DerCounts, asr_counts, der_counts


def _ratio(value: float) -> float | None:
    return None if value != value else value


def score(reference_path: str, hypothesis_path: str, collar: float, *, skip_der: bool = False) -> dict:
    references = {record.id: record for record in read_jsonl(reference_path)}
    hypotheses = {record.id: record for record in read_jsonl(hypothesis_path)}
    missing = sorted(references.keys() - hypotheses.keys())
    extra = sorted(hypotheses.keys() - references.keys())
    if missing or extra:
        raise ValueError(f"ID mismatch: missing={missing}, extra={extra}")

    total_asr = AsrCounts(0, 0, 0)
    total_der = DerCounts(0.0, 0.0, 0.0, 0.0)
    per_recording = []
    inference_seconds = audio_seconds = 0.0
    for recording_id, reference in references.items():
        hypothesis = hypotheses[recording_id]
        asr = asr_counts(reference.segments, hypothesis.segments)
        der = (
            DerCounts(0.0, 0.0, 0.0, 0.0)
            if skip_der
            else der_counts(reference.segments, hypothesis.segments, collar=collar)
        )
        total_asr = AsrCounts(
            total_asr.errors + asr.errors,
            total_asr.reference_characters + asr.reference_characters,
            total_asr.cp_errors + asr.cp_errors,
        )
        total_der = DerCounts(
            total_der.missed + der.missed,
            total_der.false_alarm + der.false_alarm,
            total_der.confusion + der.confusion,
            total_der.reference_speaker_time + der.reference_speaker_time,
        )
        if hypothesis.inference_seconds is not None:
            inference_seconds += hypothesis.inference_seconds
            duration_segments = hypothesis.segments if skip_der else reference.segments
            audio_seconds += max((s.end for s in duration_segments), default=0.0)
        per_recording.append(
            {
                "id": recording_id,
                "cer": _ratio(asr.cer),
                "cpcer": _ratio(asr.cpcer),
                "der": None if skip_der else _ratio(der.der),
                "asr_counts": asdict(asr),
                "der_counts": asdict(der),
            }
        )

    return {
        "summary": {
            "recordings": len(references),
            "cer": _ratio(total_asr.cer),
            "cpcer": _ratio(total_asr.cpcer),
            "delta_cp": _ratio(total_asr.cpcer - total_asr.cer),
            "der": None if skip_der else _ratio(total_der.der),
            "collar_seconds": None if skip_der else collar,
            "rtf": inference_seconds / audio_seconds if audio_seconds else None,
            "asr_counts": asdict(total_asr),
            "der_counts": asdict(total_der),
        },
        "recordings": per_recording,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Japanese MOSS transcription and diarization")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument(
        "--skip-der",
        action="store_true",
        help="Do not score time-based diarization (use when references have no timestamps)",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    result = score(args.reference, args.hypothesis, args.collar, skip_der=args.skip_der)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
