from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: str
    text: str

    @classmethod
    def from_dict(cls, value: dict) -> "Segment":
        segment = cls(
            start=float(value["start"]),
            end=float(value["end"]),
            speaker=str(value["speaker"]),
            text=str(value.get("text", "")),
        )
        if segment.start < 0 or segment.end <= segment.start:
            raise ValueError(f"Invalid segment interval: {value!r}")
        return segment

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "text": self.text,
        }


@dataclass(frozen=True)
class Recording:
    id: str
    audio: str | None
    segments: tuple[Segment, ...]
    inference_seconds: float | None = None

    @classmethod
    def from_dict(cls, value: dict) -> "Recording":
        if not value.get("id"):
            raise ValueError("Every recording must have a non-empty id")
        return cls(
            id=str(value["id"]),
            audio=str(value["audio"]) if value.get("audio") is not None else None,
            segments=tuple(Segment.from_dict(s) for s in value.get("segments", [])),
            inference_seconds=(
                float(value["inference_seconds"])
                if value.get("inference_seconds") is not None
                else None
            ),
        )

    def as_dict(self) -> dict:
        result = {
            "id": self.id,
            "audio": self.audio,
            "segments": [segment.as_dict() for segment in self.segments],
        }
        if self.inference_seconds is not None:
            result["inference_seconds"] = self.inference_seconds
        return result


def read_jsonl(path: str | Path) -> list[Recording]:
    records: list[Recording] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = Recording.from_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if record.id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {record.id!r}")
            seen.add(record.id)
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: Iterable[Recording]) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
