from __future__ import annotations

import argparse
from pathlib import Path

from .io import Recording, Segment, write_jsonl


def parse_speaker_transcript(path: str | Path) -> tuple[Segment, ...]:
    """Parse `speaker: text` lines, assigning synthetic times for ordering only."""
    segments: list[Segment] = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_number}: expected 'speaker: text'")
        speaker, text = line.split(":", 1)
        speaker, text = speaker.strip(), text.strip()
        if not speaker or not text:
            raise ValueError(f"{path}:{line_number}: speaker and text must be non-empty")
        # These intervals preserve turn order for CER/cpCER. They are not timestamps.
        start = float(len(segments))
        segments.append(Segment(start, start + 0.5, speaker, text))
    return tuple(segments)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a speaker-prefixed transcript to reference JSONL")
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--id")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    recording_id = args.id or Path(args.audio).stem
    record = Recording(recording_id, args.audio, parse_speaker_transcript(args.transcript))
    write_jsonl(args.output, [record])
    print(f"Wrote {args.output}: {len(record.segments)} turns, "
          f"{len({segment.speaker for segment in record.segments})} speakers")


if __name__ == "__main__":
    main()
