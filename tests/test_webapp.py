from moss_ja_eval.io import Segment
from moss_ja_eval.webapp import format_timestamp, format_transcript


def test_format_timestamp_rounds_with_carry() -> None:
    assert format_timestamp(0.48) == "00:00:00.480"
    assert format_timestamp(3599.9996) == "01:00:00.000"
    assert format_timestamp(-1) == "00:00:00.000"


def test_format_transcript_uses_start_time_and_speaker() -> None:
    segments = [
        Segment(0.48, 1.66, "[S01]", " こんにちは。 "),
        Segment(12.26, 13.81, "S02", "よろしくお願いします。"),
    ]

    assert format_transcript(segments) == (
        "[00:00:00.480] S01: こんにちは。\n"
        "[00:00:12.260] S02: よろしくお願いします。"
    )
