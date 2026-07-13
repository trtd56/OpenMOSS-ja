from moss_ja_eval.io import Segment
from moss_ja_eval.metrics import asr_counts, der_counts, edit_distance, normalize_ja
from moss_ja_eval.reference import parse_speaker_transcript


def segment(start, end, speaker, text=""):
    return Segment(start, end, speaker, text)


def test_japanese_normalization():
    assert normalize_ja("ＡＩ、です。\n") == "aiです"


def test_edit_distance():
    assert edit_distance("日本語", "日本人") == 1
    assert edit_distance("", "abc") == 3


def test_cpcer_ignores_speaker_label_names():
    reference = (segment(0, 1, "A", "はい"), segment(1, 2, "B", "いいえ"))
    hypothesis = (segment(0, 1, "S02", "はい"), segment(1, 2, "S01", "いいえ"))
    counts = asr_counts(reference, hypothesis)
    assert counts.cer == 0
    assert counts.cpcer == 0


def test_cpcer_detects_cross_speaker_attribution():
    reference = (
        segment(0, 1, "A", "甲"),
        segment(1, 2, "B", "乙"),
        segment(2, 3, "A", "丙"),
    )
    hypothesis = (
        segment(0, 1, "S01", "甲"),
        segment(1, 2, "S01", "乙"),
        segment(2, 3, "S02", "丙"),
    )
    counts = asr_counts(reference, hypothesis)
    assert counts.cer == 0
    assert counts.cpcer > 0


def test_der_label_permutation_and_confusion():
    reference = (segment(0, 2, "A"), segment(2, 4, "B"))
    correct = (segment(0, 2, "S02"), segment(2, 4, "S01"))
    wrong = (segment(0, 4, "S01"),)
    assert der_counts(reference, correct, collar=0).der == 0
    assert der_counts(reference, wrong, collar=0).der == 0.5


def test_parse_speaker_transcript(tmp_path):
    path = tmp_path / "transcript.txt"
    path.write_text("話者A: こんにちは\n話者B: はい: 承知しました", encoding="utf-8")
    segments = parse_speaker_transcript(path)
    assert [segment.speaker for segment in segments] == ["話者A", "話者B"]
    assert segments[1].text == "はい: 承知しました"
