from html.parser import HTMLParser

from moss_ja_eval.io import Recording, Segment
from moss_ja_eval.report import build_report


class _Parser(HTMLParser):
    pass


def test_report_is_standalone_html():
    reference = Recording("sample", None, (Segment(0, 1, "A", "今日は晴れです。"),))
    hypothesis = Recording("sample", None, (Segment(0, 1, "S01", "今日わ晴れです。"),), 1.0)
    score = {"summary": {"cer": 0.1, "cpcer": 0.1, "rtf": 1.0}}
    report = build_report(reference, hypothesis, score)
    parser = _Parser()
    parser.feed(report)
    assert report.startswith("<!doctype html>")
    assert 'id="normalized"' in report
    assert 'id="raw"' in report
    assert "<del" in report and "<ins" in report
    assert "http://" not in report and "https://" not in report
