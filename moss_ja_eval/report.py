from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from .io import Recording, read_jsonl
from .metrics import _minimum_assignment, normalize_ja


def _speaker_text(record: Recording, *, normalized: bool) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for segment in sorted(record.segments, key=lambda item: (item.start, item.end)):
        value = normalize_ja(segment.text) if normalized else segment.text
        grouped[segment.speaker].append(value)
    return {speaker: "".join(parts) for speaker, parts in grouped.items()}


def _plain_text(record: Recording, *, normalized: bool) -> str:
    parts = []
    for segment in sorted(record.segments, key=lambda item: (item.start, item.end)):
        parts.append(normalize_ja(segment.text) if normalized else segment.text)
    return "".join(parts) if normalized else "\n".join(parts)


def _diff_html(reference: str, hypothesis: str) -> tuple[str, dict[str, int]]:
    chunks: list[str] = []
    counts = {"equal": 0, "delete": 0, "insert": 0, "replace_ref": 0, "replace_hyp": 0}
    for opcode in Levenshtein.opcodes(reference, hypothesis):
        tag, ref_start, ref_end, hyp_start, hyp_end = opcode
        ref_value = html.escape(reference[ref_start:ref_end])
        hyp_value = html.escape(hypothesis[hyp_start:hyp_end])
        if tag == "equal":
            counts["equal"] += ref_end - ref_start
            chunks.append(ref_value)
        elif tag == "delete":
            counts["delete"] += ref_end - ref_start
            chunks.append(f'<del title="MOSSで欠落">{ref_value}</del>')
        elif tag == "insert":
            counts["insert"] += hyp_end - hyp_start
            chunks.append(f'<ins title="MOSSで追加">{hyp_value}</ins>')
        else:
            counts["replace_ref"] += ref_end - ref_start
            counts["replace_hyp"] += hyp_end - hyp_start
            chunks.append(
                f'<span class="replacement"><del title="正解側">{ref_value}</del>'
                f'<ins title="MOSS側">{hyp_value}</ins></span>'
            )
    return "".join(chunks), counts


def _speaker_mapping(reference: Recording, hypothesis: Recording) -> list[dict]:
    refs = _speaker_text(reference, normalized=True)
    hyps = _speaker_text(hypothesis, normalized=True)
    ref_names, hyp_names = list(refs), list(hyps)
    size = max(len(ref_names), len(hyp_names))
    ref_names += [""] * (size - len(ref_names))
    hyp_names += [""] * (size - len(hyp_names))
    costs = [
        [float(Levenshtein.distance(refs.get(ref, ""), hyps.get(hyp, ""))) for hyp in hyp_names]
        for ref in ref_names
    ]
    _, assignment = _minimum_assignment(costs)
    rows = []
    for row, column in enumerate(assignment):
        ref_name, hyp_name = ref_names[row], hyp_names[column]
        if not ref_name and not hyp_name:
            continue
        ref_length = len(refs.get(ref_name, ""))
        errors = int(costs[row][column])
        rows.append(
            {
                "reference": ref_name or "—",
                "hypothesis": hyp_name or "—",
                "reference_chars": ref_length,
                "hypothesis_chars": len(hyps.get(hyp_name, "")),
                "errors": errors,
                "cer": errors / ref_length if ref_length else None,
            }
        )
    return rows


def _speaker_transcript(record: Recording, *, include_time: bool) -> str:
    lines = []
    for segment in sorted(record.segments, key=lambda item: (item.start, item.end)):
        prefix = (
            f"[{segment.start:7.2f}–{segment.end:7.2f}] {segment.speaker}: "
            if include_time
            else f"{segment.speaker}: "
        )
        lines.append(prefix + segment.text)
    return "\n".join(lines)


def build_report(reference: Recording, hypothesis: Recording, score: dict) -> str:
    raw_diff, raw_counts = _diff_html(
        _plain_text(reference, normalized=False), _plain_text(hypothesis, normalized=False)
    )
    normalized_diff, normalized_counts = _diff_html(
        _plain_text(reference, normalized=True), _plain_text(hypothesis, normalized=True)
    )
    mapping = _speaker_mapping(reference, hypothesis)
    summary = score["summary"]
    duration = max((segment.end for segment in hypothesis.segments), default=0.0)

    mapping_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['reference'])}</td>"
        f"<td><span class='speaker-tag'>{html.escape(row['hypothesis'])}</span></td>"
        f"<td>{row['reference_chars']:,}</td>"
        f"<td>{row['hypothesis_chars']:,}</td>"
        f"<td>{'—' if row['cer'] is None else f'{row['cer']:.2%}'}</td>"
        "</tr>"
        for row in mapping
    )
    raw_counts_json = html.escape(json.dumps(raw_counts, ensure_ascii=False))
    normalized_counts_json = html.escape(json.dumps(normalized_counts, ensure_ascii=False))
    reference_full = html.escape(_speaker_transcript(reference, include_time=False))
    hypothesis_full = html.escape(_speaker_transcript(hypothesis, include_time=True))
    cer = summary["cer"]
    cpcer = summary["cpcer"]
    rtf = summary.get("rtf")

    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MOSS 日本語文字起こし Diff — {html.escape(reference.id)}</title>
<style>
:root {{ color-scheme: light; --ink:#172033; --muted:#657084; --line:#dfe4ec; --paper:#fff; --bg:#f4f6f9; --red:#a31621; --red-bg:#ffe1e4; --green:#08783f; --green-bg:#d9f7e7; --blue:#315be8; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Yu Gothic UI",sans-serif; }}
main {{ width:min(1180px, calc(100% - 32px)); margin:32px auto 80px; }}
h1 {{ margin:0 0 8px; font-size:clamp(24px,4vw,38px); letter-spacing:-.03em; }}
.subtitle {{ color:var(--muted); margin:0 0 24px; }}
.cards {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; margin-bottom:22px; }}
.card,.panel {{ background:var(--paper); border:1px solid var(--line); border-radius:14px; box-shadow:0 2px 8px #1720330a; }}
.card {{ padding:16px; }} .label {{ color:var(--muted); font-size:12px; }} .value {{ font-size:25px; font-weight:750; margin-top:5px; }}
.panel {{ padding:20px; margin-top:16px; }} h2 {{ margin:0 0 14px; font-size:18px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:10px 18px; color:var(--muted); font-size:13px; margin-bottom:14px; }}
.sample {{ border-radius:4px; padding:2px 5px; }}
del {{ color:var(--red); background:var(--red-bg); text-decoration:line-through 1.5px; text-decoration-color:#c13843; }}
ins {{ color:var(--green); background:var(--green-bg); text-decoration:none; }}
.replacement {{ border-bottom:2px solid #e8a825; }}
.diff {{ white-space:pre-wrap; overflow-wrap:anywhere; line-height:1.95; font-family:"SFMono-Regular",Consolas,"Hiragino Kaku Gothic ProN",monospace; font-size:14px; max-height:68vh; overflow:auto; border:1px solid var(--line); border-radius:10px; padding:18px; background:#fcfcfd; }}
.tabs {{ display:flex; gap:6px; margin-bottom:12px; }} button {{ appearance:none; border:1px solid var(--line); background:#fff; color:var(--ink); border-radius:8px; padding:8px 12px; cursor:pointer; font-weight:650; }} button.active {{ background:var(--blue); color:#fff; border-color:var(--blue); }}
[hidden] {{ display:none !important; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }} th,td {{ text-align:left; padding:10px 8px; border-bottom:1px solid var(--line); }} th {{ color:var(--muted); font-weight:650; }}
.speaker-tag {{ display:inline-block; border-radius:999px; padding:3px 8px; background:#e8edff; color:#2746ae; font-weight:700; }}
details {{ margin-top:12px; }} summary {{ cursor:pointer; font-weight:700; }} pre {{ white-space:pre-wrap; overflow-wrap:anywhere; max-height:55vh; overflow:auto; background:#fcfcfd; border:1px solid var(--line); padding:16px; border-radius:10px; line-height:1.65; }}
.note {{ color:var(--muted); font-size:13px; line-height:1.7; }}
@media(max-width:820px) {{ .cards {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .cards .card:last-child {{ grid-column:span 2; }} }}
</style>
</head>
<body><main>
<h1>日本語文字起こし Diff</h1>
<p class="subtitle">{html.escape(reference.id)} — 正解テキストと MOSS-Transcribe-Diarize 出力の比較</p>
<section class="cards">
  <div class="card"><div class="label">CER</div><div class="value">{cer:.2%}</div></div>
  <div class="card"><div class="label">cpCER</div><div class="value">{cpcer:.2%}</div></div>
  <div class="card"><div class="label">話者数</div><div class="value">{len(set(s.speaker for s in hypothesis.segments))}</div></div>
  <div class="card"><div class="label">音声末尾</div><div class="value">{duration/60:.1f}分</div></div>
  <div class="card"><div class="label">RTF</div><div class="value">{'—' if rtf is None else f'{rtf:.2f}'}</div></div>
</section>
<section class="panel">
  <h2>文字Diff</h2>
  <div class="legend"><span><span class="sample"><del>赤</del></span> 正解にあるがMOSSで欠落</span><span><span class="sample"><ins>緑</ins></span> MOSSで追加</span><span>黄色下線は置換</span></div>
  <div class="tabs"><button class="active" data-target="normalized">CER正規化後</button><button data-target="raw">原文（句読点・改行を含む）</button></div>
  <div id="normalized" class="diff" data-counts="{normalized_counts_json}">{normalized_diff}</div>
  <div id="raw" class="diff" data-counts="{raw_counts_json}" hidden>{raw_diff}</div>
  <p class="note">CER正規化後はNFKC、小文字化、空白・句読点・記号除去後の比較です。話者名は文字Diffに含みません。</p>
</section>
<section class="panel">
  <h2>話者ラベルの最適対応</h2>
  <table><thead><tr><th>正解話者</th><th>MOSS</th><th>正解文字数</th><th>MOSS文字数</th><th>話者別CER</th></tr></thead><tbody>{mapping_rows}</tbody></table>
</section>
<section class="panel">
  <h2>全文</h2>
  <details><summary>元の文字起こし</summary><pre>{reference_full}</pre></details>
  <details><summary>MOSS出力（時刻付き）</summary><pre>{hypothesis_full}</pre></details>
</section>
</main>
<script>
document.querySelectorAll('.tabs button').forEach(button => button.addEventListener('click', () => {{
  document.querySelectorAll('.tabs button').forEach(item => item.classList.toggle('active', item === button));
  document.querySelectorAll('.diff').forEach(item => item.hidden = item.id !== button.dataset.target);
}}));
</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a standalone transcription diff report")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--score", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    references = read_jsonl(args.reference)
    hypotheses = read_jsonl(args.hypothesis)
    if len(references) != 1 or len(hypotheses) != 1:
        raise ValueError("The HTML report currently expects exactly one recording")
    score = json.loads(Path(args.score).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.write_text(build_report(references[0], hypotheses[0], score), encoding="utf-8")
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
