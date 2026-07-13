from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from .io import Segment


def normalize_ja(text: str, *, keep_punctuation: bool = False) -> str:
    """Normalize Japanese text without hiding kanji/kana recognition errors."""
    text = unicodedata.normalize("NFKC", text).lower()
    if keep_punctuation:
        return "".join(ch for ch in text if not ch.isspace())
    return "".join(
        ch
        for ch in text
        if not ch.isspace() and unicodedata.category(ch)[0] not in {"P", "S", "C"}
    )


def edit_distance(reference: str, hypothesis: str) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, 1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def _speaker_text(segments: tuple[Segment, ...]) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        grouped[segment.speaker].append(normalize_ja(segment.text))
    return {speaker: "".join(parts) for speaker, parts in grouped.items()}


def _minimum_assignment(costs: list[list[float]]) -> tuple[float, tuple[int, ...]]:
    """Square assignment via bitmask DP; practical for meeting-sized speaker counts."""
    size = len(costs)
    if size == 0:
        return 0.0, ()
    if size > 16:
        raise ValueError("More than 16 speakers is not supported by the exact assignment scorer")

    @lru_cache(maxsize=None)
    def visit(row: int, used: int) -> tuple[float, tuple[int, ...]]:
        if row == size:
            return 0.0, ()
        best_cost = math.inf
        best_assignment: tuple[int, ...] = ()
        for column in range(size):
            if used & (1 << column):
                continue
            tail_cost, tail_assignment = visit(row + 1, used | (1 << column))
            candidate = costs[row][column] + tail_cost
            if candidate < best_cost:
                best_cost = candidate
                best_assignment = (column,) + tail_assignment
        return best_cost, best_assignment

    return visit(0, 0)


@dataclass(frozen=True)
class AsrCounts:
    errors: int
    reference_characters: int
    cp_errors: int

    @property
    def cer(self) -> float:
        return self.errors / self.reference_characters if self.reference_characters else float("nan")

    @property
    def cpcer(self) -> float:
        return self.cp_errors / self.reference_characters if self.reference_characters else float("nan")


def asr_counts(reference: tuple[Segment, ...], hypothesis: tuple[Segment, ...]) -> AsrCounts:
    ref_all = "".join(normalize_ja(s.text) for s in sorted(reference, key=lambda x: x.start))
    hyp_all = "".join(normalize_ja(s.text) for s in sorted(hypothesis, key=lambda x: x.start))
    ref_by_speaker = _speaker_text(reference)
    hyp_by_speaker = _speaker_text(hypothesis)
    size = max(len(ref_by_speaker), len(hyp_by_speaker))
    refs = list(ref_by_speaker.values()) + [""] * (size - len(ref_by_speaker))
    hyps = list(hyp_by_speaker.values()) + [""] * (size - len(hyp_by_speaker))
    costs = [[float(edit_distance(ref, hyp)) for hyp in hyps] for ref in refs]
    cp_errors, _ = _minimum_assignment(costs)
    return AsrCounts(edit_distance(ref_all, hyp_all), len(ref_all), int(cp_errors))


@dataclass(frozen=True)
class DerCounts:
    missed: float
    false_alarm: float
    confusion: float
    reference_speaker_time: float

    @property
    def der(self) -> float:
        numerator = self.missed + self.false_alarm + self.confusion
        return numerator / self.reference_speaker_time if self.reference_speaker_time else float("nan")


def _active(segments: tuple[Segment, ...], midpoint: float) -> set[str]:
    return {s.speaker for s in segments if s.start <= midpoint < s.end}


def der_counts(
    reference: tuple[Segment, ...], hypothesis: tuple[Segment, ...], *, collar: float = 0.25
) -> DerCounts:
    boundaries = sorted({p for s in reference + hypothesis for p in (s.start, s.end)})
    ref_boundaries = [p for s in reference for p in (s.start, s.end)]
    atoms: list[tuple[float, set[str], set[str]]] = []
    ref_speakers = sorted({s.speaker for s in reference})
    hyp_speakers = sorted({s.speaker for s in hypothesis})
    overlap = {(r, h): 0.0 for r in ref_speakers for h in hyp_speakers}

    for start, end in zip(boundaries, boundaries[1:]):
        midpoint = (start + end) / 2
        if any(abs(midpoint - boundary) < collar for boundary in ref_boundaries):
            continue
        duration = end - start
        ref_active = _active(reference, midpoint)
        hyp_active = _active(hypothesis, midpoint)
        atoms.append((duration, ref_active, hyp_active))
        for ref_speaker in ref_active:
            for hyp_speaker in hyp_active:
                overlap[(ref_speaker, hyp_speaker)] += duration

    size = max(len(ref_speakers), len(hyp_speakers))
    padded_refs: list[str | None] = ref_speakers + [None] * (size - len(ref_speakers))
    padded_hyps: list[str | None] = hyp_speakers + [None] * (size - len(hyp_speakers))
    costs = [
        [-overlap.get((ref, hyp), 0.0) if ref and hyp else 0.0 for hyp in padded_hyps]
        for ref in padded_refs
    ]
    _, assignment = _minimum_assignment(costs)
    hyp_to_ref = {
        padded_hyps[column]: padded_refs[row]
        for row, column in enumerate(assignment)
        if padded_refs[row] is not None and padded_hyps[column] is not None
    }

    missed = false_alarm = confusion = reference_time = 0.0
    for duration, ref_active, hyp_active in atoms:
        correct = sum(1 for hyp in hyp_active if hyp_to_ref.get(hyp) in ref_active)
        ref_count, hyp_count = len(ref_active), len(hyp_active)
        missed += duration * max(0, ref_count - hyp_count)
        false_alarm += duration * max(0, hyp_count - ref_count)
        confusion += duration * (min(ref_count, hyp_count) - correct)
        reference_time += duration * ref_count
    return DerCounts(missed, false_alarm, confusion, reference_time)
