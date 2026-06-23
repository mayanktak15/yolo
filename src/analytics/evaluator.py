"""Evaluation utilities for tracking assignments."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class EvaluationSummary:
    """Lightweight quality checks for generated tracks without ground truth."""

    total_tracks: int
    fragmented_tracks: int
    short_track_ratio: float


def evaluate_track_continuity(track_df: pd.DataFrame, short_track_threshold: int = 5) -> EvaluationSummary:
    """Estimate continuity quality from predicted track lengths.

    This is not a MOTChallenge metric because no ground-truth annotations are
    provided. It flags excessive short tracks as a practical diagnostic.
    """

    if track_df.empty:
        return EvaluationSummary(total_tracks=0, fragmented_tracks=0, short_track_ratio=0.0)

    durations = track_df.groupby("id")["frame"].nunique()
    fragmented = int((durations < short_track_threshold).sum())
    return EvaluationSummary(
        total_tracks=int(len(durations)),
        fragmented_tracks=fragmented,
        short_track_ratio=round(fragmented / max(len(durations), 1), 4),
    )

