"""Metric calculations for multi-object tracking outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.tracking.tracker import TrackedObject


@dataclass
class TrackingMetrics:
    """Collect frame-wise tracks and export summary analytics."""

    frame_records: list[dict[str, int]] = field(default_factory=list)
    track_records: list[dict[str, float | int | str]] = field(default_factory=list)

    def update(self, frame_index: int, tracks: list[TrackedObject]) -> None:
        self.frame_records.append({"frame": frame_index, "active_tracks": len(tracks)})
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            cx, cy = track.centroid
            self.track_records.append(
                {
                    "frame": frame_index,
                    "id": track.id,
                    "class": track.class_name,
                    "confidence": track.confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "centroid_x": cx,
                    "centroid_y": cy,
                }
            )

    @property
    def frame_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.frame_records)

    @property
    def track_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.track_records)

    def summarize(self) -> dict[str, object]:
        frame_df = self.frame_df
        track_df = self.track_df

        if track_df.empty:
            average_active = 0.0
            if not frame_df.empty:
                average_active = round(float(frame_df["active_tracks"].mean()), 4)
            return {
                "total_unique_objects": 0,
                "average_active_tracks_per_frame": average_active,
                "track_duration_statistics": {},
                "frame_wise_object_counts": frame_df.to_dict(orient="records"),
                "detection_confidence_distribution": {},
            }

        durations = track_df.groupby("id")["frame"].nunique()
        confidence = track_df["confidence"]

        return {
            "total_unique_objects": int(track_df["id"].nunique()),
            "average_active_tracks_per_frame": round(float(frame_df["active_tracks"].mean()), 4),
            "track_duration_statistics": {
                "min_frames": int(durations.min()),
                "max_frames": int(durations.max()),
                "mean_frames": round(float(durations.mean()), 4),
                "median_frames": round(float(durations.median()), 4),
            },
            "frame_wise_object_counts": frame_df.to_dict(orient="records"),
            "detection_confidence_distribution": {
                "min": round(float(confidence.min()), 4),
                "max": round(float(confidence.max()), 4),
                "mean": round(float(confidence.mean()), 4),
                "median": round(float(confidence.median()), 4),
            },
        }

    def save_tables(self, frame_counts_csv: str | Path, track_history_csv: str | Path) -> None:
        frame_counts_csv = Path(frame_counts_csv)
        track_history_csv = Path(track_history_csv)
        frame_counts_csv.parent.mkdir(parents=True, exist_ok=True)
        track_history_csv.parent.mkdir(parents=True, exist_ok=True)
        self.frame_df.to_csv(frame_counts_csv, index=False)
        self.track_df.to_csv(track_history_csv, index=False)
