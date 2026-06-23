"""Tracking data models and interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.detection.detector import BBox


@dataclass(slots=True)
class TrackedObject:
    """One tracked object at one frame."""

    id: int
    class_name: str
    confidence: float
    bbox: BBox
    frame_index: int

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": int(self.id),
            "class": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
        }


@dataclass
class TrackHistory:
    """Persistent trajectory store keyed by track ID."""

    points: dict[int, list[tuple[int, float, float]]] = field(default_factory=dict)

    def update(self, tracks: Sequence[TrackedObject]) -> None:
        for track in tracks:
            x, y = track.centroid
            self.points.setdefault(track.id, []).append((track.frame_index, x, y))

    def get_recent_points(self, track_id: int, limit: int = 40) -> list[tuple[float, float]]:
        return [(x, y) for _, x, y in self.points.get(track_id, [])[-limit:]]

    def durations(self) -> dict[int, int]:
        return {track_id: len(points) for track_id, points in self.points.items()}


class BaseTracker(ABC):
    """Contract for frame-level multi-object trackers."""

    @abstractmethod
    def update(self, frame: np.ndarray, frame_index: int) -> Sequence[TrackedObject]:
        """Return tracked objects for one BGR frame."""

