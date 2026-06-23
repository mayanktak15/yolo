"""Reusable detector abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np


BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class Detection:
    """Single object detection emitted by a detector."""

    class_name: str
    confidence: float
    bbox: BBox
    class_id: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "class": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
            "class_id": self.class_id,
        }


class BaseDetector(ABC):
    """Contract for frame-level object detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Sequence[Detection]:
        """Return detections for one BGR video frame."""

