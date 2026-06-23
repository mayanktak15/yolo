"""OpenCV annotation utilities."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import cv2
import numpy as np

from src.tracking.tracker import TrackHistory, TrackedObject
from src.visualization.trajectory import draw_trajectory


class TrackAnnotator:
    """Draw bounding boxes, labels, confidences, and trajectory trails."""

    def __init__(
        self,
        draw_trajectories: bool = True,
        trajectory_length: int = 40,
        line_thickness: int = 2,
    ) -> None:
        self.draw_trajectories = draw_trajectories
        self.trajectory_length = trajectory_length
        self.line_thickness = line_thickness

    @staticmethod
    def _color_for_id(track_id: int) -> tuple[int, int, int]:
        digest = hashlib.md5(str(track_id).encode("utf-8")).digest()
        return int(digest[0]), int(digest[1]), int(digest[2])

    def draw(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackedObject],
        history: TrackHistory,
    ) -> np.ndarray:
        annotated = frame.copy()
        for track in tracks:
            color = self._color_for_id(track.id)
            x1, y1, x2, y2 = (int(v) for v in track.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.line_thickness)

            label = f"ID: {track.id} | {track.confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            label_y = max(y1 - text_height - baseline - 4, 0)
            cv2.rectangle(
                annotated,
                (x1, label_y),
                (x1 + text_width + 8, label_y + text_height + baseline + 6),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, label_y + text_height + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if self.draw_trajectories:
                points = history.get_recent_points(track.id, self.trajectory_length)
                draw_trajectory(annotated, points, color, max(1, self.line_thickness - 1))

        return annotated

