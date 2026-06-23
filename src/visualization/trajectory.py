"""Trajectory rendering helpers."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def draw_trajectory(
    frame: np.ndarray,
    points: Sequence[tuple[float, float]],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """Draw a motion trail on a frame."""

    if len(points) < 2:
        return frame

    int_points = [(int(x), int(y)) for x, y in points]
    for start, end in zip(int_points[:-1], int_points[1:], strict=False):
        cv2.line(frame, start, end, color, thickness, lineType=cv2.LINE_AA)
    return frame


def relative_speed(points: Sequence[tuple[int, float, float]]) -> float:
    """Estimate average pixel displacement per frame for a trajectory."""

    if len(points) < 2:
        return 0.0

    speeds: list[float] = []
    for (frame_a, x_a, y_a), (frame_b, x_b, y_b) in zip(points[:-1], points[1:], strict=False):
        frame_delta = max(frame_b - frame_a, 1)
        distance = float(np.hypot(x_b - x_a, y_b - y_a))
        speeds.append(distance / frame_delta)
    return float(np.mean(speeds)) if speeds else 0.0

