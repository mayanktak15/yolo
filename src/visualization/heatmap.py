"""Movement heatmap generation from track histories."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.tracking.tracker import TrackHistory


def generate_heatmap(
    history: TrackHistory,
    frame_shape: tuple[int, int, int],
    output_path: str | Path,
    radius: int = 18,
) -> np.ndarray:
    """Create and save a heatmap image showing tracked centroid density."""

    height, width = frame_shape[:2]
    heat = np.zeros((height, width), dtype=np.float32)

    for points in history.points.values():
        for _, x, y in points:
            cx = int(np.clip(x, 0, width - 1))
            cy = int(np.clip(y, 0, height - 1))
            cv2.circle(heat, (cx, cy), radius, 1.0, -1)

    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=radius / 2)
    normalized = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), colored)
    return colored

