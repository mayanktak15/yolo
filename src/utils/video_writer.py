"""Video writing wrapper around OpenCV."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoWriter:
    """Write annotated frames to a video file."""

    def __init__(self, output_path: str | Path, fps: float, frame_size: tuple[int, int]) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, frame_size)
        if not self.writer.isOpened():
            raise OSError(f"Could not create output video: {self.output_path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

