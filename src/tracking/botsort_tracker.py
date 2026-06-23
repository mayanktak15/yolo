"""BoT-SORT tracker wrapper using Ultralytics' built-in tracker backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from ultralytics import YOLO

from src.tracking.tracker import BaseTracker, TrackedObject


class BotSortTracker(BaseTracker):
    """Runs pretrained YOLO detection and BoT-SORT ID association per frame."""

    def __init__(self, detector_config_path: str | Path, tracker_config_path: str | Path) -> None:
        self.detector_config_path = Path(detector_config_path)
        self.tracker_config_path = Path(tracker_config_path)
        self.detector_config = self._load_config(self.detector_config_path)
        self.model = YOLO(self.detector_config.get("model_name", "yolo11n.pt"))
        self.names = self.model.names
        self.class_ids = self._resolve_class_filter(self.detector_config.get("class_filter"))

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def _resolve_class_filter(self, class_filter: Sequence[str | int] | None) -> list[int] | None:
        if not class_filter:
            return None
        name_to_id = {str(name).lower(): int(idx) for idx, name in self.names.items()}
        resolved: list[int] = []
        for item in class_filter:
            if isinstance(item, int):
                resolved.append(item)
                continue
            class_id = name_to_id.get(str(item).lower())
            if class_id is not None:
                resolved.append(class_id)
        return resolved or None

    def update(self, frame: np.ndarray, frame_index: int) -> Sequence[TrackedObject]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=str(self.tracker_config_path),
            conf=float(self.detector_config.get("confidence_threshold", 0.35)),
            iou=float(self.detector_config.get("iou_threshold", 0.5)),
            imgsz=int(self.detector_config.get("image_size", 1280)),
            device=self.detector_config.get("device"),
            classes=self.class_ids,
            agnostic_nms=bool(self.detector_config.get("agnostic_nms", False)),
            verbose=False,
        )

        tracked: list[TrackedObject] = []
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            for bbox, confidence, class_id, track_id in zip(
                boxes, confidences, class_ids, track_ids, strict=False
            ):
                tracked.append(
                    TrackedObject(
                        id=int(track_id),
                        class_name=str(self.names.get(int(class_id), class_id)),
                        confidence=float(confidence),
                        bbox=tuple(float(v) for v in bbox),
                        frame_index=frame_index,
                    )
                )
        return tracked

