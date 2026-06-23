"""YOLOv11/YOLOv8 detector wrapper based on Ultralytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from ultralytics import YOLO

from src.detection.detector import Detection, BaseDetector


class YOLODetector(BaseDetector):
    """Config-driven wrapper around a pretrained Ultralytics YOLO model."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.model = YOLO(self.config.get("model_name", "yolo11n.pt"))
        self.names = self.model.names
        self.class_ids = self._resolve_class_filter(self.config.get("class_filter"))

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

    def detect(self, frame: np.ndarray) -> Sequence[Detection]:
        results = self.model.predict(
            source=frame,
            conf=float(self.config.get("confidence_threshold", 0.35)),
            iou=float(self.config.get("iou_threshold", 0.5)),
            imgsz=int(self.config.get("image_size", 1280)),
            device=self.config.get("device"),
            classes=self.class_ids,
            agnostic_nms=bool(self.config.get("agnostic_nms", False)),
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            for bbox, confidence, class_id in zip(boxes, confidences, class_ids, strict=False):
                detections.append(
                    Detection(
                        class_name=str(self.names.get(int(class_id), class_id)),
                        confidence=float(confidence),
                        bbox=tuple(float(v) for v in bbox),
                        class_id=int(class_id),
                    )
                )
        return detections

