# Technical Report: Multi-Object Detection and Persistent ID Tracking

## Objective

The system detects people in sports or public-event videos, assigns persistent track IDs, visualizes tracked subjects, and exports analytics for downstream review.

## Architecture

```text
Input Video
-> Frame Reader
-> YOLO Detector
-> BoT-SORT Tracker
-> ID Assignment
-> Visualization
-> Analytics
-> Output Video
```

The implementation uses Ultralytics YOLO for detection and its BoT-SORT backend for online multi-object tracking. OpenCV handles video IO and annotation, while Pandas and Matplotlib generate analytics and plots.

## Detection

The detector configuration defaults to `yolo11n.pt` and filters for the COCO `person` class. This covers athletes, players, participants, pedestrians, and public-event subjects without requiring custom training.

Returned detection fields:

```json
{
  "class": "person",
  "confidence": 0.92,
  "bbox": [100.0, 42.0, 180.0, 260.0]
}
```

## Tracking

BoT-SORT maintains identities over time with motion and association cues. The project stores track history by ID and frame index, allowing trajectories, durations, object counts, and relative speed estimates.

Tracked object schema:

```json
{
  "id": 17,
  "class": "person",
  "confidence": 0.92,
  "bbox": [100.0, 42.0, 180.0, 260.0]
}
```

## Robustness Considerations

- Occlusion: track buffers preserve identities through short missing intervals.
- Motion blur: confidence thresholds can be lowered for challenging footage.
- Scale changes: YOLO inference at configurable image size improves small-object recall.
- Camera movement: BoT-SORT global motion compensation is enabled with sparse optical flow.
- Similar-looking subjects: optional ReID can be enabled in `configs/tracker.yaml`.
- Partial visibility: trajectory history and association thresholds reduce ID fragmentation.

## Analytics

The pipeline exports:

1. Total unique objects tracked.
2. Average active tracks per frame.
3. Track duration statistics.
4. Frame-wise object counts.
5. Detection confidence distribution.
6. Continuity diagnostics for short-track fragmentation.
7. Relative speed in pixels per frame.

## Limitations

Without ground-truth MOT annotations, the project cannot compute formal MOTA, IDF1, or HOTA metrics. The included evaluator provides practical diagnostics based on track continuity and fragmentation.

## Recommended Improvements

- Enable ReID for crowded scenes after validating local model support.
- Add ground-truth annotation support for formal MOT evaluation.
- Calibrate pixel-to-meter conversion for physical speed estimation.
- Tune confidence and association thresholds per camera angle and sport.

