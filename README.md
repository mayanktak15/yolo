# Multi-Object Detection and Persistent ID Tracking

Production-style computer vision assignment using pretrained YOLO detection with BoT-SORT tracking, OpenCV visualization, and analytics exports.

## Features

- Detects people, athletes, players, and participants using pretrained YOLO class `person`.
- Assigns persistent IDs with BoT-SORT through Ultralytics.
- Renders bounding boxes, track IDs, confidence scores, and trajectory trails.
- Stores per-frame counts, track history, summary analytics, plots, and movement heatmaps.
- Config-driven detector, tracker, and pipeline settings.

## Project Layout

```text
multi-object-tracking/
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── configs/
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── visualization/
│   ├── analytics/
│   └── utils/
├── reports/
├── demo/
└── tests/
```

## Setup

```bash
cd multi-object-tracking
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the selected YOLO model weights, for example `yolo11n.pt`.

## Run

Place a sports or public-event video at `data/raw/input.mp4`, then run:

```bash
python -m src.main
```

Or pass paths explicitly:

```bash
python -m src.main \
  --input data/raw/basketball.mp4 \
  --output data/processed/basketball_tracked.mp4
```

Run without saving video:

```bash
python -m src.main --input data/raw/input.mp4 --no-video
```

Display frames while processing:

```bash
python -m src.main --input data/raw/input.mp4 --display
```

## Outputs

- `data/processed/tracked_output.mp4`: annotated video.
- `data/metadata/analytics.json`: summary analytics.
- `data/metadata/frame_counts.csv`: frame-wise active tracks.
- `data/metadata/track_history.csv`: bounding boxes and centroids per track.
- `reports/figures/object_count_over_time.png`: count graph.
- `reports/figures/confidence_distribution.png`: confidence histogram.
- `reports/figures/movement_heatmap.png`: movement heatmap.
- `demo/screenshots/`: directory for sample screenshots from the demo output.

## Configuration

- `configs/detector.yaml`: YOLO model, confidence threshold, image size, class filter.
- `configs/tracker.yaml`: BoT-SORT association thresholds and re-identification settings.
- `configs/pipeline.yaml`: IO paths, plotting, video writing, display, and visualization settings.

## Notes

This project intentionally uses pretrained models. For most sports and public-event videos, class `person` is the correct proxy for athletes, players, and participants. Custom training should only be added when the target footage requires domain-specific classes or uniforms.

