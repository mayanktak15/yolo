# Multi-Object Detection and Persistent ID Tracking in Sports/Event Footage

## Overview

This project implements a robust Computer Vision pipeline for **multi-object detection and persistent ID tracking** in publicly available sports or public-event videos.

The system detects relevant subjects (players, athletes, participants, etc.), assigns unique tracking IDs, and maintains identity consistency throughout the video, even under challenging real-world conditions such as:

* Occlusion
* Motion blur
* Camera movement (pan, tilt, zoom)
* Scale variation
* Partial visibility
* Similar-looking subjects

The solution uses a pretrained **YOLO detector** for object detection and **BoT-SORT** for multi-object tracking, enabling reliable and efficient tracking without custom model training.

---

## Assignment Objective

The goal of this project is to:

1. Detect all relevant subjects in a sports or public-event video.
2. Assign unique IDs to detected subjects.
3. Maintain ID consistency across the entire video.
4. Generate an annotated output video showing:

   * Bounding boxes
   * Tracking IDs
   * Detection confidence
5. Produce analytics and tracking statistics.

---

## Selected Video

### Video Source

**Video Title:** [Add Video Title]

**Source URL:** [Add Public Video URL]

**Category:** Cricket / Football / Basketball / Marathon / Other

The selected video is publicly accessible and contains multiple moving participants suitable for multi-object detection and tracking.

---

## System Architecture

```text
Input Video
      │
      ▼
Frame Reader
      │
      ▼
YOLO Detector
      │
      ▼
BoT-SORT Tracker
      │
      ▼
Persistent ID Assignment
      │
      ▼
Visualization & Analytics
      │
      ▼
Annotated Output Video
```

---

## Project Structure

```text
multi-object-tracking/

├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── configs/
│   ├── detector.yaml
│   ├── tracker.yaml
│   └── pipeline.yaml
│
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── visualization/
│   ├── analytics/
│   ├── utils/
│   └── main.py
│
├── reports/
├── demo/
├── tests/
│
├── requirements.txt
├── README.md
└── technical_report.md
```

---

## Technology Stack

| Component            | Technology          |
| -------------------- | ------------------- |
| Language             | Python 3.11+        |
| Detection            | YOLOv11 (or YOLOv8) |
| Tracking             | BoT-SORT            |
| Video Processing     | OpenCV              |
| Numerical Operations | NumPy               |
| Data Analysis        | Pandas              |
| Visualization        | Matplotlib          |
| Configuration        | YAML                |

---

## Detection Pipeline

### Model

**YOLOv11n** (or YOLOv8n)

### Target Class

* Person

In sports and event footage, the person class serves as the primary representation for:

* Players
* Athletes
* Participants
* Officials
* Spectators (when visible)

### Output

Each detection returns:

```json
{
  "class": "person",
  "confidence": 0.92,
  "bbox": [x1, y1, x2, y2]
}
```

---

## Tracking Pipeline

### Tracker

**BoT-SORT**

### Why BoT-SORT?

BoT-SORT was selected because it provides:

* Strong ID persistence
* Re-identification support
* Robust association across frames
* Improved performance under occlusion
* Better handling of camera motion

### Tracking Output

```json
{
  "id": 17,
  "class": "person",
  "confidence": 0.92,
  "bbox": [120, 45, 210, 320]
}
```

---

## ID Persistence Strategy

The tracker maintains stable identities using:

1. Motion prediction
2. Appearance embeddings
3. Detection-to-track association
4. Re-identification mechanisms
5. Historical trajectory information

This helps preserve IDs during:

* Temporary occlusions
* Overlapping subjects
* Fast movement
* Camera zoom and pan

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd multi-object-tracking
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Linux / Mac:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dependencies

Main dependencies include:

```text
ultralytics
opencv-python
numpy
pandas
matplotlib
pyyaml
scipy
tqdm
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### Default Run

```bash
python -m src.main
```

### Custom Input Video

```bash
python -m src.main \
    --input data/raw/input.mp4 \
    --output data/processed/tracked_output.mp4
```

### Display Frames During Processing

```bash
python -m src.main \
    --input data/raw/input.mp4 \
    --display
```

---

## Generated Outputs

### Annotated Video

```text
data/processed/tracked_output.mp4
```

Contains:

* Bounding boxes
* Persistent IDs
* Confidence scores
* Tracking visualization

### Analytics

```text
data/metadata/analytics.json
```

Includes:

* Total unique objects tracked
* Average active tracks per frame
* Track duration statistics
* Frame-wise object counts
* Confidence distribution

### Visualizations

```text
reports/figures/
```

Contains:

* Object count graph
* Confidence histogram
* Heatmap
* Trajectory plots

---

## Sample Results

### Screenshots

Store sample outputs in:

```text
demo/screenshots/
```

Recommended screenshots:

1. Initial detections
2. Mid-video tracking
3. Occlusion scenario
4. Long-term ID consistency
5. Final output frame

---

## Optional Enhancements Implemented

* Trajectory Visualization
* Movement Heatmap
* Object Count Over Time
* Relative Speed Estimation
* Analytics Dashboard Outputs

---

## Assumptions

* The selected video is publicly available.
* Subjects are sufficiently visible for detection.
* The person class adequately represents participants.
* Pretrained YOLO weights provide acceptable performance without fine-tuning.
* Processing is performed offline rather than real-time.

---

## Limitations

* Severe long-duration occlusions may cause ID switches.
* Extremely crowded scenes can reduce tracking accuracy.
* Similar uniforms or appearances may occasionally confuse the tracker.
* Speed estimation is relative (pixels/frame) rather than real-world velocity.
* Performance depends on video quality and camera angle.

---

## Challenges Encountered

* Maintaining ID consistency during heavy occlusion.
* Handling fast-moving players.
* Tracking small distant subjects.
* Camera zoom and rapid scene transitions.
* Similar-looking participants in team sports.

---

## Future Improvements

Potential enhancements include:

* Fine-tuned sports-specific detector
* Stronger person re-identification model
* Bird's-eye-view transformation
* Team classification
* Real-world speed estimation
* Multi-camera tracking
* Live streaming deployment

---

## Deliverables

The submission includes:

* Source Code
* README.md
* technical_report.md
* Annotated Output Video
* Original Video Source Link
* Analytics Output
* Sample Screenshots
* 3–5 Minute Demo Video

---

## Author

**Candidate Name:** [Your Name]

**Role:** Computer Vision / AI Engineer Candidate

**Submission Date:** [Date]
