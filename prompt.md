# Multi-Object Detection and Persistent ID Tracking Assignment

You are a Senior Computer Vision Engineer.

Your task is to build a complete, production-quality Computer Vision project for the following assignment.

---

# Objective

Develop a robust Multi-Object Detection and Tracking system capable of:

* Detecting all relevant subjects in a sports or public-event video.
* Assigning unique IDs to each detected subject.
* Maintaining ID consistency throughout the video.
* Handling:

  * occlusions
  * motion blur
  * scale changes
  * camera movement
  * similar-looking subjects
  * partial visibility

The final project should look like a professional AI engineering assignment submission.

---

# Technical Requirements

Use:

* Python 3.11+
* YOLOv11 (preferred) or YOLOv8 for detection
* BoT-SORT for tracking
* OpenCV for visualization
* NumPy
* Pandas
* Matplotlib

Avoid training custom models unless absolutely necessary.

Use pretrained models.

---

# Project Structure

Generate the following project structure:

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
│   │   ├── detector.py
│   │   └── yolo_detector.py
│   │
│   ├── tracking/
│   │   ├── tracker.py
│   │   └── botsort_tracker.py
│   │
│   ├── visualization/
│   │   ├── annotator.py
│   │   ├── trajectory.py
│   │   └── heatmap.py
│   │
│   ├── analytics/
│   │   ├── metrics.py
│   │   ├── statistics.py
│   │   └── evaluator.py
│   │
│   ├── utils/
│   │   ├── video_reader.py
│   │   ├── video_writer.py
│   │   └── logger.py
│   │
│   └── main.py
│
├── reports/
│
├── demo/
│
├── tests/
│
├── requirements.txt
├── README.md
└── technical_report.md

---

# Architecture

Pipeline:

Input Video
→ Frame Reader
→ YOLO Detector
→ BoT-SORT Tracker
→ ID Assignment
→ Visualization
→ Analytics
→ Output Video

---

# Detection Requirements

Detect:

* people
* athletes
* players
* participants

Return:

* bounding boxes
* confidence scores
* class names

Create reusable detector interfaces.

---

# Tracking Requirements

Implement persistent tracking.

Each object must contain:

{
"id": 17,
"class": "person",
"confidence": 0.92,
"bbox": [x1, y1, x2, y2]
}

Requirements:

* stable IDs
* re-identification support
* track history
* trajectory storage

---

# Visualization Requirements

Render:

* bounding boxes
* track IDs
* confidence scores

Example:

ID: 12 | 0.94

Add trajectory trails for each tracked object.

Store track history.

---

# Analytics Requirements

Generate:

1. Total unique objects tracked
2. Average active tracks per frame
3. Track duration statistics
4. Frame-wise object counts
5. Detection confidence distribution

Save analytics as JSON.

---

# Optional Enhancements

Implement if feasible:

## Heatmap

Generate movement heatmaps using tracked positions.

## Trajectory Visualization

Draw motion trails.

## Speed Estimation

Estimate relative speed in pixels/frame.

## Object Count Graph

Plot tracked object count over time.

---

# Coding Standards

Requirements:

* Modular architecture
* SOLID principles where practical
* Type hints
* Docstrings
* Logging
* Error handling
* Config-driven design

Avoid monolithic scripts.

Keep modules reusable.

---

# Deliverables To Generate

1. Complete Python source code.
2. requirements.txt
3. README.md
4. technical_report.md
5. Example commands to run project.
6. Sample screenshots directory structure.
7. Analytics outputs.
8. Demo workflow instructions.

---

# README Requirements

Include:

* Project Overview
* Architecture
* Installation
* Dependencies
* Running Instructions
* Video Source
* Assumptions
* Limitations
* Future Improvements

---

# Technical Report Requirements

1–2 pages.

Include:

* Problem Statement
* Detection Model Choice
* Tracking Algorithm Choice
* ID Persistence Strategy
* Challenges
* Failure Cases
* Improvements

Explain why YOLO + BoT-SORT was selected.

---

# Output Expectations

Generate code that can produce:

* tracked_output.mp4
* screenshots
* analytics.json
* heatmap.png
* trajectory visualization

The code should be executable with minimal modification.

Provide complete implementations rather than pseudocode whenever possible.

Think like a Senior Computer Vision Engineer building a portfolio-quality project for a hiring assignment.
