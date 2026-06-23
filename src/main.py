"""Command-line pipeline for multi-object detection and persistent ID tracking."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import yaml

from src.analytics.evaluator import evaluate_track_continuity
from src.analytics.metrics import TrackingMetrics
from src.analytics.statistics import confidence_histogram, plot_object_counts
from src.tracking.botsort_tracker import BotSortTracker
from src.tracking.tracker import TrackHistory
from src.utils.logger import setup_logger
from src.utils.video_reader import VideoReader
from src.utils.video_writer import VideoWriter
from src.visualization.annotator import TrackAnnotator
from src.visualization.heatmap import generate_heatmap
from src.visualization.trajectory import relative_speed


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO + BoT-SORT multi-object tracking pipeline")
    parser.add_argument("--input", type=Path, default=None, help="Input video path")
    parser.add_argument("--output", type=Path, default=None, help="Output annotated video path")
    parser.add_argument("--detector-config", type=Path, default=Path("configs/detector.yaml"))
    parser.add_argument("--tracker-config", type=Path, default=Path("configs/tracker.yaml"))
    parser.add_argument("--pipeline-config", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument("--no-video", action="store_true", help="Run analytics without writing video")
    parser.add_argument("--display", action="store_true", help="Display annotated frames while processing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_config = load_yaml(args.pipeline_config)
    logger = setup_logger(str(pipeline_config.get("log_level", "INFO")))

    input_video = args.input or Path(pipeline_config["input_video"])
    output_video = args.output or Path(pipeline_config["output_video"])
    analytics_json = Path(pipeline_config["analytics_json"])

    logger.info("Starting tracking pipeline")
    logger.info("Input video: %s", input_video)

    tracker = BotSortTracker(args.detector_config, args.tracker_config)
    history = TrackHistory()
    metrics = TrackingMetrics()
    annotator = TrackAnnotator(
        draw_trajectories=bool(pipeline_config.get("draw_trajectories", True)),
        trajectory_length=int(pipeline_config.get("trajectory_length", 40)),
        line_thickness=int(pipeline_config.get("line_thickness", 2)),
    )

    last_frame_shape: tuple[int, int, int] | None = None
    save_video = bool(pipeline_config.get("save_video", True)) and not args.no_video
    display_window = bool(pipeline_config.get("display_window", False)) or args.display

    with VideoReader(input_video) as reader:
        writer_context = (
            VideoWriter(output_video, reader.fps, (reader.width, reader.height))
            if save_video
            else None
        )

        try:
            for frame_index, frame in reader:
                tracks = list(tracker.update(frame, frame_index))
                history.update(tracks)
                metrics.update(frame_index, tracks)
                last_frame_shape = frame.shape

                annotated = annotator.draw(frame, tracks, history)
                if writer_context is not None:
                    writer_context.write(annotated)
                if display_window:
                    cv2.imshow("Multi-Object Tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if frame_index % 100 == 0:
                    logger.info("Processed frame %s with %s active tracks", frame_index, len(tracks))
        finally:
            if writer_context is not None:
                writer_context.release()
            if display_window:
                cv2.destroyAllWindows()

    summary = metrics.summarize()
    continuity = evaluate_track_continuity(metrics.track_df)
    summary["continuity_diagnostics"] = asdict(continuity)
    summary["relative_speed_pixels_per_frame"] = {
        str(track_id): round(relative_speed(points), 4)
        for track_id, points in history.points.items()
    }

    if pipeline_config.get("save_analytics", True):
        analytics_json.parent.mkdir(parents=True, exist_ok=True)
        analytics_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        metrics.save_tables(
            pipeline_config["frame_counts_csv"],
            pipeline_config["track_history_csv"],
        )

    if pipeline_config.get("save_plots", True):
        plot_object_counts(metrics.frame_df, pipeline_config["count_plot"])
        confidence_histogram(metrics.track_df, Path("reports/figures/confidence_distribution.png"))
        if last_frame_shape is not None:
            generate_heatmap(history, last_frame_shape, pipeline_config["heatmap_path"])

    logger.info("Completed tracking pipeline")
    logger.info("Unique objects tracked: %s", summary["total_unique_objects"])
    if save_video:
        logger.info("Annotated video saved to: %s", output_video)
    logger.info("Analytics saved to: %s", analytics_json)


if __name__ == "__main__":
    main()
