from src.analytics.metrics import TrackingMetrics
from src.tracking.tracker import TrackedObject


def test_tracking_metrics_summary_counts_unique_ids() -> None:
    metrics = TrackingMetrics()
    metrics.update(
        0,
        [
            TrackedObject(1, "person", 0.9, (0, 0, 10, 10), 0),
            TrackedObject(2, "person", 0.8, (10, 10, 20, 20), 0),
        ],
    )
    metrics.update(1, [TrackedObject(1, "person", 0.95, (1, 1, 11, 11), 1)])

    summary = metrics.summarize()

    assert summary["total_unique_objects"] == 2
    assert summary["average_active_tracks_per_frame"] == 1.5
    assert summary["track_duration_statistics"]["max_frames"] == 2

