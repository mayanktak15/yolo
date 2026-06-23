"""Plotting helpers for tracking statistics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_object_counts(frame_df: pd.DataFrame, output_path: str | Path) -> None:
    """Save a line chart of active tracked objects per frame."""

    if frame_df.empty:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(frame_df["frame"], frame_df["active_tracks"], color="#2364aa", linewidth=2)
    plt.title("Tracked Object Count Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Active Tracks")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def confidence_histogram(track_df: pd.DataFrame, output_path: str | Path) -> None:
    """Save a histogram of detection confidence scores."""

    if track_df.empty:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.hist(track_df["confidence"], bins=20, color="#3da35d", edgecolor="white")
    plt.title("Detection Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

