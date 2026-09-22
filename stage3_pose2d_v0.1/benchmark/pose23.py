"""Backward-compatible entry point for the Stage-3 Pose23 evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .pose23_task import evaluate_pose23_task


def evaluate_pose23(gt_path: str | Path, pred_path: str | Path) -> Dict[str, Any]:
    """Run the task-aware Pose23@t0 protocol.

    The name is retained so existing Stage-3 callers keep working while the
    implementation now reports visibility-aware, per-keypoint, group,
    difficulty-slice and provenance metrics.
    """

    return evaluate_pose23_task(gt_path, pred_path)
