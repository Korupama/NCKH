#!/usr/bin/env python3
"""Stage-3-only Phase-0 audit.

Reads an existing Stage-2 handoff/cache, runs the current Stage-3 processor,
and writes a provenance/QA report outside the Stage-2 directory. Stage 2 is
never imported or executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage3_pose2d import Stage3Config, run_stage3
from stage3_pose2d.stage2_adapter import load_stage2_bundle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_track(track: Dict[str, Any], selected_frame: int) -> Dict[str, Any]:
    observation = next((item for item in track.get("observations", []) if int(item.get("frame_index", -1)) == selected_frame), None)
    if observation is None:
        return {"track_id": track.get("track_id"), "selected_frame": selected_frame, "status": "MISSING_OBSERVATION"}
    qa = observation.get("qa") or {}
    return {
        "track_id": track.get("track_id"),
        "upstream_role": track.get("upstream_role"),
        "selected_frame": selected_frame,
        "pose_status": observation.get("pose_status"),
        "bbox_xyxy": observation.get("source_bbox_xyxy"),
        "source_backend": observation.get("source_backend"),
        "source_score_semantics": observation.get("source_score_semantics"),
        "keypoint_count": len(observation.get("keypoints_133") or []),
        "qa": qa,
        "temporal_qa": track.get("temporal_qa") or {},
        "provenance": observation.get("provenance") or {},
        "risk_flags": {
            "low_model_evidence": float(qa.get("low_model_evidence_fraction", 0.0)) > 0.25,
            "geometry_failed": not bool(qa.get("geometry_valid", False)),
            "feet_incomplete": float(qa.get("feet_completeness", 0.0)) < 0.50,
            "inside_fraction_low": float(qa.get("inside_fraction", 0.0)) < 0.75,
            "ownership_support_low": (
                (qa.get("ownership") or {}).get("ownership_status")
                in {"OUTSIDE_SOURCE", "CENTER_MISMATCH", "WEAK_SUPPORT", "NEIGHBOR_DOMINANT"}
            ),
            "ownership_status": (qa.get("ownership") or {}).get("ownership_status", "UNKNOWN"),
            "temporal_ownership_switch": bool(
                selected_frame in set((track.get("temporal_qa") or {}).get("ownership_switch_suspected_frames") or [])
            ),
            "temporal_left_right_swap": bool(
                selected_frame in set((track.get("temporal_qa") or {}).get("left_right_swap_suspected_frames") or [])
            ),
            "crop_status": (qa.get("crop_diagnostics") or {}).get("crop_status", "UNKNOWN"),
            "crop_failure": (qa.get("crop_diagnostics") or {}).get("crop_status", "UNKNOWN") not in {"OK", "UNKNOWN"},
            "temporal_reasons": [reason for reason in (qa.get("status_reasons") or []) if "temporal" in str(reason).lower()],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-3-only hallucination baseline audit")
    parser.add_argument("--stage2-dir", type=Path, required=True, help="Existing Stage-2 handoff/cache directory; read-only")
    parser.add_argument("--output-dir", type=Path, required=True, help="Stage-3 audit output directory")
    parser.add_argument("--rtmw-model", type=Path, default=ROOT.parent / "datasets" / "stage3_assets" / "rtmw_l_384x288.onnx")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    stage2_dir = args.stage2_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.rtmw_model.expanduser().resolve()

    bundle = load_stage2_bundle(stage2_dir=stage2_dir, strict=True)
    config = Stage3Config(rtmw_model=str(model_path), rtmw_device=args.device)
    state = run_stage3(stage2_dir=stage2_dir, output_dir=output_dir / "stage3_output", config=config)

    report = {
        "schema_version": "stage3-pose-hallucination-audit-1.0",
        "scope": "stage3_only",
        "stage2_read_only": True,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "inputs": {
            "stage2_dir": str(stage2_dir),
            "entity_state": str(bundle.entity_state_path),
            "rtmw_cache": str(bundle.rtmw_cache_path),
            "stage3_handoff": str(bundle.handoff_path),
            "entity_state_sha256": _sha256(bundle.entity_state_path),
            "rtmw_cache_sha256": _sha256(bundle.rtmw_cache_path),
            "stage3_handoff_sha256": _sha256(bundle.handoff_path),
            "rtmw_model": str(model_path),
            "rtmw_model_sha256": _sha256(model_path),
        },
        "selected_frame": bundle.selected_frame,
        "preflight": bundle.validation,
        "configuration": config.to_dict(),
        "metrics": state.get("metrics", {}),
        "tracks": [_audit_track(track, bundle.selected_frame) for track in state.get("tracks", [])],
        "artifacts": state.get("artifacts", {}),
        "acceptance_gate": state.get("acceptance_gate", {}),
        "notes": [
            "This report evaluates Stage 3 pose evidence only; it does not re-evaluate Stage-2 detector recall or referee classification.",
            "A VALID status is an engineering QA result, not a ground-truth accuracy claim.",
            "Stage-2 files are not modified by this command.",
        ],
    }
    report_path = output_dir / "pose_hallucination_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "selected_frame": bundle.selected_frame, "report": str(report_path), "metrics": state.get("metrics", {})}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
