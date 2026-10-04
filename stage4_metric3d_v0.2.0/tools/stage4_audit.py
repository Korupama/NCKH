#!/usr/bin/env python3
"""Create a fail-closed Phase-0 provenance report for Stage 4.

The audit inventories existing artifacts and hashes them. It never runs Stage 1
or Stage 3, creates missing inputs, or overwrites a Stage-4 experiment folder.
Missing real artifacts keep the frame baseline unavailable and metric accuracy
unevaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "stage3_state": ROOT.parent / "stage3_pose2d_v0.1" / "runs" / "stage3_real_1920" / "tracked_pose_2d_state.json",
    "stage1_camera_dir": ROOT.parent / "stage_1_camera_v12" / "outputs" / "temporal_v13" / "batch_video" / "shot_ptz" / "optimized_camera_states",
    "source_video": ROOT.parent / "stage_1_camera_v12" / "download.mp4",
    "frames_dir": ROOT.parent / "stage_1_camera_v12" / "frames",
    "sam3d_cache": ROOT / "runs" / "stage4_v051_frame104" / "sam3d_native_frame104.npz",
    "refined_output": ROOT / "runs" / "stage4_v051_frame104" / "sam3d-pitch-refined" / "world_grounded_pose_state.json",
    "direct_output": ROOT / "runs" / "stage4_v051_frame104" / "sam3d-direct" / "world_grounded_pose_state.json",
    "sam3d_checkpoint": ROOT.parent / "pretrained_models" / "sam3d_body" / "model.ckpt",
    "mhr_model": ROOT.parent / "pretrained_models" / "sam3d_body" / "assets" / "mhr_model.pt",
    "sam3d_repository": ROOT.parent / "third_party" / "sam-3d-body",
}

PRODUCTION_FILES = (
    "tools/stage4_phase0_audit.py",
    "run_sam3d_model_only.py",
    "evaluate_model_only.py",
    "compare_model_only_systems.py",
    "stage4_metric3d/model_only.py",
    "stage4_metric3d/model_only_refinement.py",
    "stage4_metric3d/model_only_evaluation.py",
    "stage4_metric3d/model_only_systems.py",
    "run_stage4.py",
    "run_sam3d_body_worker.py",
    "stage4_metric3d/camera.py",
    "stage4_metric3d/stage3_adapter.py",
    "stage4_metric3d/schemas.py",
    "stage4_metric3d/output_semantics.py",
    "stage4_metric3d/backends/sam3d_pitch_refined/config.py",
    "stage4_metric3d/backends/sam3d_pitch_refined/pipeline.py",
    "stage4_metric3d/backends/sam3d_pitch_refined/refiner.py",
    "stage4_metric3d/backends/sam3d_pitch_refined/ground_anchor.py",
    "stage4_metric3d/backends/sam3d_pitch_refined/geometry.py",
    "stage4_metric3d/backends/sam3d_pitch_refined/quality.py",
    "stage4_metric3d/backends/sam3d_pitch_refined/metrics.py",
    "stage4_metric3d/backends/sam3d_pitch_refined/visualization.py",
    "stage4_metric3d/backends/sam3d_pitch_refined/cache.py",
    "stage4_metric3d/backends/sam3d_pitch_refined/joint_mapping.py",
    "stage4_metric3d/backends/sam3d_pitch_refined/__init__.py",
    "tests_v05/test_refinement.py",
    "tests_v05/test_ground_first.py",
    "tests_v05/test_cache_and_mapping.py",
    "tests/test_output_semantics.py",
    "tests/test_visualization.py",
    "tests/test_phase0_audit.py",
    "tests/test_model_only_manifest.py",
    "tests/test_model_only_refinement.py",
    "tests/test_model_only_evaluation.py",
    "tests/test_model_only_systems.py",
)

BACKENDS = (
    {
        "name": "sam3d-pitch-refined",
        "production_candidate": True,
        "coordinate_scope": "PITCH_WORLD_METRIC",
        "role": "SAM3D MHR70 plus root-only pitch/ground refinement",
    },
    {
        "name": "sam3d-direct",
        "production_candidate": False,
        "coordinate_scope": "PITCH_WORLD_METRIC",
        "role": "SAM3D pred_cam_t direct ablation",
    },
    {
        "name": "rtmw3d",
        "production_candidate": False,
        "coordinate_scope": "ROOT_RELATIVE_METRIC",
        "role": "relative-depth initializer; not global pitch placement",
    },
    {
        "name": "kasportsformer",
        "production_candidate": False,
        "coordinate_scope": "SCALE_AMBIGUOUS",
        "role": "relative sports pose comparison",
    },
    {
        "name": "field-converter-tcn",
        "production_candidate": False,
        "coordinate_scope": "PITCH_WORLD_METRIC",
        "role": "retained optional backend; exact normalization artifact required",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    record: dict[str, Any] = {
        "path": str(resolved),
        "relative_to_stage4": _relative(resolved),
        "exists": resolved.is_file(),
    }
    if resolved.is_file():
        record.update({"bytes": resolved.stat().st_size, "sha256": _sha256(resolved)})
    return record


def _directory_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    files = sorted(p for p in resolved.rglob("*") if p.is_file()) if resolved.is_dir() else []
    entries = [{"path": str(p), "bytes": p.stat().st_size, "sha256": _sha256(p)} for p in files]
    manifest = "\n".join(f"{entry['path']}\t{entry['sha256']}" for entry in entries)
    return {
        "path": str(resolved),
        "exists": resolved.is_dir(),
        "usable": resolved.is_dir() and bool(entries),
        "file_count": len(entries),
        "manifest_sha256": hashlib.sha256(manifest.encode("utf-8")).hexdigest() if entries else None,
        "files": entries,
    }


def _json_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.suffix.lower() != ".json":
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"parseable": False}
    if not isinstance(value, dict):
        return {"parseable": True, "type": type(value).__name__}
    tracks = value.get("tracks") or []
    replay = value.get("replay_context")
    selected_frame = value.get("selected_frame")
    if selected_frame is None and isinstance(replay, dict):
        selected_frame = replay.get("selected_frame")
    return {
        "parseable": True,
        "schema_version": value.get("schema_version"),
        "stage4_version": value.get("stage4_version"),
        "selected_frame": selected_frame,
        "track_count": len(tracks) if isinstance(tracks, list) else None,
        "quality_gates": value.get("quality_gates"),
        "research_accuracy_frozen": value.get("quality_gates", {}).get("research_accuracy_frozen")
        if isinstance(value.get("quality_gates"), dict)
        else None,
    }


def _resolve(value: str | None, default: Path) -> Path:
    return Path(value).expanduser().resolve() if value else default.resolve()


MANIFEST_KEYS = (
    "stage3_state",
    "stage1_camera_dir",
    "source_video",
    "frames_dir",
    "sam3d_cache",
    "refined_output",
    "direct_output",
    "sam3d_checkpoint",
    "mhr_model",
    "sam3d_repository",
)


def _load_manifest(path: Path) -> dict[str, Path]:
    """Load explicit Phase-0 paths; relative paths are relative to the manifest."""
    manifest_path = path.expanduser().resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Phase-0 manifest must contain a JSON object")
    if data.get("schema_version") != "stage4-phase0-input-manifest-1.0":
        raise ValueError("Unsupported Phase-0 manifest schema")
    raw_paths = data.get("paths", data)
    if not isinstance(raw_paths, dict):
        raise ValueError("Phase-0 manifest paths must be a JSON object")
    result: dict[str, Path] = {}
    for key in MANIFEST_KEYS:
        value = raw_paths.get(key)
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            raise ValueError(f"Manifest path must be a string or null: {key}")
        candidate = Path(value).expanduser()
        result[key] = (candidate if candidate.is_absolute() else manifest_path.parent / candidate).resolve()
    return result


def _sam3d_package_available() -> bool:
    try:
        return importlib.util.find_spec("sam_3d_body") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _git_revision() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(ROOT.parent), "status", "--short", "--", ROOT.name],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.splitlines()
        return {"head": revision, "stage4_worktree_entries": status, "available": True}
    except (OSError, subprocess.SubprocessError):
        return {"head": None, "stage4_worktree_entries": [], "available": False}


def build_report(
    *,
    paths: dict[str, Path] | None = None,
    tests_passed: int | None = None,
    tests_failed: int | None = None,
    test_command: str = "PYTHONPATH=. python -m pytest -q tests tests_v04 tests_v05",
) -> dict[str, Any]:
    selected = dict(DEFAULTS)
    selected.update(paths or {})
    production = {name: _file_record(ROOT / relative) for name, relative in ((p, p) for p in PRODUCTION_FILES)}
    configs = {
        "stage4_v05_config": _file_record(ROOT / "configs" / "stage4_v05_sam3d_pitch_refined.json"),
        "mhr70_mapping": _file_record(ROOT / "schemas" / "sam3d_mhr70_to_rtmw_wholebody_v1.json"),
    }
    inputs = {
        "stage3_state": _file_record(selected["stage3_state"]),
        "stage1_camera_dir": _directory_record(selected["stage1_camera_dir"]),
        "source_video": _file_record(selected["source_video"]),
        "frames_dir": _directory_record(selected["frames_dir"]),
        "sam3d_native_cache": _file_record(selected["sam3d_cache"]),
        "sam3d_checkpoint": _file_record(selected["sam3d_checkpoint"]),
        "mhr_model": _file_record(selected["mhr_model"]),
        "sam3d_repository": _directory_record(selected["sam3d_repository"]),
    }
    sam3d_package_available = _sam3d_package_available()
    inputs["stage3_state"]["summary"] = _json_summary(Path(inputs["stage3_state"]["path"]))
    outputs = {
        "sam3d_pitch_refined": _file_record(selected["refined_output"]),
        "sam3d_direct": _file_record(selected["direct_output"]),
    }
    for key, record in outputs.items():
        record["summary"] = _json_summary(Path(record["path"]))

    required_replay_inputs = (
        inputs["stage3_state"],
        inputs["stage1_camera_dir"],
        inputs["sam3d_native_cache"],
    )
    source_available = inputs["source_video"]["exists"] or inputs["frames_dir"].get("usable", False)
    pretrained_inference_assets_available = (
        inputs["sam3d_checkpoint"]["exists"]
        and inputs["mhr_model"]["exists"]
        and (inputs["sam3d_repository"].get("usable", False) or sam3d_package_available)
    )
    integration_cache_generation_inputs = (
        inputs["stage3_state"],
        inputs["stage1_camera_dir"],
    )
    replay_inputs_available = all(item["exists"] if "usable" not in item else item["usable"] for item in required_replay_inputs)
    integration_cache_generation_inputs_available = source_available and pretrained_inference_assets_available and all(
        item.get("usable", item["exists"]) for item in integration_cache_generation_inputs
    )
    required_outputs = (outputs["sam3d_pitch_refined"], outputs["sam3d_direct"])
    inputs_available = replay_inputs_available
    outputs_available = all(item["exists"] for item in required_outputs)
    baseline_available = inputs_available and outputs_available
    tests_pass = tests_failed == 0 and tests_passed is not None and tests_passed > 0
    missing = [
        name for name, item in inputs.items()
        if name not in {"source_video", "frames_dir"}
        and name != "sam3d_repository"
        and (item.get("usable", item["exists"]) is False)
    ]
    if not pretrained_inference_assets_available:
        missing.extend(
            key for key in ("sam3d_checkpoint", "mhr_model")
            if not inputs[key]["exists"] and key not in missing
        )
        if not (inputs["sam3d_repository"].get("usable", False) or sam3d_package_available):
            if "sam3d_repository_or_installed_package" not in missing:
                missing.append("sam3d_repository_or_installed_package")
    if not source_available:
        if "source_video_or_frames_dir" not in missing:
            missing.append("source_video_or_frames_dir")
    missing_model_only = [
        key for key in ("sam3d_checkpoint", "mhr_model")
        if not inputs[key]["exists"]
    ]
    if not (inputs["sam3d_repository"].get("usable", False) or sam3d_package_available):
        missing_model_only.append("sam3d_repository_or_installed_package")
    missing_integration = [
        key for key in ("stage3_state", "stage1_camera_dir", "sam3d_native_cache")
        if not inputs[key].get("usable", inputs[key]["exists"])
    ]
    status = "PASS_IMPLEMENTATION" if tests_pass else "NOT_EVALUATED"
    if not inputs_available:
        baseline_status = "NOT_AVAILABLE_INTEGRATION_INPUTS"
        missing_baseline = [
            key for key in ("stage3_state", "stage1_camera_dir", "sam3d_native_cache")
            if not inputs[key].get("usable", inputs[key]["exists"])
        ]
    elif not outputs_available:
        baseline_status = "INTEGRATION_OUTPUTS_MISSING"
        missing_baseline = [key for key in ("sam3d_pitch_refined", "sam3d_direct") if not outputs[key]["exists"]]
    else:
        baseline_status = "ARTIFACTS_AVAILABLE"
        missing_baseline = []
    return {
        "schema_version": "stage4-phase0-audit-1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage4_version": "0.5.2",
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "repository": _git_revision(),
        "scope": "stage4_only",
        "upstream_read_only": True,
        "downstream_read_only": True,
        "backends": list(BACKENDS),
        "production_source_files": production,
        "configs_and_mappings": configs,
        "inputs": inputs,
        "outputs": outputs,
        "test_evidence": {
            "command": test_command,
            "passed": tests_passed,
            "failed": tests_failed,
            "status": "PASS" if tests_pass else "NOT_RECORDED",
        },
        "acceptance": {
            "implementation_gate": status,
            "model_only_development": "ALLOWED" if tests_pass else "NOT_EVALUATED",
            "model_only_pretrained_inference": "READY" if pretrained_inference_assets_available else "BLOCKED_MISSING_STAGE4_MODEL_ASSETS",
            "model_only_accuracy": "NOT_EVALUATED_NO_INDEPENDENT_BENCHMARK_RECORDED",
            "integration_frame104": baseline_status,
            "baseline_frame104": baseline_status,
            "integration_metric_accuracy": "NOT_EVALUATED",
            "metric_accuracy": "NOT_EVALUATED",
            "self_consistency": "SANITY_ONLY" if baseline_available else "NOT_AVAILABLE",
            "baseline_inputs_available": inputs_available,
            "replay_inputs_available": replay_inputs_available,
            "pretrained_inference_assets_available": pretrained_inference_assets_available,
            "sam3d_package_available_in_audit_environment": sam3d_package_available,
            "integration_cache_generation_inputs_available": integration_cache_generation_inputs_available,
            "source_frame_input_available": source_available,
            "baseline_outputs_available": outputs_available,
            "research_accuracy_frozen": False,
            "missing_input_keys": missing,
            "missing_model_only_keys": missing_model_only,
            "missing_integration_keys": missing_integration,
            "missing_baseline_keys": missing_baseline,
            "no_upstream_regeneration": True,
            "accuracy_claim_allowed": False,
        },
        "semantics": {
            "canonical_coordinate_scope": "PITCH_WORLD_METRIC",
            "root_only_refinement": True,
            "self_consistency_is_not_metric_ground_truth": True,
            "metric_ground_truth_required_for_accuracy": True,
            "selected_frame": 104,
        },
        "commands": {
            "audit_invocation": " ".join(str(x) for x in sys.argv),
            "baseline_run": "powershell -ExecutionPolicy Bypass -File .\\rerun_frame104_v051.ps1",
            "test_run": test_command,
        },
        "recovery": {
            "replay_cache": {
                "ready": replay_inputs_available,
                "missing": [
                    key for key in ("stage3_state", "stage1_camera_dir", "sam3d_native_cache")
                    if not inputs[key].get("usable", inputs[key]["exists"])
                ],
                "action": "Restore the matching Stage-3 state, Stage-1 camera directory and native SAM3D cache; do not substitute another frame or sequence.",
            },
            "model_only": {
                "development_allowed": "ALLOWED" if tests_pass else "NOT_EVALUATED",
                "pretrained_inference_ready": pretrained_inference_assets_available,
                "missing_model_assets": [
                    key for key in ("sam3d_checkpoint", "mhr_model") if not inputs[key]["exists"]
                ] + ([] if inputs["sam3d_repository"].get("usable", False) or sam3d_package_available else ["sam3d_repository_or_installed_package"]),
                "upstream_stage1_stage3_required": False,
                "action": "Use Stage-4 pretrained backends on benchmark RGB/person crops and benchmark GT; use benchmark-provided metric camera/pose only for global-coordinate claims. Stage-1/3 integration is a separate gate.",
            },
            "generate_integration_cache": {
                "ready": integration_cache_generation_inputs_available,
                "missing": [
                    key for key in ("stage3_state", "stage1_camera_dir")
                    if not inputs[key].get("usable", inputs[key]["exists"])
                ] + ([] if pretrained_inference_assets_available else ["stage4_pretrained_model_assets"])
                  + ([] if source_available else ["source_video_or_frames_dir"]),
                "action": "Optional integration lane only: after restoring matching upstream inputs and Stage-4 SAM3D assets, run the Stage-4 worker to create a native cache. This is not a prerequisite for Stage-4-only model development.",
            },
        },
        "limitations": [
            "The audit does not create Stage-1, Stage-3, SAM3D or metric-GT artifacts.",
            "Ground/contact and reprojection residuals are self-consistency diagnostics because they are optimization evidence.",
            "Metric accuracy and temporal refinement accuracy remain NOT_EVALUATED without independent metric ground truth and real cache inputs.",
        ],
    }


def _render_markdown(report: dict[str, Any]) -> str:
    acceptance = report["acceptance"]
    lines = [
        "# Stage 4 Phase 0 — Baseline provenance and acceptance",
        "",
        f"- Generated (UTC): `{report['generated_at_utc']}`",
        f"- Stage 4: `{report['stage4_version']}`",
        f"- Git HEAD: `{report['repository'].get('head') or 'NOT_AVAILABLE'}`",
        f"- Scope: `{report['scope']}`; upstream read-only: `{report['upstream_read_only']}`",
        "",
        "## Acceptance status",
        "",
        "| Gate | Result |",
        "|---|---|",
        f"| Implementation/tests | {acceptance['implementation_gate']} |",
        f"| Stage-4-only model development | {acceptance['model_only_development']} |",
        f"| Pretrained inference assets | {acceptance['model_only_pretrained_inference']} |",
        f"| Independent model accuracy | {acceptance['model_only_accuracy']} |",
        f"| Optional Stage-1/3 frame-104 integration | {acceptance['integration_frame104']} |",
        f"| Metric accuracy | {acceptance['metric_accuracy']} |",
        f"| Self-consistency diagnostics | {acceptance['self_consistency']} |",
        f"| Research accuracy frozen | {acceptance['research_accuracy_frozen']} |",
        "",
        "Accuracy is not claimed from synthetic tests, ground/contact residuals, pitch bounds, or reprojection residuals.",
        "Stage-1/Stage-3 artifacts are required only for the separate integration baseline; they do not gate Stage-4-only backend development or dataset-based model evaluation.",
        "",
        "## Test evidence",
        "",
        f"- Command: `{report['test_evidence']['command']}`",
        f"- Result: `{report['test_evidence']['status']}`; passed `{report['test_evidence']['passed']}`, failed `{report['test_evidence']['failed']}`",
        "",
        "## Input and output availability",
        "",
        "| Artifact | Exists | SHA256 / manifest SHA256 |",
        "|---|---:|---|",
    ]
    for name, record in report["inputs"].items():
        lines.append(f"| `{name}` | {record['exists']} | `{record.get('sha256') or record.get('manifest_sha256') or 'NOT_AVAILABLE'}` |")
    for name, record in report["outputs"].items():
        lines.append(f"| `{name}` | {record['exists']} | `{record.get('sha256') or 'NOT_AVAILABLE'}` |")
    lines.extend(["", "Missing optional frame-104 integration artifacts:"])
    if acceptance["missing_baseline_keys"]:
        lines.extend(f"- `{item}`" for item in acceptance["missing_baseline_keys"])
    else:
        lines.append("- None")
    stage3_summary = report["inputs"]["stage3_state"].get("summary") or {}
    lines.extend([
        "",
        "## Selected-frame compatibility",
        "",
        f"- Requested baseline frame: `{report['semantics']['selected_frame']}`",
        f"- Stage-3 selected frame: `{stage3_summary.get('selected_frame', 'NOT_AVAILABLE')}`",
        f"- Stage-3 schema: `{stage3_summary.get('schema_version', 'NOT_AVAILABLE')}`",
        "",
        "A state from another sequence/frame is not accepted as a replacement for the frame-104 baseline.",
    ])
    recovery = report["recovery"]
    lines.extend([
        "",
        "## Recovery readiness",
        "",
        "| Route | Ready | Missing |",
        "|---|---:|---|",
        f"| Replay existing native cache | {recovery['replay_cache']['ready']} | `{', '.join(recovery['replay_cache']['missing']) or 'None'}` |",
        f"| Stage-4-only pretrained inference | {recovery['model_only']['pretrained_inference_ready']} | `{', '.join(recovery['model_only']['missing_model_assets']) or 'None'}` |",
        f"| Generate integration cache (optional) | {recovery['generate_integration_cache']['ready']} | `{', '.join(recovery['generate_integration_cache']['missing']) or 'None'}` |",
        "",
        "The audit is fail-closed: it does not replace missing inputs with fixtures or rerun Stage 1/3. A missing integration baseline does not block independent Stage-4 work.",
    ])
    lines.extend(["", "## Production source and configuration hashes", "", "| File | SHA256 |", "|---|---|"])
    for name, record in report["production_source_files"].items():
        lines.append(f"| `{name}` | `{record.get('sha256') or 'MISSING'}` |")
    for name, record in report["configs_and_mappings"].items():
        lines.append(f"| `{name}` | `{record.get('sha256') or 'MISSING'}` |")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a fail-closed Stage-4 Phase-0 provenance report")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "STAGE4_PHASE0_BASELINE.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs" / "STAGE4_PHASE0_BASELINE_REPORT.md")
    parser.add_argument("--stage3-state", type=Path)
    parser.add_argument("--stage1-camera-dir", type=Path)
    parser.add_argument("--source-video", type=Path)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--sam3d-cache", type=Path)
    parser.add_argument("--refined-output", type=Path)
    parser.add_argument("--direct-output", type=Path)
    parser.add_argument("--sam3d-checkpoint", type=Path)
    parser.add_argument("--mhr-model", type=Path)
    parser.add_argument("--sam3d-repository", type=Path)
    parser.add_argument("--manifest", type=Path, help="JSON input manifest; relative paths resolve from the manifest directory")
    parser.add_argument("--tests-passed", type=int)
    parser.add_argument("--tests-failed", type=int)
    parser.add_argument("--test-command", default="PYTHONPATH=. python -m pytest -q tests tests_v04 tests_v05")
    args = parser.parse_args()
    paths: dict[str, Path] = {}
    if args.manifest is not None:
        paths.update(_load_manifest(args.manifest))
    paths.update({key: value for key, value in {
        "stage3_state": args.stage3_state,
        "stage1_camera_dir": args.stage1_camera_dir,
        "source_video": args.source_video,
        "frames_dir": args.frames_dir,
        "sam3d_cache": args.sam3d_cache,
        "refined_output": args.refined_output,
        "direct_output": args.direct_output,
        "sam3d_checkpoint": args.sam3d_checkpoint,
        "mhr_model": args.mhr_model,
        "sam3d_repository": args.sam3d_repository,
    }.items() if value is not None})
    report = build_report(paths=paths, tests_passed=args.tests_passed, tests_failed=args.tests_failed, test_command=args.test_command)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["acceptance"], "output": str(args.output.resolve())}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
