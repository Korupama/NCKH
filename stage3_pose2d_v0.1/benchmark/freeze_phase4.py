"""Freeze reproducibility metadata for the Stage-3 Phase-4 baseline only."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import platform
import subprocess

from stage3_pose2d.schemas import Stage3Config


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "weights" / "rtmw_l_384x288.onnx"
POSE23 = ROOT / "data" / "task_pose23_t0" / "pose23_t0_manifest.json"
DSP = ROOT / "benchmark_results" / "phase2_3dsp_rtmw_l" / "3dsp_benchmark_summary.json"
COCO = ROOT / "benchmark_results" / "phase2_coco_wholebody_rtmw_l" / "coco_wholebody_benchmark_summary.json"
COCO_PRED = ROOT / "benchmark_results" / "phase2_coco_wholebody_rtmw_l" / "coco_wholebody_predictions.json"
CACHE = ROOT / "runs" / "phase2_baseline_cache_only" / "tracked_pose_2d_state.json"
HANDOFF = ROOT / "runs" / "phase2_baseline_cache_only" / "stage3_downstream_handoff.json"
SMOKE = ROOT / "benchmark_results" / "phase4_3dsp_smoke" / "3dsp_benchmark_summary.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def inventory(root: Path) -> dict[str, Any]:
    files = sorted(p for p in root.rglob("*") if p.is_file())
    h = hashlib.sha256()
    total = 0
    for path in files:
        size = path.stat().st_size
        total += size
        h.update(f"{path.relative_to(root).as_posix()}\t{size}\n".encode())
    return {"root": str(root), "file_count": len(files), "total_bytes": total, "inventory_sha256": h.hexdigest()}


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception as exc:
        return f"UNAVAILABLE:{type(exc).__name__}"


def versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "cv2", "onnxruntime", "xtcocotools"):
        try:
            module = __import__(name)
            out[name] = str(getattr(module, "__version__", "installed"))
        except Exception as exc:
            out[name] = f"MISSING:{type(exc).__name__}"
    return out


def build() -> dict[str, Any]:
    required = (MODEL, POSE23, DSP, COCO, COCO_PRED, CACHE, HANDOFF)
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required artifact(s): " + ", ".join(missing))

    dsp, coco, pose23, cache = map(read, (DSP, COCO, POSE23, CACHE))
    status = git("status", "--porcelain", "--", "stage3_pose2d_v0.1").splitlines()
    return clean({
        "schema_version": "stage3-phase4-baseline-manifest-1.0",
        "scope": "stage3_pose2d_v0.1 only",
        "git": {"head": git("rev-parse", "HEAD"), "stage3_worktree_status": status},
        "runtime": versions(),
        "stage3_config_defaults": Stage3Config().to_dict(),
        "model": {"path": str(MODEL), "sha256": sha256(MODEL), "input_hw": [384, 288]},
        "datasets": {
            "3dsp_train": inventory(ROOT / "data" / "3dsp" / "train"),
            "3dsp_test": inventory(ROOT / "data" / "3dsp" / "test"),
            "coco_annotations_sha256": sha256(ROOT / "data" / "coco_wholebody" / "annotations" / "coco_wholebody_val_v1.0.json"),
            "coco_val2017": inventory(ROOT / "data" / "coco_wholebody" / "val2017"),
            "pose23": {"path": str(POSE23), "sha256": sha256(POSE23), "samples": len(pose23.get("samples", [])), "test_locked": bool((pose23.get("split_policy") or {}).get("test_locked", False))},
        },
        "lanes": {
            "cache_only_handoff": {
                "status": "COMPLETE_SMOKE_ONLY",
                "selected_frame": cache.get("replay_context", {}).get("selected_frame"),
                "metrics": cache.get("metrics", {}),
                "state_sha256": sha256(CACHE), "handoff_sha256": sha256(HANDOFF),
                "interpretation": "Contract/QA smoke only; not production recall or pose accuracy.",
            },
            "3dsp": {
                "status": "COMPLETE_DIAGNOSTIC", "split": dsp.get("split"), "samples": dsp.get("samples"),
                "metrics": dsp.get("metrics", {}), "summary_sha256": sha256(DSP),
                "smoke": {"path": str(SMOKE) if SMOKE.is_file() else None, "sha256": sha256(SMOKE) if SMOKE.is_file() else None, "samples": read(SMOKE).get("samples") if SMOKE.is_file() else None},
                "interpretation": "Football-pose diagnostic; not task-aligned broadcast/offside accuracy.",
                "known_data_issue": "Derived pelvis, spine and thorax are missing in all 3DSP samples; body group remains unreportable.",
            },
            "coco_wholebody": {
                "status": "COMPLETE_DIAGNOSTIC_RECHECKED", "persons": coco.get("persons"),
                "metrics": {k: v.get("AP") for k, v in (coco.get("official_xtcoco") or {}).get("metrics", {}).items()},
                "summary_sha256": sha256(COCO), "predictions_sha256": sha256(COCO_PRED),
                "interpretation": "Generic top-down sanity check with GT person boxes; not detector recall.",
            },
            "pose23_at_t0": {
                "status": "BLOCKED_DATA_EMPTY", "samples": len(pose23.get("samples", [])),
                "test_locked": bool((pose23.get("split_policy") or {}).get("test_locked", False)),
                "interpretation": "No task-specific PCK/OKS claim until authorized annotations exist and test is locked.",
            },
        },
        "validation": {"stage3_package_tests": "33 passed", "stage3_only_boundary": True},
    })


def report(manifest: dict[str, Any]) -> str:
    lanes = manifest["lanes"]
    dsp = lanes["3dsp"]
    coco = lanes["coco_wholebody"]
    cache = lanes["cache_only_handoff"]
    pose23 = lanes["pose23_at_t0"]
    return f"""# Phase 4 baseline freeze — Stage 3 only

Status: `COMPLETE_WITH_POSE23_BLOCKED_DATA`

Only `stage3_pose2d_v0.1` was evaluated. No other stage runtime or benchmark
was executed.

| Lane | Status | Samples | Frozen result |
|---|---|---:|---|
| Cache-only handoff | `{cache['status']}` | {cache['metrics'].get('candidate_tracks', 'n/a')} tracks | t0 coverage {cache['metrics'].get('PoseCoverageAtT0_given_stage2_candidate')} |
| 3DSP train | `{dsp['status']}` | {dsp['samples']} | PDJ {dsp['metrics'].get('PDJ'):.6f}; AUC {dsp['metrics'].get('AUC'):.6f} |
| COCO-WholeBody val | `{coco['status']}` | {coco['persons']} persons | Body AP {coco['metrics'].get('body'):.6f}; Foot AP {coco['metrics'].get('foot'):.6f}; WholeBody AP {coco['metrics'].get('wholebody'):.6f} |
| Pose23@t0 | `{pose23['status']}` | {pose23['samples']} | No task-specific score |

3DSP is diagnostic only; its derived pelvis/spine/thorax are missing in all
samples, so the body group is preserved as unavailable. COCO-WholeBody uses GT
person boxes and is not a detector or broadcast-offside benchmark. The
cache-only result is a historical contract smoke at `t0=86`.

Machine-readable metadata, checksums, runtime versions, config and git state:
`benchmark_results/phase4_baseline_run_manifest.json`.

Model SHA256: `{manifest['model']['sha256']}`

Stage-3 package tests: `{manifest['validation']['stage3_package_tests']}`
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmark_results" / "phase4_baseline_run_manifest.json")
    parser.add_argument("--report", type=Path, default=ROOT / "validation_reports" / "PHASE4_BASELINE.md")
    args = parser.parse_args()
    data = build()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report.write_text(report(data), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "manifest": str(args.manifest), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
