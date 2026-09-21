from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import subprocess
import sys

from .contracts import FieldConverterBundle


_REQUIRED_PAPER_INPUTS = {
    "use_x3d_sam_rel": True,
    "use_x2d_img": True,
    "use_x2d_box": True,
    "use_pitch_points_2d": True,
    "use_bbox_feat": True,
    "use_cam_feat": True,
    "use_ground_intersection": True,
    "use_valid_joints_as_input": True,
}


def resolve_bundle(
    *,
    field_converter_repo: str | Path | None,
    python_exe: str | Path | None,
    config_path: str | Path,
    checkpoint_path: str | Path,
    normalization_stats_path: str | Path,
    pitch_points_path: str | Path,
) -> FieldConverterBundle:
    repo = None if field_converter_repo is None else Path(field_converter_repo).expanduser().resolve()
    py = Path(python_exe or sys.executable).expanduser().resolve()
    return FieldConverterBundle(
        repo=repo,
        python_exe=py,
        config_path=Path(config_path).expanduser().resolve(),
        checkpoint_path=Path(checkpoint_path).expanduser().resolve(),
        normalization_stats_path=Path(normalization_stats_path).expanduser().resolve(),
        pitch_points_path=Path(pitch_points_path).expanduser().resolve(),
    )


def _config_contract(path: Path) -> dict[str, Any]:
    try:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(raw, dict):
        return {"ready": False, "error": "top-level YAML is not a mapping"}
    input_cfg = raw.get("input_config") or {}
    mismatches = {
        key: {"expected": expected, "actual": input_cfg.get(key)}
        for key, expected in _REQUIRED_PAPER_INPUTS.items()
        if bool(input_cfg.get(key, False)) != bool(expected)
    }
    checks = {
        "prediction_mode_delta": raw.get("prediction_mode") == "delta",
        "window_size_41": int((raw.get("dataset") or {}).get("window_size", -1)) == 41,
        "input_features_match_paper_tcn": not mismatches,
        "sam3d_relative_enabled": bool(input_cfg.get("use_x3d_sam_rel", False)),
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "input_feature_mismatches": mismatches,
        "run_name": raw.get("run_name"),
        "cam_feat_type": input_cfg.get("cam_feat_type"),
        "bbox_clean_or_noisy": input_cfg.get("bbox_clean_or_noisy"),
    }


def validate_bundle(bundle: FieldConverterBundle) -> dict[str, Any]:
    files = {
        "python_exe": bundle.python_exe,
        "config_path": bundle.config_path,
        "checkpoint_path": bundle.checkpoint_path,
        "normalization_stats_path": bundle.normalization_stats_path,
        "pitch_points_path": bundle.pitch_points_path,
    }
    checks = {name: bool(path.is_file()) for name, path in files.items()}
    checks["normalization_stats_filename"] = bundle.normalization_stats_path.name == "normalization_stats.npz"
    checks["pitch_points_filename"] = bundle.pitch_points_path.name == "pitch_points.txt"
    repo_ok = True
    if bundle.repo is not None:
        repo_ok = bool((bundle.repo / "src" / "field_converter" / "__init__.py").is_file())
    checks["field_converter_repo"] = repo_ok
    config_contract = _config_contract(bundle.config_path) if bundle.config_path.is_file() else {"ready": False, "error": "config_missing"}
    checks["paper_tcn_config_contract"] = bool(config_contract.get("ready"))
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "config_contract": config_contract,
        "bundle": bundle.to_dict(),
        "errors": [name for name, ok in checks.items() if not ok],
    }


def _python_env(bundle: FieldConverterBundle) -> dict[str, str]:
    env = os.environ.copy()
    if bundle.repo is not None:
        src = str(bundle.repo / "src")
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def probe_field_converter_import(bundle: FieldConverterBundle, timeout_s: int = 30) -> dict[str, Any]:
    if not bundle.python_exe.is_file():
        return {"ready": False, "error": "python_exe_missing"}
    code = (
        "import json, field_converter; "
        "from field_converter.inference.modeling import load_inference_model; "
        "print(json.dumps({'import_ok': True}))"
    )
    try:
        proc = subprocess.run(
            [str(bundle.python_exe), "-c", code], env=_python_env(bundle),
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except Exception as exc:
        return {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ready": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def probe_field_converter_model(bundle: FieldConverterBundle, timeout_s: int = 60) -> dict[str, Any]:
    """Load the exact TCN architecture/checkpoint on CPU without running data inference."""
    if not bundle.python_exe.is_file():
        return {"ready": False, "error": "python_exe_missing"}
    code = r'''
import json, sys
from field_converter.inference.modeling import load_inference_model
cfg, ckpt = sys.argv[1], sys.argv[2]
r = load_inference_model(model_type="tcn", config_path=cfg, checkpoint=ckpt, device_override="cpu", batch_size_override=1)
print(json.dumps({"model_type": r.model_type, "input_dim": r.input_dim, "window_size": r.window_size, "stride": r.stride, "checkpoint_metadata": r.checkpoint_metadata}))
'''
    try:
        proc = subprocess.run(
            [str(bundle.python_exe), "-c", code, str(bundle.config_path), str(bundle.checkpoint_path)],
            env=_python_env(bundle), capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except Exception as exc:
        return {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
    payload = None
    if proc.returncode == 0:
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            payload = None
    return {
        "ready": proc.returncode == 0 and isinstance(payload, dict) and payload.get("window_size") == 41,
        "returncode": int(proc.returncode),
        "model": payload,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-4000:],
    }
