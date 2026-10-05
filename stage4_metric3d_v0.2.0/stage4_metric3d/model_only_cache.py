from __future__ import annotations

"""Content-addressed provenance for the independent Stage-4 model-only lane."""

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


MODEL_ONLY_CACHE_SCHEMA = "stage4-model-only-cache-manifest-1.0"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _git_revision(path: Path) -> str | None:
    candidate = path if path.is_dir() else path.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        value = result.stdout.strip()
        return value or None
    except (OSError, subprocess.SubprocessError):
        return None


def _source_hashes(project_root: Path) -> dict[str, str]:
    relative_paths = (
        "run_sam3d_model_only.py",
        "stage4_metric3d/model_only.py",
        "stage4_metric3d/model_only_refinement.py",
        "stage4_metric3d/model_only_evaluation.py",
        "stage4_metric3d/model_only_cache.py",
        "stage4_metric3d/backends/sam3d_pitch_refined/cache.py",
        "stage4_metric3d/backends/sam3d_pitch_refined/joint_mapping.py",
    )
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = (project_root / relative).resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Cannot hash required Stage-4 source: {path}")
        result[relative] = sha256_file(path)
    return result


def build_cache_contract(
    *,
    project_root: str | Path,
    manifest_sha256: str,
    dataset: Mapping[str, Any],
    split: str,
    record_input_hashes: Mapping[str, Mapping[str, str]],
    checkpoint_path: str | Path,
    mhr_model_path: str | Path,
    sam3d_root: str | Path | None,
    device: str,
    resolved_device: str,
    inference_type: str,
    output_schema: str,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    mhr = Path(mhr_model_path).expanduser().resolve()
    repository = None if sam3d_root is None else Path(sam3d_root).expanduser().resolve()
    contract: dict[str, Any] = {
        "schema_version": MODEL_ONLY_CACHE_SCHEMA,
        "dataset": dict(dataset),
        "split": str(split),
        "manifest_sha256": str(manifest_sha256),
        "record_input_hashes": {str(k): dict(v) for k, v in sorted(record_input_hashes.items())},
        "model": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "mhr_model": str(mhr),
            "mhr_model_sha256": sha256_file(mhr),
            "sam3d_root": None if repository is None else str(repository),
            "sam3d_repository_revision": None if repository is None else _git_revision(repository),
        },
        "runtime": {
            "device_requested": str(device),
            "device_resolved": str(resolved_device),
            "inference_type": str(inference_type),
            "output_schema": str(output_schema),
        },
        "stage4_source_sha256": _source_hashes(root),
    }
    contract["cache_key"] = _sha256_bytes(_canonical_json(contract))
    return contract


def write_cache_manifest(path: str | Path, contract: Mapping[str, Any], *, output_sha256: str) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = dict(contract)
    data["output_sha256"] = str(output_sha256)
    data["cache_key"] = contract.get("cache_key")
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(target)
    return target


def load_cache_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.is_file() or target.stat().st_size <= 0:
        raise FileNotFoundError(f"Model-only cache manifest not found or empty: {target}")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot parse model-only cache manifest: {target}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != MODEL_ONLY_CACHE_SCHEMA:
        raise ValueError(f"Unsupported model-only cache manifest: {target}")
    if not isinstance(data.get("cache_key"), str) or not data["cache_key"].strip():
        raise ValueError("Model-only cache manifest cache_key is required")
    if not isinstance(data.get("output_sha256"), str) or not data["output_sha256"].strip():
        raise ValueError("Model-only cache manifest output_sha256 is required")
    return data


def validate_cached_run(
    *,
    cache_path: str | Path,
    output_path: str | Path,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    cached = load_cache_manifest(cache_path)
    expected_key = str(expected_contract.get("cache_key"))
    if cached.get("cache_key") != expected_key:
        raise ValueError("Model-only cache key does not match current manifest/model/code contract")
    for key in (
        "dataset", "split", "manifest_sha256", "record_input_hashes", "model", "runtime", "stage4_source_sha256",
    ):
        if cached.get(key) != expected_contract.get(key):
            raise ValueError(f"Model-only cache provenance mismatch: {key}")
    output = Path(output_path).expanduser().resolve()
    if not output.is_file() or output.stat().st_size <= 0:
        raise FileNotFoundError(f"Cached model-only output not found or empty: {output}")
    output_hash = sha256_file(output)
    if cached.get("output_sha256") != output_hash:
        raise ValueError("Model-only cache output_sha256 does not match output file")
    return cached
