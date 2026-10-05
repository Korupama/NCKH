from __future__ import annotations

import json
from pathlib import Path

import pytest

from stage4_metric3d.model_only_cache import (
    build_cache_contract,
    validate_cached_run,
    write_cache_manifest,
)


def _contract(tmp_path: Path) -> dict:
    checkpoint = tmp_path / "checkpoint.bin"
    mhr = tmp_path / "mhr.bin"
    checkpoint.write_bytes(b"checkpoint")
    mhr.write_bytes(b"mhr")
    root = Path(__file__).resolve().parents[1]
    return build_cache_contract(
        project_root=root,
        manifest_sha256="manifest-hash",
        dataset={"name": "dataset", "release": "release", "license_id": "license"},
        split="development",
        record_input_hashes={"record": {"image_sha256": "image", "camera_sha256": "camera", "gt_sha256": "gt"}},
        checkpoint_path=checkpoint,
        mhr_model_path=mhr,
        sam3d_root=None,
        device="cpu",
        resolved_device="cpu",
        inference_type="body",
        output_schema="stage4-model-only-sam3d-output-1.1",
    )


def test_cache_manifest_roundtrip_and_hash_validation(tmp_path: Path):
    contract = _contract(tmp_path)
    output = tmp_path / "output.npz"
    output.write_bytes(b"output")
    cache = tmp_path / "output.cache.json"
    write_cache_manifest(cache, contract, output_sha256=__import__("hashlib").sha256(b"output").hexdigest())
    result = validate_cached_run(cache_path=cache, output_path=output, expected_contract=contract)
    assert result["cache_key"] == contract["cache_key"]


def test_cache_rejects_contract_mismatch(tmp_path: Path):
    contract = _contract(tmp_path)
    output = tmp_path / "output.npz"
    output.write_bytes(b"output")
    cache = tmp_path / "output.cache.json"
    write_cache_manifest(cache, contract, output_sha256=__import__("hashlib").sha256(b"output").hexdigest())
    changed = dict(contract)
    changed["manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_cached_run(cache_path=cache, output_path=output, expected_contract=changed)


def test_cache_rejects_tampered_output(tmp_path: Path):
    contract = _contract(tmp_path)
    output = tmp_path / "output.npz"
    output.write_bytes(b"output")
    cache = tmp_path / "output.cache.json"
    write_cache_manifest(cache, contract, output_sha256=__import__("hashlib").sha256(b"output").hexdigest())
    output.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="output_sha256"):
        validate_cached_run(cache_path=cache, output_path=output, expected_contract=contract)
