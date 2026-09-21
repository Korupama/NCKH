from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import json
import shutil
import sys

from ...camera import CameraTimelineLite
from ...stage3_adapter import load_stage3_state
from .camera_adapter import camera_domain_report, camera_projection_compatibility
from .contracts import FieldConverterBundle, FieldConverterV04Config
from .exporter import eligible_tracks, export_field_converter_raw, source_frames_for_tracks
from .importer import build_downstream_handoff, import_field_converter_predictions, locate_predictions
from .model_bundle import probe_field_converter_import, probe_field_converter_model, validate_bundle
from .runner import run_field_converter
from .sam3d_cache import Sam3DCache
from .visualization import save_topdown_world_pose


def preflight_v04(
    *,
    stage3_state: str | Path,
    camera_dir: str | Path,
    sam3d_cache: str | Path | None,
    bundle: FieldConverterBundle | None,
    config: FieldConverterV04Config | None = None,
    probe_external_python: bool = True,
) -> dict[str, Any]:
    cfg = config or FieldConverterV04Config()
    cfg.validate()
    s3 = load_stage3_state(stage3_state)
    cams = CameraTimelineLite.load_dir(camera_dir)
    track_ids = eligible_tracks(s3, cfg.include_roles)
    frames = source_frames_for_tracks(s3, track_ids)
    width = int(s3.replay_context.get("image_width", -1))
    height = int(s3.replay_context.get("image_height", -1))
    source_fps = float(s3.replay_context.get("fps", 0.0))

    errors = []
    warnings = []
    if not track_ids:
        errors.append("no_player_or_goalkeeper_tracks")
    if not frames:
        errors.append("no_source_frames")
    if width <= 0 or height <= 0 or source_fps <= 0:
        errors.append("invalid_replay_context")
    missing_cam = [f for f in frames if cams.by_frame(f) is None or cams.by_frame(f).status == "INVALID"]
    mismatch_cam = [
        f for f in frames
        if cams.by_frame(f) is not None and (
            cams.by_frame(f).image_width != width or cams.by_frame(f).image_height != height
        )
    ]
    if missing_cam:
        errors.append("incomplete_camera_timeline")
    if mismatch_cam:
        errors.append("camera_image_resolution_mismatch")

    compatibility = camera_projection_compatibility(cams, frames)
    p95 = compatibility.get("p95_px")
    compatibility_status = "UNKNOWN"
    if p95 is not None:
        if float(p95) <= cfg.camera_projection_compatibility_p95_px:
            compatibility_status = "PASS"
        elif float(p95) <= cfg.camera_projection_compatibility_fail_px:
            compatibility_status = "DEGRADED"
            warnings.append("field_converter_radial_projection_differs_from_stage1_full_distortion")
        else:
            compatibility_status = "FAIL"
            errors.append("field_converter_camera_projection_incompatible")
    domain = camera_domain_report(cams, frames)
    if domain.get("status") == "OUT_OF_DOMAIN":
        warnings.append("pretrained_field_converter_camera_out_of_training_domain")

    cache_report = {"ready": False, "reason": "not_provided"}
    if sam3d_cache is not None:
        try:
            cache = Sam3DCache.load(sam3d_cache)
            expected_frames = frames
            cache_report = {
                "ready": tuple(cache.track_ids) == tuple(track_ids) and cache.frame_indices.tolist() == expected_frames,
                "path": str(cache.path),
                "T": cache.T,
                "N": cache.N,
                "semantic_mapping_validated": cache.semantic_mapping_validated,
            }
            if not cache_report["ready"]:
                errors.append("sam3d_cache_timeline_or_track_order_mismatch")
            if not cache.semantic_mapping_validated:
                warnings.append("sam3d_joint_semantics_not_validated_for_stage8")
        except Exception as exc:
            cache_report = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
            errors.append("sam3d_cache_invalid")

    bundle_report = {"ready": False, "reason": "not_provided"}
    import_probe = {"ready": False, "reason": "not_probed"}
    model_probe = {"ready": False, "reason": "not_probed"}
    if bundle is not None:
        bundle_report = validate_bundle(bundle)
        if not bundle_report["ready"]:
            errors.append("field_converter_bundle_incomplete")
        elif probe_external_python:
            import_probe = probe_field_converter_import(bundle)
            if not import_probe.get("ready"):
                errors.append("field_converter_external_python_import_failed")
            else:
                model_probe = probe_field_converter_model(bundle)
                if not model_probe.get("ready"):
                    errors.append("field_converter_checkpoint_or_config_incompatible")

    return {
        "schema_version": "stage4-v04-preflight-1.0",
        "ready": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "selected_frame": int(s3.selected_frame),
        "source_fps": source_fps,
        "image_size": [width, height],
        "track_ids": track_ids,
        "frame_indices": frames,
        "camera_missing_or_invalid_frames": missing_cam,
        "camera_resolution_mismatch_frames": mismatch_cam,
        "camera_projection_compatibility_status": compatibility_status,
        "camera_projection_compatibility": compatibility,
        "camera_domain": domain,
        "sam3d_cache": cache_report,
        "field_converter_bundle": bundle_report,
        "field_converter_import_probe": import_probe,
        "field_converter_model_probe": model_probe,
    }


def run_v04(
    *,
    stage3_state: str | Path,
    camera_dir: str | Path,
    sam3d_cache: str | Path,
    bundle: FieldConverterBundle,
    output_dir: str | Path,
    config: FieldConverterV04Config | None = None,
    timeout_s: Optional[int] = None,
) -> dict[str, Any]:
    cfg = config or FieldConverterV04Config()
    cfg.validate()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    preflight = preflight_v04(
        stage3_state=stage3_state,
        camera_dir=camera_dir,
        sam3d_cache=sam3d_cache,
        bundle=bundle,
        config=cfg,
        probe_external_python=True,
    )
    (out / "stage4_v04_preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if not preflight["ready"]:
        raise RuntimeError(f"Stage4 v0.4 preflight failed: {preflight['errors']}")

    raw_dir = out / "intermediates" / "field_converter_input"
    fc_out = out / "intermediates" / "field_converter_output"
    work = out / "intermediates" / "field_converter_work"
    export = export_field_converter_raw(
        stage3_state=stage3_state,
        camera_dir=camera_dir,
        sam3d_cache=sam3d_cache,
        output_root=raw_dir,
        sequence_name=cfg.sequence_name,
        include_roles=cfg.include_roles,
    )
    fc_run = run_field_converter(
        bundle=bundle,
        cfg=cfg,
        raw_input_dir=raw_dir,
        output_dir=fc_out,
        work_dir=work,
        source_fps=float(export["source_fps"]),
        image_size=tuple(export["image_size"]),
        timeout_s=timeout_s,
    )
    pred_path = locate_predictions(fc_out, cfg.sequence_name)
    summary_path = pred_path.with_name("summary.json")
    cache = Sam3DCache.load(sam3d_cache)
    state = import_field_converter_predictions(
        predictions_path=pred_path,
        summary_path=summary_path,
        track_ids=export["track_ids"],
        source_frames=export["frame_indices"],
        selected_frame=int(export["selected_frame"]),
        sam3d_joint_names=cache.joint_names,
        semantic_mapping_validated=cache.semantic_mapping_validated,
        camera_compatibility=preflight["camera_projection_compatibility"],
        camera_domain=preflight["camera_domain"],
        catastrophic_xy_span_m=cfg.catastrophic_xy_span_m,
        catastrophic_z_span_m=cfg.catastrophic_z_span_m,
    )
    state["configuration"] = cfg.to_dict()
    state["preflight"] = preflight
    state["provenance"].update({
        "stage3_state": str(Path(stage3_state).expanduser().resolve()),
        "camera_dir": str(Path(camera_dir).expanduser().resolve()),
        "sam3d_cache": str(Path(sam3d_cache).expanduser().resolve()),
        "field_converter_bundle": bundle.to_dict(),
        "field_converter_export_manifest": export.get("manifest_path"),
        "field_converter_log": fc_run.get("log_path"),
    })
    state_path = out / "world_grounded_pose_state.json"
    handoff_path = out / "stage4_downstream_handoff.json"
    state["artifacts"] = {
        "world_grounded_pose_state": str(state_path),
        "stage4_downstream_handoff": str(handoff_path),
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    handoff = build_downstream_handoff(state, state_path)
    handoff_path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")
    try:
        viz = save_topdown_world_pose(state, out / "selected_frame_world_pose_topdown.png")
        state["artifacts"]["selected_frame_world_pose_topdown"] = str(viz)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        state["visualization_warning"] = f"{type(exc).__name__}: {exc}"
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state
