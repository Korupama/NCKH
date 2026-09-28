from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from ...camera import CameraTimelineLite
from ...stage3_adapter import load_stage3_state, Stage3State
from .cache import Sam3DNativeCache
from .config import Sam3DPitchRefinedConfig
from .geometry import project_camera_points, camera_to_world
from .ground_anchor import choose_ground_anchor
from .joint_mapping import DEFAULT_MAPPING, mapping_provenance
from .metrics import summarize_observations
from .quality import ground_quality_reasons
from .refiner import FrameEvidence, refine_translation_sequence
from .visualization import save_topdown_world_pose, save_selected_frame_overlay

STATE_SCHEMA = "world-grounded-pose-state-1.1"
HANDOFF_SCHEMA = "stage4-downstream-handoff-2.1"
STAGE4_VERSION = "stage4-sam3d-pitch-refined-0.5.1"


def _eligible_tracks(s3: Stage3State) -> list[str]:
    return [t.track_id for t in s3.tracks if str(t.role) in {"player", "goalkeeper"}]


def _load_stage3_for_frame(stage3_state: str | Path, selected_frame: int | None) -> Stage3State:
    s3 = load_stage3_state(stage3_state)
    if selected_frame is None:
        return s3
    frame = int(selected_frame)
    available = {int(obs.frame_index) for track in s3.tracks for obs in track.observations}
    if frame not in available:
        raise ValueError(f"Selected frame {frame} is absent from Stage-3 observations")
    s3.selected_frame = frame
    s3.replay_context = dict(s3.replay_context)
    s3.replay_context["selected_frame"] = frame
    return s3


def _frames_in_window(s3: Stage3State, track_ids: list[str], radius: int) -> list[int]:
    lo, hi = int(s3.selected_frame) - radius, int(s3.selected_frame) + radius
    wanted = set(track_ids)
    return sorted({int(o.frame_index) for t in s3.tracks if t.track_id in wanted for o in t.observations if lo <= int(o.frame_index) <= hi})


def _camera_convention_report(cache: Sam3DNativeCache, cams: CameraTimelineLite, frames: list[int], track_ids: list[str], cfg: Sam3DPitchRefinedConfig) -> dict:
    frame_pos = {int(f): i for i, f in enumerate(cache.frame_indices.tolist())}
    track_pos = {t: i for i, t in enumerate(cache.track_ids)}
    errors: list[float] = []
    for frame in frames:
        ci = frame_pos.get(frame); cam = cams.by_frame(frame)
        if ci is None or cam is None or cam.status == "INVALID": continue
        for tid in track_ids:
            pj = track_pos.get(tid)
            if pj is None or not cache.valid_mask[ci, pj]: continue
            jcam = cache.skel_3d_relative_m[ci, pj].astype(np.float64) + cache.pred_cam_t_m[ci, pj].astype(np.float64)[None, :]
            uv = project_camera_points(cam, jcam, distort=False)
            ref = cache.skel_2d_px[ci, pj].astype(np.float64)
            good = np.isfinite(uv).all(axis=1) & np.isfinite(ref).all(axis=1)
            if np.any(good): errors.extend(np.linalg.norm(uv[good] - ref[good], axis=1).tolist())
    arr = np.asarray(errors, dtype=np.float64)
    p95 = None if arr.size == 0 else float(np.percentile(arr, 95))
    if p95 is None: status = "FAIL"
    elif p95 <= cfg.camera_convention_pass_p95_px: status = "PASS"
    elif p95 <= cfg.camera_convention_warn_p95_px: status = "WARN"
    else: status = "FAIL"
    return {
        "status": status, "count": int(arr.size),
        "median_px": None if arr.size == 0 else float(np.median(arr)),
        "p95_px": p95, "max_px": None if arr.size == 0 else float(np.max(arr)),
        "meaning": "SAM3D native camera-space joints + pred_cam_t reproject through Stage1 K without distortion to SAM3D's own 2D output.",
    }


def preflight_v05(*, stage3_state: str | Path, camera_dir: str | Path, sam3d_cache: str | Path, config: Sam3DPitchRefinedConfig | None = None, selected_frame: int | None = None) -> dict:
    cfg = config or Sam3DPitchRefinedConfig(); cfg.validate()
    s3 = _load_stage3_for_frame(stage3_state, selected_frame); cams = CameraTimelineLite.load_dir(camera_dir); cache = Sam3DNativeCache.load(sam3d_cache)
    track_ids = _eligible_tracks(s3); frames = _frames_in_window(s3, track_ids, cfg.window_radius_frames)
    errors: list[str] = []; warnings: list[str] = []
    try:
        source_fps = float(s3.replay_context.get("fps"))
        if not np.isfinite(source_fps) or source_fps <= 0:
            raise ValueError
    except (TypeError, ValueError):
        source_fps = None
        errors.append("stage3_source_fps_missing_or_invalid")
    try:
        image_width = int(s3.replay_context.get("image_width"))
        image_height = int(s3.replay_context.get("image_height"))
        if image_width <= 0 or image_height <= 0:
            raise ValueError
    except (TypeError, ValueError):
        image_width = None
        image_height = None
        errors.append("stage3_image_size_missing_or_invalid")
    if tuple(cache.track_ids) != tuple(track_ids): errors.append("sam3d_track_order_mismatch")
    cache_frame_set = set(int(x) for x in cache.frame_indices.tolist())
    missing_cache = [f for f in frames if f not in cache_frame_set]
    if missing_cache: warnings.append(f"sam3d_cache_missing_window_frames:{missing_cache[:12]}")
    missing_cam = [f for f in frames if cams.by_frame(f) is None or cams.by_frame(f).status == "INVALID"]
    if missing_cam: errors.append(f"camera_missing_or_invalid:{missing_cam[:12]}")
    convention = _camera_convention_report(cache, cams, frames, track_ids, cfg)
    if convention["status"] == "FAIL": errors.append("sam3d_stage1_camera_convention_failed")
    elif convention["status"] == "WARN": warnings.append("sam3d_stage1_camera_convention_warn")

    by_id = s3.track_by_id(); frame_pos = {int(f): i for i, f in enumerate(cache.frame_indices.tolist())}; track_pos = {t:i for i,t in enumerate(cache.track_ids)}
    ground_usable = 0; selected_valid = 0; selected_total = 0
    for tid in track_ids:
        tr = by_id[tid]; obs = tr.by_frame().get(s3.selected_frame); selected_total += 1
        ci = frame_pos.get(s3.selected_frame); pj = track_pos.get(tid)
        if obs is None or ci is None or pj is None or not cache.valid_mask[ci,pj]: continue
        selected_valid += 1; cam = cams.by_frame(s3.selected_frame)
        anchor, _ = choose_ground_anchor(camera=cam, observation=obs, relative_joints_m=cache.skel_3d_relative_m[ci,pj], sam_prior_cam_m=cache.pred_cam_t_m[ci,pj], min_joint_weight=cfg.min_rtmw_joint_weight, max_prior_distance_m=cfg.max_ground_prior_distance_m)
        if anchor is not None and anchor.usable: ground_usable += 1
    if selected_valid == 0: errors.append("no_valid_sam3d_player_at_selected_frame")
    if ground_usable < selected_valid: warnings.append("some_selected_players_have_no_usable_ground_anchor")
    temporal_status = "DISABLED" if not cfg.use_temporal else "TEMPORAL_NOT_AVAILABLE"
    if cfg.use_temporal:
        for tid in track_ids:
            pj = track_pos.get(tid)
            valid_frames = [f for f in frames if f in frame_pos and pj is not None and cache.valid_mask[frame_pos[f], pj]]
            if any(b-a == c-b for a,b,c in zip(valid_frames, valid_frames[1:], valid_frames[2:])):
                temporal_status = "AVAILABLE_FOR_SOME_TRACKS"
                break
        if temporal_status == "TEMPORAL_NOT_AVAILABLE": warnings.append(temporal_status)
    return {
        "schema_version": "stage4-v05-preflight-1.0",
        "ready": not errors,
        "selected_frame": int(s3.selected_frame),
        "temporal_status": temporal_status,
        "source_fps": source_fps,
        "image_size": [image_width, image_height],
        "window_frames": frames,
        "track_ids": track_ids,
        "sam3d_cache": {"schema": "stage4-sam3d-native-cache-1.0", "T": cache.T, "N": cache.N, "J": cache.J},
        "camera_convention": convention,
        "selected_frame_player_coverage": None if selected_total == 0 else float(selected_valid/selected_total),
        "ground_anchor_coverage_at_t0": None if selected_valid == 0 else float(ground_usable/selected_valid),
        "mapping": mapping_provenance(),
        "errors": errors, "warnings": warnings,
    }


def _read_selected_image(cache: Sam3DNativeCache, selected_frame: int):
    import cv2
    meta = cache.metadata or {}
    frames_dir = meta.get("frames_dir")
    if frames_dir:
        root = Path(frames_dir)
        if root.is_dir():
            import re
            for p in sorted(root.iterdir()):
                m = re.findall(r"\d+", p.stem)
                if m and int(m[-1]) == selected_frame:
                    return cv2.imread(str(p), cv2.IMREAD_COLOR)
    video = meta.get("video")
    if video and Path(video).is_file():
        cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame); ok, img = cap.read(); cap.release()
        if ok: return img
    return None


def _build_handoff(state: dict, state_path: Path) -> dict:
    tracks = []
    for tr in state.get("tracks", []):
        obs_out = []
        for obs in tr.get("observations", []):
            if not obs.get("valid"): continue
            obs_out.append({
                "frame_index": obs["frame_index"],
                "root_world_m": obs.get("root_world_m"),
                "joints_world": [
                    {"name": j["canonical_name"], "xyz_world_m": j.get("xyz_world_m"), "valid": j.get("valid", False)}
                    for j in obs.get("joints", [])
                ],
                "quality": obs.get("quality"),
            })
        tracks.append({"track_id": tr["track_id"], "role": tr.get("role"), "observations": obs_out, "selected_frame_status": tr.get("selected_frame_status")})
    return {
        "schema_version": HANDOFF_SCHEMA,
        "producer": STAGE4_VERSION,
        "source_world_grounded_pose_state": str(state_path),
        "selected_frame": state["selected_frame"],
        "coordinate_frame": state["coordinate_frame"],
        "tracks": tracks,
        "not_owned_by_stage4": ["team_id", "attacking_team", "attack_direction", "toucher", "ball_state", "legal_body_mask", "second_last_opponent", "offside_decision"],
    }


def run_v05(*, stage3_state: str | Path, camera_dir: str | Path, sam3d_cache: str | Path, output_dir: str | Path, config: Sam3DPitchRefinedConfig | None = None, refine: bool = True, selected_frame: int | None = None) -> dict:
    cfg = config or Sam3DPitchRefinedConfig(); pre = preflight_v05(stage3_state=stage3_state, camera_dir=camera_dir, sam3d_cache=sam3d_cache, config=cfg, selected_frame=selected_frame)
    if not pre["ready"]: raise RuntimeError("Stage4 v0.5 preflight failed: " + "; ".join(pre["errors"]))
    s3 = _load_stage3_for_frame(stage3_state, selected_frame); cams = CameraTimelineLite.load_dir(camera_dir); cache = Sam3DNativeCache.load(sam3d_cache)
    out = Path(output_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    (out / "stage4_preflight.json").write_text(json.dumps(pre, indent=2), encoding="utf-8")
    frame_pos = {int(f): i for i,f in enumerate(cache.frame_indices.tolist())}; track_pos = {t:i for i,t in enumerate(cache.track_ids)}
    frames_window = set(pre["window_frames"]); tracks_state=[]; all_obs=[]
    by_id=s3.track_by_id()
    for tid in pre["track_ids"]:
        tr=by_id[tid]; evidences=[]
        for obs in tr.observations:
            frame=int(obs.frame_index)
            if frame not in frames_window: continue
            ci=frame_pos.get(frame); pj=track_pos.get(tid); cam=cams.by_frame(frame)
            if ci is None or pj is None or cam is None or cam.status=="INVALID" or not cache.valid_mask[ci,pj]: continue
            rel=cache.skel_3d_relative_m[ci,pj].astype(np.float64); prior=cache.pred_cam_t_m[ci,pj].astype(np.float64)
            anchor,_=choose_ground_anchor(camera=cam,observation=obs,relative_joints_m=rel,sam_prior_cam_m=prior,min_joint_weight=cfg.min_rtmw_joint_weight,max_prior_distance_m=cfg.max_ground_prior_distance_m)
            evidences.append(FrameEvidence(frame,cam,obs,rel,prior,cache.skel_2d_px[ci,pj].astype(np.float64),anchor))
        if not evidences:
            tracks_state.append({"track_id":tid,"role":tr.role,"observations":[],"selected_frame_status":"MISSING"}); continue
        evidences.sort(key=lambda x:x.frame_index)
        if refine:
            rr=refine_translation_sequence(evidences,cfg); roots=rr.refined_cam_m
        else:
            roots=np.stack([e.sam_prior_cam_m for e in evidences]); rr=None
        obs_states=[]
        for root, ev in zip(roots,evidences):
            jcam_native=ev.relative_joints_m+root[None,:]; jworld_native=camera_to_world(ev.camera,jcam_native)
            joints=[]; reproj_errors=[]
            for entry in DEFAULT_MAPPING:
                rel=ev.relative_joints_m[entry.sam3d_index]; jcam=jcam_native[entry.sam3d_index]; jw=jworld_native[entry.sam3d_index]
                uv_pred=project_camera_points(ev.camera,jcam,distort=True)
                uv_rtmw=np.asarray(ev.observation.uv23[entry.rtmw_index],dtype=np.float64); w=float(ev.observation.state_weights23[entry.rtmw_index])
                err=None
                if np.isfinite(uv_pred).all() and np.isfinite(uv_rtmw).all() and w>=cfg.min_rtmw_joint_weight:
                    err=float(np.linalg.norm(uv_pred-uv_rtmw)); reproj_errors.append(err)
                joints.append({
                    "canonical_name":entry.canonical_name,"sam3d_index":entry.sam3d_index,"sam3d_name":entry.sam3d_name,
                    "xyz_relative_cam_m":rel.tolist() if np.isfinite(rel).all() else None,
                    "xyz_camera_m":jcam.tolist() if np.isfinite(jcam).all() else None,
                    "xyz_world_m":jw.tolist() if np.isfinite(jw).all() else None,
                    "sam3d_uv_px":ev.sam_2d_px[entry.sam3d_index].tolist() if np.isfinite(ev.sam_2d_px[entry.sam3d_index]).all() else None,
                    "rtmw_uv_px":uv_rtmw.tolist() if np.isfinite(uv_rtmw).all() else None,
                    "reprojected_uv_px":uv_pred.tolist() if np.isfinite(uv_pred).all() else None,
                    "reprojection_error_px":err,"rtmw_weight":w,"valid":bool(np.isfinite(jw).all()),
                })
            hips=[jworld_native[e.sam3d_index] for e in DEFAULT_MAPPING if e.canonical_name in {"left_hip","right_hip"}]
            root_world=np.nanmean(np.asarray(hips),axis=0) if hips else camera_to_world(ev.camera,root)
            ground_res=None
            if ev.ground_anchor is not None:
                ge=next((e for e in DEFAULT_MAPPING if e.canonical_name==ev.ground_anchor.canonical_name),None)
                if ge is not None and np.isfinite(jworld_native[ge.sam3d_index]).all(): ground_res=float(abs(jworld_native[ge.sam3d_index,2])*100.0)
            correction=float(np.linalg.norm(root-ev.sam_prior_cam_m)); obs_valid=all(np.isfinite(root))
            q={
                "root_refinement":"VALID" if obs_valid and (not refine or rr.success) else "DEGRADED",
                "ground_anchor": None if ev.ground_anchor is None else {"name":ev.ground_anchor.canonical_name,"usable":ev.ground_anchor.usable,"fallback":ev.ground_anchor.fallback,"distance_to_sam_prior_m":ev.ground_anchor.distance_to_sam_prior_m},
                "ground_contact_residual_cm":ground_res,
                "reprojection_p95_px":None if not reproj_errors else float(np.percentile(reproj_errors,95)),
                "initialization": "GROUND_CONSENSUS" if refine and ev.ground_anchor is not None and ev.ground_anchor.usable else "SAM_PRIOR",
                "ground_candidate_spread_m": None if ev.ground_anchor is None else ev.ground_anchor.candidate_spread_m,
                "ground_consensus_count": 0 if ev.ground_anchor is None else ev.ground_anchor.consensus_count,
                "ground_rejection_reason": None if ev.ground_anchor is None else ev.ground_anchor.rejection_reason,
                "ground_vs_sam_disagreement_m": None if ev.ground_anchor is None else ev.ground_anchor.distance_to_sam_prior_m,
                "ground_vs_sam_disagreement": None if ev.ground_anchor is None else ("LARGE_SAM_DEPTH_ERROR" if ev.ground_anchor.distance_to_sam_prior_m > 1.5 else "SAM_DEPTH_DISAGREEMENT" if ev.ground_anchor.distance_to_sam_prior_m >= 0.5 else "NORMAL"),
                "ground_vs_refined_m": None if ev.ground_anchor is None else float(np.linalg.norm(root-ev.ground_anchor.root_cam_m)),
            }
            obs_state={
                "frame_index":ev.frame_index,"valid":bool(obs_valid),"root_world_m":root_world.tolist() if np.isfinite(root_world).all() else None,
                "translation":{
                    "ground_hit_world_m": None if ev.ground_anchor is None else ev.ground_anchor.ground_world_m.tolist(),
                    "sam_root_world_m": camera_to_world(ev.camera,ev.sam_prior_cam_m).tolist(),
                    "refined_root_world_m": camera_to_world(ev.camera,root).tolist(),
                    "sam3d_prior_cam_m":ev.sam_prior_cam_m.tolist(),
                    "ground_init_cam_m":None if ev.ground_anchor is None else ev.ground_anchor.root_cam_m.tolist(),
                    "refined_cam_m":root.tolist(),"sam_correction_m":correction,
                },
                "joints":joints,"quality":q,
            }
            obs_states.append(obs_state); all_obs.append(obs_state)
        selected_obs=next((o for o in obs_states if o["frame_index"]==s3.selected_frame),None)
        tracks_state.append({
            "track_id":tid,"role":tr.role,"observations":obs_states,
            "selected_frame_status":"VALID" if selected_obs and selected_obs.get("valid") else "MISSING",
            "optimizer":None if rr is None else {"success":rr.success,"status":rr.status,"message":rr.message,"nfev":rr.nfev,"initial_cost":rr.initial_cost,"final_cost":rr.final_cost},
            "temporal_status": "APPLIED" if refine and cfg.use_temporal and any(b.frame_index-a.frame_index == c.frame_index-b.frame_index for a,b,c in zip(evidences,evidences[1:],evidences[2:])) else "DISABLED" if not refine or not cfg.use_temporal else "TEMPORAL_NOT_AVAILABLE",
        })
    metrics=summarize_observations(all_obs,below_pitch_tolerance_m=cfg.below_pitch_tolerance_m)
    spans=metrics["skeleton_span_m"]; reproj=metrics["reprojection_error_px"]; geometry_fail=False; reasons=[]
    for key,limit in (("x_p95",cfg.catastrophic_xy_span_m),("y_p95",cfg.catastrophic_xy_span_m),("z_p95",cfg.catastrophic_z_span_m),("diameter_p95",cfg.catastrophic_diameter_m)):
        v=spans.get(key)
        if v is not None and v>limit: geometry_fail=True; reasons.append(f"{key}>{limit}")
    if reproj.get("p95") is not None and reproj["p95"]>cfg.sanity_reprojection_p95_px: geometry_fail=True; reasons.append("reprojection_p95_above_sanity_threshold")
    selected_metrics = summarize_observations([o for o in all_obs if o["frame_index"] == s3.selected_frame], below_pitch_tolerance_m=cfg.below_pitch_tolerance_m)
    reasons.extend(ground_quality_reasons(selected_metrics, pre["ground_anchor_coverage_at_t0"]))
    geometry_fail = geometry_fail or bool(reasons)
    state={
        "schema_version":STATE_SCHEMA,"stage4_version":STAGE4_VERSION,"backend":"sam3d-pitch-refined" if refine else "sam3d-direct",
        "selected_frame":int(s3.selected_frame),
        "coordinate_frame":{"name":"STAGE1_PITCH_WORLD","units":"m","x":"goal-to-goal","y":"touchline-to-touchline","z":"up","pitch_plane":"z=0"},
        "provenance":{"stage3_state":str(Path(stage3_state).resolve()),"camera_dir":str(Path(camera_dir).resolve()),"sam3d_cache":str(Path(sam3d_cache).resolve()),"selected_frame_override":None if selected_frame is None else int(selected_frame),"joint_mapping":mapping_provenance()},
        "preflight":pre,"tracks":tracks_state,"metrics":metrics,
        "quality_gates":{"implementation_gate":"PASS","geometric_quality_gate":"FAIL" if geometry_fail else "PASS_SANITY","metric_accuracy_gate":"NOT_EVALUATED","downstream_offside_gate":"NOT_EVALUATED","research_accuracy_frozen":False,"geometry_reasons":reasons},
        "not_owned_by_stage4":["team_id","attacking_team","attack_direction","toucher","ball_state","legal_body_mask","second_last_opponent","offside_decision"],
        "artifacts":{},
    }
    state_path=out/"world_grounded_pose_state.json"; state["artifacts"]["world_grounded_pose_state"]=str(state_path)
    state_path.write_text(json.dumps(state,indent=2),encoding="utf-8")
    handoff=_build_handoff(state,state_path); handoff_path=out/"stage4_downstream_handoff.json"; handoff_path.write_text(json.dumps(handoff,indent=2),encoding="utf-8"); state["artifacts"]["stage4_downstream_handoff"]=str(handoff_path)
    qpath=out/"stage4_quality_report.json"; qpath.write_text(json.dumps({"metrics":metrics,"quality_gates":state["quality_gates"],"preflight":pre},indent=2),encoding="utf-8"); state["artifacts"]["stage4_quality_report"]=str(qpath)
    try:
        top=save_topdown_world_pose(state,out/"selected_frame_world_pose_topdown.png"); state["artifacts"]["selected_frame_world_pose_topdown"]=str(top)
    except Exception as exc: state.setdefault("warnings",[]).append(f"topdown_visualization_failed:{exc}")
    try:
        img=_read_selected_image(cache,int(s3.selected_frame))
        if img is not None:
            overlay=save_selected_frame_overlay(state,img,out/"selected_frame_overlay.png"); state["artifacts"]["selected_frame_overlay"]=str(overlay)
    except Exception as exc: state.setdefault("warnings",[]).append(f"overlay_visualization_failed:{exc}")
    state_path.write_text(json.dumps(state,indent=2),encoding="utf-8")
    return state
