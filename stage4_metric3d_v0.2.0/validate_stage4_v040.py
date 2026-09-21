from __future__ import annotations

import json
import tempfile
from pathlib import Path
import numpy as np

from stage4_metric3d.wholebody import WHOLEBODY_NAMES
from stage4_metric3d.backends.field_converter.sam3d_cache import save_sam3d_cache
from stage4_metric3d.backends.field_converter.exporter import export_field_converter_raw, build_boxes_from_stage3
from stage4_metric3d.backends.field_converter.importer import import_field_converter_predictions, build_downstream_handoff
from stage4_metric3d.backends.field_converter.pipeline import preflight_v04
from stage4_metric3d.stage3_adapter import load_stage3_state


def _look_at(C, target, up=np.array([0.0, 0.0, 1.0])):
    f = np.asarray(target, dtype=float) - np.asarray(C, dtype=float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    d = np.cross(f, r); d /= np.linalg.norm(d)
    return np.stack([r, d, f])


def _synthetic_case(root: Path):
    camera_dir = root / "camera"; camera_dir.mkdir(parents=True)
    frames = [10, 11, 12]; tids = ["track_001", "track_002"]
    K = np.array([[1000.,0,640.],[0,1000.,360.],[0,0,1.]])
    C = np.array([0.,-75.,16.]); R = _look_at(C, np.array([10.,0.,1.]))
    for f in frames:
        cam = {
            "schema_version":"1.2","frame_index":f,"status":"VALID",
            "image":{"width":1280,"height":720},"intrinsics":{"K":K.tolist()},
            "extrinsics":{"R_world_to_camera":R.tolist(),"camera_center_world_m":C.tolist()},
            "distortion":{"radial":[0,0,0,0,0,0],"tangential":[0,0],"thin_prism":[0,0,0,0]},
            "pitch":{"length_m":105.0,"width_m":68.0},
        }
        (camera_dir/f"camera_state_{f:08d}.json").write_text(json.dumps(cam))
    tracks=[]
    for n,tid in enumerate(tids):
        obs=[]
        for t,f in enumerate(frames):
            x1=300+n*300+t*2; box=[x1,180,x1+80,650]
            kps=[]
            for j,name in enumerate(WHOLEBODY_NAMES):
                if j<23: x,y,state,score=x1+40,210+j*12,"VALID",.95
                else: x=y=score=None; state="MISSING"
                kps.append({"index":j,"name":name,"x":x,"y":y,"state":state,"raw_model_score":score})
            obs.append({"frame_index":f,"source_bbox_xyxy":box,"pose_status":"VALID","keypoints_133":kps})
        tracks.append({"track_id":tid,"upstream_role":"player" if n==0 else "goalkeeper","observations":obs})
    state={"schema_version":"tracked-pose-2d-state-1.0","coordinate_space":"RAW_DISTORTED_PIXEL",
           "keypoint_schema":{"count":133,"names":list(WHOLEBODY_NAMES)},
           "replay_context":{"selected_frame":11,"fps":30.0,"frame_count":100,"image_width":1280,"image_height":720},
           "tracks":tracks}
    s3=root/"tracked_pose_2d_state.json"; s3.write_text(json.dumps(state))
    parsed=load_stage3_state(s3); boxes=build_boxes_from_stage3(parsed,frames,tids)
    s2d=np.zeros((3,2,25,2),np.float32); s3d=np.zeros((3,2,25,3),np.float32)
    for t in range(3):
        for n in range(2):
            for j in range(25):
                s2d[t,n,j]=[boxes[t,n,0]+40,210+j*10]
                s3d[t,n,j]=[j*.01,j*.005,j*.03]
    cache=save_sam3d_cache(root/"sam3d_cache.npz", frame_indices=frames,track_ids=tids,boxes_xyxy=boxes,
                            skel_2d_px=s2d,skel_3d_relative_m=s3d,valid_mask=np.ones((3,2),bool))
    return s3,camera_dir,cache,frames,tids


def _fake_official_predictions(path: Path, frames: list[int], tids: list[str]):
    N=len(tids); J=25
    frame_numbers=np.array([10.,10.5,11.,11.5,12.],dtype=np.float64); T=len(frame_numbers)
    valid=np.ones((N,T),bool); roots=np.zeros((N,T,3),np.float32)
    roots_source=np.zeros_like(roots); joints_cam=np.zeros((N,T,J,3),np.float32); joints_source=np.zeros_like(joints_cam)
    for n in range(N):
        for t in range(T):
            roots_source[n,t]=[10+n,20+t,1]
            for j in range(J):
                joints_cam[n,t,j]=[j*.01,j*.005,5+j*.03]
                joints_source[n,t,j]=[10+n+j*.01,20+t+j*.005,j*.03]
    np.savez_compressed(path, valid_mask=valid, source_frame_numbers_float=frame_numbers,
        root_pred_m=roots,root_source_world_pred_m=roots_source,root_init_cam_m=np.zeros_like(roots),
        root_delta_pred_cam_m=np.zeros_like(roots),joints_pred_cam_m=joints_cam,
        joints_pred_source_world_m=joints_source,joints_pred_2d=np.zeros((N,T,J,2),np.float32),
        world_alignment_rotation=np.eye(3,dtype=np.float32),output_fps=np.array(50.0),meta_json=np.array("{}",dtype=object))


def main() -> int:
    report={"schema_version":"stage4-v040-adapter-validation-1.0"}
    with tempfile.TemporaryDirectory(prefix="stage4_v040_") as td:
        root=Path(td); s3,camera_dir,cache,frames,tids=_synthetic_case(root)
        pre=preflight_v04(stage3_state=s3,camera_dir=camera_dir,sam3d_cache=cache,bundle=None,probe_external_python=False)
        export=export_field_converter_raw(stage3_state=s3,camera_dir=camera_dir,sam3d_cache=cache,
                                          output_root=root/"fc_input",sequence_name="synthetic")
        pred=root/"predictions.npz"; _fake_official_predictions(pred,frames,tids)
        summary=root/"summary.json"; summary.write_text(json.dumps({"proxy_reprojection_to_observed_sam2d":{"median_px":3.0}}))
        state=import_field_converter_predictions(predictions_path=pred,summary_path=summary,track_ids=tids,source_frames=frames,
             selected_frame=11,sam3d_joint_names=[f"sam3d_{i:02d}" for i in range(25)],semantic_mapping_validated=False,
             camera_compatibility=pre["camera_projection_compatibility"],camera_domain=pre["camera_domain"],
             catastrophic_xy_span_m=4.0,catastrophic_z_span_m=4.0)
        handoff=build_downstream_handoff(state,root/"world_grounded_pose_state.json")
        report["synthetic_adapter_validation"]={
            "preflight_ready":pre["ready"],"camera_compatibility_status":pre["camera_projection_compatibility_status"],
            "export_shape":export["shape"],"output_schema":state["schema_version"],"handoff_schema":handoff["schema_version"],
            "implementation_gate":state["quality_gates"]["implementation_gate"],
            "geometric_quality_gate":state["quality_gates"]["geometric_quality_gate"],
        }
    real_s3=Path('/mnt/data/stage3_src/stage3_pose2d_v0.1/runs/stage3_real_1920/tracked_pose_2d_state.json')
    real_cam=Path('/mnt/data/stage1_src/stage_1_camera_v12/outputs/temporal_v13/batch_video/shot_ptz/optimized_camera_states')
    if real_s3.is_file() and real_cam.is_dir():
        real=preflight_v04(stage3_state=real_s3,camera_dir=real_cam,sam3d_cache=None,bundle=None,probe_external_python=False)
        report["real_supplied_artifact_preflight"]={
            "adapter_ready":real["ready"],"selected_frame":real["selected_frame"],"source_fps":real["source_fps"],
            "image_size":real["image_size"],"camera_projection_compatibility_status":real["camera_projection_compatibility_status"],
            "camera_projection_p95_px":real["camera_projection_compatibility"].get("p95_px"),
            "camera_domain_status":real["camera_domain"].get("status"),
            "camera_center_source_median":real["camera_domain"].get("camera_center_source_median"),
            "aligned_max_abs_zscore":real["camera_domain"].get("aligned_max_abs_zscore"),
            "warnings":real["warnings"],
            "note":"Adapter/camera preflight only; no SAM3D cache or Field Converter pretrained inference was executed.",
        }
    passed=(report["synthetic_adapter_validation"]["preflight_ready"] and
            report["synthetic_adapter_validation"]["output_schema"]=="world-grounded-pose-state-1.0" and
            report["synthetic_adapter_validation"]["implementation_gate"]=="PASS")
    report["status"]="PASS" if passed else "FAIL"
    out=Path('validation_reports/STAGE4_V040_ADAPTER_VALIDATION_REPORT.json'); out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
