from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from stage4_metric3d.wholebody import WHOLEBODY_NAMES, POSE23_NAMES
from stage4_metric3d.camera import CameraStateLite
from stage4_metric3d.backends.sam3d_pitch_refined.cache import MHR70_NAMES, save_sam3d_native_cache
from stage4_metric3d.backends.sam3d_pitch_refined.joint_mapping import DEFAULT_MAPPING
from stage4_metric3d.backends.sam3d_pitch_refined.geometry import project_camera_points


def look_at(C, target, up=np.array([0.0, 0.0, 1.0])):
    f = np.asarray(target, dtype=float) - np.asarray(C, dtype=float); f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    d = np.cross(f, r); d /= np.linalg.norm(d)
    return np.stack([r, d, f])


def canonical_world_pose(x=10.0, y=0.0):
    return {
        "nose": [x, y, 1.75], "left_eye": [x, y+0.03, 1.78], "right_eye": [x, y-0.03, 1.78],
        "left_ear": [x, y+0.08, 1.75], "right_ear": [x, y-0.08, 1.75],
        "left_shoulder": [x, y+0.20, 1.45], "right_shoulder": [x, y-0.20, 1.45],
        "left_elbow": [x+0.04, y+0.34, 1.18], "right_elbow": [x+0.04, y-0.34, 1.18],
        "left_wrist": [x+0.08, y+0.40, 0.95], "right_wrist": [x+0.08, y-0.40, 0.95],
        "left_hip": [x, y+0.14, 0.95], "right_hip": [x, y-0.14, 0.95],
        "left_knee": [x+0.08, y+0.12, 0.50], "right_knee": [x+0.08, y-0.12, 0.50],
        "left_ankle": [x+0.13, y+0.10, 0.05], "right_ankle": [x+0.13, y-0.10, 0.05],
        "left_big_toe": [x+0.28, y+0.09, 0.0], "left_small_toe": [x+0.26, y+0.13, 0.0], "left_heel": [x+0.04, y+0.10, 0.0],
        "right_big_toe": [x+0.28, y-0.09, 0.0], "right_small_toe": [x+0.26, y-0.13, 0.0], "right_heel": [x+0.04, y-0.10, 0.0],
    }


def build_case(root: Path, prior_offset_cam=(0.45, -0.25, 0.65)):
    root.mkdir(parents=True, exist_ok=True); camera_dir=root/'cameras'; camera_dir.mkdir()
    frames=[10,11,12]; tid='track_001'
    K=np.array([[1100.,0,640.],[0,1100.,360.],[0,0,1.]])
    C=np.array([0.,-70.,15.]); R=look_at(C,np.array([10.,0.,1.]))
    tracks_obs=[]; rel_all=np.full((3,1,70,3),np.nan,np.float32); sam2d=np.full((3,1,70,2),np.nan,np.float32)
    cam_t=np.full((3,1,3),np.nan,np.float32); focal=np.full((3,1),1100.,np.float32); boxes=np.zeros((3,1,4),np.float32)
    true_trans=[]
    for ti,frame in enumerate(frames):
        camera={"schema_version":"1.2","frame_index":frame,"status":"VALID","image":{"width":1280,"height":720},"intrinsics":{"K":K.tolist()},"extrinsics":{"R_world_to_camera":R.tolist(),"camera_center_world_m":C.tolist()},"distortion":{"radial":[0,0,0,0,0,0],"tangential":[0,0],"thin_prism":[0,0,0,0]},"pitch":{"length_m":105.,"width_m":68.}}
        cp=camera_dir/f'camera_state_{frame:08d}.json'; cp.write_text(json.dumps(camera))
        cam=CameraStateLite.from_dict(camera); pose=canonical_world_pose(10.+0.12*ti,0.)
        origin_world=np.array([10.+0.12*ti,0.,1.0]); t_true=np.asarray(cam.world_to_camera(origin_world)); true_trans.append(t_true)
        rel=np.zeros((70,3),float)
        for e in DEFAULT_MAPPING:
            world=np.asarray(pose[e.canonical_name],float); rel[e.sam3d_index]=np.asarray(cam.world_to_camera(world))-t_true
        # Put unmapped hand/extra landmarks near the origin so the native cache remains finite.
        rel_all[ti,0]=rel
        prior=t_true+np.asarray(prior_offset_cam,float); cam_t[ti,0]=prior
        sam2d[ti,0]=project_camera_points(cam,rel+prior[None,:],distort=False)
        kps=[]; uv23=[]
        for j,name in enumerate(WHOLEBODY_NAMES):
            if j<23:
                uv=np.asarray(cam.project_world(np.asarray(pose[POSE23_NAMES[j]],float),distort=True)); uv23.append(uv)
                x,y=map(float,uv); state='VALID'; score=.98
            else: x=y=score=None; state='MISSING'
            kps.append({"index":j,"name":name,"x":x,"y":y,"state":state,"raw_model_score":score})
        arr=np.asarray(uv23); boxes[ti,0]=[float(arr[:,0].min()-20),float(arr[:,1].min()-20),float(arr[:,0].max()+20),float(arr[:,1].max()+20)]
        tracks_obs.append({"frame_index":frame,"source_bbox_xyxy":boxes[ti,0].tolist(),"pose_status":"VALID","keypoints_133":kps})
    state={"schema_version":"tracked-pose-2d-state-1.0","coordinate_space":"RAW_DISTORTED_PIXEL","keypoint_schema":{"count":133,"names":list(WHOLEBODY_NAMES)},"replay_context":{"selected_frame":11,"fps":30.0,"frame_count":30,"image_width":1280,"image_height":720},"tracks":[{"track_id":tid,"upstream_role":"player","upstream_identity_confidence":.99,"observations":tracks_obs}]}
    stage3=root/'tracked_pose_2d_state.json'; stage3.write_text(json.dumps(state))
    cache=save_sam3d_native_cache(root/'sam3d_native.npz',frame_indices=frames,track_ids=[tid],boxes_xyxy=boxes,skel_2d_px=sam2d,skel_3d_relative_m=rel_all,pred_cam_t_m=cam_t,focal_length_px=focal,valid_mask=np.ones((3,1),bool),joint_names=MHR70_NAMES,metadata={"producer":"synthetic-test"})
    return stage3,camera_dir,cache,np.asarray(true_trans),frames
