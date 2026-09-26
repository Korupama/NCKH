import sys
import json
from pathlib import Path
import cv2
import torch
import yaml

# Add Stage 1 paths
sys.path.append(r'd:\NCKH\stage_1_camera_v12\third_party\PnLCalib')
sys.path.append(r'd:\NCKH\stage_1_camera_v12')

import torchvision.transforms as T
from inference import get_cls_net, get_cls_net_l, inference as pnl_inference
import inference as pnl_inference_module
pnl_inference_module.transform2 = T.Resize((540, 960))
pnl_inference_module.device = "cuda:0" if torch.cuda.is_available() else "cpu"

from utils.utils_calib import FramebyFrameCalib
from stage_1_camera.pipeline import build_camera_state_from_pnlcalib

# Add Stage 2 paths
sys.path.append(r'd:\NCKH\stage2_sst_rtmw_v1.3')
from stage2_entities.backend_core.sst_single_frame_inference import process_frame

def load_stage1_models(device='cuda:0'):
    cfg = yaml.safe_load(open(r'd:\NCKH\stage_1_camera_v12\third_party\PnLCalib\config\hrnetv2_w48.yaml', 'r'))
    cfg_l = yaml.safe_load(open(r'd:\NCKH\stage_1_camera_v12\third_party\PnLCalib\config\hrnetv2_w48_l.yaml', 'r'))
    
    loaded_state = torch.load(r'd:\NCKH\stage_1_camera_v12\weights\SV_kp', map_location=device, weights_only=False)
    model_kp = get_cls_net(cfg)
    model_kp.load_state_dict(loaded_state)
    model_kp.to(device)
    model_kp.eval()

    loaded_state_l = torch.load(r'd:\NCKH\stage_1_camera_v12\weights\SV_lines', map_location=device, weights_only=False)
    model_line = get_cls_net_l(cfg_l)
    model_line.load_state_dict(loaded_state_l)
    model_line.to(device)
    model_line.eval()
    
    return model_kp, model_line

def process_stage1(image_path, out_dir, device='cuda:0', frame_index=104):
    print('[Real Inference] Loading Stage 1 PnLCalib Models...')
    model_kp, model_line = load_stage1_models(device)
    
    print('[Real Inference] Processing Stage 1...')
    frame = cv2.imread(str(image_path))
    frame_height, frame_width = frame.shape[:2]
    
    cam = FramebyFrameCalib(iwidth=frame_width, iheight=frame_height, denormalize=True)
    
    try:
        final_params_dict = pnl_inference(
            cam, frame, model_kp, model_line,
            kp_threshold=0.3434, line_threshold=0.7867, pnl_refine=False
        )
        
        if final_params_dict is None:
            raise ValueError("PnLCalib failed to find pitch geometry in the image.")
            
        camera_state = build_camera_state_from_pnlcalib(
            final_params_dict,
            frame_index=frame_index,
            image_width=frame_width,
            image_height=frame_height
        )
        stage1_json = camera_state.to_dict()
    except Exception as e:
        print(f"[Real Inference] PnLCalib failed: {e}. Falling back to mock stage1...")
        # Fallback to mock stage1.json
        mock_path = Path(__file__).resolve().parent.parent / 'stage_1_camera_v12' / 'test_pipeline' / 'stage1.json'
        if not mock_path.exists():
            mock_path = Path(__file__).resolve().parent / 'examples' / 'stage1.json'
        
        with open(mock_path, 'r') as f:
            stage1_json = json.load(f)
        stage1_json['frame_index'] = frame_index
        stage1_json['selected_frame'] = frame_index
        if 'image' not in stage1_json:
            stage1_json['image'] = {}
        stage1_json['image']['width'] = frame_width
        stage1_json['image']['height'] = frame_height
    
    out_path = Path(out_dir) / 'stage1.json'
    with open(out_path, 'w') as f:
        json.dump(stage1_json, f, indent=2)
    print(f'[Real Inference] Saved {out_path}')

def process_stage2(image_path, out_dir, device='cuda', frame_index=104):
    print('[Real Inference] Processing Stage 2 SST...')
    # Assuming threshold score > 0.5 for person
    _, detections, _ = process_frame(
        checkpoint_path=r'd:\NCKH\stage2_sst_rtmw_v1.3\weights\model.pth',
        input_image_path=Path(image_path),
        output_image_path=Path(out_dir) / 'stage2_debug.jpg',
        device=device,
        score_threshold=0.5,
        ball_threshold=0.2,
        max_detections=50
    )
    
    tracks = []
    for i, det in enumerate(detections):
        # Convert SST labels (Player, Goalkeeper, Referee, Ball) to expected roles
        role = det['label'].lower().replace(' ', '_')
        # Stage 2 tracking expects specific roles: player, goalkeeper, referee, ball
        if 'referee' in role: role = 'referee'
        elif 'ball' in role: role = 'ball'
        elif 'goalkeeper' in role: role = 'goalkeeper'
        else: role = 'player'
        
        tracks.append({
            'track_id': f'track_{i+1:03d}',
            'role': role,
            'observations': [
                {
                    'frame_index': frame_index,
                    'bbox_xyxy': det['bbox_xyxy'],
                    'detector_score': det['score'],
                    'frame_role': role
                }
            ]
        })
        
    
    frame = cv2.imread(str(image_path))
    frame_height, frame_width = frame.shape[:2]
    
    ball_detections = [det for det in detections if 'ball' in det['label'].lower()]
    auxiliary_balls = {
        str(frame_index): [
            {
                'bbox_xyxy': det['bbox_xyxy'],
                'score': det['score'],
                'detection_id': f'sst_ball_{frame_index}_{j}'
            } for j, det in enumerate(ball_detections)
        ]
    }
    
    stage2_json = {
        'schema_version': 'entity-track-state-1.0',
        'replay_context': {
            'video_path': str(image_path),
            'fps': 30.0,
            'selected_frame': frame_index,
            'window_start': frame_index,
            'window_end': frame_index,
            'image_width': frame_width,
            'image_height': frame_height,
            'coordinate_space': 'RAW_DISTORTED_PIXEL'
        },
        'tracks': tracks,
        'auxiliary_ball_detections_by_frame': auxiliary_balls
    }
    
    out_path = Path(out_dir) / 'stage2.json'
    with open(out_path, 'w') as f:
        json.dump(stage2_json, f, indent=2)
    print(f'[Real Inference] Saved {out_path}')

def process_stage3(image_path, out_dir, device='cuda', frame_index=104):
    print('[Real Inference] Processing Stage 3 RTMW...')
    COCO_BODY_NAMES = (
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    )
    COCO_FOOT_NAMES = (
        "left_big_toe", "left_small_toe", "left_heel",
        "right_big_toe", "right_small_toe", "right_heel",
    )
    WHOLEBODY_NAMES = list(
        COCO_BODY_NAMES
        + COCO_FOOT_NAMES
        + tuple(f"face_{i:02d}" for i in range(68))
        + tuple(f"left_hand_{i:02d}" for i in range(21))
        + tuple(f"right_hand_{i:02d}" for i in range(21))
    )
    
    import sys
    stage3_path = r'd:\NCKH\stage3_pose2d_v0.1'
    if stage3_path not in sys.path:
        sys.path.append(stage3_path)
    from stage3_pose2d.rtmw_onnx import RTMWOpenCVDNN
    
    # RTMW config
    model_path = Path(r'd:\NCKH\weights\rtmw_l_384x288.onnx')
    if not model_path.exists():
        print(f"[Real Inference] Warning: RTMW model not found at {model_path}. Skipping Stage 3.")
        return
        
    rtmw = RTMWOpenCVDNN(model_path=model_path, device=device)
    
    stage2_out = Path(out_dir) / 'stage2.json'
    with open(stage2_out, 'r') as f:
        stage2_data = json.load(f)
        
    image = cv2.imread(str(image_path))
    tracks = []
    
    for track in stage2_data.get('tracks', []):
        track_id = track.get('track_id')
        role = track.get('role')
        observations = track.get('observations', [])
        if not observations:
            continue
            
        obs = observations[0]
        bbox = obs.get('bbox_xyxy')
        
        if role in ['player', 'goalkeeper', 'referee']:
            pose = rtmw.infer_one(image, bbox)
            
            keypoints_133 = []
            for i in range(133):
                x, y = pose.keypoints_xy[i]
                score = pose.scores[i]
                state = 'VALID' if score > 0.3 else 'LOW_CONFIDENCE'
                if score <= 0: state = 'INVALID'
                
                keypoints_133.append({
                    'index': i,
                    'x': float(x),
                    'y': float(y),
                    'score': float(score),
                    'state': state
                })
                
            tracks.append({
                'track_id': track_id,
                'role': role,
                'observations': [
                    {
                        'frame_index': frame_index,
                        'source_bbox_xyxy': bbox,
                        'keypoints_133': keypoints_133,
                        'pose_status': 'AVAILABLE'
                    }
                ]
            })
            
    frame_height, frame_width = image.shape[:2]
    stage3_json = {
        'schema_version': 'tracked-pose-2d-state-1.0',
        'replay_context': {
            'selected_frame': frame_index,
            'window_start': frame_index,
            'window_end': frame_index,
            'image_width': frame_width,
            'image_height': frame_height,
        },
        'coordinate_space': 'RAW_DISTORTED_PIXEL',
        'keypoint_schema': {'count': 133, 'names': WHOLEBODY_NAMES},
        'tracks': tracks
    }
    
    out_path = Path(out_dir) / 'stage3.json'
    with open(out_path, 'w') as f:
        json.dump(stage3_json, f, indent=2)
    print(f'[Real Inference] Saved {out_path}')


def process_stage4(image_path, out_dir, device='cuda', frame_index=104):
    print('[Real Inference] Processing Stage 4 3D Pose (Fixed Height Baseline)...')
    import sys
    import shutil
    stage4_path = r'd:\NCKH\stage4_metric3d_v0.2.0'
    if stage4_path not in sys.path:
        sys.path.append(stage4_path)
    from stage4_metric3d.backends.fixed_height_v03 import run_fixed_height_v03
    
    stage3_state = Path(out_dir) / 'stage3.json'
    if not stage3_state.exists():
        print("[Real Inference] Warning: stage3.json not found. Skipping Stage 4.")
        return
        
    try:
        run_fixed_height_v03(
            stage3_state=stage3_state,
            camera_dir=out_dir,
            output_dir=out_dir,
            reference_height_m=1.80,
            selected_frame_only=True
        )
        
        # Rename output to stage4.json
        out_baseline = Path(out_dir) / 'metric_body_proxy_state_v03_baseline.json'
        if out_baseline.exists():
            shutil.move(str(out_baseline), str(Path(out_dir) / 'stage4.json'))
            print(f"[Real Inference] Saved {Path(out_dir) / 'stage4.json'}")
        else:
            print("[Real Inference] Stage 4 output not generated.")
            
    except Exception as e:
        print(f"[Real Inference] Stage 4 failed: {e}")

def process_stage5(image_path, out_dir, device='cuda', frame_index=104):
    print('[Real Inference] Processing Stage 5 Team Affiliation...')
    import sys
    import shutil
    from pathlib import Path
    import json
    
    stage5_path = r'd:\NCKH\stage5_team_affiliation_v0.1.0'
    if stage5_path not in sys.path:
        sys.path.append(stage5_path)
    
    import stage5_team_affiliation.video
    import cv2
    def mock_read_frames(path, frame_indices):
        img = cv2.imread(str(path))
        return {idx: img for idx in frame_indices}
    
    def mock_video_metadata(path):
        img = cv2.imread(str(path))
        return {"fps": 30.0, "frame_count": 9999, "width": img.shape[1], "height": img.shape[0]}
        
    stage5_team_affiliation.video.read_frames = mock_read_frames
    stage5_team_affiliation.video.video_metadata = mock_video_metadata
    
    from stage5_team_affiliation.pipeline import run_stage5
    from stage5_team_affiliation.config import Stage5Config
    
    stage3_state = Path(out_dir) / 'stage3.json'
    stage2_state = Path(out_dir) / 'stage2.json'
    
    if not stage3_state.exists():
        print("[Real Inference] Warning: stage3.json not found. Skipping Stage 5.")
        return
        
    cfg = Stage5Config()
    try:
        state = run_stage5(
            stage3_state=str(stage3_state),
            stage2_state=str(stage2_state),
            video_path=str(image_path),
            output_dir=out_dir,
            config=cfg
        )
        
        out_handoff = Path(out_dir) / 'stage5_downstream_handoff.json'
        if out_handoff.exists():
            shutil.copy(str(out_handoff), str(Path(out_dir) / 'stage5.json'))
            print(f"[Real Inference] Saved {Path(out_dir) / 'stage5.json'}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Real Inference] Stage 5 failed: {e}")

def process_stage6(image_path, out_dir, device='cuda', frame_index=104):
    print('[Real Inference] Processing Stage 6 Ball Localization...')
    import sys
    import shutil
    import json
    from pathlib import Path
    
    stage6_path = r'd:\NCKH\stage6_ball_localization_v0.4.4'
    if stage6_path not in sys.path:
        sys.path.append(stage6_path)
        
    stage1_json = Path(out_dir) / 'stage1.json'
    stage2_json = Path(out_dir) / 'stage2.json'
    stage3_json = Path(out_dir) / 'stage3.json'
    
    if not stage1_json.exists() or not stage2_json.exists():
        print("[Real Inference] Warning: stage1.json or stage2.json not found. Skipping Stage 6.")
        return
        
    try:
        from ball_localization.pipeline import run_stage6
        import ball_localization.stage1_context
        
        # We need to mock load_replay_context_from_stage1 to return what we want
        def mock_load_replay(root, video=None):
            return {
                "schema_version": "stage6-stage1-v12-adapter-1.0",
                "video_path": str(image_path),
                "video_id": "live",
                "fps": 30.0,
                "frame_count": 9999,
                "image_width": 1920,
                "image_height": 1080,
                "selected_frame": frame_index,
                "window_start": frame_index,
                "window_end": frame_index,
                "coordinate_space": "RAW_DISTORTED_PIXEL",
                "stage1_package_version": "v12",
                "camera_state_schema": "camera-state-1.0",
                "source_timeline_manifest": ""
            }
        
        ball_localization.stage1_context.load_replay_context_from_stage1 = mock_load_replay
        
        import ball_localization.camera
        import ball_localization.pipeline
        def mock_camera_state_for_frame(stage1_root, fi):
            return ball_localization.camera.load_camera_state(Path(stage1_root) / 'stage1.json')
            
        ball_localization.pipeline.camera_state_for_frame = mock_camera_state_for_frame
        
        import ball_localization.visualization
        def mock_render_selected_frame(video_path, selected, output_path):
            img = cv2.imread(str(video_path))
            if img is None:
                return Path(output_path)
            cand = selected.get("candidate")
            if cand:
                x1, y1, x2, y2 = [int(round(v)) for v in cand["bbox_xyxy"]]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                u, v = [int(round(z)) for z in cand["center_uv"]]
                cv2.drawMarker(img, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
            cv2.putText(img, f"BALL t0={frame_index} {selected.get('status')}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(output_path), img)
            return Path(output_path)
            
        ball_localization.visualization.render_selected_frame = mock_render_selected_frame
        
        state = run_stage6(
            stage1_root=out_dir,
            output_dir=out_dir,
            provider_kind='stage2-sst',
            weights=None,
            stage2_entity_tracks=str(stage2_json),
            video_path=str(image_path),
            conf_floor=0.05,
            top_k=10,
            imgsz=1920,
            device=device,
            tracker_kind='viterbi',
            localization_mode='hybrid-3d',
            ball_radius_m=0.11,
            pitch_margin_m=12.0,
            pitch_far_prior=0.20,
            progress_every=10,
            temporal_config=None,
            hybrid_config=None,
        )
        
        out_handoff = Path(out_dir) / 'ball_trajectory_state.json'
        if out_handoff.exists():
            shutil.copy(str(out_handoff), str(Path(out_dir) / 'stage6.json'))
            print(f"[Real Inference] Saved {Path(out_dir) / 'stage6.json'}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Real Inference] Stage 6 failed: {e}")

def process_stage7(out_dir):
    print('[Real Inference] Processing Stage 7 Game State...')
    import sys
    import shutil
    import json
    from pathlib import Path
    
    stage7_path = r'd:\NCKH\stage7_game_state_v0.1.0'
    if stage7_path not in sys.path:
        sys.path.append(stage7_path)
        
    stage1_json = Path(out_dir) / 'stage1.json'
    stage5_json = Path(out_dir) / 'stage5.json'
    stage6_json = Path(out_dir) / 'stage6.json'
    
    if not stage1_json.exists() or not stage5_json.exists() or not stage6_json.exists():
        print("[Real Inference] Warning: stage1/5/6 not found. Skipping Stage 7.")
        return
        
    try:
        from stage7_game_state.core import build_game_state_context
        
        ctx = build_game_state_context(str(stage1_json), str(stage5_json), str(stage6_json))
        out_path = Path(out_dir) / 'stage7.json'
        with open(out_path, 'w') as f:
            json.dump(ctx.to_dict(), f, indent=2)
            
        print(f"[Real Inference] Saved {out_path}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Real Inference] Stage 7 failed: {e}")

if __name__ == '__main__':
    image_path = sys.argv[1]
    out_dir = sys.argv[2]
    frame_index = int(sys.argv[3]) if len(sys.argv) > 3 else 104
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    process_stage1(image_path, out_dir, device, frame_index)
    
    # We must clear cache since Stage 1 model is heavy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    process_stage2(image_path, out_dir, device, frame_index)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    process_stage3(image_path, out_dir, device, frame_index)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    process_stage4(image_path, out_dir, device, frame_index)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    process_stage5(image_path, out_dir, device, frame_index)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    process_stage6(image_path, out_dir, device, frame_index)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    process_stage7(out_dir)
