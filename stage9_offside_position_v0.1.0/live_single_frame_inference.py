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
        raise RuntimeError(
            f"Stage 1 PnLCalib failed for frame {frame_index}; fresh pipeline refuses mock fallback"
        ) from e
    
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
            'video_path': str(Path(image_path).resolve()),
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
                    'name': WHOLEBODY_NAMES[i],
                    'x': float(x),
                    'y': float(y),
                    'raw_model_score': float(score),
                    'state': state
                })
                
            tracks.append({
                'track_id': track_id,
                'upstream_role': role,
                'upstream_identity_confidence': track.get('identity_confidence'),
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
            'video_path': str(Path(image_path).resolve()),
            'video_id': 'stage9_live_upload',
            'fps': 30.0,
            'frame_count': frame_index + 1,
            'selected_frame': frame_index,
            'window_start': frame_index,
            'window_end': frame_index,
            'image_width': frame_width,
            'image_height': frame_height,
            'coordinate_space': 'RAW_DISTORTED_PIXEL',
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
    print('[Real Inference] Running the existing Stage 4 SAM3D + v0.5.1 pipeline...')
    from datetime import datetime, timezone
    import os
    import subprocess
    import shutil
    import uuid
    project_root = Path(__file__).resolve().parent.parent
    stage4_root = project_root / 'stage4_metric3d_v0.2.0'
    stage3_state = Path(out_dir) / 'stage3.json'
    camera_state = Path(out_dir) / 'stage1.json'
    missing = [str(path) for path in (stage3_state, camera_state, Path(image_path)) if not path.is_file()]
    if missing:
        raise FileNotFoundError('Stage 4 inputs are missing: ' + ', '.join(missing))

    # The web layer only prepares an isolated runtime directory. All 3D
    # inference, validation and refinement stay owned by the existing Stage-4
    # entry points in stage4_metric3d_v0.2.0.
    # Some already-running Stage-9 servers still use the legacy live_session
    # output directory. Keep every Stage-4 invocation isolated even in that
    # directory so retrying a frame never collides with an earlier run and
    # never reuses its native SAM3D artifact.
    run_stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    stage4_run_id = f'{run_stamp}_f{int(frame_index):08d}_{uuid.uuid4().hex[:8]}'
    runtime = Path(out_dir) / 'stage4_runs' / stage4_run_id
    camera_dir = runtime / 'camera'
    frames_dir = runtime / 'frames'
    result_dir = runtime / 'sam3d-pitch-refined'
    camera_dir.mkdir(parents=True, exist_ok=False)
    frames_dir.mkdir(parents=True, exist_ok=False)
    camera_copy = camera_dir / f'camera_state_{frame_index:08d}.json'
    frame_copy = frames_dir / f'frame_{frame_index:08d}.jpg'
    shutil.copy2(camera_state, camera_copy)
    shutil.copy2(image_path, frame_copy)

    worker_python = project_root / '.venv-sam3d-cpu' / 'Scripts' / 'python.exe'
    stage_python = Path(r'D:\anaconda3\envs\stage4-rtmw3d\python.exe')
    checkpoint = project_root / 'third_party' / 'model.ckpt'
    mhr_model = project_root / 'third_party' / 'mhr_model.pt'
    sam3d_root = project_root / 'third_party' / 'sam-3d-body'
    required = [worker_python, stage_python, checkpoint, mhr_model, sam3d_root / 'sam_3d_body']
    absent = [str(path) for path in required if not path.exists()]
    if absent:
        raise FileNotFoundError('Stage 4 runtime dependency is missing: ' + ', '.join(absent))

    native_artifact = runtime / 'sam3d_native_this_run.npz'
    worker_command = [
        str(worker_python), '-u', str(stage4_root / 'run_sam3d_body_worker.py'),
        '--stage3-state', str(stage3_state),
        '--camera-dir', str(camera_dir),
        '--frames-dir', str(frames_dir),
        '--checkpoint', str(checkpoint),
        '--mhr-path', str(mhr_model),
        '--sam3d-root', str(sam3d_root),
        '--device', 'cuda' if str(device).startswith('cuda') else 'cpu',
        '--cpu-threads', '4',
        '--person-batch-size', '1',
        '--inference-type', 'body',
        '--frame-index', str(frame_index),
        '--output-cache', str(native_artifact),
    ]
    worker_env = os.environ.copy()
    worker_env['SAM3D_DINOV3_ROOT'] = str(project_root / 'third_party' / 'dinov3')
    subprocess.run(worker_command, cwd=str(stage4_root), env=worker_env, check=True)
    if not native_artifact.is_file():
        raise RuntimeError('Existing Stage-4 SAM3D worker did not produce its native artifact')

    common = [
        str(stage_python), str(stage4_root / 'run_stage4.py'), 'sam3d-pitch-refined',
        '--stage3-state', str(stage3_state),
        '--camera-dir', str(camera_dir),
        '--sam3d-cache', str(native_artifact),
        '--selected-frame', str(frame_index),
        '--window-radius', '0',
        '--disable-temporal',
    ]
    subprocess.run(common + ['--preflight-only'], cwd=str(stage4_root), check=True)
    subprocess.run(common + ['--output-dir', str(result_dir)], cwd=str(stage4_root), check=True)

    handoff = result_dir / 'stage4_downstream_handoff.json'
    quality = result_dir / 'stage4_quality_report.json'
    if not handoff.is_file() or not quality.is_file():
        raise RuntimeError('Existing Stage-4 pipeline did not produce its required artifacts')
    shutil.copy2(handoff, Path(out_dir) / 'stage4.json')
    manifest = {
        'schema_version': 'stage9-stage4-invocation-1.0',
        'stage4_run_id': stage4_run_id,
        'frame_index': int(frame_index),
        'pipeline_owner': str(stage4_root),
        'worker_entry_point': str(stage4_root / 'run_sam3d_body_worker.py'),
        'stage4_entry_point': str(stage4_root / 'run_stage4.py'),
        'native_artifact_generated_in_this_run': str(native_artifact),
        'reused_preexisting_native_artifact': False,
        'handoff': str(handoff),
        'quality_report': str(quality),
    }
    (Path(out_dir) / 'stage4_invocation.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f"[Real Inference] Existing Stage 4 pipeline saved {Path(out_dir) / 'stage4.json'}")

def process_stage5(image_path, out_dir, device='cuda', frame_index=104):
    print('[Real Inference] Running the existing Stage 5 team-affiliation pipeline...')
    import shutil
    import subprocess
    from datetime import datetime, timezone
    from pathlib import Path
    import uuid

    out_root = Path(out_dir).resolve()
    stage2_state = out_root / 'stage2.json'
    stage3_state = out_root / 'stage3.json'
    stage4_handoff = out_root / 'stage4.json'
    source_image = Path(image_path).resolve()
    for required in (stage2_state, stage3_state, stage4_handoff, source_image):
        if not required.is_file():
            raise FileNotFoundError(f'Existing Stage-5 pipeline input is missing: {required}')

    project_root = Path(__file__).resolve().parent
    stage5_root = project_root.parent / 'stage5_team_affiliation_v0.1.0'
    stage5_entry = stage5_root / 'run_stage5.py'
    if not stage5_entry.is_file():
        raise FileNotFoundError(f'Existing Stage-5 entry point is missing: {stage5_entry}')

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    stage5_run_id = f'{stamp}_f{int(frame_index):08d}_{uuid.uuid4().hex[:8]}'
    result_dir = out_root / 'stage5_runs' / stage5_run_id
    result_dir.mkdir(parents=True, exist_ok=False)

    command = [
        sys.executable,
        str(stage5_entry),
        '--stage3-state', str(stage3_state),
        '--stage2-state', str(stage2_state),
        '--stage4-handoff', str(stage4_handoff),
        '--video', str(source_image),
        '--output-dir', str(result_dir),
        '--method', 'legacy-v0',
        '--sample-every', '1',
        '--max-samples', '1',
        '--min-torso-frames', '1',
        '--min-lower-frames', '1',
    ]
    subprocess.run(command, cwd=str(stage5_root), check=True)

    handoff = result_dir / 'stage5_downstream_handoff.json'
    state = result_dir / 'team_affiliation_state.json'
    overlay = result_dir / 'selected_frame_team_affiliation.png'
    if not handoff.is_file() or not state.is_file():
        raise RuntimeError('Existing Stage-5 pipeline did not produce its required artifacts')

    public_handoff = out_root / 'stage5.json'
    public_state = out_root / 'stage5_team_affiliation_state.json'
    shutil.copy2(handoff, public_handoff)
    shutil.copy2(state, public_state)
    if overlay.is_file():
        shutil.copy2(overlay, out_root / 'stage5_overlay.png')

    manifest = {
        'schema_version': 'stage9-stage5-invocation-1.0',
        'stage5_run_id': stage5_run_id,
        'frame_index': int(frame_index),
        'pipeline_owner': str(stage5_root),
        'stage5_entry_point': str(stage5_entry),
        'method': 'legacy-v0',
        'input_mode': 'SINGLE_SELECTED_FRAME',
        'temporal_aggregation': False,
        'source_image': str(source_image),
        'stage2_state': str(stage2_state),
        'stage3_state': str(stage3_state),
        'stage4_handoff': str(stage4_handoff),
        'result_directory': str(result_dir),
        'handoff': str(handoff),
        'state': str(state),
        'overlay': str(overlay) if overlay.is_file() else None,
        'reused_preexisting_artifact': False,
    }
    (out_root / 'stage5_invocation.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'[Real Inference] Existing Stage 5 pipeline saved {public_handoff}')

def process_stage6(image_path, out_dir, device='cuda', frame_index=104):
    print('[Real Inference] Running the existing Stage 6 v0.5.1 contact-aware pipeline...')
    import shutil
    import subprocess
    from datetime import datetime, timezone
    from pathlib import Path
    import uuid

    out_root = Path(out_dir).resolve()
    source_image = Path(image_path).resolve()
    inputs = {
        'stage1': out_root / 'stage1.json',
        'stage2': out_root / 'stage2.json',
        'stage3': out_root / 'stage3.json',
        'stage4': out_root / 'stage4.json',
        'image': source_image,
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Existing Stage-6 pipeline inputs are missing: ' + ', '.join(missing))

    project_root = Path(__file__).resolve().parent
    stage6_root = project_root.parent / 'stage6_ball_localization_v0.4.4'
    stage6_entry = stage6_root / 'run_stage6_ball.py'
    if not stage6_entry.is_file():
        raise FileNotFoundError(f'Existing Stage-6 entry point is missing: {stage6_entry}')

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    stage6_run_id = f'{stamp}_f{int(frame_index):08d}_{uuid.uuid4().hex[:8]}'
    runtime = out_root / 'stage6_runs' / stage6_run_id
    result_dir = runtime / 'contact-aware'

    # Build the directory contract already consumed by Stage 6's camera adapter.
    # The copied payload remains the Stage-1 result generated for this request.
    camera_dir = runtime / 'stage1_bundle' / 'outputs' / 'temporal_v13' / 'batch_video' / 'shot_ptz' / 'optimized_camera_states'
    camera_dir.mkdir(parents=True, exist_ok=False)
    camera_copy = camera_dir / f'camera_state_{int(frame_index):08d}.json'
    shutil.copy2(inputs['stage1'], camera_copy)

    command = [
        sys.executable,
        str(stage6_entry),
        '--stage1-root', str(runtime / 'stage1_bundle'),
        '--output-dir', str(result_dir),
        '--provider', 'stage2-sst',
        '--stage2-entity-tracks', str(inputs['stage2']),
        '--video', str(source_image),
        '--stage3-state', str(inputs['stage3']),
        '--stage4-handoff', str(inputs['stage4']),
        '--tracker', 'viterbi',
        '--localization-mode', 'contact-aware',
        '--device', str(device),
        '--progress-every', '1',
    ]
    subprocess.run(command, cwd=str(stage6_root), check=True)

    state = result_dir / 'ball_trajectory_state.json'
    handoff = result_dir / 'stage6_downstream_handoff.json'
    overlay = result_dir / 'selected_frame_ball.png'
    report = result_dir / 'contact_refinement_report.md'
    if not state.is_file() or not handoff.is_file():
        raise RuntimeError('Existing Stage-6 pipeline did not produce its required artifacts')

    public_state = out_root / 'stage6.json'
    shutil.copy2(state, public_state)
    shutil.copy2(handoff, out_root / 'stage6_downstream_handoff.json')
    if overlay.is_file():
        shutil.copy2(overlay, out_root / 'stage6_overlay.png')
    if report.is_file():
        shutil.copy2(report, out_root / 'stage6_contact_report.md')

    manifest = {
        'schema_version': 'stage9-stage6-invocation-1.0',
        'stage6_run_id': stage6_run_id,
        'frame_index': int(frame_index),
        'pipeline_owner': str(stage6_root),
        'stage6_entry_point': str(stage6_entry),
        'provider': 'stage2-sst',
        'tracker': 'viterbi',
        'localization_mode': 'contact-aware',
        'input_mode': 'SINGLE_SELECTED_FRAME',
        'temporal_window_frames': 1,
        'stage1_camera_copy': str(camera_copy),
        'stage2_state': str(inputs['stage2']),
        'stage3_state': str(inputs['stage3']),
        'stage4_handoff': str(inputs['stage4']),
        'source_image': str(source_image),
        'result_directory': str(result_dir),
        'state': str(state),
        'downstream_handoff': str(handoff),
        'overlay': str(overlay) if overlay.is_file() else None,
        'reused_preexisting_artifact': False,
    }
    (out_root / 'stage6_invocation.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'[Real Inference] Existing Stage 6 pipeline saved {public_state}')

def process_stage7(out_dir, frame_index=104):
    print('[Real Inference] Running the existing Stage 7 game-state pipeline...')
    import shutil
    import subprocess
    from datetime import datetime, timezone
    from pathlib import Path
    import uuid

    out_root = Path(out_dir).resolve()
    inputs = {
        'stage1': out_root / 'stage1.json',
        'stage5': out_root / 'stage5.json',
        'stage6': out_root / 'stage6.json',
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Existing Stage-7 pipeline inputs are missing: ' + ', '.join(missing))

    project_root = Path(__file__).resolve().parent
    stage7_root = project_root.parent / 'stage7_game_state_v0.1.0'
    preflight_entry = stage7_root / 'preflight_stage7.py'
    stage7_entry = stage7_root / 'run_stage7.py'
    missing_entries = [str(path) for path in (preflight_entry, stage7_entry) if not path.is_file()]
    if missing_entries:
        raise FileNotFoundError('Existing Stage-7 entry point is missing: ' + ', '.join(missing_entries))

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    stage7_run_id = f'{stamp}_f{int(frame_index):08d}_{uuid.uuid4().hex[:8]}'
    result_dir = out_root / 'stage7_runs' / stage7_run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    preflight_report = result_dir / 'stage7_preflight.json'
    state = result_dir / 'game_state_context.json'

    common_inputs = [
        '--stage1', str(inputs['stage1']),
        '--stage5', str(inputs['stage5']),
        '--stage6', str(inputs['stage6']),
    ]
    preflight_command = [
        sys.executable, str(preflight_entry),
        *common_inputs,
        '--output', str(preflight_report),
    ]
    preflight = subprocess.run(preflight_command, cwd=str(stage7_root))
    if preflight.returncode not in (0, 2) or not preflight_report.is_file():
        raise RuntimeError(f'Existing Stage-7 preflight failed with exit code {preflight.returncode}')

    stage7_command = [
        sys.executable, str(stage7_entry),
        *common_inputs,
        '--output', str(state),
    ]
    execution = subprocess.run(stage7_command, cwd=str(stage7_root))
    if execution.returncode not in (0, 2) or not state.is_file():
        raise RuntimeError(f'Existing Stage-7 pipeline failed with exit code {execution.returncode}')

    stage7_state = json.loads(state.read_text(encoding='utf-8'))
    public_state = out_root / 'stage7.json'
    shutil.copy2(state, public_state)
    shutil.copy2(preflight_report, out_root / 'stage7_preflight.json')
    manifest = {
        'schema_version': 'stage9-stage7-invocation-1.0',
        'stage7_run_id': stage7_run_id,
        'frame_index': int(frame_index),
        'pipeline_owner': str(stage7_root),
        'preflight_entry_point': str(preflight_entry),
        'stage7_entry_point': str(stage7_entry),
        'preflight_exit_code': int(preflight.returncode),
        'pipeline_exit_code': int(execution.returncode),
        'semantic_status': stage7_state.get('status'),
        'reasons': stage7_state.get('reasons', []),
        'stage1_state': str(inputs['stage1']),
        'stage5_handoff': str(inputs['stage5']),
        'stage6_state': str(inputs['stage6']),
        'result_directory': str(result_dir),
        'preflight_report': str(preflight_report),
        'state': str(state),
        'reused_preexisting_artifact': False,
    }
    (out_root / 'stage7_invocation.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'[Real Inference] Existing Stage 7 pipeline saved {public_state} ({stage7_state.get("status")})')

def process_stage8(out_dir, frame_index=104):
    print('[Real Inference] Running the existing Stage 8 offside-reference pipeline...')
    import shutil
    import subprocess
    from datetime import datetime, timezone
    from pathlib import Path
    import uuid

    out_root = Path(out_dir).resolve()
    inputs = {
        'stage4': out_root / 'stage4.json',
        'stage6': out_root / 'stage6_downstream_handoff.json',
        'stage7': out_root / 'stage7.json',
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Existing Stage-8 pipeline inputs are missing: ' + ', '.join(missing))

    project_root = Path(__file__).resolve().parent
    stage8_root = project_root.parent / 'stage8_offside_reference_v0.1.0'
    preflight_entry = stage8_root / 'preflight_stage8.py'
    stage8_entry = stage8_root / 'run_stage8.py'
    missing_entries = [str(path) for path in (preflight_entry, stage8_entry) if not path.is_file()]
    if missing_entries:
        raise FileNotFoundError('Existing Stage-8 entry point is missing: ' + ', '.join(missing_entries))

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    stage8_run_id = f'{stamp}_f{int(frame_index):08d}_{uuid.uuid4().hex[:8]}'
    result_dir = out_root / 'stage8_runs' / stage8_run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    preflight_report = result_dir / 'stage8_preflight.json'
    state = result_dir / 'offside_reference_state.json'
    plot = result_dir / 'stage8_longitudinal_qa.png'

    common_inputs = [
        '--stage4', str(inputs['stage4']),
        '--stage6', str(inputs['stage6']),
        '--stage7', str(inputs['stage7']),
    ]
    preflight = subprocess.run(
        [sys.executable, str(preflight_entry), *common_inputs, '--output', str(preflight_report)],
        cwd=str(stage8_root),
    )
    if preflight.returncode not in (0, 2) or not preflight_report.is_file():
        raise RuntimeError(f'Existing Stage-8 preflight failed with exit code {preflight.returncode}')

    execution = subprocess.run(
        [sys.executable, str(stage8_entry), *common_inputs, '--output', str(state)],
        cwd=str(stage8_root),
    )
    if execution.returncode not in (0, 2) or not state.is_file():
        raise RuntimeError(f'Existing Stage-8 pipeline failed with exit code {execution.returncode}')

    stage8_state = json.loads(state.read_text(encoding='utf-8'))
    public_state = out_root / 'stage8.json'
    shutil.copy2(state, public_state)
    shutil.copy2(preflight_report, out_root / 'stage8_preflight.json')
    if plot.is_file():
        shutil.copy2(plot, out_root / 'stage8_longitudinal_qa.png')

    manifest = {
        'schema_version': 'stage9-stage8-invocation-1.0',
        'stage8_run_id': stage8_run_id,
        'frame_index': int(frame_index),
        'pipeline_owner': str(stage8_root),
        'preflight_entry_point': str(preflight_entry),
        'stage8_entry_point': str(stage8_entry),
        'preflight_exit_code': int(preflight.returncode),
        'pipeline_exit_code': int(execution.returncode),
        'semantic_status': stage8_state.get('status'),
        'reasons': stage8_state.get('reasons', []),
        'stage4_handoff': str(inputs['stage4']),
        'stage6_handoff': str(inputs['stage6']),
        'stage7_state': str(inputs['stage7']),
        'result_directory': str(result_dir),
        'preflight_report': str(preflight_report),
        'state': str(state),
        'plot': str(plot) if plot.is_file() else None,
        'reused_preexisting_artifact': False,
    }
    (out_root / 'stage8_invocation.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'[Real Inference] Existing Stage 8 pipeline saved {public_state} ({stage8_state.get("status")})')

def process_stage9(image_path, out_dir, frame_index=104):
    print('[Real Inference] Running the existing Stage 9 offside-position pipeline in strict mode...')
    import shutil
    import subprocess
    from datetime import datetime, timezone
    from pathlib import Path
    import uuid

    out_root = Path(out_dir).resolve()
    inputs = {
        'stage1': out_root / 'stage1.json',
        'stage3': out_root / 'stage3.json',
        'stage4': out_root / 'stage4.json',
        'stage6': out_root / 'stage6.json',
        'stage7': out_root / 'stage7.json',
        'stage8': out_root / 'stage8.json',
        'image': Path(image_path).resolve(),
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Existing Stage-9 pipeline inputs are missing: ' + ', '.join(missing))

    stage9_root = Path(__file__).resolve().parent
    stage9_entry = stage9_root / 'run_stage9.py'
    if not stage9_entry.is_file():
        raise FileNotFoundError(f'Existing Stage-9 entry point is missing: {stage9_entry}')

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    stage9_run_id = f'{stamp}_f{int(frame_index):08d}_{uuid.uuid4().hex[:8]}'
    result_dir = out_root / 'stage9_runs' / stage9_run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    state = result_dir / 'offside_position_state.json'
    overlay = result_dir / 'stage9_overlay.jpg'
    command = [
        sys.executable, str(stage9_entry),
        '--stage4', str(inputs['stage4']),
        '--stage6', str(inputs['stage6']),
        '--stage7', str(inputs['stage7']),
        '--stage8', str(inputs['stage8']),
        '--stage1', str(inputs['stage1']),
        '--stage3', str(inputs['stage3']),
        '--image', str(inputs['image']),
        '--output', str(state),
        '--overlay', str(overlay),
        '--strict',
    ]
    execution = subprocess.run(command, cwd=str(stage9_root))
    if execution.returncode not in (0, 2) or not state.is_file():
        raise RuntimeError(f'Existing Stage-9 pipeline failed with exit code {execution.returncode}')

    stage9_state = json.loads(state.read_text(encoding='utf-8'))
    public_state = out_root / 'stage9.json'
    shutil.copy2(state, public_state)
    if overlay.is_file():
        shutil.copy2(overlay, out_root / 'stage9_overlay.jpg')
    manifest = {
        'schema_version': 'stage9-live-invocation-1.0',
        'stage9_run_id': stage9_run_id,
        'frame_index': int(frame_index),
        'pipeline_owner': str(stage9_root),
        'stage9_entry_point': str(stage9_entry),
        'mode': 'STRICT',
        'pipeline_exit_code': int(execution.returncode),
        'semantic_status': stage9_state.get('status'),
        'stage4_handoff': str(inputs['stage4']),
        'stage6_state': str(inputs['stage6']),
        'stage7_state': str(inputs['stage7']),
        'stage8_state': str(inputs['stage8']),
        'result_directory': str(result_dir),
        'state': str(state),
        'overlay': str(overlay) if overlay.is_file() else None,
        'reused_preexisting_artifact': False,
    }
    (out_root / 'stage9_invocation.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'[Real Inference] Existing Stage 9 pipeline saved {public_state} ({stage9_state.get("status")})')

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
        
    process_stage7(out_dir, frame_index)

    process_stage8(out_dir, frame_index)

    process_stage9(image_path, out_dir, frame_index)
