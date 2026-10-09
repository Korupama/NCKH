"""Fresh single-image RTMW input adapter for the existing Stage-3 processor."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import uuid

import cv2
import numpy as np


def run_live_stage3(image_path, out_dir, device='cpu', frame_index=104):
    project = Path(__file__).resolve().parents[2]
    package = str(project / 'stage3_pose2d_v0.1')
    if package not in sys.path:
        sys.path.insert(0, package)
    from stage3_pose2d.processor import run_stage3
    from stage3_pose2d.rtmw_onnx import RTMWOpenCVDNN
    from stage3_pose2d.schemas import Stage3Config
    from stage3_pose2d.wholebody133 import WHOLEBODY_KEYPOINT_NAMES

    image_path, out_dir = Path(image_path).resolve(), Path(out_dir).resolve()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'Cannot read Stage-3 source image: {image_path}')
    source = out_dir / 'stage2.json'
    entity = copy.deepcopy(json.loads(source.read_text(encoding='utf-8')))
    replay = entity.get('replay_context', {})
    height, width = image.shape[:2]
    if replay.get('selected_frame') != frame_index:
        raise ValueError('Stage-2/Stage-3 frame mismatch')
    if (replay.get('image_width'), replay.get('image_height')) != (width, height):
        raise ValueError('Stage-2/Stage-3 image size mismatch')
    if replay.get('coordinate_space') != 'RAW_DISTORTED_PIXEL':
        raise ValueError('Stage 3 requires RAW_DISTORTED_PIXEL coordinates')

    # Keep the replay identity shared with Stage 2/6. Disable the optional
    # video overlay explicitly below instead of erasing its source path.
    replay.update(source_image_path=str(image_path),
                  window_start=frame_index, window_end=frame_index)
    candidates, seen = [], set()
    for track in entity.get('tracks', []):
        tid = track['track_id']
        if tid in seen:
            raise ValueError(f'Duplicate Stage-2 track: {tid}')
        seen.add(tid)
        observations = [o for o in track.get('observations', []) if o.get('frame_index') == frame_index]
        track['observations'] = observations
        is_person = track.get('role') in ('player', 'goalkeeper', 'referee')
        track['candidate_for_stage3'] = is_person and bool(observations)
        if track['candidate_for_stage3']:
            if len(observations) != 1:
                raise ValueError(f'Ambiguous Stage-2 observation: {tid}')
            bbox = np.asarray(observations[0].get('bbox_xyxy'), dtype=float)
            if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] <= bbox[:2]):
                raise ValueError(f'Invalid Stage-2 person bbox: {tid}')
            candidates.append(tid)

    model_path = project / 'weights/rtmw_l_384x288.onnx'
    if not model_path.is_file():
        raise FileNotFoundError(f'Stage-3 RTMW model is missing: {model_path}')
    config = Stage3Config(rtmw_model=str(model_path), rtmw_device=device)
    model = RTMWOpenCVDNN(model_path, input_width=config.rtmw_input_width,
                         input_height=config.rtmw_input_height,
                         bbox_padding=config.bbox_padding, device=device)
    runtime = out_dir / 'stage3_runs' / uuid.uuid4().hex
    runtime.mkdir(parents=True, exist_ok=False)
    raw_tracks = {}
    for track in entity['tracks']:
        if not track['candidate_for_stage3']:
            continue
        bbox = track['observations'][0]['bbox_xyxy']
        pose = model.infer_one(image, bbox)
        diagnostics = pose.inference_diagnostics
        records = [dict(index=i, name=name,
                        x=float(pose.keypoints_xy[i, 0]) if np.isfinite(pose.keypoints_xy[i, 0]) else None,
                        y=float(pose.keypoints_xy[i, 1]) if np.isfinite(pose.keypoints_xy[i, 1]) else None,
                        raw_score=float(pose.scores[i]) if np.isfinite(pose.scores[i]) else None)
                   for i, name in enumerate(WHOLEBODY_KEYPOINT_NAMES)]
        raw_tracks[track['track_id']] = {'observations': [{
            'frame_index': frame_index, 'bbox_xyxy': bbox,
            'pose': dict(keypoints=records, backend='opencv_dnn',
                         score_semantics='RTMW_RAW_SIMCC_MAX_NOT_CALIBRATED_PROBABILITY',
                         model_input_size_width_height=diagnostics['input_size_wh'],
                         inference_diagnostics=diagnostics),
        }]}

    # This raw interchange artifact is produced on every invocation. It is
    # never looked up or reused from a previous run. QA remains Stage-3-owned.
    raw = dict(schema_version='stage2-raw-rtmw-track-cache-1.0',
               coordinate_space='RAW_DISTORTED_PIXEL',
               score_semantics='RTMW_RAW_SIMCC_MAX_NOT_CALIBRATED_PROBABILITY',
               provenance={'source': 'STAGE9_FRESH_RTMW', 'reused_preexisting_artifact': False},
               tracks=raw_tracks)
    for name, payload in [('stage2_entity_tracks.json', entity),
                          ('stage2_rtmw_track_cache.json', raw),
                          ('stage3_handoff.json', {'candidate_track_ids': candidates})]:
        (runtime / name).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    state = run_stage3(stage2_dir=runtime, output_dir=runtime / 'result', config=config,
                       render_overlay=False)
    for track in state['tracks']:
        for observation in track['observations']:
            observation['provenance'].update(source='STAGE9_FRESH_RTMW',
                                             fresh_inference=True, reused_preexisting_artifact=False)
            for point in observation['keypoints_133']:
                point['source'] = 'RTMW_FRESH_INFERENCE'
    state['live_execution'] = dict(mode='EXISTING_STAGE3_PIPELINE_FRESH_SINGLE_FRAME_INFERENCE',
                                   source_stage2=str(source), source_image=str(image_path),
                                   temporal_status='NOT_AVAILABLE_SINGLE_FRAME')
    encoded = json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False)
    Path(state['artifacts']['tracked_pose_2d_state']).write_text(encoded, encoding='utf-8')
    (out_dir / 'stage3.json').write_text(encoded, encoding='utf-8')
    invocation = dict(execution=state['live_execution']['mode'], pipeline_owner=package,
                      processor='stage3_pose2d.processor.run_stage3',
                      frame_index=frame_index, runtime=str(runtime),
                      reused_preexisting_artifact=False, metrics=state['metrics'])
    (out_dir / 'stage3_invocation.json').write_text(json.dumps(invocation, indent=2), encoding='utf-8')
    return state
