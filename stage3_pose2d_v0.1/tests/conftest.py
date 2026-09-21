from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pytest

from stage3_pose2d.wholebody133 import WHOLEBODY_KEYPOINT_NAMES


def make_pose_records(bbox=(100.0,100.0,200.0,300.0), score=3.0):
    x1,y1,x2,y2=bbox; w=x2-x1; h=y2-y1
    # Plausible body/feet for first 23, compact face/hands around torso/arms.
    template = [
        (.50,.12),(.47,.11),(.53,.11),(.44,.13),(.56,.13),
        (.38,.27),(.62,.27),(.32,.42),(.68,.42),(.28,.56),(.72,.56),
        (.43,.52),(.57,.52),(.42,.70),(.58,.70),(.41,.88),(.59,.88),
        (.38,.96),(.42,.96),(.40,.91),(.58,.96),(.62,.96),(.60,.91),
    ]
    out=[]
    for i,name in enumerate(WHOLEBODY_KEYPOINT_NAMES):
        if i < 23:
            rx,ry=template[i]
        elif i < 91:
            a=(i-23)/67.0; rx=.44+.12*a; ry=.08+.08*np.sin(a*np.pi)
        elif i < 112:
            a=(i-91)/20.0; rx=.26-.04*a; ry=.55+.08*a
        else:
            a=(i-112)/20.0; rx=.74+.04*a; ry=.55+.08*a
        out.append({"index":i,"name":name,"x":float(x1+rx*w),"y":float(y1+ry*h),"raw_score":float(score)})
    return out


def write_stage2_fixture(base: Path, *, frames=(9,10,11), selected_frame=10, missing_t0=False, bad_coord=False):
    base.mkdir(parents=True,exist_ok=True)
    bbox=[100.0,100.0,200.0,300.0]
    observations=[]; cache_obs=[]
    for f in frames:
        observations.append({
            "frame_index":f,"bbox_xyxy":bbox,"detector_score":.9,
            "physical_human_id":f"human_{f}","pose_cache_key":f"track_001:frame_{f}"
        })
        if not (missing_t0 and f==selected_frame):
            records=make_pose_records(tuple(bbox))
            # small horizontal motion by frame
            for r in records:
                r["x"] += float(f-selected_frame)*2.0
            cache_obs.append({
                "frame_index":f,"physical_human_id":f"human_{f}",
                "pose_cache_key":f"track_001:frame_{f}","bbox_xyxy":bbox,
                "pose":{
                    "bbox_xyxy":bbox,"expanded_pose_bbox_xyxy":[95,90,205,310],
                    "backend":"rtmw_onnx","is_learned_model":True,
                    "score_semantics":"RTMW_RAW_SIMCC_MAX_NOT_CALIBRATED_PROBABILITY",
                    "keypoints":records,
                }
            })
    coord="UNDISTORTED_PIXEL" if bad_coord else "RAW_DISTORTED_PIXEL"
    entity={
        "schema_version":"entity-track-state-1.0","stage2_version":"stage2-sst-rtmw-1.3.0",
        "replay_context":{
            "selected_frame":selected_frame,"window_start":min(frames),"window_end":max(frames),
            "image_width":1920,"image_height":1080,"fps":25.0,"coordinate_space":coord,"video_path":""
        },
        "tracks":[{
            "track_id":"track_001","role":"player","role_score":.95,"role_margin":.4,
            "candidate_for_stage3":True,"identity_confidence":.9,"observations":observations
        }]
    }
    cache={
        "schema_version":"stage2-raw-rtmw-track-cache-1.0",
        "score_semantics":"RTMW_RAW_SIMCC_MAX_NOT_CALIBRATED_PROBABILITY",
        "coordinate_space":coord,
        "tracks":{"track_001":{"candidate_for_stage3":True,"final_stage2_role":"player","observations":cache_obs}}
    }
    handoff={
        "stage3_input_ready":not missing_t0,"candidate_track_ids":["track_001"],
        "candidate_tracks_missing_rtmw_at_selected_frame":["track_001"] if missing_t0 else [],
        "entity_track_state_schema":"entity-track-state-1.0",
        "raw_rtmw_track_cache":str(base/"stage2_rtmw_track_cache.json")
    }
    for name,data in (("stage2_entity_tracks.json",entity),("stage2_rtmw_track_cache.json",cache),("stage3_handoff.json",handoff)):
        (base/name).write_text(json.dumps(data,indent=2),encoding="utf-8")
    return base


@pytest.fixture
def stage2_fixture(tmp_path):
    return write_stage2_fixture(tmp_path/"stage2")
