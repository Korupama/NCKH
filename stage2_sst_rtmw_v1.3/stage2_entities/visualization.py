from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Mapping
import json

import cv2
import numpy as np

from .contracts import EntityTrackState, ReplayContext

ROLE_COLOR = {
    "player": (70, 220, 70),
    "goalkeeper": (0, 190, 255),
    "referee": (220, 180, 40),
    "other": (160, 160, 160),
}
ROLE_SHORT = {"player": "P", "goalkeeper": "G", "referee": "R", "other": "O"}


def _draw_box(image, entity, *, show_pose_cache=False):
    h, w = image.shape[:2]
    role = entity.get("role", "other")
    color = ROLE_COLOR.get(role, ROLE_COLOR["other"])
    x1,y1,x2,y2 = [int(round(float(x))) for x in entity["bbox_xyxy"]]
    x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w-1,x2),min(h-1,y2)
    thickness = 3 if entity.get("status") == "VALID" else 2
    cv2.rectangle(image, (x1,y1), (x2,y2), color, thickness)
    label = f"{ROLE_SHORT.get(role,'?')}{str(entity['track_id']).split('_')[-1]} {role.upper()}"
    if show_pose_cache and entity.get("pose_available"):
        label += " RTMW"
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    lx = max(0, min(x1, w-tw-8)); ly=max(th+6,y1-4)
    cv2.rectangle(image,(lx,ly-th-5),(lx+tw+6,ly+2),(0,0,0),-1)
    cv2.putText(image,label,(lx+3,ly-2),cv2.FONT_HERSHEY_SIMPLEX,0.52,color,1,cv2.LINE_AA)


def render_selected_frame(
    context: ReplayContext,
    state: EntityTrackState,
    output_path: str | Path,
    *,
    show_excluded: bool = False,
) -> Path:
    cap=cv2.VideoCapture(context.video_path)
    if not cap.isOpened(): raise FileNotFoundError(context.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, context.selected_frame)
    ok, frame=cap.read(); cap.release()
    if not ok or frame is None: raise RuntimeError(f"Cannot decode frame {context.selected_frame}")
    shown=[]
    for e in state.selected_frame_entities:
        if not show_excluded and not e.get("candidate_for_stage3", False):
            continue
        _draw_box(frame,e)
        shown.append(e)
    counts={r:sum(e.get("role")==r for e in shown) for r in ROLE_COLOR}
    lines=[
        f"Stage 2 | frame {context.selected_frame} | {state.status}",
        f"candidates={counts['player']+counts['goalkeeper']} player={counts['player']} GK={counts['goalkeeper']}" +
        (f" ref={counts['referee']} other={counts['other']}" if show_excluded else ""),
    ]
    y=28
    for line in lines:
        cv2.putText(frame,line,(18,y),cv2.FONT_HERSHEY_SIMPLEX,0.68,(255,255,255),2,cv2.LINE_AA); y+=28
    output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(output_path),frame): raise OSError(output_path)
    return output_path


def render_tracking_video(
    context: ReplayContext,
    state: EntityTrackState,
    output_path: str | Path,
    *,
    show_excluded: bool = False,
    trail_length: int = 8,
) -> Path:
    by_frame=defaultdict(list)
    for t in state.tracks:
        if not show_excluded and not t.candidate_for_stage3: continue
        for o in t.observations:
            by_frame[o.frame_index].append({
                "track_id":t.track_id,"role":t.role,"status":t.status,
                "candidate_for_stage3":t.candidate_for_stage3,"bbox_xyxy":o.bbox_xyxy,
                "pose_available":o.pose_cache_key is not None,
            })
    cap=cv2.VideoCapture(context.video_path)
    if not cap.isOpened(): raise FileNotFoundError(context.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES,context.window_start)
    output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc(*"mp4v"),context.fps,(context.image_width,context.image_height))
    trails=defaultdict(lambda:deque(maxlen=trail_length))
    for fi in context.frame_indices:
        ok,frame=cap.read()
        if not ok or frame is None: break
        for e in by_frame.get(fi,[]):
            _draw_box(frame,e,show_pose_cache=True)
            x1,y1,x2,y2=e["bbox_xyxy"]
            trails[e["track_id"]].append((int((x1+x2)/2),int((y1+y2)/2)))
        for tid,pts in trails.items():
            role=next((t.role for t in state.tracks if t.track_id==tid),"other")
            points=list(pts)
            for i in range(1,len(points)):
                cv2.line(frame,points[i-1],points[i],ROLE_COLOR.get(role,ROLE_COLOR["other"]),2,cv2.LINE_AA)
        caption="SELECTED" if fi==context.selected_frame else f"frame {fi}"
        cv2.putText(frame,caption,(18,context.image_height-22),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)
        writer.write(frame)
    cap.release(); writer.release()
    return output_path
