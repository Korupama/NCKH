from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json
from ..contracts import BallCandidate2D

def load_stage2_candidates(path:str|Path)->tuple[Dict[str,Any],Dict[int,list[BallCandidate2D]]]:
    p=Path(path).expanduser().resolve(); data=json.loads(p.read_text(encoding="utf-8")); replay=dict(data.get("replay_context") or {}); out={}
    for key,rows in (data.get("auxiliary_ball_detections_by_frame") or {}).items():
        fi=int(key); candidates=[]
        for rank,d in enumerate(sorted(rows or [],key=lambda x:float(x.get("score",0.0)),reverse=True),start=1):
            x1,y1,x2,y2=[float(x) for x in d["bbox_xyxy"]]
            candidates.append(BallCandidate2D(fi,str(d.get("detection_id") or f"sst_ball_{fi}_{rank}"),[x1,y1,x2,y2],[(x1+x2)/2,(y1+y2)/2],float(d.get("score",0.0)),"stage2-sst-ball",0.5*((x2-x1)+(y2-y1)),metadata={"stage2_source":str(p)}))
        out[fi]=candidates
    return replay,out
