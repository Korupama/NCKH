from __future__ import annotations
from dataclasses import dataclass
from typing import Dict,List,Optional
import math, numpy as np
from ..contracts import BallCandidate2D

@dataclass
class ViterbiConfig:
    missing_cost:float=2.6
    enter_exit_cost:float=0.8
    motion_weight:float=7.0
    size_weight:float=0.35
    max_normalized_jump:float=0.30
    epsilon:float=1e-6

def _emission(c:Optional[BallCandidate2D],cfg:ViterbiConfig)->float:
    if c is None:return cfg.missing_cost
    return -math.log(max(cfg.epsilon,min(1.0,float(c.ranking_score))))

def _transition(a:Optional[BallCandidate2D],b:Optional[BallCandidate2D],diag:float,cfg:ViterbiConfig)->float:
    if a is None and b is None:return 0.0
    if a is None or b is None:return cfg.enter_exit_cost
    jump=float(np.linalg.norm(np.asarray(b.center_uv)-np.asarray(a.center_uv))/max(diag,1.0)); cost=cfg.motion_weight*jump
    if jump>cfg.max_normalized_jump: cost+=8.0*(jump-cfg.max_normalized_jump)
    cost+=cfg.size_weight*abs(math.log(max(b.diameter_px,1e-3)/max(a.diameter_px,1e-3)))
    return cost

def select_ball_path(candidates_by_frame:Dict[int,List[BallCandidate2D]],frame_indices:List[int],*,image_width:int,image_height:int,config:ViterbiConfig|None=None)->Dict[int,Optional[BallCandidate2D]]:
    cfg=config or ViterbiConfig();
    if not frame_indices:return {}
    diag=float(math.hypot(image_width,image_height)); states=[list(candidates_by_frame.get(fi,[]))+[None] for fi in frame_indices]; costs=[np.asarray([_emission(s,cfg) for s in states[0]],float)]; back=[np.full(len(states[0]),-1,int)]
    for t in range(1,len(frame_indices)):
        cur=np.full(len(states[t]),np.inf); b=np.full(len(states[t]),-1,int)
        for j,sj in enumerate(states[t]):
            vals=[costs[t-1][i]+_transition(si,sj,diag,cfg) for i,si in enumerate(states[t-1])]; i=int(np.argmin(vals)); cur[j]=vals[i]+_emission(sj,cfg); b[j]=i
        costs.append(cur); back.append(b)
    j=int(np.argmin(costs[-1])); chosen={}
    for t in range(len(frame_indices)-1,-1,-1):
        chosen[frame_indices[t]]=states[t][j]; j=int(back[t][j]) if t>0 else -1
    return chosen
