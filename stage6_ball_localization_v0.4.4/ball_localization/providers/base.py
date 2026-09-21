from __future__ import annotations
from typing import Protocol
import numpy as np
from ..contracts import BallCandidate2D
class BallCandidateProvider(Protocol):
    name:str
    def detect(self,image_bgr:np.ndarray,frame_index:int)->list[BallCandidate2D]: ...
    def info(self)->dict: ...
