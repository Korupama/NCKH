from __future__ import annotations
from pathlib import Path
from typing import Any
import hashlib, numpy as np
from ..contracts import BallCandidate2D

def sha256_file(path:str|Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()

class SoccerNetV3DYOLOProvider:
    name="soccernet-v3d-yolo"
    def __init__(self,weights:str|Path,*,conf_floor:float=0.05,top_k:int=10,imgsz:int=1920,device:str="cpu",iou:float=0.5)->None:
        try: from ultralytics import YOLO
        except Exception as exc: raise RuntimeError('Install YOLO support with: pip install -e ".[yolo]"') from exc
        self.weights=str(Path(weights).expanduser().resolve())
        if not Path(self.weights).is_file(): raise FileNotFoundError(self.weights)
        self.conf_floor=float(conf_floor); self.top_k=int(top_k); self.imgsz=int(imgsz); self.device=str(device); self.iou=float(iou); self.model=YOLO(self.weights); self.weights_sha256=sha256_file(self.weights)
    def detect(self,image_bgr:np.ndarray,frame_index:int)->list[BallCandidate2D]:
        results=self.model.predict(source=image_bgr,conf=self.conf_floor,iou=self.iou,imgsz=self.imgsz,device=self.device,verbose=False,max_det=max(self.top_k*4,self.top_k))
        if not results or results[0].boxes is None or len(results[0].boxes)==0: return []
        boxes=results[0].boxes; xyxy=boxes.xyxy.detach().cpu().numpy(); conf=boxes.conf.detach().cpu().numpy(); cls=boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros(len(conf)); order=np.argsort(-conf)[:self.top_k]; out=[]
        for rank,idx in enumerate(order.tolist(),start=1):
            x1,y1,x2,y2=[float(x) for x in xyxy[idx]]; w=max(0.0,x2-x1); h=max(0.0,y2-y1)
            out.append(BallCandidate2D(int(frame_index),f"ball_{frame_index:08d}_{rank:02d}",[x1,y1,x2,y2],[(x1+x2)/2,(y1+y2)/2],float(conf[idx]),"yolo-sn-v3d",0.5*(w+h),metadata={"class_id":int(cls[idx]),"rank_by_detector_score":rank}))
        return out
    def info(self)->dict[str,Any]:
        return {"name":self.name,"weights":self.weights,"weights_sha256":self.weights_sha256,"conf_floor":self.conf_floor,"top_k":self.top_k,"imgsz":self.imgsz,"device":self.device,"nms_iou":self.iou}
