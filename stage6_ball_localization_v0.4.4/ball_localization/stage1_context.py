from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json, cv2

def _find(root:Path,rel:str)->Path:
    for p in (root/rel,root/"stage_1_camera"/rel):
        if p.is_file(): return p
    raise FileNotFoundError(rel)

def load_replay_context_from_stage1(stage1_root:str|Path,video_path:str|Path|None=None)->Dict[str,Any]:
    root=Path(stage1_root).expanduser().resolve(); mp=_find(root,"outputs/stage1_final/camera_timeline_target_window_manifest.json"); m=json.loads(mp.read_text(encoding="utf-8")); frames=[int(x["frame_index"]) for x in m.get("frames",[])]
    if not frames: raise ValueError("timeline manifest has no frames")
    if video_path is None:
        vids=list(root.glob("*.mp4"))+list((root/"stage_1_camera").glob("*.mp4"));
        if not vids: raise FileNotFoundError("video not supplied and no .mp4 found in Stage-1 workspace")
        video=vids[0].resolve()
    else: video=Path(video_path).expanduser().resolve()
    cap=cv2.VideoCapture(str(video));
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video}")
    fps=float(cap.get(cv2.CAP_PROP_FPS)); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    return {"schema_version":"stage6-stage1-v12-adapter-1.0","video_path":str(video),"video_id":video.stem,"fps":fps,"frame_count":count,"image_width":width,"image_height":height,"selected_frame":int(m["target_frame_index"]),"window_start":min(frames),"window_end":max(frames),"coordinate_space":"RAW_DISTORTED_PIXEL","stage1_package_version":str(m.get("package_version")),"camera_state_schema":str(m.get("camera_state_schema")),"source_timeline_manifest":str(mp)}
