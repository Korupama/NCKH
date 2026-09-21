from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any,Dict,Optional
import ast,csv,math,json,re,zipfile

import cv2
import numpy as np

from ..coordinates import coordinate_transform_metadata, soccernet_xyz_to_stage6

def _clean(v:Any)->str:
    s=str(v).strip()
    try:
        f=float(s)
        if f.is_integer():return str(int(f))
    except Exception:pass
    return s

def _is_missing_scalar(v:Any)->bool:
    if v is None:
        return True
    s=str(v).strip().lower()
    return s in {"", "nan", "none", "null"}


def parse_bool(v:Any)->bool:
    """Parse boolean-ish CSV cells, including pandas-style numeric values such as ``1.0``."""
    if _is_missing_scalar(v):
        return False
    s=str(v).strip().lower()
    if s in {"true","t","yes","y"}:
        return True
    if s in {"false","f","no","n"}:
        return False
    try:
        f=float(s)
        return math.isfinite(f) and f != 0.0
    except Exception:
        return False
def parse_literal(v:Any)->Any:
    if v is None:return None
    s=str(v).strip()
    if not s or s.lower() in {"nan","none","null"}:return None
    try:return ast.literal_eval(s)
    except Exception:pass
    try:return json.loads(s)
    except Exception:return None

def parse_bbox(v:Any)->Optional[list[float]]:
    x=parse_literal(v)
    if x is None:return None
    if isinstance(x,dict):
        for keys in (("x1","y1","x2","y2"),("left","top","right","bottom")):
            if all(k in x for k in keys):return [float(x[k]) for k in keys]
        if all(k in x for k in ("x","y","w","h")):return [float(x["x"]),float(x["y"]),float(x["x"])+float(x["w"]),float(x["y"])+float(x["h"])]
    if isinstance(x,(list,tuple)) and len(x)==4:
        # SoccerNet-v3 dataloader convention: (x_top, y_top, width, height).
        x0,y0,w,h=[float(z) for z in x]
        return [x0,y0,x0+w,y0+h]
    return None

_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

def _finite_xyz(values:Any)->Optional[list[float]]:
    try:
        xyz=[float(values[0]),float(values[1]),float(values[2])]
    except Exception:
        return None
    return xyz if all(math.isfinite(z) for z in xyz) else None

def parse_xyz(v:Any)->Optional[list[float]]:
    """Parse a raw SoccerNet-v3D ``ball_3D`` textual value without changing axes."""
    x=parse_literal(v)
    if isinstance(x,(list,tuple)) and len(x)>=3:
        return _finite_xyz(x)
    if isinstance(x,dict):
        for keys in (("x","y","z"),("X","Y","Z")):
            if all(k in x for k in keys):
                return _finite_xyz([x[k] for k in keys])
    if v is None:return None
    raw=str(v).strip()
    if not raw or raw.lower() in {"nan","none","null"}:return None
    nums=[float(m.group(0)) for m in _FLOAT_RE.finditer(raw)]
    if len(nums)>=3:
        return _finite_xyz(nums[:3])
    return None

def parse_ball_3d_canonical(v:Any)->tuple[Optional[list[float]],Optional[list[float]]]:
    """Return (canonical_stage6_xyz, raw_soccernet_xyz)."""
    raw=parse_xyz(v)
    if raw is None:return None,None
    canonical=soccernet_xyz_to_stage6(raw).astype(float).tolist()
    return canonical,[float(x) for x in raw]

@dataclass
class SNv3DRecord:
    row_index:int
    league:str
    season:str
    match:str
    main_action:str
    action:bool
    replay:str
    img_w:int
    img_h:int
    is_ball:bool
    ball_bbox:Optional[list[float]]
    # Canonical Stage-6 coordinates: X goal-to-goal, Y touchline-to-touchline,
    # Z up-positive. This is the only field evaluators should use.
    ball_3d:Optional[list[float]]
    # Original SoccerNet convention retained for provenance/debugging.
    ball_3d_soccernet:Optional[list[float]]
    optimized_d:Optional[float]
    set_name:str
    raw:Dict[str,str]
    @property
    def is_action_frame(self)->bool:
        # Official SoccerNet-v3 names action frames as <main_action>.png and
        # replay frames as <main_action>_<replay>.png. The public SNv3D.csv
        # may serialize the action flag numerically (e.g. 1.0/0.0), and older
        # exports can leave replay blank on action rows. Treat missing replay
        # as a defensive action-frame signal so we never generate "14_.png".
        return bool(self.action or _is_missing_scalar(self.replay))
    @property
    def basename(self)->str:
        if self.is_action_frame:
            return f"{_clean(self.main_action)}.png"
        return f"{_clean(self.main_action)}_{_clean(self.replay)}.png"
    @property
    def record_id(self)->str:return f"{self.league}/{self.season}/{self.match}/{self.basename}"


def optimized_bbox_from_record(record: SNv3DRecord) -> Optional[list[float]]:
    """Reconstruct the SoccerNet-v3D optimized square box.

    The public CSV exposes the original ``ball_bbox`` and the optimized image-space
    diameter ``optimized_d``.  The optimized box is therefore represented as a
    square centered on the original annotation center with side length
    ``optimized_d``.  No clipping is applied so the geometry remains in the same
    raw image coordinate system as the annotation.
    """
    if record.ball_bbox is None or record.optimized_d is None:
        return None
    d=float(record.optimized_d)
    if not math.isfinite(d) or d <= 0.0:
        return None
    x1,y1,x2,y2=[float(v) for v in record.ball_bbox]
    cx=0.5*(x1+x2)
    cy=0.5*(y1+y2)
    h=0.5*d
    return [cx-h,cy-h,cx+h,cy+h]


def gt_bbox_for_record(record: SNv3DRecord, mode: str = "optimized") -> Optional[list[float]]:
    """Return the requested 2D GT box without silently mixing conventions."""
    mode=str(mode).strip().lower()
    if not record.is_ball:
        return None
    if mode == "original":
        return None if record.ball_bbox is None else [float(v) for v in record.ball_bbox]
    if mode == "optimized":
        return optimized_bbox_from_record(record)
    raise ValueError("gt box mode must be 'original' or 'optimized'")


class SoccerNetV3DCSV:
    def __init__(self,csv_path:str|Path)->None:
        self.csv_path=Path(csv_path).expanduser().resolve()
        if not self.csv_path.is_file():raise FileNotFoundError(self.csv_path)
        self.records=self._read()
    def _read(self)->list[SNv3DRecord]:
        out=[]
        with self.csv_path.open("r",encoding="utf-8-sig",newline="") as f:
            reader=csv.DictReader(f); req={"league","season","match","main_action","action","replay","img_w","img_h","is_ball","ball_bbox","ball_3D","optimized_d","set"}; missing=req-set(reader.fieldnames or [])
            if missing:raise ValueError(f"SNv3D.csv missing columns: {sorted(missing)}")
            for i,row in enumerate(reader):
                def fnum(k):
                    try:
                        n=float(row.get(k,"nan")); return None if math.isnan(n) else n
                    except Exception:return None
                ball_3d,ball_3d_sn=parse_ball_3d_canonical(row["ball_3D"])
                out.append(SNv3DRecord(
                    row_index=i,
                    league=row["league"],season=row["season"],match=row["match"],main_action=row["main_action"],
                    action=parse_bool(row["action"]),replay=row["replay"],img_w=int(float(row["img_w"])),img_h=int(float(row["img_h"])),
                    is_ball=parse_bool(row["is_ball"]),ball_bbox=parse_bbox(row["ball_bbox"]),ball_3d=ball_3d,ball_3d_soccernet=ball_3d_sn,
                    optimized_d=fnum("optimized_d"),set_name=str(row["set"]).strip().lower(),raw=dict(row),
                ))
        return out
    def split(self,name:str)->list[SNv3DRecord]:return [r for r in self.records if r.set_name==name.strip().lower()]
    def summary(self)->Dict[str,Any]:
        sets={}
        for r in self.records:
            d=sets.setdefault(r.set_name,{
                "rows":0,"ball_rows":0,"ball_3d_rows":0,
                "ball_3d_raw_nonempty_rows":0,"ball_3d_parse_failures":0,
                "ball_3d_parse_failure_examples":[],
                "source_negative_z_rows":0,
                "canonical_nonnegative_z_rows":0,
                "action_frame_rows":0,
                "replay_frame_rows":0,
                "action_inferred_from_missing_replay_rows":0,
                "optimized_bbox_rows":0,
            })
            d["rows"]+=1
            d["ball_rows"]+=int(r.is_ball and r.ball_bbox is not None)
            d["ball_3d_rows"]+=int(r.ball_3d is not None)
            raw=str(r.raw.get("ball_3D","")).strip()
            raw_nonempty=bool(raw and raw.lower() not in {"nan","none","null"})
            d["ball_3d_raw_nonempty_rows"]+=int(raw_nonempty)
            if raw_nonempty and r.ball_3d is None:
                d["ball_3d_parse_failures"]+=1
                if len(d["ball_3d_parse_failure_examples"])<3:
                    d["ball_3d_parse_failure_examples"].append(raw[:200])
            if r.ball_3d_soccernet is not None:
                d["source_negative_z_rows"]+=int(float(r.ball_3d_soccernet[2])<0.0)
            if r.ball_3d is not None:
                d["canonical_nonnegative_z_rows"]+=int(float(r.ball_3d[2])>=0.0)
            d["action_frame_rows"]+=int(r.is_action_frame)
            d["replay_frame_rows"]+=int(not r.is_action_frame)
            d["action_inferred_from_missing_replay_rows"]+=int((not r.action) and _is_missing_scalar(r.replay))
            d["optimized_bbox_rows"]+=int(optimized_bbox_from_record(r) is not None)
        return {
            "csv":str(self.csv_path),
            "rows":len(self.records),
            "coordinate_transform":coordinate_transform_metadata(),
            "sets":sets,
        }

@dataclass(frozen=True)
class ResolvedSoccerNetImage:
    """A SoccerNet-v3 frame resolved either as a loose file or a ZIP member."""
    kind:str
    path:Path
    member:Optional[str]=None

    @property
    def reference(self)->str:
        if self.kind=="filesystem":
            return str(self.path.resolve())
        if self.kind=="frames-v3-zip" and self.member:
            return make_zip_image_uri(self.path,self.member)
        raise ValueError(f"Unsupported image source: {self.kind}")


def make_zip_image_uri(archive:str|Path,member:str)->str:
    archive_path=Path(archive).expanduser().resolve()
    member_clean=str(member).replace("\\","/").lstrip("/")
    return f"zip://{archive_path}!/{member_clean}"


def parse_zip_image_uri(reference:str)->Optional[tuple[Path,str]]:
    text=str(reference)
    if not text.startswith("zip://") or "!/" not in text:
        return None
    payload=text[len("zip://"):]
    archive_raw,member=payload.split("!/",1)
    if not archive_raw or not member:
        return None
    return Path(archive_raw).expanduser(),member


def read_image_reference(reference:str|Path)->Optional[np.ndarray]:
    """Read a normal image path or ``zip://...Frames-v3.zip!/member.png`` reference."""
    text=str(reference)
    parsed=parse_zip_image_uri(text)
    if parsed is None:
        image=cv2.imread(text,cv2.IMREAD_COLOR)
        return image
    archive,member=parsed
    if not archive.is_file():
        return None
    try:
        with zipfile.ZipFile(archive,"r") as zf:
            data=zf.read(member)
    except (KeyError,zipfile.BadZipFile,OSError):
        return None
    array=np.frombuffer(data,dtype=np.uint8)
    if array.size==0:
        return None
    return cv2.imdecode(array,cv2.IMREAD_COLOR)


class SoccerNetImageResolver:
    """Resolve SNv3D frames from the Hugging Face ``frames-v3`` layout.

    The primary layout is exactly
    ``<root>/<league>/<season>/<match>/Frames-v3.zip``. Loose/extracted images
    remain supported as a convenience. ZIP members are decoded in memory.
    """
    def __init__(self,image_root:str|Path)->None:
        self.root=Path(image_root).expanduser().resolve()
        self.cache:Dict[str,Optional[ResolvedSoccerNetImage]]={}
        self._zip_member_indexes:Dict[Path,Dict[str,str]]={}

    def expected_archive_path(self,r:SNv3DRecord)->Path:
        """Exact local path mirroring the Hugging Face ``frames-v3`` revision."""
        return self.root/r.league/r.season/r.match/"Frames-v3.zip"

    def _bases(self,r:SNv3DRecord)->tuple[Path,...]:
        return (
            self.root/r.league/r.season/r.match,
            self.root/r.match,
            self.root,
        )

    def _zip_index(self,archive:Path)->Dict[str,str]:
        archive=archive.resolve()
        if archive in self._zip_member_indexes:
            return self._zip_member_indexes[archive]
        index:Dict[str,str]={}
        try:
            with zipfile.ZipFile(archive,"r") as zf:
                for member in zf.namelist():
                    normalized=member.replace("\\","/")
                    if normalized.endswith("/"):
                        continue
                    basename=normalized.rsplit("/",1)[-1]
                    current=index.get(basename)
                    if current is None or len(normalized)<len(current):
                        index[basename]=member
        except (zipfile.BadZipFile,OSError):
            index={}
        self._zip_member_indexes[archive]=index
        return index

    def _archive_candidates(self,base:Path)->list[Path]:
        archives:list[Path]=[]
        direct=base/"Frames-v3.zip"
        if direct.is_file():
            archives.append(direct)
        # Never recursively scan the entire dataset root for every record.
        # Recursive fallback is limited to a resolved match directory.
        if base!=self.root and base.is_dir() and not archives:
            archives.extend(sorted(p for p in base.rglob("Frames-v3.zip") if p.is_file()))
        return archives

    def resolve_source(self,r:SNv3DRecord)->Optional[ResolvedSoccerNetImage]:
        if r.record_id in self.cache:
            return self.cache[r.record_id]
        source:Optional[ResolvedSoccerNetImage]=None
        for base in self._bases(r):
            if not base.exists():
                continue
            direct=base/r.basename
            if direct.is_file():
                source=ResolvedSoccerNetImage("filesystem",direct.resolve())
                break
            # Preserve support for already-extracted archives, including a nested
            # Frames-v3/ directory if the archive tool created one.
            if base!=self.root:
                loose=sorted(p for p in base.rglob(r.basename) if p.is_file() and p.suffix.lower() in {".png",".jpg",".jpeg"})
                if loose:
                    source=ResolvedSoccerNetImage("filesystem",loose[0].resolve())
                    break
            for archive in self._archive_candidates(base):
                member=self._zip_index(archive).get(r.basename)
                if member is not None:
                    source=ResolvedSoccerNetImage("frames-v3-zip",archive.resolve(),member)
                    break
            if source is not None:
                break
        self.cache[r.record_id]=source
        return source

    def resolve(self,r:SNv3DRecord)->Optional[Path]:
        """Backward-compatible loose-file resolver.

        Returns a Path only for an extracted image. ZIP-backed records are available
        through ``resolve_source`` / ``read`` instead.
        """
        source=self.resolve_source(r)
        return source.path if source is not None and source.kind=="filesystem" else None

    def read(self,r:SNv3DRecord)->tuple[Optional[np.ndarray],Optional[ResolvedSoccerNetImage]]:
        source=self.resolve_source(r)
        if source is None:
            return None,None
        image=read_image_reference(source.reference)
        return image,source
