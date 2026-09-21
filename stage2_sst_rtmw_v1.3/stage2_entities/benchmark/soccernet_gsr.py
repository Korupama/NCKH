from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import re

import cv2

HUMAN_ROLES = {"player", "goalkeeper", "referee", "other"}
CANDIDATE_ROLES = {"player", "goalkeeper"}


def _role(value: Any) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower().replace(" ", "_")
    mapping = {
        "player": "player",
        "goalkeeper": "goalkeeper",
        "goal_keeper": "goalkeeper",
        "keeper": "goalkeeper",
        "referee": "referee",
        "main_referee": "referee",
        "side_referee": "referee",
        "assistant_referee": "referee",
        "other": "other",
        "staff": "other",
    }
    return mapping.get(v)


def _bbox_xyxy(ann: Mapping[str, Any]) -> Optional[List[float]]:
    b = ann.get("bbox_image")
    if isinstance(b, Mapping):
        if all(k in b for k in ("x", "y", "w", "h")):
            x, y, w, h = map(float, (b["x"], b["y"], b["w"], b["h"]))
            return [x, y, x + w, y + h]
        if all(k in b for k in ("x1", "y1", "x2", "y2")):
            return [float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])]
    b = ann.get("bbox")
    if isinstance(b, (list, tuple)) and len(b) >= 4:
        # SoccerNet/COCO-style boxes are left, top, width, height.
        x, y, w, h = map(float, b[:4])
        return [x, y, x + w, y + h]
    return None


def _version_tuple(v: Any) -> Tuple[int, ...]:
    if v is None:
        return tuple()
    nums = re.findall(r"\d+", str(v))
    return tuple(int(x) for x in nums[:3])


@dataclass(frozen=True)
class GTObject:
    frame_index: int
    image_id: str
    track_id: str
    bbox_xyxy: List[float]
    role: str
    attributes: Dict[str, Any]

    @property
    def candidate(self) -> bool:
        return self.role in CANDIDATE_ROLES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "image_id": self.image_id,
            "track_id": self.track_id,
            "bbox_xyxy": list(self.bbox_xyxy),
            "role": self.role,
            "candidate_for_stage3": self.candidate,
            "attributes": dict(self.attributes),
        }


@dataclass
class SoccerNetSequence:
    sequence_id: str
    sequence_dir: Path
    labels_path: Path
    version: str
    fps: float
    width: int
    height: int
    images: List[Dict[str, Any]]
    image_id_to_frame: Dict[str, int]
    gt_by_frame: Dict[int, List[GTObject]]
    video_path: Optional[Path]

    @property
    def num_frames(self) -> int:
        return len(self.images)

    @property
    def labelled_frames(self) -> List[int]:
        return [i for i, rec in enumerate(self.images) if bool(rec.get("is_labeled", True))]

    def evaluation_frames(self, start: int, end: int) -> List[int]:
        return [i for i in range(max(0, int(start)), min(self.num_frames - 1, int(end)) + 1) if bool(self.images[i].get("is_labeled", True))]

    def gt(self, frame_index: int, roles: Optional[set[str]] = None) -> List[GTObject]:
        out = list(self.gt_by_frame.get(int(frame_index), []))
        if roles is not None:
            out = [x for x in out if x.role in roles]
        return out

    def frame_record(self, frame_index: int) -> Dict[str, Any]:
        return self.images[int(frame_index)]

    def _candidate_image_paths(self, frame_index: int) -> List[Path]:
        rec = self.frame_record(frame_index)
        raw = rec.get("file_name") or rec.get("path") or rec.get("file_path")
        result: List[Path] = []
        if raw:
            p = Path(str(raw))
            if p.is_absolute():
                result.append(p)
            else:
                result.extend([
                    self.sequence_dir / p,
                    self.sequence_dir.parent / p,
                    self.sequence_dir.parent.parent / p,
                ])
        # Common extracted-frame layouts.
        for ext in ("jpg", "jpeg", "png"):
            result.extend([
                self.sequence_dir / f"{frame_index:06d}.{ext}",
                self.sequence_dir / f"{frame_index+1:06d}.{ext}",
                self.sequence_dir / "img1" / f"{frame_index+1:06d}.{ext}",
                self.sequence_dir / "frames" / f"{frame_index:06d}.{ext}",
            ])
        return result

    def resolve_image_path(self, frame_index: int) -> Optional[Path]:
        for p in self._candidate_image_paths(frame_index):
            if p.is_file():
                return p.resolve()
        return None

    def materialize_frames(self, frame_indices: Sequence[int], output_dir: str | Path) -> Dict[str, Any]:
        """Return Stage-2 frame_map for requested local sequence frame indices.

        Direct image files are reused. Missing images are decoded from a discovered sequence video
        and written as lossless PNG. Every output retains the sequence-local frame index exactly.
        """
        indices = sorted(set(int(x) for x in frame_indices))
        if not indices:
            raise ValueError("No frame indices requested")
        if indices[0] < 0 or indices[-1] >= self.num_frames:
            raise IndexError(f"Requested frame outside 0..{self.num_frames-1}: {indices[0]}..{indices[-1]}")
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        paths: Dict[int, Path] = {}
        missing: List[int] = []
        for fi in indices:
            p = self.resolve_image_path(fi)
            if p is not None:
                paths[fi] = p
            else:
                cached = out / f"frame_{fi:09d}.png"
                if cached.is_file():
                    paths[fi] = cached
                else:
                    missing.append(fi)

        if missing:
            if self.video_path is None or not self.video_path.is_file():
                raise FileNotFoundError(
                    f"{self.sequence_id}: {len(missing)} benchmark frames have no image file and no video was found. "
                    "Expected image records that resolve on disk or an mp4 in the sequence directory."
                )
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video {self.video_path}")
            for fi in missing:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    raise RuntimeError(f"{self.sequence_id}: failed decoding video frame {fi}")
                h, w = frame.shape[:2]
                if self.width > 0 and self.height > 0 and (w, h) != (self.width, self.height):
                    cap.release()
                    raise RuntimeError(
                        f"{self.sequence_id}: decoded {w}x{h}, labels report {self.width}x{self.height}"
                    )
                target = out / f"frame_{fi:09d}.png"
                cv2.imwrite(str(target), frame)
                paths[fi] = target
            cap.release()

        return {
            "schema_version": "stage2-benchmark-frame-map-1.0",
            "sequence_id": self.sequence_id,
            "coordinate_space": "RAW_DISTORTED_PIXEL",
            "frames": [
                {
                    "local_frame_index": pos,
                    "global_frame_index": fi,
                    "path": str(paths[fi].resolve()),
                }
                for pos, fi in enumerate(indices)
            ],
        }


class SoccerNetGSRDataset:
    """Lightweight reader for the released SoccerNet-GSR `Labels-GameState.json` files.

    It deliberately reads only image-space bbox, track id and role needed by Stage 2.
    Team, jersey and pitch annotations are ignored.
    """

    def __init__(self, root: str | Path, split: str = "valid", *, require_version_13: bool = True) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = str(split)
        self.split_dir = self.root / self.split
        self.require_version_13 = bool(require_version_13)
        if not self.split_dir.is_dir():
            raise FileNotFoundError(
                f"SoccerNet-GSR split not found: {self.split_dir}. Expected <root>/{self.split}/..."
            )

    def discover(self) -> List[str]:
        files = sorted(self.split_dir.rglob("Labels-GameState.json"))
        return [p.parent.name for p in files]

    def labels_paths(self) -> Dict[str, Path]:
        result: Dict[str, Path] = {}
        for p in sorted(self.split_dir.rglob("Labels-GameState.json")):
            sid = p.parent.name
            if sid in result:
                raise RuntimeError(f"Duplicate sequence id {sid}: {result[sid]} and {p}")
            result[sid] = p
        return result

    def load(self, sequence_id: str) -> SoccerNetSequence:
        paths = self.labels_paths()
        if sequence_id not in paths:
            raise KeyError(f"Unknown sequence {sequence_id!r}; available examples={list(paths)[:10]}")
        labels_path = paths[sequence_id]
        data = json.loads(labels_path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping) or "images" not in data or "annotations" not in data:
            raise ValueError(f"{labels_path} is not SoccerNet/COCO-style GameState JSON")

        info = dict(data.get("info") or {})
        version = str(info.get("version", "unknown"))
        parsed_version = _version_tuple(version)
        if self.require_version_13 and (not parsed_version or parsed_version < (1, 3)):
            raise ValueError(f"{sequence_id}: SoccerNet-GSR v1.3+ required, found {version!r}")

        images_raw = list(data.get("images") or [])
        if not images_raw:
            raise ValueError(f"{sequence_id}: labels contain no images")

        # The released files store images chronologically; explicit frame fields win when present.
        explicit = []
        explicit_ok = True
        for idx, rec in enumerate(images_raw):
            value = rec.get("frame")
            if value is None:
                value = rec.get("frame_id")
            if value is None:
                explicit_ok = False
                break
            try:
                explicit.append((int(value), idx, dict(rec)))
            except Exception:
                explicit_ok = False
                break
        if explicit_ok:
            explicit.sort(key=lambda x: (x[0], x[1]))
            images = [x[2] for x in explicit]
        else:
            images = [dict(x) for x in images_raw]

        image_id_to_frame: Dict[str, int] = {}
        for fi, rec in enumerate(images):
            iid = rec.get("image_id", rec.get("id"))
            if iid is None:
                raise ValueError(f"{sequence_id}: image record {fi} has no image_id/id")
            image_id_to_frame[str(iid)] = fi

        first = images[0]
        width = int(first.get("width", info.get("width", 0)) or 0)
        height = int(first.get("height", info.get("height", 0)) or 0)
        fps = float(info.get("frame_rate", info.get("fps", 25.0)) or 25.0)

        gt_by_frame: Dict[int, List[GTObject]] = {}
        skipped_nonhuman = 0
        skipped_nobox = 0
        for ann in data.get("annotations") or []:
            if ann.get("supercategory") not in (None, "object"):
                continue
            attrs = dict(ann.get("attributes") or {})
            role = _role(attrs.get("role", ann.get("role")))
            if role is None or role not in HUMAN_ROLES:
                skipped_nonhuman += 1
                continue
            iid = str(ann.get("image_id"))
            if iid not in image_id_to_frame:
                continue
            box = _bbox_xyxy(ann)
            if box is None:
                skipped_nobox += 1
                continue
            tid = ann.get("track_id", attrs.get("track_id", ann.get("id")))
            if tid is None:
                continue
            fi = image_id_to_frame[iid]
            gt_by_frame.setdefault(fi, []).append(GTObject(
                frame_index=fi,
                image_id=iid,
                track_id=str(tid),
                bbox_xyxy=box,
                role=role,
                attributes=attrs,
            ))

        seq_dir = labels_path.parent
        video_candidates = []
        for name in ("video.mp4", "video.720p.mp4", f"{sequence_id}.mp4"):
            p = seq_dir / name
            if p.is_file():
                video_candidates.append(p)
        if not video_candidates:
            video_candidates = sorted(seq_dir.glob("*.mp4"))
        video_path = video_candidates[0].resolve() if video_candidates else None
        if (width <= 0 or height <= 0 or fps <= 0) and video_path is not None:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                if width <= 0: width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                if height <= 0: height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if fps <= 0: fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
        if width <= 0 or height <= 0:
            sample_path = None
            for fi in range(len(images)):
                raw = images[fi].get("file_name") or images[fi].get("path") or images[fi].get("file_path")
                if raw:
                    for base in (seq_dir, seq_dir.parent, seq_dir.parent.parent):
                        q = Path(str(raw)) if Path(str(raw)).is_absolute() else base / str(raw)
                        if q.is_file(): sample_path = q; break
                if sample_path is not None: break
            if sample_path is not None:
                im = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
                if im is not None: height, width = im.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError(f"{sequence_id}: could not determine image width/height from labels, images, or video")

        return SoccerNetSequence(
            sequence_id=sequence_id,
            sequence_dir=seq_dir,
            labels_path=labels_path,
            version=version,
            fps=fps,
            width=width,
            height=height,
            images=images,
            image_id_to_frame=image_id_to_frame,
            gt_by_frame=gt_by_frame,
            video_path=video_path,
        )

    def inspect(self, max_sequences: int = 5) -> Dict[str, Any]:
        ids = self.discover()
        items = []
        for sid in ids[:max_sequences]:
            seq = self.load(sid)
            role_counts: Dict[str, int] = {r: 0 for r in sorted(HUMAN_ROLES)}
            for objs in seq.gt_by_frame.values():
                for obj in objs:
                    role_counts[obj.role] += 1
            items.append({
                "sequence_id": sid,
                "version": seq.version,
                "frames": seq.num_frames,
                "fps": seq.fps,
                "resolution": [seq.width, seq.height],
                "labelled_frames": len(seq.gt_by_frame),
                "human_annotations": sum(role_counts.values()),
                "role_counts": role_counts,
                "video_path": str(seq.video_path) if seq.video_path else None,
                "sample_image_resolves": bool(seq.resolve_image_path(seq.labelled_frames[0])) if seq.labelled_frames else False,
            })
        return {
            "root": str(self.root),
            "split": self.split,
            "num_sequences": len(ids),
            "sequences": items,
        }
