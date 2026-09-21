from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np

from .stage3_adapter import PoseObservation2D
from .wholebody import LEFT_FOOT, RIGHT_FOOT


@dataclass
class FootProxy2D:
    side: str
    ankle_xy: Optional[np.ndarray]
    distal_xy: Optional[np.ndarray]
    heel_xy: Optional[np.ndarray]
    contact_xy: Optional[np.ndarray]
    available_landmarks: int
    quality_weight: float
    bottom_gap_ratio: Optional[float]
    normalized_xy: Optional[np.ndarray]


@dataclass
class ContactFrame:
    frame_index: int
    state: str
    left_likelihood: float
    right_likelihood: float
    left: FootProxy2D
    right: FootProxy2D
    diagnostics: Dict[str, float]


def _mean_valid(points: Sequence[np.ndarray], weights: Sequence[float]) -> Optional[np.ndarray]:
    vals = []
    ws = []
    for point, weight in zip(points, weights):
        point = np.asarray(point, dtype=np.float64)
        if np.isfinite(point).all() and float(weight) > 0:
            vals.append(point)
            ws.append(float(weight))
    if not vals:
        return None
    arr = np.stack(vals)
    w = np.asarray(ws, dtype=np.float64)
    return np.sum(arr * w[:, None], axis=0) / max(np.sum(w), 1e-12)


def foot_proxy(obs: PoseObservation2D, side: str) -> FootProxy2D:
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    if side == "left":
        ankle, big, small, heel = 15, 17, 18, 19
    else:
        ankle, big, small, heel = 16, 20, 21, 22

    uv = obs.uv23
    w = obs.state_weights23
    ankle_xy = uv[ankle].copy() if np.isfinite(uv[ankle]).all() and w[ankle] > 0 else None
    heel_xy = uv[heel].copy() if np.isfinite(uv[heel]).all() and w[heel] > 0 else None
    distal_xy = _mean_valid([uv[big], uv[small]], [w[big], w[small]])

    surface_candidates = []
    surface_weights = []
    for idx in (big, small, heel):
        if np.isfinite(uv[idx]).all() and w[idx] > 0:
            surface_candidates.append(uv[idx])
            surface_weights.append(w[idx])
    contact_xy = None
    if surface_candidates:
        arr = np.stack(surface_candidates)
        ww = np.asarray(surface_weights)
        # Weighted horizontal centre plus the lower observed surface y. This is
        # an approximate broadcast-foot contact proxy, not a toe claim.
        x = float(np.sum(arr[:, 0] * ww) / max(np.sum(ww), 1e-12))
        y = float(np.max(arr[:, 1]))
        contact_xy = np.asarray([x, y], dtype=np.float64)
    elif ankle_xy is not None:
        contact_xy = ankle_xy.copy()

    valid_indices = [idx for idx in (ankle, big, small, heel) if np.isfinite(uv[idx]).all() and w[idx] > 0]
    quality_weight = float(np.mean(w[valid_indices])) if valid_indices else 0.0
    bbox = obs.bbox_xyxy
    bottom_gap_ratio = None
    normalized_xy = None
    if contact_xy is not None and np.isfinite(bbox).all():
        bw = max(float(bbox[2] - bbox[0]), 1.0)
        bh = max(float(bbox[3] - bbox[1]), 1.0)
        bottom_gap_ratio = float((bbox[3] - contact_xy[1]) / bh)
        normalized_xy = np.asarray([
            (contact_xy[0] - 0.5 * (bbox[0] + bbox[2])) / bw,
            (contact_xy[1] - 0.5 * (bbox[1] + bbox[3])) / bh,
        ], dtype=np.float64)

    return FootProxy2D(
        side=side,
        ankle_xy=ankle_xy,
        distal_xy=distal_xy,
        heel_xy=heel_xy,
        contact_xy=contact_xy,
        available_landmarks=len(valid_indices),
        quality_weight=quality_weight,
        bottom_gap_ratio=bottom_gap_ratio,
        normalized_xy=normalized_xy,
    )


def infer_contact_sequence(
    observations: Sequence[PoseObservation2D],
    *,
    likelihood_threshold: float = 0.58,
    both_y_tolerance_ratio: float = 0.035,
) -> Dict[int, ContactFrame]:
    obs = sorted(observations, key=lambda x: x.frame_index)
    proxies = {
        item.frame_index: (foot_proxy(item, "left"), foot_proxy(item, "right"), item)
        for item in obs
    }

    def stability(frame_pos: int, side_index: int) -> float:
        cur = proxies[obs[frame_pos].frame_index][side_index]
        if cur.normalized_xy is None:
            return 0.0
        neighbours = []
        for off in (-1, 1):
            p = frame_pos + off
            if 0 <= p < len(obs):
                other = proxies[obs[p].frame_index][side_index]
                if other.normalized_xy is not None:
                    neighbours.append(np.linalg.norm(cur.normalized_xy - other.normalized_xy))
        if not neighbours:
            return 0.5
        speed = float(np.median(neighbours))
        return float(np.exp(-((speed / 0.12) ** 2)))

    out: Dict[int, ContactFrame] = {}
    for pos, item in enumerate(obs):
        left, right, raw_obs = proxies[item.frame_index]

        def base_likelihood(proxy: FootProxy2D, stable: float) -> float:
            if proxy.contact_xy is None or proxy.bottom_gap_ratio is None:
                return 0.0
            gap = abs(float(proxy.bottom_gap_ratio))
            bottom_score = float(np.exp(-((gap / 0.10) ** 2)))
            completeness = min(1.0, proxy.available_landmarks / 3.0)
            return float(np.clip(
                0.48 * bottom_score + 0.25 * stable + 0.27 * completeness * proxy.quality_weight,
                0.0, 1.0,
            ))

        l_stab = stability(pos, 0)
        r_stab = stability(pos, 1)
        l = base_likelihood(left, l_stab)
        r = base_likelihood(right, r_stab)

        # The lower foot in bbox-normalized image coordinates gets weak support,
        # but close feet can both remain plausible contacts.
        y_delta_ratio = 0.0
        if left.contact_xy is not None and right.contact_xy is not None and np.isfinite(raw_obs.bbox_xyxy).all():
            bh = max(float(raw_obs.bbox_xyxy[3] - raw_obs.bbox_xyxy[1]), 1.0)
            y_delta_ratio = float((left.contact_xy[1] - right.contact_xy[1]) / bh)
            if abs(y_delta_ratio) > both_y_tolerance_ratio:
                if y_delta_ratio > 0:
                    l = min(1.0, l + 0.10)
                    r = max(0.0, r - 0.08)
                else:
                    r = min(1.0, r + 0.10)
                    l = max(0.0, l - 0.08)

        if l >= likelihood_threshold and r >= likelihood_threshold and abs(y_delta_ratio) <= both_y_tolerance_ratio:
            state = "BOTH_CONTACT"
        elif l >= likelihood_threshold and l >= r + 0.05:
            state = "LEFT_CONTACT"
        elif r >= likelihood_threshold and r >= l + 0.05:
            state = "RIGHT_CONTACT"
        elif max(l, r) >= likelihood_threshold:
            state = "CONTACT_UNCERTAIN"
        else:
            state = "NO_CONFIDENT_CONTACT"

        out[item.frame_index] = ContactFrame(
            frame_index=item.frame_index,
            state=state,
            left_likelihood=l,
            right_likelihood=r,
            left=left,
            right=right,
            diagnostics={
                "left_stability": l_stab,
                "right_stability": r_stab,
                "foot_y_delta_ratio_left_minus_right": y_delta_ratio,
            },
        )
    return out


def contact_to_dict(contact: ContactFrame) -> Dict[str, object]:
    def proxy_dict(p: FootProxy2D):
        return {
            "side": p.side,
            "ankle_xy": None if p.ankle_xy is None else p.ankle_xy.tolist(),
            "distal_xy": None if p.distal_xy is None else p.distal_xy.tolist(),
            "heel_xy": None if p.heel_xy is None else p.heel_xy.tolist(),
            "contact_xy": None if p.contact_xy is None else p.contact_xy.tolist(),
            "available_landmarks": p.available_landmarks,
            "quality_weight": p.quality_weight,
            "bottom_gap_ratio": p.bottom_gap_ratio,
        }
    return {
        "frame_index": contact.frame_index,
        "state": contact.state,
        "left_likelihood": contact.left_likelihood,
        "right_likelihood": contact.right_likelihood,
        "left": proxy_dict(contact.left),
        "right": proxy_dict(contact.right),
        "diagnostics": contact.diagnostics,
    }
