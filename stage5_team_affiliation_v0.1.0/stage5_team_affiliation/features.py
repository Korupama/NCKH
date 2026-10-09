from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from .config import Stage5Config
from .regions import rasterize_polygon


def _hist(channel: np.ndarray, mask: np.ndarray, bins: int, value_range: Tuple[int, int]) -> np.ndarray:
    h = cv2.calcHist([channel], [0], mask, [bins], list(value_range)).reshape(-1).astype(np.float32)
    s = float(h.sum())
    if s > 0:
        h /= s
    return h


def extract_color_feature(frame_bgr: np.ndarray, polygon: np.ndarray, config: Stage5Config) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    mask = rasterize_polygon(frame_bgr.shape, polygon, config.torso_erode_fraction)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)

    h, s, v = cv2.split(hsv)
    green = (
        (h >= config.green_hue_min)
        & (h <= config.green_hue_max)
        & (s >= config.green_min_saturation)
    )
    quality = (
        (s >= config.min_pixel_saturation)
        & (v >= config.min_pixel_value)
        & (v <= config.max_pixel_value)
        & (~green)
    )
    if config.include_neutral_kit_pixels:
        # Dark fabric has unstable hue (often green from pitch colour spill).
        # Hue/saturation alone must not remove black or white kits. Restrict
        # this recovery to the existing eroded body-region mask below.
        neutral = ((s < config.min_pixel_saturation)
                   | ((v <= config.dark_kit_max_value) & green)
                   | (v < config.min_pixel_value))
        quality |= neutral
        hsv = hsv.copy()
        hsv[neutral, 0] = 0
        hsv[neutral, 1] = 0
        h, s, v = cv2.split(hsv)
    use = ((mask > 0) & quality).astype(np.uint8) * 255
    count = int(np.count_nonzero(use))
    region_count = int(np.count_nonzero(mask))
    if count < 12:
        return None, {"usable_pixels": count, "region_pixels": region_count, "status": "TOO_FEW_PIXELS"}

    L, a, b = cv2.split(lab)
    feats = [
        _hist(h, use, 18, (0, 180)),
        _hist(s, use, 8, (0, 256)),
        _hist(a, use, 8, (0, 256)),
        _hist(b, use, 8, (0, 256)),
    ]
    pix_hsv = hsv[use > 0]
    pix_lab = lab[use > 0]
    robust = np.concatenate([
        np.median(pix_hsv, axis=0) / np.asarray([180.0, 255.0, 255.0]),
        np.median(pix_lab, axis=0) / 255.0,
    ]).astype(np.float32)
    feature = np.concatenate(feats + [robust]).astype(np.float32)
    n = float(np.linalg.norm(feature))
    if n > 0:
        feature /= n
    return feature, {
        "usable_pixels": count,
        "region_pixels": region_count,
        "usable_fraction": float(count / max(1, region_count)),
        "status": "VALID",
        "neutral_kit_pixels_enabled": config.include_neutral_kit_pixels,
        "median_hsv": [float(x) for x in np.median(pix_hsv, axis=0)],
        "median_lab": [float(x) for x in np.median(pix_lab, axis=0)],
    }


def aggregate_features(features: list[np.ndarray]) -> Optional[np.ndarray]:
    if not features:
        return None
    x = np.median(np.stack(features, axis=0), axis=0).astype(np.float32)
    n = float(np.linalg.norm(x))
    if n > 0:
        x /= n
    return x

def fuse_region_features(
    torso: Optional[np.ndarray],
    lower: Optional[np.ndarray],
    *,
    torso_weight: float = 0.75,
    lower_weight: float = 0.25,
) -> Optional[np.ndarray]:
    """Fuse independently sampled regions while preserving cosine geometry.

    Missing regions are renormalized rather than replaced by zeros. This makes the
    method robust to occluded legs and keeps the legacy torso-only behavior when no
    lower-body evidence is available.
    """
    parts = []
    if torso is not None and torso_weight > 0:
        parts.append((np.asarray(torso, dtype=np.float32), float(torso_weight)))
    if lower is not None and lower_weight > 0:
        parts.append((np.asarray(lower, dtype=np.float32), float(lower_weight)))
    if not parts:
        return None
    dim = parts[0][0].shape
    if any(x.shape != dim for x, _ in parts):
        raise ValueError("torso and lower features must have identical dimensions")
    total = sum(w for _, w in parts)
    fused = sum((w / total) * x for x, w in parts)
    norm = float(np.linalg.norm(fused))
    return (fused / norm).astype(np.float32) if norm > 0 else None
