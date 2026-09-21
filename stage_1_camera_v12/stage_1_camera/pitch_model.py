from __future__ import annotations

from typing import List, Tuple
import numpy as np

from .contracts import PitchSpec

Polyline3D = Tuple[np.ndarray, str]
Segment3D = Tuple[np.ndarray, np.ndarray, str]


def _sample_segment(a, b, spacing_m: float = 0.5) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    length = float(np.linalg.norm(b - a))
    n = max(2, int(np.ceil(length / max(spacing_m, 1e-6))) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return a[None, :] * (1.0 - t) + b[None, :] * t


def pitch_segments(p: PitchSpec) -> List[Segment3D]:
    """Straight canonical pitch markings/goal-frame segments.

    This legacy-compatible helper is kept for simple 3D plotting. For image
    reprojection use :func:`pitch_polylines`, which densely samples straight
    markings so OpenCV lens distortion is respected after projection.
    """
    L, W = p.length_m, p.width_m
    x0, x1 = -L / 2.0, L / 2.0
    y0, y1 = -W / 2.0, W / 2.0
    seg: List[Segment3D] = []

    def add(a, b, name):
        seg.append((np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), name))

    add((x0, y0, 0), (x1, y0, 0), "touchline")
    add((x0, y1, 0), (x1, y1, 0), "touchline")
    add((x0, y0, 0), (x0, y1, 0), "goal_line")
    add((x1, y0, 0), (x1, y1, 0), "goal_line")
    add((0, y0, 0), (0, y1, 0), "halfway")

    for sign in (-1, 1):
        xg = sign * L / 2.0
        for depth, half_width, name in [
            (p.goal_area_depth_m, 18.32 / 2.0, "goal_area"),
            (p.penalty_area_depth_m, 40.32 / 2.0, "penalty_area"),
        ]:
            xi = xg - sign * depth
            add((xg, -half_width, 0), (xi, -half_width, 0), name)
            add((xi, -half_width, 0), (xi, half_width, 0), name)
            add((xi, half_width, 0), (xg, half_width, 0), name)

        # Goal frame. Height is positive in project canonical world.
        for yy in (-p.goal_width_m / 2.0, p.goal_width_m / 2.0):
            add((xg, yy, 0), (xg, yy, p.goal_height_m), "goal_post")
        add(
            (xg, -p.goal_width_m / 2.0, p.goal_height_m),
            (xg, p.goal_width_m / 2.0, p.goal_height_m),
            "crossbar",
        )

    return seg


def pitch_circles(p: PitchSpec) -> List[Polyline3D]:
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    c = np.column_stack(
        [
            p.centre_circle_radius_m * np.cos(theta),
            p.centre_circle_radius_m * np.sin(theta),
            np.zeros_like(theta),
        ]
    )
    return [(c, "centre_circle")]


def pitch_polylines(p: PitchSpec, spacing_m: float = 0.5) -> List[Polyline3D]:
    """Dense metric 3D model of the pitch used for image reprojection QA.

    Includes outer boundary, halfway line, goal/penalty areas, centre circle,
    penalty arcs/marks and the goal frames. Straight markings are sampled
    densely because radial/tangential distortion can make their projections
    curved in raw broadcast pixel space.
    """
    polylines: List[Polyline3D] = []

    for a, b, name in pitch_segments(p):
        polylines.append((_sample_segment(a, b, spacing_m), name))

    polylines.extend(pitch_circles(p))

    L = p.length_m
    x_left_mark = -L / 2.0 + p.penalty_mark_m
    x_right_mark = L / 2.0 - p.penalty_mark_m
    r = p.centre_circle_radius_m
    # Penalty area main line is 16.5 m from goal line, penalty mark 11 m.
    dx = p.penalty_area_depth_m - p.penalty_mark_m
    phi = float(np.arccos(np.clip(dx / r, -1.0, 1.0)))

    # Left arc is the circle portion outside the penalty area, facing +X.
    th_left = np.linspace(-phi, phi, 121)
    left_arc = np.column_stack(
        [
            x_left_mark + r * np.cos(th_left),
            r * np.sin(th_left),
            np.zeros_like(th_left),
        ]
    )
    polylines.append((left_arc, "penalty_arc"))

    # Right arc faces -X.
    th_right = np.linspace(np.pi - phi, np.pi + phi, 121)
    right_arc = np.column_stack(
        [
            x_right_mark + r * np.cos(th_right),
            r * np.sin(th_right),
            np.zeros_like(th_right),
        ]
    )
    polylines.append((right_arc, "penalty_arc"))

    # Tiny circles make penalty/centre marks visible without inventing a large
    # geometric primitive that could bias visual alignment.
    dot_r = 0.12
    theta = np.linspace(0.0, 2.0 * np.pi, 33)
    for x, name in [(x_left_mark, "penalty_mark"), (0.0, "centre_mark"), (x_right_mark, "penalty_mark")]:
        dot = np.column_stack(
            [
                x + dot_r * np.cos(theta),
                dot_r * np.sin(theta),
                np.zeros_like(theta),
            ]
        )
        polylines.append((dot, name))

    return polylines
