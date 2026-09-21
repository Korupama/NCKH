from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import math
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .camera import CameraStateLite
from .foot_contact import ContactFrame
from .initializers.cache import InitializerObservation
from .schemas import Stage4Config
from .wholebody import BONE_PRIORS, NOMINAL_Z_FRACTION, POSE23_NAMES


@dataclass
class FrameInput:
    frame_index: int
    camera: CameraStateLite
    uv23: np.ndarray
    state_weights23: np.ndarray
    bbox_xyxy: np.ndarray
    contact: ContactFrame
    initializer: Optional[InitializerObservation] = None
    uv_source23: Optional[Tuple[str, ...]] = None


@dataclass
class OptimizationResult:
    frame_indices: np.ndarray
    xyz_world_m: np.ndarray
    available_mask: np.ndarray
    lambdas_m: np.ndarray
    estimated_height_m: float
    depth_beta_m: float
    initializer_used: bool
    success: bool
    status: str
    cost: float
    nfev: int
    njev: Optional[int]
    elapsed_s: float
    message: str
    median_reprojection_error_px: float
    p90_reprojection_error_px: float
    median_bone_abs_error_m: float
    ground_residual_median_m: Optional[float]
    negative_z_joint_fraction: float
    diagnostics: Dict[str, object]


class RayBundleProblem:
    """Calibrated-camera hard-ray metric 3D optimization.

    Each observed 2D joint defines a world ray and the only per-joint unknown is
    its positive distance lambda along that ray. This is equivalent to enforcing
    a hard reprojection constraint and avoids moving Stage-3 image evidence to
    satisfy a learned 3D prior.
    """

    def __init__(
        self,
        frames: Sequence[FrameInput],
        config: Stage4Config,
        *,
        fixed_height_m: Optional[float] = None,
        fixed_depth_beta_m: Optional[float] = None,
        initial_xyz_world_m: Optional[np.ndarray] = None,
    ) -> None:
        if not frames:
            raise ValueError("RayBundleProblem requires at least one frame")
        self.frames = sorted(frames, key=lambda x: x.frame_index)
        self.config = config
        self.fixed_height_m = fixed_height_m
        self.fixed_depth_beta_m = fixed_depth_beta_m
        self.T = len(self.frames)
        self.J = 23
        self.origins = np.full((self.T, self.J, 3), np.nan, dtype=np.float64)
        self.dirs = np.full((self.T, self.J, 3), np.nan, dtype=np.float64)
        self.valid = np.zeros((self.T, self.J), dtype=bool)
        self.var_keys: List[Tuple[int, int]] = []
        self.var_index: Dict[Tuple[int, int], int] = {}

        for ti, frame in enumerate(self.frames):
            uv = np.asarray(frame.uv23, dtype=np.float64)
            weights = np.asarray(frame.state_weights23, dtype=np.float64)
            if uv.shape != (23, 2) or weights.shape != (23,):
                raise ValueError("FrameInput uv23/state_weights23 shape mismatch")
            good = np.isfinite(uv).all(axis=1) & (weights > 0)
            self.valid[ti] = good
            if np.any(good):
                origins, dirs = frame.camera.world_ray(uv[good])
                self.origins[ti, good] = origins
                self.dirs[ti, good] = dirs
            for j in np.where(good)[0]:
                key = (ti, int(j))
                self.var_index[key] = len(self.var_keys)
                self.var_keys.append(key)

        if len(self.var_keys) < 4:
            raise ValueError("Too few Stage-3 rays for metric reconstruction")

        self.height_idx: Optional[int] = None
        self.beta_idx: Optional[int] = None
        n = len(self.var_keys)
        if fixed_height_m is None:
            self.height_idx = n
            n += 1
        self.initializer_used = any(f.initializer is not None for f in self.frames)
        if self.initializer_used and fixed_depth_beta_m is None:
            self.beta_idx = n
            n += 1
        self.nvars = n

        self.x0 = np.zeros(self.nvars, dtype=np.float64)
        self.lb = np.full(self.nvars, -np.inf, dtype=np.float64)
        self.ub = np.full(self.nvars, np.inf, dtype=np.float64)
        self._init_lambdas()
        if initial_xyz_world_m is not None:
            initial_xyz = np.asarray(initial_xyz_world_m, dtype=np.float64)
            if initial_xyz.shape != (self.T, self.J, 3):
                raise ValueError(
                    f"initial_xyz_world_m expected {(self.T, self.J,3)}, got {initial_xyz.shape}"
                )
            for idx, (ti, j) in enumerate(self.var_keys):
                xyz = initial_xyz[ti, j]
                if not np.isfinite(xyz).all():
                    continue
                # The perturbed pixel defines a new ray. Project the frozen
                # solution onto that ray to obtain a nearby, feasible depth.
                lam = float(np.dot(xyz - self.origins[ti, j], self.dirs[ti, j]))
                if np.isfinite(lam):
                    self.x0[idx] = float(np.clip(lam, config.lambda_min_m, config.lambda_max_m))
        if self.height_idx is not None:
            self.x0[self.height_idx] = config.nominal_height_m
            self.lb[self.height_idx] = config.height_min_m
            self.ub[self.height_idx] = config.height_max_m
        self.beta0 = self._estimate_beta0()
        if self.beta_idx is not None:
            self.x0[self.beta_idx] = self.beta0
            self.lb[self.beta_idx] = config.depth_beta_min_m
            self.ub[self.beta_idx] = config.depth_beta_max_m

    def _contact_target_z(self, ti: int, j: int, default_z: float) -> float:
        c = self.frames[ti].contact
        threshold = self.config.contact_likelihood_threshold
        if j in (15, 17, 18, 19) and c.left_likelihood >= threshold:
            return 0.075 if j == 15 else 0.015
        if j in (16, 20, 21, 22) and c.right_likelihood >= threshold:
            return 0.075 if j == 16 else 0.015
        return default_z

    def _init_lambdas(self) -> None:
        h = self.config.nominal_height_m
        for idx, (ti, j) in enumerate(self.var_keys):
            name = POSE23_NAMES[j]
            z = h * float(NOMINAL_Z_FRACTION.get(name, 0.5))
            z = self._contact_target_z(ti, j, z)
            o = self.origins[ti, j]
            d = self.dirs[ti, j]
            lam = np.nan
            if abs(d[2]) > 1e-10:
                lam = (z - o[2]) / d[2]
            if not np.isfinite(lam) or lam <= self.config.lambda_min_m:
                # Fallback to a point near the pitch intersection or a typical
                # broadcast distance. This is numerical initialization only.
                if abs(d[2]) > 1e-10:
                    ground_lam = -o[2] / d[2]
                    if np.isfinite(ground_lam) and ground_lam > 0:
                        lam = ground_lam
                if not np.isfinite(lam) or lam <= 0:
                    lam = 60.0
            lam = float(np.clip(lam, self.config.lambda_min_m, self.config.lambda_max_m))
            self.x0[idx] = lam
            self.lb[idx] = self.config.lambda_min_m
            self.ub[idx] = self.config.lambda_max_m

    def _height(self, x: np.ndarray) -> float:
        if self.fixed_height_m is not None:
            return float(self.fixed_height_m)
        assert self.height_idx is not None
        return float(x[self.height_idx])

    def _beta(self, x: np.ndarray) -> float:
        if not self.initializer_used:
            return 0.0
        if self.fixed_depth_beta_m is not None:
            return float(self.fixed_depth_beta_m)
        assert self.beta_idx is not None
        return float(x[self.beta_idx])

    def xyz_from_x(self, x: np.ndarray) -> np.ndarray:
        xyz = np.full((self.T, self.J, 3), np.nan, dtype=np.float64)
        for idx, (ti, j) in enumerate(self.var_keys):
            xyz[ti, j] = self.origins[ti, j] + float(x[idx]) * self.dirs[ti, j]
        return xyz

    @staticmethod
    def _normalized_initializer_depth(init: InitializerObservation) -> Optional[np.ndarray]:
        z = np.asarray(init.relative_depth_23, dtype=np.float64).copy()
        finite = np.isfinite(z)
        if finite.sum() < 5:
            return None
        if np.isfinite(z[[11, 12]]).any():
            root = float(np.nanmean(z[[11, 12]]))
        else:
            root = float(np.nanmedian(z[finite]))
        z -= root
        body = np.abs(z[:17][np.isfinite(z[:17])])
        if body.size == 0:
            return None
        scale = float(np.nanpercentile(body, 75))
        if not np.isfinite(scale) or scale < 1e-6:
            return None
        return z / scale

    def _estimate_beta0(self) -> float:
        if not self.initializer_used:
            return 0.0
        xyz0 = self.xyz_from_x(self.x0)
        a: List[float] = []
        b: List[float] = []
        for ti, frame in enumerate(self.frames):
            if frame.initializer is None:
                continue
            zrel = self._normalized_initializer_depth(frame.initializer)
            if zrel is None:
                continue
            roots = [j for j in (11, 12) if self.valid[ti, j]]
            if not roots:
                continue
            root_depth = float(np.mean([frame.camera.camera_depth(xyz0[ti, j]) for j in roots]))
            for j in range(self.J):
                if not self.valid[ti, j] or not np.isfinite(zrel[j]):
                    continue
                dz = float(frame.camera.camera_depth(xyz0[ti, j]) - root_depth)
                a.append(float(zrel[j]))
                b.append(dz)
        if len(a) < 5:
            return float(self.config.depth_beta_init_m)
        aa = np.asarray(a)
        bb = np.asarray(b)
        denom = float(np.dot(aa, aa))
        beta = float(np.dot(aa, bb) / denom) if denom > 1e-10 else self.config.depth_beta_init_m
        if abs(beta) < 0.04:
            # Preserve depth orientation from correlation while avoiding a
            # flat initial prior.
            corr = np.corrcoef(aa, bb)[0, 1] if np.std(aa) > 0 and np.std(bb) > 0 else 1.0
            beta = math.copysign(self.config.depth_beta_init_m, corr if np.isfinite(corr) else 1.0)
        return float(np.clip(beta, self.config.depth_beta_min_m, self.config.depth_beta_max_m))

    def residuals(self, x: np.ndarray) -> np.ndarray:
        cfg = self.config
        xyz = self.xyz_from_x(x)
        height = self._height(x)
        beta = self._beta(x)
        r: List[float] = []

        if self.fixed_height_m is None:
            r.append((height - cfg.nominal_height_m) / max(cfg.height_sigma_m, 1e-6))

        # Broad metric anthropometric priors. Stage-3 state weights attenuate
        # joints already marked unstable.
        bone_scale = math.sqrt(max(cfg.bone_prior_weight, 0.0))
        for ti, frame in enumerate(self.frames):
            for prior in BONE_PRIORS:
                if not (self.valid[ti, prior.a] and self.valid[ti, prior.b]):
                    continue
                length = float(np.linalg.norm(xyz[ti, prior.a] - xyz[ti, prior.b]))
                target = prior.length_ratio_to_height * height
                sigma = max(0.035, prior.sigma_ratio_to_height * height)
                w = float(math.sqrt(max(0.05, frame.state_weights23[prior.a] * frame.state_weights23[prior.b])))
                r.append(bone_scale * w * (length - target) / sigma)

        # Learned relative-depth prior. Stage-3 x/y remain hard constraints via
        # rays; RTMW3D contributes only relative depth evidence in v0.1.
        depth_scale = math.sqrt(max(cfg.depth_prior_weight, 0.0))
        if self.initializer_used and depth_scale > 0:
            for ti, frame in enumerate(self.frames):
                init = frame.initializer
                if init is None:
                    continue
                zrel = self._normalized_initializer_depth(init)
                if zrel is None:
                    continue
                roots = [j for j in (11, 12) if self.valid[ti, j]]
                if not roots:
                    continue
                root_depth = float(np.mean([frame.camera.camera_depth(xyz[ti, j]) for j in roots]))
                for j in range(self.J):
                    if not self.valid[ti, j] or not np.isfinite(zrel[j]):
                        continue
                    dz = float(frame.camera.camera_depth(xyz[ti, j]) - root_depth)
                    # Do not treat raw model score as probability. It is only a
                    # mild relative evidence multiplier after clipping.
                    score_weight = 1.0
                    try:
                        raw_score = float(init.scores_23[j])
                        if np.isfinite(raw_score):
                            score_weight = float(np.clip(raw_score / 3.0, 0.35, 1.25))
                    except Exception:
                        pass
                    stage3_weight = float(np.sqrt(max(0.05, frame.state_weights23[j])))
                    r.append(
                        depth_scale * stage3_weight * score_weight
                        * (dz - beta * float(zrel[j])) / max(cfg.depth_prior_sigma_m, 1e-6)
                    )
            # Weakly prevent beta from becoming an arbitrary very large scale.
            r.append(0.20 * (beta - self.beta0) / 0.40)

        # Ground-contact soft constraints and non-penetration.
        ground_scale = math.sqrt(max(cfg.ground_weight, 0.0))
        nonpen_scale = math.sqrt(max(cfg.nonpenetration_weight, 0.0))
        threshold = cfg.contact_likelihood_threshold
        for ti, frame in enumerate(self.frames):
            for side, likelihood, ankle, surface in (
                ("left", frame.contact.left_likelihood, 15, (17, 18, 19)),
                ("right", frame.contact.right_likelihood, 16, (20, 21, 22)),
            ):
                if likelihood >= threshold:
                    if self.valid[ti, ankle]:
                        r.append(ground_scale * likelihood * (xyz[ti, ankle, 2] - 0.075) / 0.055)
                    for j in surface:
                        if self.valid[ti, j]:
                            r.append(ground_scale * likelihood * (xyz[ti, j, 2] - 0.015) / 0.040)
            for j in range(self.J):
                if not self.valid[ti, j]:
                    continue
                penetration = max(0.0, -0.03 - float(xyz[ti, j, 2]))
                # Keep residual dimensionality constant across optimizer calls.
                r.append(nonpen_scale * penetration / 0.035)

        # World-space temporal acceleration. Only truly consecutive frames are
        # coupled; this prevents accidental smoothing across track gaps.
        temporal_scale = math.sqrt(max(cfg.temporal_weight, 0.0))
        if temporal_scale > 0 and self.T >= 3:
            for ti in range(1, self.T - 1):
                if not (
                    self.frames[ti].frame_index - self.frames[ti - 1].frame_index == 1
                    and self.frames[ti + 1].frame_index - self.frames[ti].frame_index == 1
                ):
                    continue
                for j in range(self.J):
                    if not (self.valid[ti - 1, j] and self.valid[ti, j] and self.valid[ti + 1, j]):
                        continue
                    dd = xyz[ti + 1, j] - 2.0 * xyz[ti, j] + xyz[ti - 1, j]
                    if j in (17, 18, 19, 20, 21, 22):
                        sigma = 0.20
                    elif j in (13, 14, 15, 16, 7, 8, 9, 10):
                        sigma = 0.13
                    else:
                        sigma = 0.075
                    r.extend((temporal_scale * dd / sigma).tolist())

        # Keep pelvis centre near the physical pitch region when observable.
        pitch_scale = math.sqrt(max(cfg.pitch_bound_weight, 0.0))
        if pitch_scale > 0:
            for ti, frame in enumerate(self.frames):
                hips = [j for j in (11, 12) if self.valid[ti, j]]
                if not hips:
                    continue
                pelvis = np.mean(xyz[ti, hips], axis=0)
                length = float(frame.camera.pitch.get("length_m", 105.0))
                width = float(frame.camera.pitch.get("width_m", 68.0))
                xlim = 0.5 * length + cfg.pitch_margin_m
                ylim = 0.5 * width + cfg.pitch_margin_m
                excess_x = max(0.0, abs(float(pelvis[0])) - xlim)
                excess_y = max(0.0, abs(float(pelvis[1])) - ylim)
                # Hinge losses still emit zeros so least_squares sees a fixed
                # residual vector length for every parameter evaluation.
                r.append(pitch_scale * excess_x / 1.0)
                r.append(pitch_scale * excess_y / 1.0)

        if not r:
            return np.asarray([0.0], dtype=np.float64)
        return np.asarray(r, dtype=np.float64)

    def jacobian_sparsity(self):
        """Return a conservative sparse finite-difference dependency pattern.

        Rows are grouped by frame where useful.  This is a safe superset of
        the true local dependency graph: it preserves the objective while
        letting SciPy avoid dense numerical differentiation.
        """

        frame_columns = [
            [self.var_index[(ti, j)] for j in range(self.J) if self.valid[ti, j]]
            for ti in range(self.T)
        ]
        dependencies: List[List[int]] = []

        def add(columns: Sequence[int], count: int = 1) -> None:
            dependencies.extend([list(columns)] * count)

        if self.fixed_height_m is None:
            add([self.height_idx])
        for ti, frame in enumerate(self.frames):
            for prior in BONE_PRIORS:
                if self.valid[ti, prior.a] and self.valid[ti, prior.b]:
                    columns = list(frame_columns[ti])
                    if self.height_idx is not None:
                        columns.append(self.height_idx)
                    add(columns)
        if self.initializer_used and self.config.depth_prior_weight > 0:
            for ti, frame in enumerate(self.frames):
                zrel = None if frame.initializer is None else self._normalized_initializer_depth(frame.initializer)
                roots = [j for j in (11, 12) if self.valid[ti, j]]
                if zrel is None or not roots:
                    continue
                for j in range(self.J):
                    if self.valid[ti, j] and np.isfinite(zrel[j]):
                        columns = list(frame_columns[ti])
                        if self.beta_idx is not None:
                            columns.append(self.beta_idx)
                        add(columns)
            # The weak beta residual is present even when beta is frozen for
            # uncertainty re-solves. In that case it is a constant row with no
            # free-variable dependency, but the sparsity matrix must still
            # preserve the residual row count.
            add([] if self.beta_idx is None else [self.beta_idx])
        for ti, frame in enumerate(self.frames):
            for likelihood, ankle, surface in (
                (frame.contact.left_likelihood, 15, (17, 18, 19)),
                (frame.contact.right_likelihood, 16, (20, 21, 22)),
            ):
                if likelihood >= self.config.contact_likelihood_threshold:
                    if self.valid[ti, ankle]:
                        add(frame_columns[ti])
                    for joint in surface:
                        if self.valid[ti, joint]:
                            add(frame_columns[ti])
            for joint in range(self.J):
                if self.valid[ti, joint]:
                    add(frame_columns[ti])
        if self.config.temporal_weight > 0 and self.T >= 3:
            for ti in range(1, self.T - 1):
                if not (
                    self.frames[ti].frame_index - self.frames[ti - 1].frame_index == 1
                    and self.frames[ti + 1].frame_index - self.frames[ti].frame_index == 1
                ):
                    continue
                columns = frame_columns[ti - 1] + frame_columns[ti] + frame_columns[ti + 1]
                for joint in range(self.J):
                    if self.valid[ti - 1, joint] and self.valid[ti, joint] and self.valid[ti + 1, joint]:
                        add(columns, count=3)
        if self.config.pitch_bound_weight > 0:
            for ti in range(self.T):
                if any(self.valid[ti, joint] for joint in (11, 12)):
                    add(frame_columns[ti], count=2)

        residual_count = int(self.residuals(self.x0).size)
        if len(dependencies) != residual_count:
            raise RuntimeError(f"Jacobian sparsity row mismatch: {len(dependencies)} != {residual_count}")
        pattern = lil_matrix((residual_count, self.nvars), dtype=np.int8)
        for row, columns in enumerate(dependencies):
            for column in columns:
                if column is not None:
                    pattern[row, int(column)] = 1
        return pattern.tocsr()

    def solve(self) -> OptimizationResult:
        sparsity = self.jacobian_sparsity() if self.config.sparse_jacobian else None
        started = time.perf_counter()
        res = least_squares(
            self.residuals,
            self.x0,
            bounds=(self.lb, self.ub),
            loss=self.config.optimizer_loss,
            f_scale=self.config.optimizer_f_scale,
            max_nfev=self.config.optimizer_max_nfev,
            x_scale="jac",
            jac_sparsity=sparsity,
            verbose=int(self.config.optimizer_verbose),
        )
        elapsed_s = time.perf_counter() - started
        xyz = self.xyz_from_x(res.x)
        lambdas = np.full((self.T, self.J), np.nan, dtype=np.float64)
        for idx, (ti, j) in enumerate(self.var_keys):
            lambdas[ti, j] = res.x[idx]

        reproj: List[float] = []
        bone_abs: List[float] = []
        ground_abs: List[float] = []
        negative_count = 0
        valid_count = int(self.valid.sum())
        height = self._height(res.x)
        for ti, frame in enumerate(self.frames):
            good = self.valid[ti]
            if np.any(good):
                pred_uv = frame.camera.project_world(xyz[ti, good])
                err = np.linalg.norm(pred_uv - frame.uv23[good], axis=1)
                reproj.extend(err.tolist())
                negative_count += int(np.sum(xyz[ti, good, 2] < -0.03))
            for prior in BONE_PRIORS:
                if self.valid[ti, prior.a] and self.valid[ti, prior.b]:
                    length = float(np.linalg.norm(xyz[ti, prior.a] - xyz[ti, prior.b]))
                    bone_abs.append(abs(length - prior.length_ratio_to_height * height))
            if frame.contact.left_likelihood >= self.config.contact_likelihood_threshold:
                for j, zt in ((15, 0.075), (17, 0.015), (18, 0.015), (19, 0.015)):
                    if self.valid[ti, j]:
                        ground_abs.append(abs(float(xyz[ti, j, 2] - zt)))
            if frame.contact.right_likelihood >= self.config.contact_likelihood_threshold:
                for j, zt in ((16, 0.075), (20, 0.015), (21, 0.015), (22, 0.015)):
                    if self.valid[ti, j]:
                        ground_abs.append(abs(float(xyz[ti, j, 2] - zt)))

        reproj_arr = np.asarray(reproj, dtype=np.float64)
        finite_xyz = bool(np.isfinite(xyz[self.valid]).all())
        median_bone_abs_error = float(np.median(bone_abs)) if bone_abs else float("nan")
        ground_residual_median = float(np.median(ground_abs)) if ground_abs else None
        negative_fraction = float(negative_count / max(valid_count, 1))
        geometry_gate = {
            "finite_xyz": finite_xyz,
            "height_in_bounds": bool(self.config.height_min_m <= height <= self.config.height_max_m),
            "median_bone_abs_error_m": median_bone_abs_error,
            "bone_plausible": bool(np.isfinite(median_bone_abs_error) and median_bone_abs_error <= self.config.geometry_max_bone_error_m),
            "ground_residual_median_m": ground_residual_median,
            "ground_plausible": bool(ground_residual_median is None or ground_residual_median <= self.config.geometry_max_ground_residual_m),
            "negative_z_joint_fraction": negative_fraction,
            "z_plausible": bool(negative_fraction <= self.config.geometry_max_negative_z_fraction),
        }
        geometry_gate["plausible"] = bool(all(
            geometry_gate[key] for key in ("finite_xyz", "height_in_bounds", "bone_plausible", "ground_plausible", "z_plausible")
        ))
        if bool(res.success):
            status = "CONVERGED"
        elif int(res.status) == 0 or int(res.nfev) >= int(self.config.optimizer_max_nfev):
            status = "MAX_NFEV"
        elif not np.isfinite(res.x).all():
            status = "NUMERICAL_FAILURE"
        else:
            status = "FAILED"
        return OptimizationResult(
            frame_indices=np.asarray([f.frame_index for f in self.frames], dtype=np.int64),
            xyz_world_m=xyz,
            available_mask=self.valid.copy(),
            lambdas_m=lambdas,
            estimated_height_m=float(height),
            depth_beta_m=float(self._beta(res.x)),
            initializer_used=self.initializer_used,
            success=bool(res.success),
            status=status,
            cost=float(res.cost),
            nfev=int(res.nfev),
            njev=None if getattr(res, "njev", None) is None else int(res.njev),
            elapsed_s=float(elapsed_s),
            message=str(res.message),
            median_reprojection_error_px=float(np.median(reproj_arr)) if reproj_arr.size else float("nan"),
            p90_reprojection_error_px=float(np.percentile(reproj_arr, 90)) if reproj_arr.size else float("nan"),
            median_bone_abs_error_m=median_bone_abs_error,
            ground_residual_median_m=ground_residual_median,
            negative_z_joint_fraction=negative_fraction,
            diagnostics={
                "n_lambda_variables": len(self.var_keys),
                "n_total_variables": self.nvars,
                "initial_height_m": self.config.nominal_height_m,
                "initial_depth_beta_m": self.beta0 if self.initializer_used else None,
                "final_residual_count": int(self.residuals(res.x).size),
                "hard_ray_parameterization": True,
                "stage3_xy_moved_by_optimizer": False,
                "scipy_status": int(res.status),
                "termination_reason": status,
                "njev": None if getattr(res, "njev", None) is None else int(res.njev),
                "elapsed_s": float(elapsed_s),
                "jacobian_mode": "sparse" if self.config.sparse_jacobian else "dense",
                "jacobian_nnz": None if sparsity is None else int(sparsity.nnz),
                "jacobian_density": None if sparsity is None else float(sparsity.nnz / max(1, sparsity.shape[0] * sparsity.shape[1])),
                "geometry_quality_gate": geometry_gate,
            },
        )


def solve_metric_pose(
    frames: Sequence[FrameInput],
    config: Stage4Config,
    *,
    fixed_height_m: Optional[float] = None,
    fixed_depth_beta_m: Optional[float] = None,
    initial_xyz_world_m: Optional[np.ndarray] = None,
) -> OptimizationResult:
    return RayBundleProblem(
        frames,
        config,
        fixed_height_m=fixed_height_m,
        fixed_depth_beta_m=fixed_depth_beta_m,
        initial_xyz_world_m=initial_xyz_world_m,
    ).solve()
