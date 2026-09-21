import numpy as np
from stage4_metric3d.evaluation import (
    goalward_anchor_longitudinal_error,
    mpjpe,
    pairwise_longitudinal_ordering_accuracy,
    uncertainty_interval_coverage,
)


def test_mpjpe_and_gale():
    gt = np.zeros((23,3))
    pred = gt.copy(); pred[:,0] += .1
    assert abs(mpjpe(pred,gt)-.1) < 1e-12
    assert abs(goalward_anchor_longitudinal_error(pred,gt,attack_sign=1)-.1) < 1e-12


def test_pairwise_ordering():
    assert pairwise_longitudinal_ordering_accuracy([1,2,3],[1,2,3]) == 1.0
    assert pairwise_longitudinal_ordering_accuracy([3,2,1],[1,2,3]) == 0.0


def test_uncertainty_interval_coverage_is_axis_specific():
    gt = np.asarray([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]])
    lo = gt - np.asarray([[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]])
    hi = gt + np.asarray([[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]])
    hi[0, 1, 0] = 1.5
    result = uncertainty_interval_coverage(gt, lo, hi)
    assert result["coverage_x"] == 0.5
    assert result["coverage_y"] == 1.0
    assert result["coverage_z"] == 1.0
