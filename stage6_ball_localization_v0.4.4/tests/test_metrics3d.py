from ball_localization.evaluation import summarize_3d


def test_3d_metrics_perfect():
    m=summarize_3d([{
        'gt_xyz':[1,2,3],
        'pred_xyz':[1,2,3],
        'camera_to_gt_distance_m':10,
        'self_reprojection_error_px':0,
        'geometry_diagnostics':{'size_prior_validity':'VALID'},
    }])
    assert m['coverage']==1 and m['MAE_3D_m']==0 and m['BLE_X_MAE_m']==0 and m['Precision_at_2m']==1
    assert m['SelfReprojectionErrorPx_mean']==0
    assert m['validity_breakdown']['counts']['VALID']==1


def test_validity_breakdown_keeps_rejection_reason():
    m=summarize_3d([{
        'gt_xyz':[1,2,3],
        'pred_xyz':None,
        'geometry_diagnostics':{'size_prior_validity':'INVALID_HEIGHT'},
    }])
    assert m['coverage']==0
    assert m['validity_breakdown']['counts']['INVALID_HEIGHT']==1
