from stage_1_camera.candidates import collect_pnlcalib_candidates


class FakeCalib:
    def get_cam_params(self, mode, use_ransac, refine=False, refine_w_lines=False):
        if mode == 'full' and use_ransac == 0:
            return {'x_focal_length':1000, 'y_focal_length':1000, 'position_meters':[0,0,-10]}, 4.0
        if mode == 'ground_plane' and use_ransac == 5:
            return {'x_focal_length':1100, 'y_focal_length':1100, 'position_meters':[0,0,-10]}, 2.0
        return None, None


def test_candidate_grid_keeps_full_even_when_ground_is_best_residual():
    d = collect_pnlcalib_candidates(
        FakeCalib(), modes=('full','ground_plane'), ransac_values=(0,5),
        selected_result={'mode':'full','use_ransac':0,'rep_err':4.0},
    )
    assert d['num_attempted'] == 4
    assert d['num_successful'] == 2
    assert d['best_overall']['mode'] == 'ground_plane'
    assert d['full_no_ransac']['rep_err_px'] == 4.0
    assert any(x['selected'] for x in d['candidates'])
