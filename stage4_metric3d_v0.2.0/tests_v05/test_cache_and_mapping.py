import numpy as np
from stage4_metric3d.backends.sam3d_pitch_refined.cache import MHR70_NAMES, save_sam3d_native_cache, Sam3DNativeCache
from stage4_metric3d.backends.sam3d_pitch_refined.joint_mapping import DEFAULT_MAPPING


def test_official_mhr70_mapping_is_complete():
    assert len(MHR70_NAMES)==70
    assert len(DEFAULT_MAPPING)==23
    d={e.canonical_name:e.sam3d_index for e in DEFAULT_MAPPING}
    assert d['left_wrist']==62 and d['right_wrist']==41
    assert d['left_big_toe']==15 and d['right_heel']==20


def test_native_cache_roundtrip(tmp_path):
    p=save_sam3d_native_cache(tmp_path/'c.npz',frame_indices=[1,2],track_ids=['t'],boxes_xyxy=np.ones((2,1,4)),skel_2d_px=np.ones((2,1,70,2)),skel_3d_relative_m=np.ones((2,1,70,3)),pred_cam_t_m=np.ones((2,1,3)),focal_length_px=np.ones((2,1)),valid_mask=np.ones((2,1),bool))
    c=Sam3DNativeCache.load(p)
    assert c.T==2 and c.N==1 and c.J==70
