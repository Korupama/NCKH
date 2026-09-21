from pathlib import Path
import zipfile

import cv2
import numpy as np

from ball_localization.datasets import make_zip_image_uri
from ball_localization.evaluation.failure_analysis import classify_2d_failure, classify_3d_failure, render_failure_overlays


def test_2d_failure_wrong_top1_but_candidate_recoverable():
    row = {
        "record_id": "x",
        "gt_bbox": [0,0,10,10],
        "img_w": 100,
        "img_h": 100,
        "predictions": [
            {"bbox_xyxy": [50,50,60,60], "score": .9},
            {"bbox_xyxy": [0,0,10,10], "score": .8},
        ],
    }
    f = classify_2d_failure(row, candidate_k=2)
    assert "WRONG_TOP1_BUT_GT_IN_TOPK" in f["failure_types"]


def test_3d_failure_blex():
    row = {"record_id":"x","gt_xyz":[0,0,0],"pred_xyz":[2,0,0],"detector_required":False}
    f = classify_3d_failure(row, error_3d_threshold_m=10, blex_threshold_m=1)
    assert "LARGE_BLE_X" in f["failure_types"]


def test_failure_overlay_reads_zip_reference(tmp_path:Path):
    image=np.zeros((60,80,3),dtype=np.uint8)
    ok,encoded=cv2.imencode('.png',image)
    assert ok
    archive=tmp_path/'Frames-v3.zip'
    with zipfile.ZipFile(archive,'w') as zf:
        zf.writestr('nested/3.png',encoded.tobytes())
    failures=[{
        'record_id':'L/S/M/3.png',
        'failure_types':['TEST'],
        'severity':1.0,
        'image_path':make_zip_image_uri(archive,'nested/3.png'),
        'gt_bbox':[5,5,15,15],
        'predictions':[{'bbox_xyxy':[6,6,16,16]}],
    }]
    rendered=render_failure_overlays(failures,tmp_path/'out',max_images=1)
    assert len(rendered)==1
    assert Path(rendered[0]).is_file()
