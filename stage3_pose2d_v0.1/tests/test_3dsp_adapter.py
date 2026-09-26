import json
import numpy as np
import pytest
from benchmark.dsp3_adapter import (
    _select_crop_candidate,
    inspect_3dsp,
    iter_3dsp,
    normalize_crop_scales,
    parse_3dsp_keypoints_2d,
    run_3dsp_benchmark,
)


def test_parse_public_typo_key():
    data={"keypont_2d":{str(i):{"name":str(i),"x":i,"y":i+1} for i in range(17)}}
    arr=parse_3dsp_keypoints_2d(data)
    assert arr.shape==(17,2)
    assert np.allclose(arr[3],[3,4])


def test_iter_synthetic_3dsp(tmp_path):
    import cv2
    shot=tmp_path/"train"/"00001"; (shot/"img").mkdir(parents=True); (shot/"posture").mkdir()
    cv2.imwrite(str(shot/"img"/"001.jpg"),np.zeros((32,24,3),np.uint8))
    (shot/"posture"/"001.json").write_text(json.dumps({"keypont_2d":{str(i):{"x":i,"y":i} for i in range(17)}}))
    samples=list(iter_3dsp(tmp_path,"train"))
    assert len(samples)==1 and samples[0].gt_h36m17.shape==(17,2)
    assert inspect_3dsp(tmp_path)["splits"]["train"]["posture_json"]==1


def test_iter_shot_manifest_filters_and_preserves_order(tmp_path):
    import cv2
    for shot_id in ("00001", "00002", "00003"):
        shot=tmp_path/"train"/shot_id; (shot/"img").mkdir(parents=True); (shot/"posture").mkdir()
        cv2.imwrite(str(shot/"img"/"001.jpg"),np.zeros((32,24,3),np.uint8))
        (shot/"posture"/"001.json").write_text(json.dumps({"keypont_2d":{str(i):{"x":i,"y":i} for i in range(17)}}))
    manifest=tmp_path/"holdout.json"
    manifest.write_text(json.dumps({"shot_ids":["00003","00001"]}))
    samples=list(iter_3dsp(tmp_path,"train",shot_manifest=manifest))
    assert [s.shot_id for s in samples] == ["00001", "00003"]


def test_crop_scales_keep_control_first_and_remove_duplicates():
    assert normalize_crop_scales([1.1, 1.0, 1.1, 0.9]) == [1.0, 1.1, 0.9]
    assert normalize_crop_scales([0.9, 1.1]) == [1.0, 0.9, 1.1]


@pytest.mark.parametrize("values", [[], [0.0], [-1.0], [float("nan")], [float("inf")]])
def test_crop_scales_reject_invalid_values(values):
    with pytest.raises(ValueError):
        normalize_crop_scales(values)


def test_crop_selector_uses_qa_only_and_prefers_control_on_tie():
    qa = {
        "pose_status": "VALID",
        "body_completeness": 1.0,
        "core_completeness": 1.0,
        "feet_completeness": 1.0,
        "inside_fraction": 1.0,
        "bone_outlier_fraction": 0.0,
    }
    candidates = [
        {"crop_scale": 1.0, "qa": dict(qa), "ground_truth_metric": 0.0},
        {"crop_scale": 1.1, "qa": dict(qa), "ground_truth_metric": 999.0},
    ]
    assert _select_crop_candidate(candidates) == 0


def test_crop_selector_prefers_better_stage3_quality_before_crop_distance():
    candidates = [
        {
            "crop_scale": 1.0,
            "qa": {
                "pose_status": "DEGRADED",
                "body_completeness": 0.8,
                "core_completeness": 0.7,
                "feet_completeness": 0.3,
                "inside_fraction": 0.8,
                "bone_outlier_fraction": 0.1,
            },
        },
        {
            "crop_scale": 1.1,
            "qa": {
                "pose_status": "VALID",
                "body_completeness": 1.0,
                "core_completeness": 1.0,
                "feet_completeness": 1.0,
                "inside_fraction": 1.0,
                "bone_outlier_fraction": 0.0,
            },
        },
    ]
    assert _select_crop_candidate(candidates) == 1
