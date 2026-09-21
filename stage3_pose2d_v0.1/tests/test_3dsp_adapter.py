import json
import numpy as np
from benchmark.dsp3_adapter import parse_3dsp_keypoints_2d, iter_3dsp, inspect_3dsp


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
