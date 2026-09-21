import json
from benchmark.pose23 import evaluate_pose23


def test_pose23_eval(tmp_path):
    k=[{"x":float(i),"y":float(i)} for i in range(23)]
    gt={"annotations":[{"id":"a","bbox_xyxy":[0,0,100,200],"keypoints_23":k}]}
    pr={"predictions":[{"id":"a","keypoints_23":k}]}
    g=tmp_path/"g.json"; p=tmp_path/"p.json"; g.write_text(json.dumps(gt));p.write_text(json.dumps(pr))
    r=evaluate_pose23(g,p)
    assert r["metrics"]["PCK@0.05"]==1.0
    assert r["metrics"]["groups"]["heels"]["PCK@0.05"]==1.0
