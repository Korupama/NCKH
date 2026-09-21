import numpy as np
from benchmark.metrics import normalized_joint_errors,pdj_from_errors,pdj_auc_from_errors,summarize_pdj,summarize_pck


def h36m_pose():
    # shoulders at +/-1, y=-1; hips at +/-0.6,y=0 -> torso-centre distance =1
    p=np.zeros((17,2),float)
    p[11]=[-1,-1]; p[14]=[1,-1]; p[1]=[.6,0]; p[4]=[-.6,0]
    p[2]=[.6,1];p[3]=[.6,2];p[5]=[-.6,1];p[6]=[-.6,2]
    p[7]=[0,-.2];p[8]=[0,-.7];p[9]=[0,-1.2];p[10]=[0,-1.5]
    p[12]=[-1.4,-.5];p[13]=[-1.6,0];p[15]=[1.4,-.5];p[16]=[1.6,0]
    return p


def test_pdj_perfect_and_auc():
    gt=h36m_pose()[None]
    s=summarize_pdj(gt.copy(),gt)
    assert s["PDJ"]==1.0
    assert s["AUC"] > .99


def test_pdj_threshold():
    gt=h36m_pose()[None]; pred=gt.copy(); pred[:,0,0]+=.6
    e=normalized_joint_errors(pred,gt)
    assert pdj_from_errors(e,.5) < 1.0


def test_pck_bbox_normalized():
    gt=np.zeros((1,23,2)); pred=gt.copy(); pred[0,0,0]=12
    s=summarize_pck(pred,gt,[[0,0,100,200]],thresholds=(.05,.10))
    assert 0 < s["PCK@0.05"] < 1
