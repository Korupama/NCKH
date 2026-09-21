from ball_localization.contracts import BallCandidate2D
from ball_localization.tracking import select_ball_path

def c(fi,cid,x,score):return BallCandidate2D(fi,cid,[x,10,x+4,14],[x+2,12],score,'synthetic',4)

def test_viterbi_prefers_temporally_consistent_lower_score_path():
    by={0:[c(0,'true0',10,.8),c(0,'false0',100,.95)],1:[c(1,'true1',12,.8),c(1,'false1',300,.95)],2:[c(2,'true2',14,.8),c(2,'false2',500,.95)]}
    p=select_ball_path(by,[0,1,2],image_width=640,image_height=360)
    assert [p[i].candidate_id for i in [0,1,2]]==['true0','true1','true2']
