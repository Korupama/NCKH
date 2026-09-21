from stage5_team_affiliation.evaluation import evaluate_team_assignments


def test_permutation_invariant_accuracy():
    gt={'a':0,'b':0,'c':1,'d':1}
    pred={'a':1,'b':1,'c':0,'d':0}
    m=evaluate_team_assignments(gt,pred)
    assert m['hungarian_matched_accuracy']==1.0
    assert m['ARI']==1.0
