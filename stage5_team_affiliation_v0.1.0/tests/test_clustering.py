import numpy as np
from stage5_team_affiliation.config import Stage5Config
from stage5_team_affiliation.clustering import fit_two_teams, assign_to_centroids


def test_two_team_clustering():
    feats={}
    for i in range(5): feats[f'a{i}']=np.array([1,0,0,0],np.float32)+i*0.005
    for i in range(5): feats[f'b{i}']=np.array([0,1,0,0],np.float32)+i*0.005
    r=fit_two_teams(feats,Stage5Config(min_cluster_margin=0.01))
    la={r.labels[f'a{i}'] for i in range(5)}
    lb={r.labels[f'b{i}'] for i in range(5)}
    assert len(la)==1 and len(lb)==1 and la!=lb


def test_assign_centroids_fail_closed():
    c=np.array([[1,0],[0,1]],float)
    r=assign_to_centroids(np.array([0.5,0.5]),c,min_margin=.2)
    assert r['status']=='UNKNOWN'
