from stage5_team_affiliation.soccernet_gsr import extract_track_team_labels


def test_gsr_team_extraction():
    labels={'annotations':[
        {'track_id':1,'attributes':{'role':'player','team':'left'}},
        {'track_id':1,'attributes':{'role':'player','team':'left'}},
        {'track_id':2,'attributes':{'role':'referee','team':None}},
        {'track_id':3,'attributes':{'role':'goalkeeper','team':'right'}},
    ]}
    out=extract_track_team_labels(labels)
    assert out=={'1':'left','3':'right'}
