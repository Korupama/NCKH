import json
from pathlib import Path
import pytest
from stage5_team_affiliation.adapters import load_stage3_state


def test_stage3_rejects_wrong_coordinate_space(tmp_path):
    p=tmp_path/'s.json'
    p.write_text(json.dumps({'schema_version':'tracked-pose-2d-state-1.0','coordinate_space':'RESIZED','keypoint_schema':{'names':[str(i) for i in range(133)]}}))
    with pytest.raises(ValueError): load_stage3_state(p)
