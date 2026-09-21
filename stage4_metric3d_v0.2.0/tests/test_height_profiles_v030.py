from stage4_metric3d.height_profiles import load_height_profile
from stage4_metric3d.wholebody import POSE23_NAMES


def test_canonical_profile_covers_pose23():
    profile = load_height_profile("canonical-body-height-v1")
    assert set(profile.fractions) == set(POSE23_NAMES)
    assert profile.z_m("left_shoulder", 1.8) == 0.82 * 1.8
    assert profile.raw["provenance"]["kind"] == "engineering_prior"
