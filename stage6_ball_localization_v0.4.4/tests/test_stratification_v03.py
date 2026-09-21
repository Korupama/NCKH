from ball_localization.evaluation.stratification import diameter_bin, height_bin, camera_distance_bin


def test_stage6_bins():
    assert diameter_bin(4.9) == "<5px"
    assert diameter_bin(7) == "5-10px"
    assert diameter_bin(12) == "10-15px"
    assert diameter_bin(20) == ">=15px"
    assert height_bin(0.11) == "ground<=0.35m"
    assert height_bin(0.7) == "0.35-1m"
    assert height_bin(2.0) == "1-3m"
    assert height_bin(4.0) == ">3m"
    assert camera_distance_bin(20) == "<30m"
    assert camera_distance_bin(40) == "30-50m"
