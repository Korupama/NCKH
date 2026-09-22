from stage3_pose2d.quality import summarize_t0_anatomy_evidence
from stage3_pose2d.wholebody133 import WHOLEBODY_KEYPOINT_NAMES


def _records(*, missing_name=None, temporal_name=None):
    records = []
    for index, name in enumerate(WHOLEBODY_KEYPOINT_NAMES):
        missing = name == missing_name
        record = {
            "index": index,
            "name": name,
            "x": None if missing else float(index),
            "y": None if missing else float(index + 1),
            "state": "MISSING" if missing else "VALID",
            "temporal_estimate_xy": None,
        }
        if name == temporal_name:
            record["temporal_estimate_xy"] = [float(index), float(index + 1)]
            record["coordinate_evidence_kind"] = "TEMPORAL_ESTIMATE_ONLY"
        records.append(record)
    return records


def test_temporal_estimate_never_counts_as_raw_observed():
    evidence = summarize_t0_anatomy_evidence(
        _records(temporal_name="left_ankle"),
        pose_status="DEGRADED",
    )

    core = evidence["stage4_core_metric_anchors"]
    assert core["all_raw_observed"] is False
    assert "left_ankle" in core["missing_raw_names"]
    assert "left_ankle" in core["temporal_estimate_only_names"]
    assert "TEMPORAL_ESTIMATE_ONLY_AT_T0" in evidence["consumer_review_flags"]


def test_anatomy_summary_reports_missing_raw_group_without_legal_semantics():
    evidence = summarize_t0_anatomy_evidence(
        _records(missing_name="left_big_toe"),
        pose_status="VALID",
    )

    left_foot = evidence["groups"]["left_foot"]
    assert left_foot["raw_observed_count"] == 3
    assert left_foot["all_raw_observed"] is False
    assert "left_big_toe" in left_foot["missing_raw_names"]
    assert evidence["scope"].startswith("raw image-space evidence")
    assert "Law-11 legal-body membership" in evidence["scope"]
