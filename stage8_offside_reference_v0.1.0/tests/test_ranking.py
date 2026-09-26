from stage8_offside_reference.ranking import rank_opponents, select_second_last


def row(tid, q, x=None):
    return {"track_id": tid, "goalward_q_m": q, "goalward_x_m": q if x is None else x, "anchor": {"name": "nose"}, "status": "VALID"}


def test_ranking_descending_goalward():
    r = rank_opponents([row("a", 40), row("b", 50), row("c", 45)])
    assert [x["track_id"] for x in r] == ["b", "c", "a"]
    assert [x["rank"] for x in r] == [1, 2, 3]


def test_second_last_single_identity():
    r = rank_opponents([row("a", 50), row("b", 45), row("c", 40)])
    s = select_second_last(r)
    assert s["track_id"] == "b"
    assert s["candidate_track_ids"] == ["b"]
    assert s["identity_ambiguous"] is False


def test_second_last_tie_preserves_reference_q_but_identity_ambiguous():
    r = rank_opponents([row("a", 50), row("b", 45), row("c", 45), row("d", 30)])
    s = select_second_last(r)
    assert s["track_id"] is None
    assert set(s["candidate_track_ids"]) == {"b", "c"}
    assert s["goalward_q_m"] == 45


def test_less_than_two_returns_none():
    assert select_second_last(rank_opponents([row("a", 50)])) is None
