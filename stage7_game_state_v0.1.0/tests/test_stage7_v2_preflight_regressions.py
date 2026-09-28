from stage7_game_state.benchmark_v2 import _prf, _case_independent


def test_all_missed_referees_are_zero_f1():
    assert _prf(0, 0, 10)['f1'] == 0.0
    assert _prf(0, 4, 10)['f1'] == 0.0
    assert _prf(0, 0, 0)['f1'] is None


def test_string_false_cannot_certify_independent_gt():
    assert not _case_independent({'provenance': {'independent_gt': 'false'}})
    assert _case_independent({'provenance': {'independent_gt': True}})
