from pathlib import Path
import pytest

from ball_localization.evaluation.checkpointing import RecordCheckpointStore


def test_record_checkpoint_resume_and_identity_guard(tmp_path: Path):
    identity = {"benchmark": "x", "value": 1}
    store = RecordCheckpointStore(tmp_path / "run", "schema", identity, 2)
    store.mark_running()
    store.commit("a", {"value": 10})
    assert store.status()["completed_units"] == 1
    resumed = RecordCheckpointStore(tmp_path / "run", "schema", identity, 2)
    assert resumed.load_result("a")["value"] == 10
    with pytest.raises(RuntimeError):
        RecordCheckpointStore(tmp_path / "run", "schema", {"benchmark": "x", "value": 2}, 2)
