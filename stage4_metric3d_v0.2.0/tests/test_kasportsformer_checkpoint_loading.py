import pickle

import pytest

from stage4_metric3d.initializers.kasportsformer_worker import (
    OFFICIAL_WORLDPOSE_DET_CHECKPOINT_SHA256,
    _load_checkpoint_payload,
)


def test_unknown_checkpoint_hash_never_uses_unsafe_pickle_fallback(tmp_path):
    checkpoint = tmp_path / "unknown.pth"
    checkpoint.write_bytes(b"not a real checkpoint")

    class _LoadTorch:
        @staticmethod
        def load(*args, **kwargs):
            if kwargs.get("weights_only") is True:
                raise pickle.UnpicklingError("unsupported numpy metadata")
            raise AssertionError("unsafe fallback must not be attempted")

    with pytest.raises(RuntimeError, match="refusing unsafe fallback"):
        _load_checkpoint_payload(
            _LoadTorch,
            checkpoint,
            "0" * 64,
        )


def test_official_hash_allows_numpy_metadata_fallback(tmp_path):
    checkpoint = tmp_path / "official.pth"
    checkpoint.write_bytes(b"fixture")
    calls = []

    class _LoadTorch:
        @staticmethod
        def load(*args, **kwargs):
            calls.append(kwargs.get("weights_only"))
            if kwargs.get("weights_only") is True:
                raise pickle.UnpicklingError("unsupported numpy metadata")
            return {"model": {"weight": 1}}

    payload = _load_checkpoint_payload(
        _LoadTorch,
        checkpoint,
        OFFICIAL_WORLDPOSE_DET_CHECKPOINT_SHA256,
    )
    assert payload == {"model": {"weight": 1}}
    assert calls == [True, False]


def test_official_hash_constant_is_pinned():
    assert OFFICIAL_WORLDPOSE_DET_CHECKPOINT_SHA256 == (
        "a4e0b9018e4755676a8421edbf8063375d7d22eb2f423f6aa4ba3883ee90a767"
    )
