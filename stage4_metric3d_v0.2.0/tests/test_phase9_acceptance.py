from tools.stage4_phase9_acceptance import build_acceptance


def test_phase9_defers_accuracy_when_assets_are_missing():
    result = build_acceptance({"acceptance": {
        "implementation_gate": "PASS_IMPLEMENTATION",
        "model_only_pretrained_inference": "BLOCKED_MISSING_STAGE4_MODEL_ASSETS",
        "model_only_accuracy": "NOT_EVALUATED_NO_INDEPENDENT_BENCHMARK_RECORDED",
        "accuracy_claim_allowed": False,
        "research_accuracy_frozen": False,
    }})
    assert result["decision"] == "DEFER_ACCURACY_CLAIM"
    assert result["release_candidate"] is False


def test_phase9_rejects_failed_implementation():
    result = build_acceptance({"acceptance": {
        "implementation_gate": "NOT_EVALUATED",
        "model_only_pretrained_inference": "READY",
        "model_only_accuracy": "EVALUATED",
        "accuracy_claim_allowed": True,
        "research_accuracy_frozen": True,
    }})
    assert result["decision"] == "REJECT_IMPLEMENTATION"
