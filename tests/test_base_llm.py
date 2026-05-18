import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from base_llm import DummyLLMVerifier, _normalize_decision, _parse_json_decision


def test_dummy_verifier_preserves_legacy_string_contract():
    verifier = DummyLLMVerifier()
    assert verifier.verify("name A", "name A") == "unsure"


def test_dummy_verifier_details_are_compatible():
    verifier = DummyLLMVerifier()
    decision = verifier.verify_with_details("name A", "name A")
    assert decision.decision == "unsure"
    assert decision.confidence == 0.0


def test_parse_json_decision_clamps_confidence():
    decision = _parse_json_decision('{"decision":"match","confidence":1.7,"reason":"same name"}')
    assert decision.decision == "match"
    assert decision.confidence == 1.0
    assert decision.reason == "same name"


def test_normalize_decision_aliases_non_match():
    assert _normalize_decision("not match") == "non-match"
    assert _normalize_decision("different") == "non-match"
    assert _normalize_decision("unexpected") == "unsure"
