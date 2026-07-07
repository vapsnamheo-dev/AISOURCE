"""tests/test_femto_llm_guard.py — 환각 방어 3층 게이트 단위 테스트."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.femto_llm_guard import (
    KNOWN_STATUS_CLASSES,
    check_input_gate,
    confidence_tone,
    generate_report_guarded_mock,
    _validate_guarded_result,
)


NORMAL_SENSOR = {"h_rms": 0.3, "h_kurt": 3.1, "v_rms": 0.25, "temp_mean": 28.0}


# ── 2층: 입력 게이트 ──────────────────────────────────────────────────────
def test_input_gate_passes_normal_sensor():
    passed, reason = check_input_gate(NORMAL_SENSOR)
    assert passed is True
    assert reason == ""


def test_input_gate_blocks_impossible_rms():
    sensor = {**NORMAL_SENSOR, "h_rms": 999.0}
    passed, reason = check_input_gate(sensor)
    assert passed is False
    assert "h_rms" in reason
    assert "재측정" in reason


def test_input_gate_blocks_impossible_temperature():
    sensor = {**NORMAL_SENSOR, "temp_mean": -273.0}
    passed, reason = check_input_gate(sensor)
    assert passed is False
    assert "temp_mean" in reason


# ── 1층: 신뢰도 톤 분기 ───────────────────────────────────────────────────
def test_confidence_tone_assertive_far_from_threshold():
    assert confidence_tone(ml_prob=0.95, ml_threshold=0.50) == "assertive"
    assert confidence_tone(ml_prob=0.05, ml_threshold=0.50) == "assertive"


def test_confidence_tone_hedged_near_threshold():
    assert confidence_tone(ml_prob=0.52, ml_threshold=0.50) == "hedged"
    assert confidence_tone(ml_prob=0.55, ml_threshold=0.50, margin=0.10) == "hedged"


# ── 3층: 클래스 한정성 자기인지 — 근거 없으면 "판단불가" ────────────────
def test_guarded_mock_returns_unknown_when_no_evidence_and_ambiguous():
    result = generate_report_guarded_mock(
        sensor=NORMAL_SENSOR, ml_prob=0.51, ml_threshold=0.50,
        rul_min=None, rag_cases=None, doc_snippets=None,
    )
    assert result["status"] == "판단불가"
    assert "근거" in result["doc_basis"]


def test_guarded_mock_returns_normal_status_when_confident_and_no_evidence():
    result = generate_report_guarded_mock(
        sensor=NORMAL_SENSOR, ml_prob=0.05, ml_label=0, ml_threshold=0.50,
        rul_min=None, rag_cases=None, doc_snippets=None,
    )
    assert result["status"] in KNOWN_STATUS_CLASSES
    assert result["status"] != "판단불가"


def test_guarded_mock_blocks_ood_input_without_calling_llm():
    sensor = {**NORMAL_SENSOR, "h_rms": 999.0}
    result = generate_report_guarded_mock(sensor=sensor, ml_prob=0.9, ml_threshold=0.50)
    assert result["status"] == "판단불가"
    assert "입력 게이트" in result["action"]["description"]


# ── 근거 구조 감사: 스키마 검증 ──────────────────────────────────────────
def test_validate_guarded_result_accepts_valid_shape():
    valid = {
        "status": "정상", "anomalies": [],
        "action": {"urgency": "정기 점검", "description": "이상 없음"},
        "similar_case_note": "없음", "doc_basis": "해당 없음",
    }
    ok, err = _validate_guarded_result(valid)
    assert ok is True
    assert err == ""


def test_validate_guarded_result_rejects_unknown_status():
    invalid = {
        "status": "완전정상", "anomalies": [],
        "action": {"urgency": "정기 점검", "description": "이상 없음"},
        "similar_case_note": "없음", "doc_basis": "해당 없음",
    }
    ok, err = _validate_guarded_result(invalid)
    assert ok is False
    assert "status" in err


def test_validate_guarded_result_rejects_missing_field():
    invalid = {"status": "정상", "anomalies": []}
    ok, err = _validate_guarded_result(invalid)
    assert ok is False
    assert "필수 필드" in err
