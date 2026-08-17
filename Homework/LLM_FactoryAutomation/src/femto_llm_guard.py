# -*- coding: utf-8 -*-
"""
Proposal A — LLM 환각(hallucination) 방어 3층 게이트.

1층 신뢰도 톤 분기      : ml_prob가 판정 임계값에서 먼 경우(고확신)는 단정적 어투,
                          가까운 경우(애매)는 확인 권유 어투로 시스템 프롬프트를 분기한다.
2층 입력 게이트          : 센서값이 물리적으로 불가능한 범위(OOD)면 LLM 호출 없이
                          차단하고 재측정을 요청한다 — 잘못된 입력에 대한 그럴듯한
                          답변(환각)을 원천 차단.
3층 클래스 한정성 자기인지 : 판정 가능 범위를 정상/주의/위험/판단불가 4종으로 고정하고,
                          근거(RAG 사례·문서·RUL)가 불충분하면 추측 대신 "판단불가"를
                          답하도록 프롬프트·스키마 검증으로 강제한다.

+ 근거 구조 감사: 구조화 응답이 스키마를 위반하면 1회 재시도하고, 그래도 실패하면
  추측 대신 "판단불가"로 안전하게 폴백한다.

사용:
    from src.femto_llm_guard import generate_report_guarded, generate_report_guarded_mock
    result = generate_report_guarded(sensor, ml_prob, ml_label, ..., rag_cases=..., doc_snippets=...)
"""
from __future__ import annotations

import copy
import json
import logging
from typing import Any

from src.femto_llm_report import (
    STRUCTURED_REPORT_SCHEMA,
    STRUCTURED_SYSTEM_PROMPT,
    _build_context,
    _estimate_cost_usd,
    _get_client,
    generate_report_structured_mock,
)

logger = logging.getLogger(__name__)

# ── 2층: 입력 게이트 — 물리적으로 불가능한 범위(OOD) ─────────────────────────
# FEMTO-ST 실측 데이터(정상~고장 직전 구간)보다 넉넉한 한계값. 이 범위를 벗어나면
# 센서 오류·단위 실수(g ↔ mm/s 등)·다른 장비 데이터일 가능성이 높다고 판단한다.
SENSOR_PHYSICAL_RANGE: dict[str, tuple[float, float]] = {
    "h_rms": (0.0, 5.0),
    "v_rms": (0.0, 5.0),
    "h_kurt": (0.0, 50.0),
    "temp_mean": (-10.0, 100.0),
}

# ── 3층: 클래스 한정성 자기인지 — 판정 가능 범위 고정 ────────────────────────
KNOWN_STATUS_CLASSES = ["정상", "주의", "위험", "판단불가"]


# ── 종합 판정 — ML/DL/RAG 신호를 규칙으로 미리 합쳐 LLM에 앵커로 제공 ────────────
def combined_verdict(
    ml_label: int,
    rul_min: float | None,
    rul_alarm_min: float,
    rul_reliable: bool,
) -> tuple[str, str]:
    """ML 분류 + (신뢰 가능한 경우만) DL RUL을 규칙 기반으로 하나의 판정으로 합친다.

    화면에 "ML=정상"과 "RUL=긴급"이 동시에 떠서 결론 없이 사용자를 혼란스럽게
    하는 문제를 풀기 위한 것 — LLM에게 최종 결론을 맡기지 않고 코드가 결정론적으로
    정하며, 그 결과를 LLM에게 "이 판정과 일치하는 status로 답하라"는 앵커로 준다.
    RUL이 신뢰 불가(rul_reliable=False, 예: 단일 스냅샷 반복 입력)면 RUL은 판정에서
    제외하고 ML 결과만 사용한다 — 신뢰할 수 없는 신호를 종합에 섞으면 종합 결과 자체가
    오염되기 때문이다.

    Returns
    -------
    (verdict, reason) : verdict는 KNOWN_STATUS_CLASSES 중 하나("판단불가" 제외 3종),
                         reason은 그 판정에 이른 근거를 사람이 읽을 문장으로 설명.
    """
    rul_urgent = rul_reliable and rul_min is not None and rul_min <= rul_alarm_min

    if ml_label == 1 and rul_urgent:
        return "위험", "ML 열화 감지와 DL 잔여수명(RUL) 임계 이하가 서로 일치 — 두 모델이 같은 결론"
    if ml_label == 1:
        rul_note = "RUL은 신뢰 불가(저신뢰)로 판정에서 제외" if not rul_reliable else "RUL은 양호"
        return "주의", f"ML 열화 감지({rul_note})"
    if rul_urgent:
        return "주의", "ML은 정상이나 DL 잔여수명(RUL)이 긴급 기준 이하 — 두 신호가 불일치, 재측정 권고"
    rul_note = "(RUL은 저신뢰로 판정 제외)" if not rul_reliable else ""
    return "정상", f"ML 정상{rul_note}"


def _with_unknown_status(schema: dict[str, Any]) -> dict[str, Any]:
    guarded = copy.deepcopy(schema)
    guarded["properties"]["status"]["enum"] = KNOWN_STATUS_CLASSES
    return guarded


GUARDED_REPORT_SCHEMA: dict[str, Any] = _with_unknown_status(STRUCTURED_REPORT_SCHEMA)

GUARDED_SYSTEM_PROMPT_SUFFIX = """

[환각 방어 규칙 — 반드시 준수]
- 판정 가능 범위는 정상/주의/위험/판단불가 4종으로 고정되어 있습니다.
- 유사 사례(RAG-Level1)와 문서 근거(RAG-Level2)가 모두 없고 RUL도 예측 불가한
  상태에서는 절대 추측하지 말고 status를 "판단불가"로 답하세요.
- "판단불가"일 때 doc_basis에는 반드시 무엇이 부족해 판단할 수 없는지 명시하세요.
- 모르는 것을 아는 것처럼 답하는 것보다, 모른다고 답하는 것이 항상 낫습니다."""


def check_input_gate(sensor: dict[str, float]) -> tuple[bool, str]:
    """센서 입력이 물리적으로 타당한 범위인지 확인한다 (2층 입력 게이트).

    Returns
    -------
    (passed, reason) : passed=False면 reason에 차단 사유(재측정 안내 포함)를 담는다.
    """
    violations = []
    for key, (lo, hi) in SENSOR_PHYSICAL_RANGE.items():
        if key not in sensor:
            continue
        val = sensor[key]
        if val < lo or val > hi:
            violations.append(f"{key}={val:.4f} (허용범위 {lo}~{hi})")

    if violations:
        reason = (
            "입력 게이트 차단 — 센서값이 물리적으로 불가능한 범위입니다: "
            + ", ".join(violations)
            + ". 센서 배선·단위(g/mm/s 등)·장비 매칭을 확인 후 재측정하세요."
        )
        return False, reason
    return True, ""


def confidence_tone(ml_prob: float, ml_threshold: float, margin: float = 0.15) -> str:
    """ml_prob와 판정 임계값의 거리로 서술 톤을 분기한다 (1층 신뢰도 톤 분기).

    margin 이상 떨어져 있으면 "assertive"(단정), 그 이내면 "hedged"(확인 권유).
    """
    return "assertive" if abs(ml_prob - ml_threshold) >= margin else "hedged"


def _tone_instruction(tone: str) -> str:
    if tone == "assertive":
        return "이번 판정은 임계값과 충분히 떨어진 고확신 구간입니다. 단정적으로 상태를 서술하세요."
    return (
        "이번 판정은 임계값 부근의 애매한 구간입니다. 단정적으로 말하지 말고 "
        "'추가 확인을 권장합니다' 같은 확인 유도 표현을 반드시 포함하세요."
    )


def _validate_guarded_result(result: Any) -> tuple[bool, str]:
    if not isinstance(result, dict):
        return False, "결과가 dict가 아님"
    missing = [k for k in GUARDED_REPORT_SCHEMA["required"] if k not in result]
    if missing:
        return False, f"필수 필드 누락: {missing}"
    if result["status"] not in KNOWN_STATUS_CLASSES:
        return False, f"알 수 없는 status 값: {result['status']}"
    action = result.get("action")
    if not isinstance(action, dict) or "urgency" not in action or "description" not in action:
        return False, "action 필드 구조 오류(urgency/description 필요)"
    return True, ""


def _blocked_result(reason: str) -> dict[str, Any]:
    return {
        "status": "판단불가",
        "anomalies": [],
        "action": {"urgency": "정기 점검", "description": reason},
        "similar_case_note": "입력 게이트 차단으로 LLM 호출 생략",
        "doc_basis": "해당 없음",
    }


def _fallback_result(reason: str) -> dict[str, Any]:
    return {
        "status": "판단불가",
        "anomalies": [],
        "action": {
            "urgency": "정기 점검",
            "description": f"LLM 응답 검증 실패({reason}) — 재시도 후에도 신뢰할 수 있는 "
                            f"판정을 얻지 못해 추측 대신 판단을 보류합니다.",
        },
        "similar_case_note": "해당 없음",
        "doc_basis": "해당 없음",
    }


def generate_report_guarded(
    sensor: dict[str, float],
    ml_prob: float = 0.0,
    ml_label: int = 0,
    ml_threshold: float = 0.50,
    rul_min: float | None = None,
    rul_alarm_min: float = 60.0,
    rag_cases: list[dict[str, Any]] | None = None,
    doc_snippets: list[str] | None = None,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 600,
    max_retries: int = 1,
    return_usage: bool = False,
    combined_verdict: str | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """환각 방어 3층 게이트를 적용한 LLM 진단 (Structured Output).

    1) 입력 게이트(2층): 센서값이 OOD면 LLM 호출 없이 차단.
    2) 신뢰도 톤 분기(1층): 임계값 근접도에 따라 단정/확인유도 어투 지시.
    3) 클래스 한정성 자기인지(3층): 근거 불충분 시 "판단불가" 강제.
    4) 근거 구조 감사: 스키마 위반 시 max_retries회 재시도, 그래도 실패하면
       추측 대신 "판단불가" 폴백을 반환한다.

    combined_verdict: 호출부(streamlit_femto.py)가 combined_verdict()로 미리 계산한
    ML+DL 종합 판정("정상"/"주의"/"위험"). 주어지면 컨텍스트에 앵커로 추가해 LLM의
    status 필드가 화면 상단 종합 판정 배지와 다른 결론을 내지 않도록 강제한다
    (LLM에게 종합 판단 자체를 맡기지 않고, 코드가 이미 정한 값을 따르게 함 — 3층
    클래스 한정성과 같은 원리, "판단불가"만 예외로 이 앵커보다 우선한다).

    return_usage=True면 (결과, usage딕셔너리) 튜플을 반환한다. usage는 재시도로
    발생한 모든 API 호출의 토큰을 합산한 값이다(입력 게이트 차단으로 API를
    아예 안 부른 경우는 0으로 채운다) — generate_report()의 return_usage 계약과
    동일한 형태로 맞춰 Streamlit 쪽 비용 캡션을 그대로 재사용할 수 있게 한다.
    """
    def _usage_dict(input_tokens: int, output_tokens: int) -> dict[str, Any]:
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": _estimate_cost_usd(input_tokens, output_tokens),
        }

    passed, reason = check_input_gate(sensor)
    if not passed:
        blocked = _blocked_result(reason)
        return (blocked, _usage_dict(0, 0)) if return_usage else blocked

    tone = confidence_tone(ml_prob, ml_threshold)
    context = _build_context(
        sensor=sensor, ml_prob=ml_prob, ml_label=ml_label, ml_threshold=ml_threshold,
        rul_min=rul_min, rul_alarm_min=rul_alarm_min, rag_cases=rag_cases or [],
        doc_snippets=doc_snippets,
    )
    context += f"\n\n[서술 톤 지시]\n{_tone_instruction(tone)}"
    if combined_verdict:
        context += (
            f"\n\n[종합 판정 — status는 반드시 이 값과 일치시키세요]\n"
            f"ML·DL 신호를 규칙 기반으로 미리 종합한 결과는 \"{combined_verdict}\"입니다. "
            f"근거가 명백히 부족해 3층 규칙상 \"판단불가\"를 써야 하는 경우가 아니라면, "
            f"status를 반드시 \"{combined_verdict}\"로 답하세요."
        )

    system_prompt = STRUCTURED_SYSTEM_PROMPT + GUARDED_SYSTEM_PROMPT_SUFFIX
    client = _get_client()

    last_error = "알 수 없는 오류"
    total_input_tokens = 0
    total_output_tokens = 0
    for _attempt in range(max_retries + 1):
        response = client.messages.create(
            model=model, max_tokens=max_tokens, system=system_prompt,
            messages=[{"role": "user", "content": context}],
            output_config={"format": {"type": "json_schema", "schema": GUARDED_REPORT_SCHEMA}},
        )
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        text = next(b.text for b in response.content if b.type == "text")
        try:
            result = json.loads(text)
        except json.JSONDecodeError as e:
            last_error = f"JSON 파싱 실패: {e}"
            context += f"\n\n[재시도 지시] 이전 응답이 유효한 JSON이 아니었습니다: {last_error}. 스키마를 정확히 따르세요."
            continue

        ok, err = _validate_guarded_result(result)
        if ok:
            return (result, _usage_dict(total_input_tokens, total_output_tokens)) if return_usage else result
        last_error = err
        context += f"\n\n[재시도 지시] 이전 응답이 스키마를 위반했습니다: {err}. 다시 스키마에 맞게 답하세요."

    logger.warning("구조화 검증 %d회 실패 — 판단불가로 폴백: %s", max_retries + 1, last_error)
    fallback = _fallback_result(last_error)
    return (fallback, _usage_dict(total_input_tokens, total_output_tokens)) if return_usage else fallback


def generate_report_guarded_mock(
    sensor: dict[str, float],
    ml_prob: float = 0.0,
    ml_label: int = 0,
    ml_threshold: float = 0.50,
    rul_min: float | None = None,
    rul_alarm_min: float = 60.0,
    rag_cases: list[dict[str, Any]] | None = None,
    doc_snippets: list[str] | None = None,
    combined_verdict: str | None = None,
) -> dict[str, Any]:
    """API 키 없이 게이트 로직만 검증하는 Mock (데모·테스트용).

    실제 LLM 호출 없이 1층·2층·3층 게이트 로직을 그대로 통과시켜, 근거가
    부족할 때 "판단불가"를 반환하는 흐름을 확인할 수 있다.

    combined_verdict가 주어지면(generate_report_guarded()와 동일 계약) "판단불가"로
    빠지는 경우를 제외하고 status를 그 값으로 덮어써, Mock 모드에서도 화면 상단
    종합 판정 배지와 보고서 본문이 다른 결론을 내지 않게 한다.
    """
    passed, reason = check_input_gate(sensor)
    if not passed:
        return _blocked_result(reason)

    has_evidence = bool(rag_cases) or bool(doc_snippets) or (rul_min is not None and rul_min > 0)
    tone = confidence_tone(ml_prob, ml_threshold)

    if not has_evidence and tone == "hedged":
        return {
            "status": "판단불가",
            "anomalies": [],
            "action": {
                "urgency": "정기 점검",
                "description": "임계값 부근의 애매한 확률이며 유사사례·문서·RUL 근거가 "
                                "전혀 없어 추측 대신 판단을 보류합니다.",
            },
            "similar_case_note": "근거 없음",
            "doc_basis": "근거 없음 — RAG·RUL 데이터 부족",
        }

    result = generate_report_structured_mock(
        sensor=sensor, ml_prob=ml_prob, ml_label=ml_label,
        rul_min=rul_min, rul_alarm_min=rul_alarm_min, doc_snippets=doc_snippets,
    )
    if tone == "hedged":
        result["action"]["description"] += " (확인 권장 — 임계값 인근 애매 구간)"
    if combined_verdict and result["status"] != "판단불가":
        result["status"] = combined_verdict
    return result
