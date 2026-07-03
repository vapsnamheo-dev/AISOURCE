# -*- coding: utf-8 -*-
"""
Proposal A — LLM 진단 보고서 생성
RAG(FAISS 유사사례) + ML 열화 확률 + DL RUL 예측 결과를
Claude API에 전달하여 자연어 정비 권고 보고서를 생성한다.

사용:
    from src.femto_llm_report import generate_report, generate_report_mock
    report = generate_report(sensor_values, ml_prob, ml_label, ..., rag_cases)

Structured Output (v0.5):
    from src.femto_llm_report import generate_report_structured, generate_report_structured_mock
    result = generate_report_structured(sensor_values, ml_prob, ml_label, ..., rag_cases)
    # result: {"status": "정상|주의|위험", "anomalies": [...], "action": {...},
    #          "similar_case_note": "...", "doc_basis": "..."}
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# claude-haiku-4-5 공식 단가(2026-06-24 기준, $/1M 토큰). 다른 모델을 쓰면 이 값은
# 참고용 추정치가 된다.
_HAIKU_INPUT_PRICE_PER_MTOK = 1.00
_HAIKU_OUTPUT_PRICE_PER_MTOK = 5.00


def _estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * _HAIKU_INPUT_PRICE_PER_MTOK
        + output_tokens / 1_000_000 * _HAIKU_OUTPUT_PRICE_PER_MTOK
    )


def _get_client():
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("anthropic 패키지 필요: pip install anthropic") from e

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.\n"
            ".env 파일 또는 터미널에서 설정 후 재시작하세요:\n"
            "  set ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=api_key)


SYSTEM_PROMPT = """당신은 베어링 설비 예지보전(PdM) 전문가입니다.
센서 데이터, ML 분류 결과, DL 잔여수명(RUL) 예측, 과거 유사 사례를 종합하여
현장 정비 담당자가 즉시 이해하고 조치할 수 있는 진단 보고서를 작성합니다.

보고서 형식 (5개 섹션, 총 350자 이내):
1. [현재 상태] 베어링 상태를 한 줄로 요약 (정상/주의/위험)
2. [주요 이상 신호] 센서값에서 관찰된 이상 패턴 2~3가지
3. [정비 권고] 구체적 조치 사항 (즉시/24시간 내/1주일 내) — [정비 지식 문서 근거]에
   해당 조치의 근거가 있으면 반드시 인용하세요.
4. [유사 사례 참고] 가장 유사한 과거 사례의 결과와 교훈
5. [문서 근거] [정비 지식 문서 근거] 섹션 중 이번 판단에 사용한 내용을 1줄로 요약.
   근거가 없으면 "해당 없음"이라고 쓰세요.

전문 용어보다 현장 언어로 작성하세요."""


def _build_context(
    sensor: dict[str, float],
    ml_prob: float,
    ml_label: int,
    ml_threshold: float,
    rul_min: float | None,
    rul_alarm_min: float,
    rag_cases: list[dict[str, Any]],
    doc_snippets: list[str] | None = None,
) -> str:
    lines = ["=== 베어링 진단 요청 ===\n"]

    lines.append("[현재 센서 측정값]")
    key_sensors = ["h_rms", "h_kurt", "v_rms", "temp_mean"]
    for k in key_sensors:
        if k in sensor:
            lines.append(f"  {k:12s} = {sensor[k]:.4f}")

    lines.append(f"\n[ML 열화 분류]")
    status = "열화(이상)" if ml_label == 1 else "정상"
    lines.append(f"  상태: {status}  (열화 확률 {ml_prob:.1%} / 임계값 {ml_threshold:.2f})")

    lines.append(f"\n[DL 잔여수명(RUL) 예측]")
    if rul_min is not None and rul_min > 0:
        urgency = "즉시 점검" if rul_min < rul_alarm_min else "정상 범위"
        lines.append(f"  예측 RUL: {rul_min:.0f}분  (경보 기준: {rul_alarm_min:.0f}분) → {urgency}")
    else:
        lines.append("  RUL 예측 불가 (DL 모델 미로드)")

    lines.append(f"\n[과거 유사 사례 Top-{min(3, len(rag_cases))}]")
    if rag_cases:
        for r in rag_cases[:3]:
            rul_str = f"RUL={r['rul']:.0f}분" if r.get("rul") else "RUL미상"
            status_str = "열화" if r.get("label") == 1 else "정상"
            lines.append(
                f"  {r['rank']}위: {r['bearing']} "
                f"유사도={r['similarity']:.1f}%  {rul_str}  상태={status_str}"
            )
    else:
        lines.append("  유사 사례 없음 (RAG 인덱스 미구축)")

    lines.append(f"\n[정비 지식 문서 근거 (RAG-Level2)]")
    if doc_snippets:
        for i, snippet in enumerate(doc_snippets, 1):
            lines.append(f"  ({i}) {snippet.strip()}")
    else:
        lines.append("  문서 근거 없음 (RAG-Level2 인덱스 미구축 또는 미조회)")

    return "\n".join(lines)


def generate_report(
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
    return_usage: bool = False,
) -> str | tuple[str, dict[str, Any]]:
    """
    LLM 진단 보고서 생성.

    Parameters
    ----------
    sensor       : {"h_rms": 1.2, "h_kurt": 3.1, ...}
    ml_prob      : ML 열화 확률 (0~1)
    ml_label     : ML 분류 결과 (0=정상, 1=열화)
    ml_threshold : ML 판정 임계값
    rul_min      : DL 예측 잔여수명 (분, None이면 미예측)
    rul_alarm_min: DL 경보 기준 (분)
    rag_cases    : femto_rag_search.search() 반환값 (RAG-Level1, 수치 유사사례)
    doc_snippets : femto_doc_rag.retrieve_docs() 반환값 (RAG-Level2, 정비 지식 문서)
    model        : Claude 모델 ID
    max_tokens   : 최대 출력 토큰
    return_usage : True면 (보고서, usage딕셔너리) 튜플을 반환한다.
                   usage = {"input_tokens", "output_tokens", "cost_usd"}

    Returns
    -------
    str, 또는 return_usage=True일 때 tuple[str, dict]
    """
    context = _build_context(
        sensor=sensor,
        ml_prob=ml_prob,
        ml_label=ml_label,
        ml_threshold=ml_threshold,
        rul_min=rul_min,
        rul_alarm_min=rul_alarm_min,
        rag_cases=rag_cases or [],
        doc_snippets=doc_snippets,
    )

    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )
    text = response.content[0].text

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost_usd = _estimate_cost_usd(input_tokens, output_tokens)
    logger.info(
        "LLM 보고서 생성 완료 model=%s input_tokens=%d output_tokens=%d cost_usd=%.5f",
        model, input_tokens, output_tokens, cost_usd,
    )

    if return_usage:
        return text, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        }
    return text


def generate_report_mock(
    sensor: dict[str, float],
    ml_prob: float = 0.0,
    ml_label: int = 0,
    rul_min: float | None = None,
    rul_alarm_min: float = 60.0,
    doc_snippets: list[str] | None = None,
) -> str:
    """API 키 없이 규칙 기반 Mock 보고서 생성 (데모용)."""
    status = (
        "위험" if ml_label == 1 and (rul_min or 999) < rul_alarm_min
        else "주의" if ml_label == 1
        else "정상"
    )

    anomalies = []
    if sensor.get("h_rms", 0) > 0.5:
        anomalies.append(f"수평 진동 RMS {sensor['h_rms']:.4f} 상승 (정상 기준 0.5 이하)")
    if sensor.get("h_kurt", 0) > 5.0:
        anomalies.append(f"수평 첨도 {sensor['h_kurt']:.2f} 급등 (베어링 충격 증가)")
    if sensor.get("temp_mean", 0) > 40:
        anomalies.append(f"온도 {sensor['temp_mean']:.1f}°C 상승 (윤활 부족 가능성)")
    if not anomalies:
        anomalies.append("센서값 정상 범위 내")

    if ml_label == 1 and (rul_min or 999) < rul_alarm_min:
        action = "즉시 점검 및 교체 준비 필요"
    elif ml_label == 1:
        action = "24시간 내 정밀 점검 권고"
    else:
        action = "정기 점검 일정대로 유지"

    rul_str = f"{rul_min:.0f}분" if rul_min else "측정 불가"

    doc_line = (
        f"  • {doc_snippets[0].strip()[:80]}..."
        if doc_snippets else "  • 문서 근거 없음 (RAG-Level2 인덱스 미구축)"
    )

    return (
        f"[현재 상태] {status} — 열화 확률 {ml_prob:.1%}, 예측 잔여수명 {rul_str}\n\n"
        f"[주요 이상 신호]\n" +
        "\n".join(f"  • {a}" for a in anomalies) +
        f"\n\n[정비 권고] {action}\n\n"
        f"[유사 사례 참고] RAG 인덱스 기반 유사 베어링 사례 검색 결과를 참고하세요.\n\n"
        f"[문서 근거]\n{doc_line}\n\n"
        f"※ 이 보고서는 Mock 모드입니다. ANTHROPIC_API_KEY 설정 시 AI 보고서가 생성됩니다."
    )


STRUCTURED_SYSTEM_PROMPT = """당신은 베어링 설비 예지보전(PdM) 전문가입니다.
센서 데이터, ML 분류 결과, DL 잔여수명(RUL) 예측, 과거 유사 사례를 종합하여
현장 정비 담당자가 시스템에서 바로 활용할 수 있는 정형 진단 결과를 산출합니다.
전문 용어보다 현장 언어로 작성하고, 각 필드는 간결하게 채우세요."""

STRUCTURED_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["정상", "주의", "위험"],
            "description": "베어링 현재 상태",
        },
        "anomalies": {
            "type": "array",
            "items": {"type": "string"},
            "description": "센서값에서 관찰된 이상 패턴 (없으면 빈 배열)",
        },
        "action": {
            "type": "object",
            "properties": {
                "urgency": {
                    "type": "string",
                    "enum": ["즉시", "24시간 내", "1주일 내", "정기 점검"],
                },
                "description": {"type": "string"},
            },
            "required": ["urgency", "description"],
            "additionalProperties": False,
        },
        "similar_case_note": {
            "type": "string",
            "description": "가장 유사한 과거 사례의 결과와 교훈 (RAG-Level1 근거)",
        },
        "doc_basis": {
            "type": "string",
            "description": "정비 지식 문서 근거 요약 (RAG-Level2). 근거 없으면 '해당 없음'",
        },
    },
    "required": ["status", "anomalies", "action", "similar_case_note", "doc_basis"],
    "additionalProperties": False,
}


def generate_report_structured(
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
) -> dict[str, Any]:
    """
    LLM 진단 결과를 JSON Schema로 강제된 정형(Structured Output) 형태로 생성한다.

    generate_report()와 입력 파라미터는 동일하지만, 자유 텍스트 대신
    {"status", "anomalies", "action", "similar_case_note", "doc_basis"} 구조의
    dict를 반환한다. 다운스트림 시스템(대시보드·알람·DB 저장) 연동에 적합하다.

    Returns
    -------
    dict: STRUCTURED_REPORT_SCHEMA를 따르는 진단 결과
    """
    context = _build_context(
        sensor=sensor,
        ml_prob=ml_prob,
        ml_label=ml_label,
        ml_threshold=ml_threshold,
        rul_min=rul_min,
        rul_alarm_min=rul_alarm_min,
        rag_cases=rag_cases or [],
        doc_snippets=doc_snippets,
    )

    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=STRUCTURED_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
        output_config={"format": {"type": "json_schema", "schema": STRUCTURED_REPORT_SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def generate_report_structured_mock(
    sensor: dict[str, float],
    ml_prob: float = 0.0,
    ml_label: int = 0,
    rul_min: float | None = None,
    rul_alarm_min: float = 60.0,
    doc_snippets: list[str] | None = None,
) -> dict[str, Any]:
    """API 키 없이 규칙 기반 Mock 정형 진단 결과 생성 (데모용)."""
    status = (
        "위험" if ml_label == 1 and (rul_min or 999) < rul_alarm_min
        else "주의" if ml_label == 1
        else "정상"
    )

    anomalies = []
    if sensor.get("h_rms", 0) > 0.5:
        anomalies.append(f"수평 진동 RMS {sensor['h_rms']:.4f} 상승 (정상 기준 0.5 이하)")
    if sensor.get("h_kurt", 0) > 5.0:
        anomalies.append(f"수평 첨도 {sensor['h_kurt']:.2f} 급등 (베어링 충격 증가)")
    if sensor.get("temp_mean", 0) > 40:
        anomalies.append(f"온도 {sensor['temp_mean']:.1f}°C 상승 (윤활 부족 가능성)")

    if ml_label == 1 and (rul_min or 999) < rul_alarm_min:
        urgency, description = "즉시", "즉시 점검 및 교체 준비 필요"
    elif ml_label == 1:
        urgency, description = "24시간 내", "24시간 내 정밀 점검 권고"
    else:
        urgency, description = "정기 점검", "정기 점검 일정대로 유지"

    rul_str = f"{rul_min:.0f}분" if rul_min else "측정 불가"

    return {
        "status": status,
        "anomalies": anomalies,
        "action": {"urgency": urgency, "description": description},
        "similar_case_note": (
            f"예측 잔여수명 {rul_str} — RAG 인덱스 기반 유사 베어링 사례 검색 결과를 참고하세요."
        ),
        "doc_basis": (
            doc_snippets[0].strip()[:80] if doc_snippets else "해당 없음 (RAG-Level2 인덱스 미구축)"
        ),
    }
