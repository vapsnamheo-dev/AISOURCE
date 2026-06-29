# -*- coding: utf-8 -*-
"""
Proposal A — LLM 진단 보고서 생성
RAG(FAISS 유사사례) + ML 열화 확률 + DL RUL 예측 결과를
Claude API에 전달하여 자연어 정비 권고 보고서를 생성한다.

사용:
    from src.femto_llm_report import generate_report, generate_report_mock
    report = generate_report(sensor_values, ml_prob, ml_label, ..., rag_cases)
"""
from __future__ import annotations

import os
from typing import Any


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

보고서 형식 (4개 섹션, 총 300자 이내):
1. [현재 상태] 베어링 상태를 한 줄로 요약 (정상/주의/위험)
2. [주요 이상 신호] 센서값에서 관찰된 이상 패턴 2~3가지
3. [정비 권고] 구체적 조치 사항 (즉시/24시간 내/1주일 내)
4. [유사 사례 참고] 가장 유사한 과거 사례의 결과와 교훈

전문 용어보다 현장 언어로 작성하세요."""


def _build_context(
    sensor: dict[str, float],
    ml_prob: float,
    ml_label: int,
    ml_threshold: float,
    rul_min: float | None,
    rul_alarm_min: float,
    rag_cases: list[dict[str, Any]],
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

    return "\n".join(lines)


def generate_report(
    sensor: dict[str, float],
    ml_prob: float = 0.0,
    ml_label: int = 0,
    ml_threshold: float = 0.50,
    rul_min: float | None = None,
    rul_alarm_min: float = 60.0,
    rag_cases: list[dict[str, Any]] | None = None,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 600,
) -> str:
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
    rag_cases    : femto_rag_search.search() 반환값
    model        : Claude 모델 ID
    max_tokens   : 최대 출력 토큰

    Returns
    -------
    str: 자연어 진단 보고서
    """
    context = _build_context(
        sensor=sensor,
        ml_prob=ml_prob,
        ml_label=ml_label,
        ml_threshold=ml_threshold,
        rul_min=rul_min,
        rul_alarm_min=rul_alarm_min,
        rag_cases=rag_cases or [],
    )

    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )
    return response.content[0].text


def generate_report_mock(
    sensor: dict[str, float],
    ml_prob: float = 0.0,
    ml_label: int = 0,
    rul_min: float | None = None,
    rul_alarm_min: float = 60.0,
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

    return (
        f"[현재 상태] {status} — 열화 확률 {ml_prob:.1%}, 예측 잔여수명 {rul_str}\n\n"
        f"[주요 이상 신호]\n" +
        "\n".join(f"  • {a}" for a in anomalies) +
        f"\n\n[정비 권고] {action}\n\n"
        f"[유사 사례 참고] RAG 인덱스 기반 유사 베어링 사례 검색 결과를 참고하세요.\n\n"
        f"※ 이 보고서는 Mock 모드입니다. ANTHROPIC_API_KEY 설정 시 AI 보고서가 생성됩니다."
    )
