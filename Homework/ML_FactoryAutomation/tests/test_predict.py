# ════════════════════════════════════════════════════════════════════
# [역할] 추론 함수(확률 범위·반환 스키마) 단위 테스트.
# [단계] 테스트 (CI)
# (이 헤더는 이전/이번 작업 맥락을 요약한 주석입니다 — 기능에는 영향 없음)
# ════════════════════════════════════════════════════════════════════
from src import train, predict, config


def _ensure_models():
    if not config.MODEL_PATHS["xgb"].exists():
        train.train(save=True)


def test_predict_schema_and_range():
    _ensure_models()
    inp = {"type": "M", "air_temperature": 300.0, "process_temperature": 310.0,
           "rotational_speed": 1500, "torque": 40.0, "tool_wear": 100}
    res = predict.predict_one(inp)
    assert set(res) == {"pred_label", "pred_proba"}
    assert res["pred_label"] in (0, 1)
    assert 0.0 <= res["pred_proba"] <= 1.0


def test_high_stress_increases_risk():
    _ensure_models()
    low = {"type": "L", "air_temperature": 298.0, "process_temperature": 308.0,
           "rotational_speed": 1500, "torque": 30.0, "tool_wear": 0}
    high = {"type": "L", "air_temperature": 304.0, "process_temperature": 313.0,
            "rotational_speed": 1350, "torque": 70.0, "tool_wear": 250}
    assert predict.predict_one(high)["pred_proba"] >= predict.predict_one(low)["pred_proba"]
