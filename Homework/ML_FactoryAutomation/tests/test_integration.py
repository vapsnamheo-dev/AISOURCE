# ════════════════════════════════════════════════════════════════════
# [역할] CSV→전처리→학습→예측→DB 저장 end-to-end 통합 테스트.
# [단계] 테스트 (CI)
# [작업 메모] 임시 SQLite 격리.
# (이 헤더는 이전/이번 작업 맥락을 요약한 주석입니다 — 기능에는 영향 없음)
# ════════════════════════════════════════════════════════════════════
"""end-to-end: 로드 -> 전처리 -> 학습 -> 예측 -> DB 저장."""
from src import train, predict, db


def test_end_to_end(tmp_path):
    # 학습
    out = train.train(save=True)
    assert out["xgb"] is not None
    # 격리된 임시 DB
    engine = db.init_db(db.get_engine(f"sqlite:///{tmp_path/'t.db'}"))
    inp = {"type": "H", "air_temperature": 302.0, "process_temperature": 312.0,
           "rotational_speed": 1400, "torque": 55.0, "tool_wear": 200,
           "product_id": "ITG1"}
    with db.get_session(engine) as s:
        res = predict.predict_and_log(inp, s)
        assert res["prediction_id"] is not None
        cnt = s.query(db.Prediction).count()
    assert cnt == 1
