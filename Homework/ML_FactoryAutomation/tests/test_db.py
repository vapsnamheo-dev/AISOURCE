# ════════════════════════════════════════════════════════════════════
# [역할] DB CRUD 단위 테스트.
# [단계] 테스트 (CI)
# (이 헤더는 이전/이번 작업 맥락을 요약한 주석입니다 — 기능에는 영향 없음)
# ════════════════════════════════════════════════════════════════════
from src import db


def test_crud_in_memory():
    engine = db.init_db(db.get_engine("sqlite:///:memory:"))
    with db.get_session(engine) as s:
        m = db.Machine(product_id="T1", type="M")
        s.add(m)
        s.flush()
        r = db.SensorReading(machine_id=m.machine_id, air_temperature=300,
                             process_temperature=310, rotational_speed=1500,
                             torque=40, tool_wear=100)
        s.add(r)
        s.flush()
        p = db.Prediction(reading_id=r.reading_id, pred_label=1, pred_proba=0.87)
        s.add(p)
        s.commit()
        assert p.prediction_id is not None
        assert s.get(db.FailureType, "TWF").name == "Tool Wear Failure"
