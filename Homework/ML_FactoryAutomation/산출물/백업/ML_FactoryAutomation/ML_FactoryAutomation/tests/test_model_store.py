"""DB 모델 스토어: 저장→조회→예측 테스트."""
from sqlalchemy.orm import sessionmaker
from src import db, model_store


def _session(tmp_path):
    eng = db.get_engine(f"sqlite:///{tmp_path}/t.db")
    db.init_db(eng)
    return sessionmaker(bind=eng, future=True)()


def test_save_load_predict_from_db(tmp_path):
    s = _session(tmp_path)
    mid = model_store.save_model_to_db(which="xgb", name="test-model",
                                       threshold=0.5, session=s)
    assert mid > 0
    _id, bundle = model_store.load_model_from_db(name="test-model", session=s)
    assert "model" in bundle and "scaler" in bundle and "columns" in bundle

    out = model_store.predict_from_db(
        {"type": "L", "air_temperature": 300.0, "process_temperature": 310.0,
         "rotational_speed": 1500, "torque": 40.0, "tool_wear": 120},
        name="test-model", session=s)
    assert out["source"] == "db"
    assert out["pred_label"] in (0, 1)
    assert 0.0 <= out["pred_proba"] <= 1.0
