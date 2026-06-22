"""데모 일괄 검증(predict_dataframe_from_db)과 물리 실제라벨 테스트."""
from src import db, model_store, synth_ai4i


def test_actual_label_physics():
    # 정상 / 과부하(OSF) / 열누적(HDF)
    assert synth_ai4i.actual_label("M", 300, 310, 1500, 40, 100) == 0
    assert synth_ai4i.actual_label("L", 300, 310, 1400, 60, 215) == 1
    assert synth_ai4i.actual_label("H", 303, 309, 1300, 45, 120) == 1


def test_batch_predict_from_db(tmp_path):
    url = f"sqlite:///{tmp_path / 'b.db'}"
    eng = db.init_db(db.get_engine(url))
    with db.get_session(eng) as s:
        model_store.save_model_to_db("xgb", session=s)
    demo = synth_ai4i.generate(200, seed=99)
    with db.get_session(eng) as s:
        out, mid, has_actual = model_store.predict_dataframe_from_db(demo, session=s)
    assert has_actual and mid >= 1
    assert {"pred_label", "pred_proba", "actual", "match"} <= set(out.columns)
    assert len(out) == 200
    acc = (out["pred_label"] == out["actual"]).mean()
    assert acc > 0.9  # 동일 물리분포이므로 높은 정확도
