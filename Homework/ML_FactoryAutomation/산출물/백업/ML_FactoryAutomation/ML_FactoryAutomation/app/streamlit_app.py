"""Streamlit 예측 웹앱: 단건 예측(실제 일치 표시) + CSV 일괄 검증(실제 vs 예측)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from src import db, predict, model_store, synth_ai4i

st.set_page_config(page_title="설비 고장 예측 (PdM-Guard)", page_icon="🛠️", layout="wide")


@st.cache_resource
def _setup():
    engine = db.init_db()
    artifacts = predict.load_artifacts("xgb")
    model_store.ensure_model_in_db("xgb")  # DB에 활성 모델 보장(일괄 예측에서 사용)
    return engine, artifacts


engine, artifacts = _setup()

st.title("🛠️ 설비 고장 예측 (PdM-Guard)")
st.caption("센서값으로 XGBoost가 고장 확률을 예측합니다. (라벨: 0=정상, 1=고장)")

with st.sidebar:
    st.header("센서 데이터 입력 (단건)")
    type_ = st.selectbox("제품 등급 (Type)", ["L", "M", "H"], index=1)
    air = st.slider("공기 온도 [K]", 295.0, 305.0, 300.0, 0.1)
    proc = st.slider("공정 온도 [K]", 305.0, 314.0, 310.0, 0.1)
    rpm = st.slider("회전속도 [rpm]", 1160, 2890, 1500, 1)
    torque = st.slider("토크 [Nm]", 3.0, 77.0, 40.0, 0.1)
    wear = st.slider("공구 마모 [min]", 0, 260, 100, 1)
    go = st.button("고장 예측하기", type="primary")

tab1, tab2 = st.tabs(["🔮 단건 예측", "📁 CSV 일괄 검증"])

with tab1:
    c1, c2 = st.columns([1, 1])
    if go:
        inp = {"type": type_, "air_temperature": air, "process_temperature": proc,
               "rotational_speed": rpm, "torque": torque, "tool_wear": wear}
        with db.get_session(engine) as s:
            res = predict.predict_and_log(inp, s, artifacts=artifacts)
        actual = synth_ai4i.actual_label(type_, air, proc, rpm, torque, wear)
        with c1:
            st.subheader("예측 결과")
            st.metric("고장 확률", f"{res['pred_proba']*100:.1f} %")
            if res["pred_label"] == 1:
                st.error("⚠️ 예측: 고장 위험 (점검 권장)")
            else:
                st.success("✅ 예측: 정상 범위")
            st.markdown("**실제(물리규칙) 판정 vs 예측**")
            st.write(f"- 실제 결과(물리규칙): **{'고장' if actual == 1 else '정상'}**")
            st.write(f"- 모델 예측: **{'고장' if res['pred_label'] == 1 else '정상'}**")
            if res["pred_label"] == actual:
                st.success("🎯 예측 일치 (정답)")
            else:
                st.warning("❌ 예측 불일치 (오답)")
        with c2:
            st.subheader("최근 예측 이력")
            try:
                with engine.connect() as conn:
                    h = pd.read_sql("SELECT prediction_id, pred_label, pred_proba, predicted_at "
                                    "FROM prediction ORDER BY prediction_id DESC LIMIT 10", conn)
                st.dataframe(h, width="stretch")
            except Exception as e:
                st.info(f"이력 없음: {e}")
    else:
        st.info("좌측 사이드바에서 센서값을 입력하고 '고장 예측하기'를 누르세요.")

with tab2:
    st.subheader("CSV 일괄 검증 — DB 모델로 예측 후 실제 결과와 비교")
    st.caption("demo data/demo_1000.csv 처럼 Type·센서값(+Target 실제라벨)이 든 CSV를 업로드하세요.")
    up = st.file_uploader("CSV 파일 선택", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        out, model_id, has_actual = model_store.predict_dataframe_from_db(df)
        st.write(f"총 **{len(out):,}건** 예측 완료 (DB 활성 모델 id={model_id}).")
        if has_actual:
            tp = int(((out["pred_label"] == 1) & (out["actual"] == 1)).sum())
            tn = int(((out["pred_label"] == 0) & (out["actual"] == 0)).sum())
            fp = int(((out["pred_label"] == 1) & (out["actual"] == 0)).sum())
            fn = int(((out["pred_label"] == 0) & (out["actual"] == 1)).sum())
            acc = (tp + tn) / len(out)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("정확도", f"{acc * 100:.1f} %")
            m2.metric("일치 / 불일치", f"{int(out['match'].sum())} / {int((~out['match']).sum())}")
            m3.metric("고장 탐지", f"{tp} / {tp + fn}")
            m4.metric("오경보(FP)", f"{fp}")
            st.markdown(f"**혼동행렬** — 정상→정상 {tn} · 정상→고장(오경보) {fp} · "
                        f"고장→정상(놓침) {fn} · 고장→고장(탐지) {tp}")
            show = out.copy()
            show["판정"] = show["match"].map({True: "✅ 일치", False: "❌ 불일치"})
            show["실제"] = show["actual"].map({0: "정상", 1: "고장"})
            show["예측"] = show["pred_label"].map({0: "정상", 1: "고장"})
            cols = ["Type", "Air temperature [K]", "Process temperature [K]",
                    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
                    "실제", "예측", "pred_proba", "판정"]
            st.markdown("**❌ 불일치(오답) 건**")
            st.dataframe(show[~show["match"]][cols], width="stretch")
            st.markdown("**전체 결과 (앞 100건)**")
            st.dataframe(show[cols].head(100), width="stretch")
        else:
            st.info("Target(실제라벨) 열이 없어 예측 결과만 표시합니다.")
            st.dataframe(out.head(100), width="stretch")
