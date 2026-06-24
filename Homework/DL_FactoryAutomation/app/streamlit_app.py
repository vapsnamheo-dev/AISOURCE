"""
Streamlit 앱 — DL 기반 설비 고장 예측 (1D-CNN/LSTM)
ML 프로젝트(XGBoost)의 DL 후속으로, 시계열 시퀀스를 입력받아 고장 확률 예측.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import FEATURE_COLS, SEQ_LEN, DEMO_DIR
from src.generate_ts_data import generate_run

st.set_page_config(page_title="PdM-Guard DL", page_icon="🔬", layout="wide")
st.title("🔬 PdM-Guard DL — 시계열 설비 고장 예측 (1D-CNN/LSTM)")
st.caption("ML(XGBoost) 프로젝트의 딥러닝 후속: 센서 시계열로 고장 패턴 학습")

tab1, tab2 = st.tabs(["🔮 단건 시뮬레이션", "📁 데모 CSV 일괄 예측"])

# ── 모델 로드 (캐시) ────────────────────────────────────────────────────────
@st.cache_resource
def load_dl_model():
    try:
        from src.predict import load_model_and_norm
        return load_model_and_norm("cnn_lstm", "binary")
    except Exception as e:
        return None, None, None, str(e)

model, mean, std, err = (lambda r: (*r, None) if len(r)==3 else r)(*((lambda r: r + (None,) if len(r)==3 else r)(load_dl_model() or (None, None, None, "로드 실패"))))

with tab1:
    st.subheader("설비 파라미터 입력 → 시뮬레이션 시퀀스 생성 → 고장 확률 출력")

    col1, col2 = st.columns([1, 2])
    with col1:
        equip_type = st.selectbox("설비 등급", ["L", "M", "H"])
        force_fail = st.checkbox("강제 고장 시뮬레이션")
        if st.button("🔁 시퀀스 생성 & 예측", type="primary"):
            seq_df = generate_run(equip_type=equip_type, force_failure=force_fail)
            st.session_state["current_seq"] = seq_df

    if "current_seq" in st.session_state:
        seq_df = st.session_state["current_seq"]
        with col2:
            st.line_chart(seq_df[["torque_nm", "tool_wear_min", "rotational_speed_rpm"]].rename(
                columns={"torque_nm": "토크(Nm)", "tool_wear_min": "공구마모(min)",
                         "rotational_speed_rpm": "회전속도(rpm)"}))

        if model is not None:
            from src.predict import predict_sequence
            result = predict_sequence(seq_df, model=model, mean=mean, std=std)
            prob = result["prob"]
            risk = result["risk_level"]
            color = {"정상": "green", "주의": "orange", "위험": "darkorange", "긴급": "red"}[risk]
            st.markdown(f"### 고장 확률: **:{color}[{prob}%]** — :{color}[{risk}]")
        else:
            st.info(f"모델 미학습 상태입니다. `python src/train.py` 실행 후 이용하세요.\n\n오류: {err}")

with tab2:
    st.subheader("데모 CSV 업로드 (시퀀스 1개 = 50행)")
    demo_files = sorted(DEMO_DIR.glob("demo_*.csv"))

    if demo_files:
        selected = st.selectbox("데모 파일 선택", [f.name for f in demo_files[:20]])
        demo_path = DEMO_DIR / selected
        seq_df = pd.read_csv(demo_path)
        st.dataframe(seq_df[FEATURE_COLS].describe(), use_container_width=True)

        if model is not None:
            from src.predict import predict_sequence
            result = predict_sequence(seq_df, model=model, mean=mean, std=std)
            st.metric("고장 확률", f"{result['prob']}%", delta=result["risk_level"])
        else:
            st.info("모델 학습 필요: `python src/train.py`")
    else:
        st.warning("데모 데이터 없음. `python src/generate_ts_data.py` 먼저 실행하세요.")
