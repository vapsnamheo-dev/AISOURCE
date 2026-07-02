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

tab1, tab2, tab3 = st.tabs(["🔮 단건 시뮬레이션", "📁 데모 CSV 일괄 예측", "🔍 RAG 유사 사례 검색"])

# ── 모델 로드 (캐시) ────────────────────────────────────────────────────────
@st.cache_resource
def load_dl_model():
    try:
        from src.predict import load_model_and_norm
        return load_model_and_norm("cnn_lstm", "binary")
    except Exception as e:
        return None, None, None, str(e)

model, mean, std, err = (lambda r: (*r, None) if len(r)==3 else r)(*((lambda r: r + (None,) if len(r)==3 else r)(load_dl_model() or (None, None, None, "로드 실패"))))

@st.cache_data
def _load_rag_feature_stats():
    try:
        from src.femto_rag_search import FEAT_PATH, _load_feature_list, _fill_nan
        if not FEAT_PATH.exists():
            return {}, []
        features = _load_feature_list()
        df = pd.read_csv(FEAT_PATH)
        df = _fill_nan(df, features)
        return {
            f: {
                "min": float(df[f].min()),
                "max": float(df[f].max()),
                "med": float(df[f].median()),
            }
            for f in features
        }, features
    except Exception:
        return {}, []

@st.cache_data
def _load_rag_dataset():
    try:
        from src.femto_rag_search import FEAT_PATH, _load_feature_list, _fill_nan
        if not FEAT_PATH.exists():
            return pd.DataFrame(), []
        features = _load_feature_list()
        df = pd.read_csv(FEAT_PATH)
        df = _fill_nan(df, features)
        return df, features
    except Exception:
        return pd.DataFrame(), []

with tab1:
    st.subheader("설비 파라미터 입력 → 시퀀스 생성 → 고장 확률 출력")

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

with tab3:
    st.header("🔍 RAG 유사 사례 검색")
    st.caption("현재 센서값 기반으로 FAISS 유사 사례를 검색하고 과거 베어링 RUL을 추정합니다.")

    stats, rag_features = _load_rag_feature_stats()
    if not rag_features:
        st.warning("RAG 피처 데이터가 없습니다. 먼저 `python -m src.femto_rag_search`로 인덱스를 빌드하세요.")
        st.stop()

    df_all, rag_features = _load_rag_dataset()
    query_source = st.radio(
        "쿼리 입력 방식",
        ["슬라이더 직접 입력", "과거 데이터 선택"],
        horizontal=True,
    )

    with st.sidebar.expander("RAG 검색 설정", expanded=False):
        k = st.slider("Top-K 유사 사례", 1, 20, 5)
        exclude_same = st.checkbox("동일 베어링 제외", value=True)
        st.caption("검색 결과에서 현재 쿼리와 같은 베어링을 제외할지 선택합니다.")
        st.divider()
        st.markdown("**인덱스 상태**")
        try:
            from src.femto_rag_search import INDEX_PATH, META_PATH
            index_ready = INDEX_PATH.exists() and META_PATH.exists()
        except Exception:
            index_ready = False
        st.write("✅ 준비됨" if index_ready else "❌ 없음")
        if st.button("🔨 RAG 인덱스 빌드 / 재빌드", type="primary"):
            from src.femto_rag_search import build_index
            try:
                build_index(verbose=False)
                st.success("RAG 인덱스 빌드 완료")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"RAG 인덱스 빌드 실패: {e}")

    if query_source == "과거 데이터 선택" and not df_all.empty:
        bearing_list = sorted(df_all["bearing"].unique().tolist())
        selected_bearing = st.selectbox("베어링 선택", bearing_list)
        bearing_df = df_all[df_all["bearing"] == selected_bearing].sort_values("minute")
        selected_minute = st.selectbox("시점(분) 선택", bearing_df["minute"].tolist(), index=len(bearing_df) // 2)
        row = bearing_df[bearing_df["minute"] == selected_minute].iloc[0]
        query = {f: float(row[f]) for f in rag_features if f in row.index}
        query["bearing"] = selected_bearing

        st.info(f"선택된 사례: **{selected_bearing}** / t={selected_minute}분")
        st.markdown("### 선택된 쿼리 입력값")
        st.dataframe(
            pd.DataFrame(query, index=[0]).T.rename(columns={0: "값"}),
            use_container_width=True,
        )
    else:
        query = {}
        input_features = [
            f
            for f in [
                "h_rms",
                "h_kurt",
                "h_skew",
                "h_crest",
                "v_rms",
                "v_kurt",
                "v_skew",
                "v_crest",
                "temp_mean",
                "energy",
                "health_idx",
                "rms_ratio",
            ]
            if f in stats
        ]
        st.markdown("### 센서 입력 값")
        columns = st.columns(3)
        for idx, feature in enumerate(input_features):
            stat = stats[feature]
            query[feature] = columns[idx % 3].slider(
                feature,
                float(stat["min"]),
                float(stat["max"]),
                float(stat["med"]),
                step=max((stat["max"] - stat["min"]) / 100, 0.0001),
                format="%.4f",
                key=f"rag_{feature}",
            )

        st.markdown("### 입력 쿼리 요약")
        st.dataframe(
            pd.DataFrame(query, index=[0]).T.rename(columns={0: "값"}),
            use_container_width=True,
        )

    if st.button("🔎 RAG 유사 사례 검색 실행", type="primary"):
        try:
            from src.femto_rag_search import load_index, search_and_estimate_rul
            index, meta, features, scaler = load_index()
            result = search_and_estimate_rul(query, k=k, exclude_same_bearing=exclude_same)
        except Exception as e:
            st.error(f"RAG 검색 실패: {e}")
            st.stop()

        cases = result.get("similar_cases", [])
        st.metric("검색된 사례 수", f"{len(cases)}개")
        st.metric("평균 유사도", f"{result.get('avg_similarity', 0):.1f}%")
        rul = result.get("estimated_rul")
        st.metric("RAG 추정 잔여수명", f"{rul:.0f} 분" if rul is not None else "N/A")

        if cases:
            df_cases = pd.DataFrame([
                {
                    "순위": c.get("rank", idx + 1),
                    "유사도(%)": c.get("similarity", 0),
                    "베어링": c.get("bearing", "-"),
                    "시각(분)": c.get("minute", "-"),
                    "RUL(분)": int(c["rul"]) if c.get("rul") is not None else "-",
                    "RUL(%)": f"{c.get('rul_pct', 0) * 100:.1f}" if c.get("rul_pct") is not None else "-",
                    "상태": "🔴 열화" if c.get("label") == 1 else "🟢 정상",
                    "데이터셋": c.get("split", "-"),
                }
                for idx, c in enumerate(cases)
            ])

            st.markdown("### 유사 사례 Top-K")
            st.dataframe(df_cases, use_container_width=True)
            st.markdown("### 유사도 분포")
            st.bar_chart(df_cases.set_index("순위")["유사도(%)"])
        else:
            st.info("검색된 유사 사례가 없습니다.")
