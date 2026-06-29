"""
Streamlit 앱 — FEMTO-ST 베어링 예지보전 (ML+DL 통합 진단)
ML만 적용시(열화 분류) + LSTM RUL 예측 결합 시스템
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.io as pio
    _t = pio.templates["plotly_white"]
    _t.layout.font.family = "Malgun Gothic, Apple Gothic, sans-serif"
    pio.templates["korean"] = _t
    pio.templates.default = "korean"
except Exception:
    pass

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "FEMTO_processed"
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="FEMTO-ST 베어링 예지보전",
    page_icon="⚙️",
    layout="wide",
)
st.title("⚙️ FEMTO-ST 베어링 예지보전 — ML+DL 통합 진단")
st.caption("ML만 적용시(열화 분류) + LSTM(잔여수명 예측) 결합 시스템")

# ── Cloud 자동 전처리 ──────────────────────────────────────────────────────────
import subprocess as _sp
_feat_path = PROCESSED_DIR / "femto_features.csv"
if not _feat_path.exists():
    with st.spinner("⏳ 전처리 데이터 생성 중... (최초 1회, 약 1~2분)"):
        _sp.run([sys.executable, "-m", "src.femto_preprocess"],
                capture_output=True, cwd=str(ROOT))
    st.rerun()

# ── 사이드바: 진단 설정 ────────────────────────────────────────────────────────
st.sidebar.header("⚙️ 진단 설정")

# UI 1: ML 결정 임계값 (기존 ML 프로젝트 방식 그대로 이식)
st.sidebar.subheader("ML 열화 판정 임계값")
ml_threshold = st.sidebar.slider(
    "P(열화) 기준",
    0.0, 1.0, 0.5, 0.01,
    help="낮추면 재현율↑(놓침↓), 높이면 정밀도↑(오경보↓)",
)
if "thr_log" not in st.session_state:
    st.session_state.thr_log = []
if (not st.session_state.thr_log) or st.session_state.thr_log[-1][1] != ml_threshold:
    st.session_state.thr_log.append((datetime.now().strftime("%H:%M:%S"), ml_threshold))
with st.sidebar.expander("임계값 변경 이력"):
    for ts, thv in st.session_state.thr_log[-10:]:
        st.write(f"- {ts} → {thv:.2f}")

st.sidebar.divider()

# UI 2: DL RUL 경보 임계값 (FEMTO-ST 차별화 기능)
st.sidebar.subheader("DL RUL 경보 기준")
rul_threshold = st.sidebar.slider(
    "잔여수명 경보 기준 (분)",
    10, 300, 60, 5,
    help="LSTM이 예측한 잔여수명이 이 값 이하이면 경보 발령",
)
if "rul_log" not in st.session_state:
    st.session_state.rul_log = []
if (not st.session_state.rul_log) or st.session_state.rul_log[-1][1] != rul_threshold:
    st.session_state.rul_log.append((datetime.now().strftime("%H:%M:%S"), rul_threshold))
with st.sidebar.expander("RUL 임계값 변경 이력"):
    for ts, rv in st.session_state.rul_log[-10:]:
        st.write(f"- {ts} → {rv}분")

st.sidebar.divider()


# ── 캐시: 데이터 및 모델 로딩 ─────────────────────────────────────────────────

@st.cache_data
def load_feature_data() -> tuple[pd.DataFrame, list[str]]:
    """전처리된 FEMTO 피처 데이터를 로딩한다."""
    feat_path = PROCESSED_DIR / "femto_features.csv"
    sel_path = PROCESSED_DIR / "selected_features.csv"

    if not feat_path.exists():
        return pd.DataFrame(), []

    df = pd.read_csv(feat_path)
    if sel_path.exists():
        features = pd.read_csv(sel_path)["feature"].tolist()
    else:
        features = ["h_rms", "h_kurt", "h_skew", "h_crest",
                    "v_rms", "v_kurt", "v_skew", "v_crest", "temp_mean"]
    return df, features


@st.cache_data
def load_vif_results() -> pd.DataFrame:
    """VIF 분석 결과를 로딩한다."""
    vif_path = PROCESSED_DIR / "vif_results.csv"
    if not vif_path.exists():
        return pd.DataFrame()
    return pd.read_csv(vif_path)


@st.cache_resource
def load_ml_model() -> tuple[object, object, dict]:
    """ML 최고 모델 + 스케일러 + 결과 JSON을 로딩한다."""
    model_path = MODEL_DIR / "femto_best_clf.pkl"
    scaler_path = MODEL_DIR / "femto_scaler.pkl"
    results_path = MODEL_DIR / "femto_ml_results.json"

    model, scaler, results = None, None, {}
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
    return model, scaler, results


@st.cache_data
def load_dl_compare_results() -> dict:
    """5종 DL 아키텍처 비교 결과를 로딩한다."""
    path = MODEL_DIR / "femto_dl_compare_results.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_rul_models() -> tuple[object, object, object, dict]:
    """RF RUL + LSTM + 스케일러 + 결과 JSON을 로딩한다."""
    rf_path = MODEL_DIR / "femto_rf_rul.pkl"
    lstm_path = MODEL_DIR / "femto_lstm_rul.keras"
    seq_sc_path = MODEL_DIR / "femto_seq_scaler.pkl"
    y_sc_path = MODEL_DIR / "femto_y_scaler.pkl"
    results_path = MODEL_DIR / "femto_rul_results.json"

    rf_model, lstm_model, seq_scaler, y_scaler = None, None, None, None
    rul_results = {}

    if rf_path.exists():
        with open(rf_path, "rb") as f:
            rf_model = pickle.load(f)
    if seq_sc_path.exists():
        with open(seq_sc_path, "rb") as f:
            seq_scaler = pickle.load(f)
    if y_sc_path.exists():
        with open(y_sc_path, "rb") as f:
            y_scaler = pickle.load(f)
    if lstm_path.exists():
        try:
            import tensorflow as tf
            lstm_model = tf.keras.models.load_model(lstm_path)
        except Exception:
            pass
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            rul_results = json.load(f)

    return rf_model, lstm_model, seq_scaler, y_scaler, rul_results


# ── 데이터 로딩 실행 ───────────────────────────────────────────────────────────
df, features = load_feature_data()
vif_df = load_vif_results()
ml_model, ml_scaler, ml_results = load_ml_model()
rf_rul, lstm_rul, seq_scaler, y_scaler, rul_results = load_rul_models()
dl_compare = load_dl_compare_results()

# 데이터 로딩 실패 시 (자동 전처리가 완료된 후에도 비어있으면 데이터 파일 문제)
if df.empty:
    st.error("데이터 로딩 실패. 앱을 새로고침하세요.")

# ── 탭 구성 ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 데이터 탐색",
    "🤖 ML 성능",
    "🔮 DL RUL 예측",
    "🏭 통합 진단 (실시간)",
    "🔬 DL 아키텍처 비교 (5종)",
])

# ════════════════════════════════════════════════════════
# Tab 1: 데이터 탐색
# ════════════════════════════════════════════════════════
with tab1:
    st.header("📊 FEMTO-ST 데이터 탐색")

    if df.empty:
        st.info("데이터를 먼저 전처리하세요.")
    else:
        # 데이터셋 요약
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("총 베어링 수", df["bearing"].nunique())
        col_b.metric("총 스냅샷 수", f"{len(df):,}")
        col_c.metric("열화 비율", f"{df['label'].mean()*100:.1f}%")
        col_d.metric("데이터 모드", "Demo(합성)" if "Syn" in df["bearing"].iloc[0] else "실데이터")

        st.divider()

        # h_rms 추이 (베어링별)
        st.subheader("베어링별 h_rms 진동 추이")
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B", "#B279A2"]
            for idx, (bearing, bdf) in enumerate(df.groupby("bearing")):
                bdf = bdf.sort_values("minute")
                color = colors[idx % len(colors)]
                fig.add_trace(go.Scatter(
                    x=bdf["minute"], y=bdf["h_rms"],
                    name=bearing, line=dict(color=color, width=1.5),
                ))
                # 열화 임계값 수평선 (베어링별)
                thr = bdf["threshold"].iloc[0]
                fig.add_hline(
                    y=thr, line_dash="dash", line_color=color, opacity=0.5,
                    annotation_text=f"{bearing} thr",
                )
            fig.update_layout(
                xaxis_title="Time (min)",
                yaxis_title="h_rms (Vibration RMS)",
                legend_title="Bearing",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.line_chart(df.pivot_table(index="minute", columns="bearing", values="h_rms"))

        st.divider()

        # 라벨 분포 + 피처 분포
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("열화 라벨 분포")
            label_counts = df["label"].value_counts().rename({0: "Normal", 1: "Degraded"})
            try:
                import plotly.express as px
                fig2 = px.pie(
                    values=label_counts.values,
                    names=label_counts.index,
                    color_discrete_map={"Normal": "#4C78A8", "Degraded": "#E45756"},
                )
                st.plotly_chart(fig2, use_container_width=True)
            except ImportError:
                st.bar_chart(label_counts)

        with col_r:
            st.subheader("피처 분포 (박스플롯)")
            if features:
                sel_feat = st.selectbox("피처 선택", features, key="feat_box")
                try:
                    import plotly.express as px
                    fig3 = px.box(
                        df, x="bearing", y=sel_feat, color="bearing",
                        labels={"bearing": "Bearing", sel_feat: sel_feat},
                    )
                    fig3.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig3, use_container_width=True)
                except ImportError:
                    st.dataframe(df.groupby("bearing")[sel_feat].describe())


# ════════════════════════════════════════════════════════
# Tab 2: ML 성능
# ════════════════════════════════════════════════════════
with tab2:
    st.header("🤖 ML 열화 분류 성능")

    # VIF 분석표
    st.subheader("VIF 다중공선성 분석")
    if vif_df.empty:
        st.info("VIF 분석 결과가 없습니다. `python -m src.femto_preprocess` 를 실행하세요.")
    else:
        def _color_vif(val: float) -> str:
            if not isinstance(val, (int, float)) or np.isnan(val):
                return ""
            if val >= 10:
                return "background-color: #FFCCCC"
            if val >= 5:
                return "background-color: #FFFFCC"
            return "background-color: #CCFFCC"

        styled = vif_df.style.applymap(_color_vif, subset=["VIF"])
        st.dataframe(styled, use_container_width=True)
        st.caption("VIF: 양호(녹색, <5) / 주의(노랑, 5~10) / 심각(빨강, ≥10)")

    st.divider()

    # 모델 성능 비교표
    st.subheader("모델 3종 성능 비교")
    if not ml_results:
        st.warning("ML 모델 없음. 먼저 실행하세요: `python -m src.femto_ml`")
    else:
        perf_rows = []
        for name, r in ml_results.items():
            if name.startswith("_"):
                continue
            perf_rows.append({
                "Model": name,
                "Accuracy": r.get("accuracy", "-"),
                "Precision": r.get("precision", "-"),
                "Recall": r.get("recall", "-"),
                "F1": r.get("f1", "-"),
                "ROC-AUC": r.get("roc_auc", "-"),
            })
        if perf_rows:
            perf_df = pd.DataFrame(perf_rows).set_index("Model")
            best_name = ml_results.get("_best_model", "")
            st.dataframe(
                perf_df.style.highlight_max(axis=0, color="#CCFFCC", subset=["Recall", "F1", "ROC-AUC"]),
                use_container_width=True,
            )
            st.success(f"최고 모델 (Recall 기준): **{best_name}**")

        # Feature Importance
        best_name = ml_results.get("_best_model", "")
        if best_name and "feature_importance" in ml_results.get(best_name, {}):
            st.subheader(f"Feature Importance ({best_name})")
            imp = ml_results[best_name]["feature_importance"]
            imp_df = pd.DataFrame({"Feature": list(imp.keys()), "Importance": list(imp.values())})
            imp_df = imp_df.sort_values("Importance", ascending=True)
            try:
                import plotly.express as px
                fig4 = px.bar(
                    imp_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="Blues",
                )
                fig4.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig4, use_container_width=True)
            except ImportError:
                st.dataframe(imp_df)

    st.divider()

    # Confusion Matrix (ml_threshold 적용)
    st.subheader("최고 모델 Confusion Matrix")
    if ml_results and not df.empty and ml_model is not None and features:
        best_name = ml_results.get("_best_model", "")
        best_r = ml_results.get(best_name, {})
        if "confusion_matrix" in best_r:
            st.write(f"임계값 = **{ml_threshold:.2f}** (사이드바 슬라이더로 조정)")
            # 기본 CM (threshold=0.5 기준 저장값 사용, 실시간 반영은 아래에서)
            cm_data = best_r["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm_data,
                index=["Actual Normal", "Actual Degraded"],
                columns=["Pred Normal", "Pred Degraded"],
            )
            st.dataframe(cm_df)
            st.caption("※ Confusion Matrix는 학습 시 threshold=0.5 기준. 실시간 임계값 효과는 Tab 4에서 확인.")


# ════════════════════════════════════════════════════════
# Tab 3: DL RUL 예측
# ════════════════════════════════════════════════════════
with tab3:
    st.header("🔮 DL 잔여수명(RUL) 예측")

    if not rul_results:
        st.warning("DL 모델 없음. 먼저 실행하세요: `python -m src.femto_dl_rul`")
    else:
        # ML vs DL 성능 비교표
        st.subheader("ML(RF) vs DL(LSTM) RUL 예측 성능")
        rf_res = rul_results.get("rf", {})
        lstm_res = rul_results.get("lstm", {})
        improvement = rul_results.get("improvement_pct", None)

        cmp_df = pd.DataFrame({
            "Method": ["RF Baseline", "LSTM"],
            "RMSE (min)": [rf_res.get("rmse", "-"), lstm_res.get("rmse", "-")],
            "MAE (min)": [rf_res.get("mae", "-"), lstm_res.get("mae", "-")],
        })
        st.dataframe(cmp_df.set_index("Method"), use_container_width=True)

        if improvement is not None:
            if improvement > 0:
                st.success(f"LSTM이 RF 대비 RMSE **{improvement:.1f}%** 개선")
            elif improvement < 0:
                st.info(f"RF가 LSTM 대비 RMSE **{abs(improvement):.1f}%** 우수 (데이터 규모에 따라 역전 가능)")
            else:
                st.info("두 모델 성능 동일")

        st.divider()

        # 학습 곡선
        history = rul_results.get("history", {})
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])

        if train_loss:
            st.subheader("LSTM 학습 곡선 — Fold 1 대표 (GroupKFold 5-Fold 중 1번째 분할)")
            st.caption(
                "📊 **Fold 1**: 전체 데이터를 5등분하여 첫 번째 그룹을 검증셋으로 사용한 분할. "
                "5개 Fold 모두 표시하면 복잡해져 Fold 1만 시각화. "
                "**성능 수치(RMSE/MAE)는 5-Fold 평균값.**"
            )
            loss_df = pd.DataFrame({
                "Epoch": list(range(1, len(train_loss) + 1)),
                "Train Loss": train_loss,
                "Val Loss": val_loss if val_loss else [None] * len(train_loss),
            }).set_index("Epoch")
            st.line_chart(loss_df)

        st.divider()

        # RUL 추이 (베어링별 실제 vs 예측)
        st.subheader("베어링별 실제 RUL 추이")
        if not df.empty:
            try:
                import plotly.graph_objects as go
                fig5 = go.Figure()
                for bearing, bdf in df.groupby("bearing"):
                    bdf = bdf.sort_values("minute")
                    fig5.add_trace(go.Scatter(
                        x=bdf["minute"], y=bdf["rul"],
                        name=f"{bearing} (Actual)",
                        line=dict(width=1.5),
                    ))
                fig5.add_hline(
                    y=rul_threshold, line_dash="dash", line_color="red",
                    annotation_text=f"RUL 경보 기준 {rul_threshold}분",
                )
                fig5.update_layout(
                    xaxis_title="Time (min)",
                    yaxis_title="RUL (min)",
                    height=400,
                )
                st.plotly_chart(fig5, use_container_width=True)
            except ImportError:
                rul_pivot = df.pivot_table(index="minute", columns="bearing", values="rul")
                st.line_chart(rul_pivot)

            # 현재 RUL 경보 기준 설명
            st.info(
                f"현재 RUL 경보 기준: **{rul_threshold}분** (사이드바에서 조정)\n\n"
                f"LSTM 예측 RUL이 이 값 이하일 때 경보 발령됩니다."
            )


# ════════════════════════════════════════════════════════
# Tab 4: 통합 진단 (실시간)
# ════════════════════════════════════════════════════════
with tab4:
    st.header("🏭 통합 진단 — 실시간 베어링 상태 평가")

    if ml_model is None:
        st.warning("ML 모델 없음. 먼저 실행하세요: `python -m src.femto_ml`")
    else:
        st.subheader("진동 특성 직접 입력")
        st.caption("현재 측정값을 입력하면 ML(열화 판정) + DL(잔여수명 예측)을 실시간으로 수행합니다.")

        col1, col2 = st.columns(2)
        with col1:
            h_rms = st.slider("h_rms (수평 진동 RMS)", 0.0, 10.0, 0.5, 0.01)
            h_kurt = st.slider("h_kurt (첨도)", 0.0, 20.0, 3.0, 0.1)
            h_skew = st.slider("h_skew (왜도)", -5.0, 5.0, 0.0, 0.1)
            h_crest = st.slider("h_crest (파고율)", 1.0, 20.0, 4.0, 0.1)
        with col2:
            v_rms = st.slider("v_rms (수직 진동 RMS)", 0.0, 10.0, 0.45, 0.01)
            v_kurt = st.slider("v_kurt (첨도)", 0.0, 20.0, 3.0, 0.1)
            v_skew = st.slider("v_skew (왜도)", -5.0, 5.0, 0.0, 0.1)
            v_crest = st.slider("v_crest (파고율)", 1.0, 20.0, 4.0, 0.1)

        temp = st.slider("온도 (°C)", 20.0, 80.0, 30.0, 0.5)

        if st.button("진단하기", type="primary"):
            # 현재 입력값을 피처 순서에 맞게 정렬
            feature_values = {
                "h_rms": h_rms, "h_kurt": h_kurt, "h_skew": h_skew, "h_crest": h_crest,
                "v_rms": v_rms, "v_kurt": v_kurt, "v_skew": v_skew, "v_crest": v_crest,
                "temp_mean": temp,
                # 파생 피처 기본값 (selected_features에 포함될 수 있는 항목)
                "energy": h_rms ** 2 + v_rms ** 2,
                "health_idx": 1.0 / (1.0 + h_kurt + v_kurt),
                "rms_ratio": h_rms / (v_rms + 1e-9),
            }
            # features가 비어 있을 때(전처리 캐시 초기화 전) 기본 피처 목록 사용
            _feat_list = features if features else [
                "h_rms", "h_kurt", "h_skew", "h_crest",
                "v_rms", "v_kurt", "v_skew", "v_crest", "temp_mean",
            ]
            input_vals = np.array([[feature_values.get(f, 0.0) for f in _feat_list]])

            # ML 열화 판정
            try:
                if ml_scaler is not None and input_vals.shape[1] > 0 and input_vals.shape[1] == ml_scaler.n_features_in_:
                    input_scaled = ml_scaler.transform(input_vals)
                elif ml_scaler is not None and input_vals.shape[1] > 0:
                    # 스케일러 피처 수 불일치 시 앞쪽 n개만 사용
                    n = ml_scaler.n_features_in_
                    input_scaled = ml_scaler.transform(input_vals[:, :n])
                else:
                    input_scaled = input_vals

                proba = ml_model.predict_proba(input_scaled)[0][1]
                pred = int(proba >= ml_threshold)
            except Exception as e:
                st.error(f"ML 예측 오류: {e}")
                proba, pred = 0.0, 0

            # DL RUL 예측 (30분 시퀀스 필요 → 단건 입력 시 상수 시퀀스로 근사)
            predicted_rul = None
            if rf_rul is not None:
                try:
                    # RF RUL: 마지막 타임스텝 피처
                    if seq_scaler is not None:
                        input_scaled_seq = seq_scaler.transform(input_vals)
                    else:
                        input_scaled_seq = input_vals

                    rul_raw = rf_rul.predict(input_scaled_seq)[0]
                    # 역스케일링
                    if y_scaler is not None:
                        predicted_rul = float(y_scaler.inverse_transform([[rul_raw]])[0][0])
                    else:
                        predicted_rul = float(rul_raw)
                    predicted_rul = max(0.0, predicted_rul)
                except Exception:
                    predicted_rul = None

            # LSTM RUL (30타임스텝 상수 시퀀스 근사)
            lstm_rul_pred = None
            if lstm_rul is not None and seq_scaler is not None:
                try:
                    seq_input = np.tile(input_vals, (30, 1))  # (30, n_feat)
                    seq_input_scaled = seq_scaler.transform(seq_input)
                    seq_input_3d = seq_input_scaled[np.newaxis, :, :]  # (1, 30, n_feat)
                    rul_raw_lstm = lstm_rul.predict(seq_input_3d, verbose=0)[0][0]
                    if y_scaler is not None:
                        lstm_rul_pred = float(y_scaler.inverse_transform([[rul_raw_lstm]])[0][0])
                    else:
                        lstm_rul_pred = float(rul_raw_lstm)
                    lstm_rul_pred = max(0.0, lstm_rul_pred)
                except Exception:
                    lstm_rul_pred = None

            # 결과 표시
            st.divider()
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("ML 열화 판정")
                st.metric("열화 확률", f"{proba * 100:.1f}%")
                if pred == 1:
                    st.error(f"열화 감지 (P={proba:.2f} > 임계값 {ml_threshold:.2f})")
                else:
                    st.success(f"정상 (P={proba:.2f} <= 임계값 {ml_threshold:.2f})")

                # 임계값 변경에 따른 판정 변화 안내
                st.caption(
                    f"임계값을 낮추면(현재 {ml_threshold:.2f}) 더 민감하게 감지합니다."
                )

            with col_b:
                st.subheader("DL 잔여수명 예측")
                use_rul = lstm_rul_pred if lstm_rul_pred is not None else predicted_rul
                method = "LSTM" if lstm_rul_pred is not None else ("RF" if predicted_rul is not None else None)

                if use_rul is not None:
                    st.metric("예측 잔여수명", f"{use_rul:.0f} 분", help=f"예측 방법: {method}")
                    if use_rul <= rul_threshold:
                        st.error(f"긴급 경보: 잔여수명 {use_rul:.0f}분 (기준 {rul_threshold}분 이하)")
                    elif use_rul <= rul_threshold * 2:
                        st.warning(f"주의: 잔여수명 {use_rul:.0f}분 (기준의 2배 이내)")
                    else:
                        st.success(f"양호: 잔여수명 {use_rul:.0f}분")
                else:
                    st.info("DL 모델 미학습 — `python -m src.femto_dl_rul` 실행 후 사용 가능")
                    if pred == 0:
                        st.info("정상 판정 → RUL 예측 불필요 (잔여 수명 충분)")

            # 입력값 요약
            with st.expander("입력값 요약"):
                summary = pd.DataFrame({
                    "Feature": list(feature_values.keys()),
                    "Value": list(feature_values.values()),
                })
                st.dataframe(summary, use_container_width=True)


# ════════════════════════════════════════════════════════
# Tab 5: DL 아키텍처 비교 (5종)
# ════════════════════════════════════════════════════════
with tab5:
    st.header("🔬 DL 아키텍처 비교 — LSTM / GRU / BiLSTM / 1D-CNN / CNN-LSTM")
    st.caption(
        "femto_dl_compare.py 결과: GroupKFold CV + OOS(Full_Test_Set) 평가 "
        "| EarlyStopping(monitor=val_loss, patience=7) 적용"
    )

    if not dl_compare:
        st.warning(
            "비교 결과 없음. 먼저 실행하세요: `python -m src.femto_dl_compare`"
        )
    else:
        best_model = dl_compare.get("_best_model", "")

        # ── 성능 비교표 ─────────────────────────────────────────────────────────
        st.subheader("5종 아키텍처 성능 비교 (OOS RMSE 기준)")

        rows = []
        for name, r in dl_compare.items():
            if name.startswith("_"):
                continue
            rows.append({
                "아키텍처": name,
                "CV RMSE (분)": r.get("cv_rmse", "-"),
                "OOS RMSE (분)": r.get("oos_rmse", "-"),
                "OOS MAE (분)": r.get("oos_mae", "-"),
                "실행 Epoch": r.get("actual_epochs", "-"),
                "최적": "⭐ 최적" if name == best_model else "",
            })

        if rows:
            cmp_df = pd.DataFrame(rows).set_index("아키텍처")
            numeric_cols = ["CV RMSE (분)", "OOS RMSE (분)", "OOS MAE (분)"]
            for c in numeric_cols:
                cmp_df[c] = pd.to_numeric(cmp_df[c], errors="coerce")

            try:
                styled = cmp_df.style.highlight_min(
                    subset=["OOS RMSE (분)"], color="#CCFFCC", axis=0
                ).format("{:.1f}", subset=numeric_cols, na_rep="-")
                st.dataframe(styled, use_container_width=True)
            except Exception:
                st.dataframe(cmp_df, use_container_width=True)

            if best_model:
                best_r = dl_compare.get(best_model, {})
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("최적 아키텍처", best_model)
                col_b.metric("OOS RMSE", f"{best_r.get('oos_rmse', '-')} 분")
                col_c.metric("실행 Epoch",
                             f"{best_r.get('actual_epochs', '-')} / 50 (EarlyStopping)")

        st.divider()

        # ── EarlyStopping 분석 ──────────────────────────────────────────────────
        st.subheader("⏱️ EarlyStopping 효과 분석")
        st.info(
            "**설정**: monitor='val_loss', patience=7, restore_best_weights=True, max_epochs=50  \n"
            "EarlyStopping이 발동하면 설정(50 epoch) 전에 자동 종료 → 과적합 방지 + 학습 시간 절약"
        )

        early_rows = []
        for name, r in dl_compare.items():
            if name.startswith("_"):
                continue
            actual = r.get("actual_epochs")
            if actual is not None:
                saved = 50 - int(actual)
                early_rows.append({
                    "아키텍처": name,
                    "최대 Epoch": 50,
                    "실행 Epoch": int(actual),
                    "절약 Epoch": saved,
                    "조기 종료": "✅ 발동" if saved > 0 else "⬜ 미발동",
                })

        if early_rows:
            early_df = pd.DataFrame(early_rows).set_index("아키텍처")
            st.dataframe(early_df, use_container_width=True)

        st.divider()

        # ── 학습 곡선 PNG ─────────────────────────────────────────────────────
        st.subheader(f"📈 학습 곡선 — 최적 모델 ({best_model})")

        curve_png = MODEL_DIR / f"femto_dl_{best_model}_training_curve.png"
        if curve_png.exists():
            st.image(str(curve_png), caption=f"{best_model} Training & Validation Loss·MAE",
                     use_container_width=True)
            st.caption(
                "**해석**: Train Loss 계속 감소 + Val Loss 수렴 후 정체 → EarlyStopping 발동 지점에서 "
                "최적 가중치(restore_best_weights=True) 복원. Val Loss가 상승 반전하면 과적합 시작."
            )
        else:
            st.info(
                f"학습 곡선 이미지 없음: `models/femto_dl_{best_model}_training_curve.png`  \n"
                "`python -m src.femto_dl_compare` 실행 후 표시됩니다."
            )

        st.divider()

        # ── OOS 예측 결과 PNG ─────────────────────────────────────────────────
        st.subheader(f"🎯 저장 모델 로드 후 OOS 예측 결과 — {best_model}")

        pred_png = MODEL_DIR / f"femto_dl_{best_model}_oos_prediction.png"
        if pred_png.exists():
            st.image(str(pred_png), caption=f"{best_model} — 실제 RUL vs 예측 RUL (OOS 첫 200 샘플)",
                     use_container_width=True)
            st.caption(
                "**파란선**: 실제 RUL (분) · **주황선**: 저장 모델 로드 후 예측 RUL  \n"
                "모델 파일: `models/femto_best_dl_{best_model}.keras`"
            )
        else:
            st.info(
                f"OOS 예측 이미지 없음: `models/femto_dl_{best_model}_oos_prediction.png`  \n"
                "`python -m src.femto_dl_compare` 실행 후 표시됩니다."
            )

        # 저장 모델 파일 존재 여부 체크
        keras_path = MODEL_DIR / f"femto_best_dl_{best_model}.keras"
        if keras_path.exists():
            size_mb = keras_path.stat().st_size / (1024 * 1024)
            st.success(f"✅ 저장 모델 확인: `{keras_path.name}` ({size_mb:.2f} MB)")
        else:
            st.warning(f"저장 모델 없음: `models/femto_best_dl_{best_model}.keras`")
