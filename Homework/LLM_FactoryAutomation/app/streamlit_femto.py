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

# ── ML 프로젝트 링크 ──────────────────────────────────────────────────────────
_col_link, _col_spacer = st.columns([1, 5])
with _col_link:
    st.link_button(
        "🏭 ML 설비 진단 앱으로 →",
        "https://mlfactoryautomation.streamlit.app/",
        help="AI4I 제조 데이터 기반 ML 진단 앱 (CSV 업로드 판정 포함)",
        use_container_width=True,
    )

# ── Cloud 자동 전처리 ──────────────────────────────────────────────────────────
import subprocess as _sp
_feat_path = PROCESSED_DIR / "femto_features.csv"
if not _feat_path.exists():
    _tried = st.session_state.get("_preprocess_tried", False)
    if not _tried:
        st.session_state["_preprocess_tried"] = True
        with st.spinner("⏳ 전처리 데이터 생성 중... (최초 1회, 약 1~2분)"):
            _sp.run([sys.executable, "-m", "src.femto_preprocess"],
                    capture_output=True, cwd=str(ROOT))
        st.cache_data.clear()
        st.rerun()
    else:
        st.error("❌ 전처리 실패: femto_features.csv를 생성할 수 없습니다. 앱을 재시작하거나 관리자에게 문의하세요.")
        st.stop()

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
def _load_features_cached() -> tuple[pd.DataFrame, list[str]]:
    """파일이 존재할 때만 호출 — 캐시 대상."""
    feat_path = PROCESSED_DIR / "femto_features.csv"
    sel_path = PROCESSED_DIR / "selected_features.csv"
    df = pd.read_csv(feat_path)
    if sel_path.exists():
        features = pd.read_csv(sel_path)["feature"].tolist()
    else:
        features = ["h_rms", "h_kurt", "h_skew", "h_crest",
                    "v_rms", "v_kurt", "v_skew", "v_crest", "temp_mean"]
    return df, features


def load_feature_data() -> tuple[pd.DataFrame, list[str]]:
    """파일 존재 여부 확인 후 캐시 함수 호출 — 파일 없으면 캐시하지 않음."""
    if not (PROCESSED_DIR / "femto_features.csv").exists():
        return pd.DataFrame(), []
    return _load_features_cached()


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

# 데이터 로딩 실패 시 — st.stop()으로 이후 탭 렌더링 차단
if df.empty:
    st.error("❌ 데이터 로딩 실패. 앱을 새로고침(F5)하거나 잠시 후 다시 시도하세요.")
    st.info("로컬 실행 시: `python -m src.femto_preprocess` 후 재시작")
    st.stop()

# ── 탭 구성 ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 데이터 탐색 (demo data loading)",
    "🤖 ML 성능",
    "🔮 DL RUL 예측",
    "🏭 통합 진단 (실시간·CSV 진단)",
    "🔬 DL 아키텍처 비교 (5종)",
    "💡 AI 정비 권고 (LLM)",
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
# ════════════════════════════════════════════════════════
# Tab 4: 통합 진단 (실시간·CSV 진단)
# ════════════════════════════════════════════════════════
with tab4:
    st.header("🏭 통합 진단 — 실시간 베어링 상태 평가")

    # ML+DL 통합 효과 설명 배너
    with st.expander("ℹ️ ML+DL 통합 시스템이 PdM을 어떻게 강화하는가", expanded=False):
        st.markdown("""
**이 시스템 내 ML+DL 2단 진단의 효과**

| 단계 | 모델 | 역할 | 강점 |
|------|------|------|------|
| 1단 | ML (LogisticRegression) | **이진 알람** — 지금 열화 중인가? | 빠름·경량·설명 가능 |
| 2단 | DL (LSTM) | **정량 예측** — 잔여수명 몇 분? | 시계열 패턴 학습 |
| **결합** | ML → DL | 열화 감지 + RUL 정량화 + 경보 발령 | 오경보↓·정보량↑ |

> **ML만**: "고장 가능성 있음"
> **ML+DL**: "열화 감지됨 + 잔여수명 약 47분 → 다음 교대조 전 교체 권장"

**ML 과제(CNC 공작기계) + DL 과제(베어링 진동)의 통합 효과**

| 시스템 | 대상 장비 | 고장 유형 |
|--------|-----------|-----------|
| ML 과제 (mlfactoryautomation.streamlit.app) | CNC 공작기계 (선삭·밀링) | 공구마모·열발산·전력과부하·오버스트레인 5종 |
| DL 과제 (이 앱) | 회전기계 베어링 | 피로균열·표면마모 등 진동 기반 열화 |
| **두 시스템 통합** | 제조 라인 전체 커버 | 가공 장비 + 회전체 = 스마트 팩토리 PdM 플랫폼 |

두 시스템을 결합하면 **장비 유형별 전문 모델**로 제조 공정 전체의 예지보전이 가능합니다.
        """)

    if ml_model is None:
        st.warning("ML 모델 없음. 먼저 실행하세요: `python -m src.femto_ml`")
    else:
        diag_sub1, diag_sub2 = st.tabs(["🎛️ 슬라이더 직접 입력", "📂 FEMTO CSV 파일 일괄 진단"])

        # ── Sub1: 슬라이더 직접 입력 ─────────────────────────────────────────────
        with diag_sub1:
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
                feature_values = {
                    "h_rms": h_rms, "h_kurt": h_kurt, "h_skew": h_skew, "h_crest": h_crest,
                    "v_rms": v_rms, "v_kurt": v_kurt, "v_skew": v_skew, "v_crest": v_crest,
                    "temp_mean": temp,
                    "energy": h_rms ** 2 + v_rms ** 2,
                    "health_idx": 1.0 / (1.0 + h_kurt + v_kurt),
                    "rms_ratio": h_rms / (v_rms + 1e-9),
                }
                _feat_list = features if features else [
                    "h_rms", "h_kurt", "h_skew", "h_crest",
                    "v_rms", "v_kurt", "v_skew", "v_crest", "temp_mean",
                ]
                input_vals = np.array([[feature_values.get(f, 0.0) for f in _feat_list]])

                try:
                    n_sc = ml_scaler.n_features_in_ if ml_scaler is not None else input_vals.shape[1]
                    X_sc = ml_scaler.transform(input_vals[:, :n_sc]) if ml_scaler is not None else input_vals
                    proba = ml_model.predict_proba(X_sc)[0][1]
                    pred = int(proba >= ml_threshold)
                except Exception as e:
                    st.error(f"ML 예측 오류: {e}")
                    proba, pred = 0.0, 0

                predicted_rul = None
                if rf_rul is not None:
                    try:
                        sc_in = seq_scaler.transform(input_vals) if seq_scaler is not None else input_vals
                        rul_raw = rf_rul.predict(sc_in)[0]
                        predicted_rul = max(0.0, float(
                            y_scaler.inverse_transform([[rul_raw]])[0][0]
                            if y_scaler is not None else rul_raw
                        ))
                    except Exception:
                        predicted_rul = None

                lstm_rul_pred = None
                if lstm_rul is not None and seq_scaler is not None:
                    try:
                        seq_input = np.tile(input_vals, (30, 1))
                        seq_sc = seq_scaler.transform(seq_input)[np.newaxis, :, :]
                        rul_raw_l = lstm_rul.predict(seq_sc, verbose=0)[0][0]
                        lstm_rul_pred = max(0.0, float(
                            y_scaler.inverse_transform([[rul_raw_l]])[0][0]
                            if y_scaler is not None else rul_raw_l
                        ))
                    except Exception:
                        lstm_rul_pred = None

                st.divider()
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("ML 열화 판정")
                    st.metric("열화 확률", f"{proba * 100:.1f}%")
                    if pred == 1:
                        st.error(f"열화 감지 (P={proba:.2f} > 임계값 {ml_threshold:.2f})")
                    else:
                        st.success(f"정상 (P={proba:.2f} <= 임계값 {ml_threshold:.2f})")
                    st.caption(f"임계값을 낮추면(현재 {ml_threshold:.2f}) 더 민감하게 감지합니다.")

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

                with st.expander("입력값 요약"):
                    st.dataframe(
                        pd.DataFrame({"Feature": list(feature_values.keys()),
                                      "Value": list(feature_values.values())}),
                        use_container_width=True,
                    )

        # ── Sub2: FEMTO CSV 파일 일괄 진단 ──────────────────────────────────────
        with diag_sub2:
            st.subheader("📂 FEMTO CSV 파일 일괄 진단")
            st.caption(
                "FEMTO-ST 피처 CSV (femto_features.csv 형식 또는 h_rms 등 컬럼 포함 파일)를 "
                "업로드하면 각 행에 ML 열화 판정 + DL RUL 예측 결과를 추가하여 보여줍니다."
            )
            st.info(
                "필수 컬럼: h_rms, h_kurt, h_skew, h_crest, v_rms, v_kurt, v_skew, v_crest  "
                "| temp_mean / energy / health_idx / rms_ratio 없으면 자동 계산"
            )

            uploaded = st.file_uploader(
                "FEMTO 피처 CSV 업로드",
                type=["csv"],
                key="femto_csv_upload",
            )

            if uploaded is not None:
                try:
                    up_df = pd.read_csv(uploaded)
                    st.write(f"로드된 데이터: {len(up_df):,}행 × {len(up_df.columns)}열")
                    st.dataframe(up_df.head(3), use_container_width=True)

                    REQUIRED = ["h_rms", "h_kurt", "h_skew", "h_crest",
                                "v_rms", "v_kurt", "v_skew", "v_crest"]
                    missing = [c for c in REQUIRED if c not in up_df.columns]
                    if missing:
                        st.error(f"필수 컬럼 없음: {missing}")
                    else:
                        if "temp_mean" not in up_df.columns:
                            up_df["temp_mean"] = 0.0
                        if "energy" not in up_df.columns:
                            up_df["energy"] = up_df["h_rms"] ** 2 + up_df["v_rms"] ** 2
                        if "health_idx" not in up_df.columns:
                            up_df["health_idx"] = 1.0 / (1.0 + up_df["h_kurt"] + up_df["v_kurt"])
                        if "rms_ratio" not in up_df.columns:
                            up_df["rms_ratio"] = up_df["h_rms"] / (up_df["v_rms"] + 1e-9)

                        _feat_list = features if features else REQUIRED + ["temp_mean"]
                        X_up = up_df[[f for f in _feat_list if f in up_df.columns]].fillna(0).values

                        if st.button("일괄 진단 실행", type="primary", key="batch_run"):
                            with st.spinner("진단 중..."):
                                try:
                                    n_sc = ml_scaler.n_features_in_ if ml_scaler is not None else X_up.shape[1]
                                    X_sc = ml_scaler.transform(X_up[:, :n_sc]) if ml_scaler is not None else X_up
                                    probas = ml_model.predict_proba(X_sc)[:, 1]
                                    preds = (probas >= ml_threshold).astype(int)
                                except Exception as e:
                                    st.error(f"ML 오류: {e}")
                                    probas = np.zeros(len(up_df))
                                    preds = np.zeros(len(up_df), dtype=int)

                                rul_preds = np.full(len(up_df), np.nan)
                                if rf_rul is not None:
                                    try:
                                        X_rsc = seq_scaler.transform(X_up) if seq_scaler is not None else X_up
                                        raw_rul = rf_rul.predict(X_rsc)
                                        if y_scaler is not None:
                                            raw_rul = y_scaler.inverse_transform(
                                                raw_rul.reshape(-1, 1)).flatten()
                                        rul_preds = np.maximum(0, raw_rul)
                                    except Exception:
                                        pass

                            keep_cols = (["minute", "bearing"] + REQUIRED
                                         if "bearing" in up_df.columns else REQUIRED)
                            result_df = up_df[[c for c in keep_cols if c in up_df.columns]].copy()
                            result_df["ML_열화확률(%)"] = (probas * 100).round(1)
                            result_df["ML_판정"] = np.where(preds == 1, "열화", "정상")
                            result_df["RF_RUL_분"] = np.where(
                                np.isnan(rul_preds), "-",
                                np.round(rul_preds).astype(int).astype(str),
                            )

                            def _row_color(row):
                                c = "background-color: #FFDDDD" if row["ML_판정"] == "열화" else ""
                                return [c] * len(row)

                            st.dataframe(
                                result_df.style.apply(_row_color, axis=1),
                                use_container_width=True,
                            )
                            n_deg = int(preds.sum())
                            st.metric(
                                "열화 감지 행 수",
                                f"{n_deg}/{len(up_df)} ({n_deg/len(up_df)*100:.1f}%)",
                            )
                            csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                "결과 CSV 다운로드",
                                data=csv_bytes,
                                file_name="femto_diagnosis_result.csv",
                                mime="text/csv",
                            )

                except Exception as e:
                    st.error(f"파일 처리 오류: {e}")

            else:
                with st.expander("업로드 CSV 샘플 형식 보기"):
                    sample = pd.DataFrame([
                        {"minute": 100, "h_rms": 0.55, "h_kurt": 3.1, "h_skew": 0.01,
                         "h_crest": 3.6, "v_rms": 0.44, "v_kurt": 3.0, "v_skew": 0.00,
                         "v_crest": 3.7, "temp_mean": 0.0},
                        {"minute": 200, "h_rms": 2.80, "h_kurt": 8.5, "h_skew": 0.40,
                         "h_crest": 9.2, "v_rms": 2.10, "v_kurt": 7.0, "v_skew": 0.30,
                         "v_crest": 8.1, "temp_mean": 0.0},
                    ])
                    st.dataframe(sample, use_container_width=True)
                    st.caption("femto_features.csv를 직접 업로드해도 됩니다.")


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

# ════════════════════════════════════════════════════════
# Tab 6: AI 정비 권고 (LLM — Proposal A)
# ════════════════════════════════════════════════════════
with tab6:
    st.header("💡 AI 정비 권고 — LLM 진단 보고서 (Proposal A)")
    st.caption(
        "RAG 유사 사례 + ML 열화 확률 + DL 잔여수명(RUL) 예측을 Claude AI에 전달하여 "
        "자연어 정비 권고 보고서를 생성합니다."
    )

    # ── API 키 상태 표시 ──────────────────────────────────────────────────────
    import os as _os
    _has_api_key = bool(_os.environ.get("ANTHROPIC_API_KEY", ""))
    if _has_api_key:
        st.success("✅ ANTHROPIC_API_KEY 감지됨 — AI 보고서 생성 가능")
    else:
        st.warning(
            "⚠️ ANTHROPIC_API_KEY 미설정 — Mock 모드로 실행됩니다.  \n"
            "실제 AI 보고서: 터미널에서 `set ANTHROPIC_API_KEY=sk-ant-...` 후 재시작"
        )

    st.divider()

    # ── 센서 입력 ─────────────────────────────────────────────────────────────
    st.subheader("1️⃣ 현재 센서값 입력")
    _col1, _col2 = st.columns(2)
    with _col1:
        _h_rms   = st.slider("h_rms (수평 진동 RMS)",  0.0, 10.0, 0.5,  0.01, key="llm_h_rms")
        _h_kurt  = st.slider("h_kurt (첨도)",           0.0, 20.0, 3.0,  0.1,  key="llm_h_kurt")
        _h_skew  = st.slider("h_skew (왜도)",          -5.0,  5.0, 0.0,  0.1,  key="llm_h_skew")
        _h_crest = st.slider("h_crest (파고율)",        1.0, 20.0, 4.0,  0.1,  key="llm_h_crest")
    with _col2:
        _v_rms   = st.slider("v_rms (수직 진동 RMS)",  0.0, 10.0, 0.45, 0.01, key="llm_v_rms")
        _v_kurt  = st.slider("v_kurt (첨도)",           0.0, 20.0, 3.0,  0.1,  key="llm_v_kurt")
        _v_skew  = st.slider("v_skew (왜도)",          -5.0,  5.0, 0.0,  0.1,  key="llm_v_skew")
        _v_crest = st.slider("v_crest (파고율)",        1.0, 20.0, 4.0,  0.1,  key="llm_v_crest")
    _temp = st.slider("온도 (°C)", 20.0, 80.0, 30.0, 0.5, key="llm_temp")

    _sensor_vals = {
        "h_rms": _h_rms, "h_kurt": _h_kurt, "h_skew": _h_skew, "h_crest": _h_crest,
        "v_rms": _v_rms, "v_kurt": _v_kurt, "v_skew": _v_skew, "v_crest": _v_crest,
        "temp_mean": _temp,
        "energy":     _h_rms ** 2 + _v_rms ** 2,
        "health_idx": 1.0 / (1.0 + _h_kurt + _v_kurt),
        "rms_ratio":  _h_rms / (_v_rms + 1e-9),
    }

    st.divider()
    st.subheader("2️⃣ AI 보고서 생성")

    if st.button("🤖 AI 정비 권고 보고서 생성", type="primary", key="llm_generate"):

        with st.spinner("진단 중..."):

            # ── ML 진단 ───────────────────────────────────────────────────────
            _proba, _pred = 0.0, 0
            if ml_model is not None:
                try:
                    _feat_list = features if features else [
                        "h_rms","h_kurt","h_skew","h_crest",
                        "v_rms","v_kurt","v_skew","v_crest","temp_mean",
                    ]
                    _input_arr = np.array([[_sensor_vals.get(f, 0.0) for f in _feat_list]])
                    _n_sc = ml_scaler.n_features_in_ if ml_scaler is not None else _input_arr.shape[1]
                    _X_sc = ml_scaler.transform(_input_arr[:, :_n_sc]) if ml_scaler is not None else _input_arr
                    _proba = float(ml_model.predict_proba(_X_sc)[0][1])
                    _pred  = int(_proba >= ml_threshold)
                except Exception as _e:
                    st.error(f"ML 오류: {_e}")

            # ── DL RUL 예측 ───────────────────────────────────────────────────
            _rul_val = None
            _feat_list2 = features if features else [
                "h_rms","h_kurt","h_skew","h_crest",
                "v_rms","v_kurt","v_skew","v_crest","temp_mean",
            ]
            _input_arr2 = np.array([[_sensor_vals.get(f, 0.0) for f in _feat_list2]])
            if lstm_rul is not None and seq_scaler is not None:
                try:
                    _seq = np.tile(_input_arr2, (30, 1))
                    _seq_sc = seq_scaler.transform(_seq)[np.newaxis, :, :]
                    _r = float(lstm_rul.predict(_seq_sc, verbose=0)[0][0])
                    _rul_val = max(0.0, float(
                        y_scaler.inverse_transform([[_r]])[0][0] if y_scaler else _r
                    ))
                except Exception:
                    pass
            if _rul_val is None and rf_rul is not None and seq_scaler is not None:
                try:
                    _sc2 = seq_scaler.transform(_input_arr2)
                    _r2 = float(rf_rul.predict(_sc2)[0])
                    _rul_val = max(0.0, float(
                        y_scaler.inverse_transform([[_r2]])[0][0] if y_scaler else _r2
                    ))
                except Exception:
                    pass

            # ── RAG 유사 사례 검색 ────────────────────────────────────────────
            _rag_cases = []
            try:
                from src.femto_rag_search import load_index, search as rag_search
                _idx, _meta, _feats, _sc = load_index()
                _rag_cases = rag_search(
                    _sensor_vals, index=_idx, meta=_meta, features=_feats, scaler=_sc, k=3
                )
            except Exception as _re:
                st.caption(f"RAG 검색 미지원 (FAISS 미설치 또는 인덱스 없음): {_re}")

            # ── 중간 결과 표시 ────────────────────────────────────────────────
            _c1, _c2, _c3 = st.columns(3)
            with _c1:
                st.metric("ML 열화 확률", f"{_proba*100:.1f}%")
                if _pred == 1:
                    st.error(f"열화 감지 (임계값 {ml_threshold:.2f})")
                else:
                    st.success("정상")
            with _c2:
                if _rul_val is not None:
                    st.metric("예측 잔여수명", f"{_rul_val:.0f} 분")
                    if _rul_val <= rul_threshold:
                        st.error(f"긴급 ({rul_threshold}분 이하)")
                    else:
                        st.success("양호")
                else:
                    st.metric("예측 잔여수명", "DL 모델 없음")
            with _c3:
                st.metric("RAG 유사 사례", f"{len(_rag_cases)}건")
                if _rag_cases:
                    st.caption(f"최유사: {_rag_cases[0]['bearing']} ({_rag_cases[0]['similarity']:.1f}%)")

            # ── LLM 보고서 생성 ───────────────────────────────────────────────
            st.divider()
            st.subheader("🤖 AI 진단 보고서")
            try:
                from src.femto_llm_report import generate_report, generate_report_mock
                if _has_api_key:
                    _report = generate_report(
                        sensor=_sensor_vals,
                        ml_prob=_proba, ml_label=_pred, ml_threshold=ml_threshold,
                        rul_min=_rul_val, rul_alarm_min=float(rul_threshold),
                        rag_cases=_rag_cases,
                    )
                else:
                    _report = generate_report_mock(
                        sensor=_sensor_vals,
                        ml_prob=_proba, ml_label=_pred,
                        rul_min=_rul_val, rul_alarm_min=float(rul_threshold),
                    )
                st.markdown(
                    f"<div style='background:#f0f7ff;border-left:4px solid #2E75B6;"
                    f"padding:16px;border-radius:4px;white-space:pre-wrap;font-size:14px'>"
                    f"{_report}</div>",
                    unsafe_allow_html=True,
                )
            except Exception as _le:
                st.error(f"보고서 생성 오류: {_le}")

    # ── LLM 아키텍처 설명 ─────────────────────────────────────────────────────
    with st.expander("📐 Proposal A — LLM 통합 아키텍처 설명"):
        st.markdown("""
**데이터 흐름 (Proposal A)**

```
센서 측정값 (h_rms, h_kurt, v_rms, temp_mean ...)
       │
       ├─► ML 모델 (RF/XGB) ─────► 열화 확률 (0~1) + 판정(정상/열화)
       │
       ├─► DL 모델 (GRU/LSTM) ──► 잔여수명(RUL) 예측 (분)
       │
       └─► RAG (FAISS 인덱스) ──► 유사 사례 Top-3 (bearing, 유사도, RUL, 상태)
                    │
                    ▼
           Claude API (claude-haiku-4-5)
           System: PdM 전문가 프롬프트
           User: 센서 + ML + DL + RAG 컨텍스트
                    │
                    ▼
           자연어 진단 보고서 (4개 섹션 300자)
           현재상태 | 이상신호 | 정비권고 | 유사사례
```

**모델**: `claude-haiku-4-5-20251001` (빠른 응답, 낮은 비용)
**환경변수**: `ANTHROPIC_API_KEY` 미설정 시 규칙 기반 Mock 보고서 자동 전환
**RAG**: FAISS IndexFlatIP + 12-dim 특성 벡터 + 코사인 유사도 (`python -m src.femto_rag_search` 로 인덱스 빌드)
        """)
