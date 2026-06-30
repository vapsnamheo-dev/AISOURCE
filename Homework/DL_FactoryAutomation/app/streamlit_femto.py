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
    page_title="설비 고장 예측 (PdM-Guard)",
    page_icon="⚙️",
    layout="wide",
)
st.title("⚙️ 설비 고장 예측 (PdM-Guard)")
st.caption("FEMTO-ST 베어링 예지보전 — ML 열화 분류 + DL RUL 예측 통합 시스템")

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
        except Exception as e:
            import streamlit as _st
            _st.session_state["lstm_load_error"] = str(e)
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
    "🖼️ CNN 이미지 분류",
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
        import html as _html
        def _vif_bg(val):
            if not isinstance(val, (int, float)) or (isinstance(val, float) and np.isnan(val)):
                return "#444444", "#FFFFFF"
            if val >= 10:
                return "#CC3333", "#FFFFFF"
            if val >= 5:
                return "#CC9900", "#000000"
            return "#2d7a2d", "#FFFFFF"

        header = "<tr>" + "".join(
            f'<th style="background:#1a4a7a;color:#FFFFFF;padding:8px 12px;text-align:left;border-bottom:2px solid #4a8abf;">{_html.escape(str(c))}</th>'
            for c in vif_df.columns
        ) + "</tr>"

        rows_html = ""
        for i, row in vif_df.iterrows():
            cells = ""
            for col in vif_df.columns:
                val = row[col]
                if col == "VIF":
                    bg, fg = _vif_bg(val)
                    txt = f"{val:.4f}" if isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val)) else str(val)
                    cells += f'<td style="background:{bg};color:{fg};padding:8px 12px;font-weight:bold;text-align:right;border-bottom:1px solid #333;">{_html.escape(txt)}</td>'
                else:
                    cells += f'<td style="color:#E0E0E0;padding:8px 12px;border-bottom:1px solid #333;">{_html.escape(str(val))}</td>'
            rows_html += f"<tr>{cells}</tr>"

        st.html(f"""
        <style>.vif-tbl{{border-collapse:collapse;width:100%;font-size:14px;}}
        .vif-tbl tr:hover td{{background-color:rgba(255,255,255,0.05)!important;}}</style>
        <div style="overflow-x:auto;border-radius:6px;border:1px solid #333;">
        <table class="vif-tbl"><thead>{header}</thead><tbody>{rows_html}</tbody></table></div>
        """)
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
                perf_df.style.highlight_max(axis=0, color="#2d7a2d", subset=["Recall", "F1", "ROC-AUC"]),
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

    # Confusion Matrix — OOS 테스트 데이터로 실시간 생성 (임계값 반영)
    st.subheader("최고 모델 Confusion Matrix")
    if ml_results and not df.empty and ml_model is not None and features:
        best_name = ml_results.get("_best_model", "")
        st.write(f"임계값 = **{ml_threshold:.2f}** (사이드바 슬라이더로 실시간 조정)")
        try:
            from sklearn.metrics import confusion_matrix as _sk_cm
            import plotly.graph_objects as _go
            _split_col = "split" if "split" in df.columns else None
            _test_df = df[df[_split_col] == "test"].copy() if _split_col else df.copy()
            if _test_df.empty:
                _test_df = df.copy()
            _feat_cols = [f for f in features if f in _test_df.columns]
            _X = _test_df[_feat_cols].fillna(0).values
            _y_true = _test_df["label"].values
            _n = ml_scaler.n_features_in_ if ml_scaler is not None else _X.shape[1]
            _X_sc = ml_scaler.transform(_X[:, :_n]) if ml_scaler is not None else _X
            _y_prob = ml_model.predict_proba(_X_sc)[:, 1]
            _y_pred = (_y_prob >= ml_threshold).astype(int)
            _cm = _sk_cm(_y_true, _y_pred)
            _fig_cm = _go.Figure(data=_go.Heatmap(
                z=_cm,
                x=["예측: 정상", "예측: 열화"],
                y=["실제: 정상", "실제: 열화"],
                colorscale=[[0, "#1a4a7a"], [0.5, "#c47d00"], [1, "#2d7a2d"]],
                text=_cm, texttemplate="%{text}",
                textfont={"size": 24, "color": "white"}, showscale=False,
            ))
            _fig_cm.update_layout(height=340, margin=dict(t=50, b=50, l=80, r=40),
                title=dict(text=f"{best_name} — OOS Confusion Matrix (임계값 {ml_threshold:.2f})", font=dict(size=13)),
                xaxis=dict(side="bottom"))
            _col_cm, _col_cm_stat = st.columns([3, 2])
            with _col_cm:
                st.plotly_chart(_fig_cm, use_container_width=True)
            with _col_cm_stat:
                _tn, _fp, _fn, _tp = _cm.ravel()
                _prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
                _rec  = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
                _f1   = 2 * _prec * _rec / (_prec + _rec) if (_prec + _rec) > 0 else 0.0
                st.metric("TP — 열화 정탐 ✅", int(_tp))
                st.metric("FP — 정상 오경보 ⚠️", int(_fp))
                st.metric("FN — 열화 미탐 🔴", int(_fn))
                st.metric("TN — 정상 정탐 ✅", int(_tn))
                st.divider()
                st.metric("Precision", f"{_prec:.4f}")
                st.metric("Recall",    f"{_rec:.4f}")
                st.metric("F1-Score",  f"{_f1:.4f}")
            st.caption(f"※ OOS 테스트 베어링 {_test_df['bearing'].nunique() if 'bearing' in _test_df else '?'}종 기준 | 임계값 {ml_threshold:.2f} 실시간 반영")
        except Exception as _cm_err:
            st.error(f"Confusion Matrix 생성 실패: {_cm_err}")


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
                        _err = st.session_state.get("lstm_load_error")
                        if _err:
                            st.error(f"DL 모델 로드 실패: {_err}")
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

            # ── 입력 방법 선택 ────────────────────────────────────────────────
            _csv_mode = st.radio(
                "입력 방법",
                ["📁 파일 업로드", "📦 레포 샘플 불러오기"],
                horizontal=True,
                key="csv_input_mode",
            )

            _DEMO_CSV_PATHS = {
                "femto_demo.csv (3,305행 · 정상+열화 혼합)": ROOT / "demo data" / "femto_demo.csv",
            }

            up_df = None

            if _csv_mode == "📁 파일 업로드":
                # 모드 전환 시 이전 샘플 캐시 초기화
                st.session_state.pop("_femto_sample_df", None)
                _csv_file = st.file_uploader(
                    "FEMTO 피처 CSV 업로드",
                    type=["csv"],
                    key="femto_csv_upload",
                )
                if _csv_file is not None:
                    up_df = pd.read_csv(_csv_file)
            else:
                _avail = {k: v for k, v in _DEMO_CSV_PATHS.items() if v.exists()}
                if not _avail:
                    st.warning("레포에 샘플 파일이 없습니다.")
                else:
                    _sel_csv = st.selectbox("샘플 데이터 선택", list(_avail.keys()), key="csv_sample_sel")
                    if st.button("📦 레포 샘플 로드", key="load_sample_csv"):
                        st.session_state["_femto_sample_df"] = pd.read_csv(_avail[_sel_csv])
                    if "_femto_sample_df" in st.session_state:
                        up_df = st.session_state["_femto_sample_df"]
                    else:
                        st.info("위에서 샘플을 선택하고 [레포 샘플 로드] 버튼을 누르세요.")

            if up_df is not None:
                try:
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

                            n_deg    = int(preds.sum())
                            n_total  = len(up_df)
                            n_norm   = n_total - n_deg
                            deg_rate = n_deg / n_total * 100

                            # ── 요약 지표 (4열) ───────────────────────────────
                            _sm1, _sm2, _sm3, _sm4 = st.columns(4)
                            _sm1.metric("총 분석 행", f"{n_total:,} 행")
                            _sm2.metric("정상 판정 🟢", f"{n_norm:,} 행",
                                        delta=f"{100-deg_rate:.1f}%",
                                        delta_color="normal")
                            _sm3.metric("열화 감지 🔴", f"{n_deg:,} 행",
                                        delta=f"{deg_rate:.1f}%",
                                        delta_color="inverse")
                            if "bearing" in result_df.columns:
                                _risk_top = (result_df.groupby("bearing")["ML_열화확률(%)"]
                                             .mean().sort_values(ascending=False))
                                _sm4.metric("최고위험 베어링",
                                            f"{_risk_top.index[0]}",
                                            delta=f"열화확률 {_risk_top.iloc[0]:.1f}%",
                                            delta_color="inverse")
                            else:
                                _sm4.metric("평균 열화확률", f"{probas.mean()*100:.1f}%")

                            # ── 상태 배너 ─────────────────────────────────────
                            if deg_rate >= 50:
                                st.error(f"🔴 경고: 전체 데이터의 {deg_rate:.1f}%가 열화 구간으로 판정되었습니다. 즉시 점검이 필요합니다.")
                            elif deg_rate >= 20:
                                st.warning(f"⚠️ 주의: 전체의 {deg_rate:.1f}%가 열화 구간입니다. 모니터링을 강화하세요.")
                            else:
                                st.success(f"✅ 양호: 전체의 {deg_rate:.1f}%만 열화로 감지되었습니다.")

                            st.divider()

                            # ── 진단 결과 테이블 ──────────────────────────────
                            _n_bear = result_df["bearing"].nunique() if "bearing" in result_df.columns else "-"
                            _rul_valid = result_df["RF_RUL_분"][result_df["RF_RUL_분"] != "-"]
                            _rul_info = (f"평균 RF_RUL {pd.to_numeric(_rul_valid, errors='coerce').mean():.0f}분"
                                         if not _rul_valid.empty else "RUL 모델 미적용")

                            st.subheader("📋 베어링별 ML 진단 결과")
                            st.caption(
                                f"총 **{n_total:,}행** · 베어링 **{_n_bear}종** · "
                                f"열화 **{n_deg:,}행({deg_rate:.1f}%)** / 정상 **{n_norm:,}행** · "
                                f"{_rul_info}  |  "
                                f"🔴 분홍 행 = ML 열화 판정 (임계값 {ml_threshold:.2f}) · "
                                f"RF_RUL_분 = 랜덤포레스트 잔여수명 예측값(분)"
                            )

                            def _row_color(row):
                                c = "background-color: #FFDDDD" if row["ML_판정"] == "열화" else ""
                                return [c] * len(row)

                            st.dataframe(
                                result_df.style.apply(_row_color, axis=1),
                                use_container_width=True,
                                height=320,
                            )

                            csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                "⬇️ 결과 CSV 다운로드",
                                data=csv_bytes,
                                file_name="femto_diagnosis_result.csv",
                                mime="text/csv",
                            )

                            # ── Grad-CAM: CSV 진동 → 이미지 → CNN → 히트맵 ────────
                            st.divider()
                            st.subheader("🔥 Grad-CAM — 최고위험 베어링 진동 분석")
                            st.caption(
                                "열화 확률 최고 베어링의 진동 RMS를 이미지로 변환 → CNN 판정 → "
                                "Grad-CAM으로 **어느 시점/구간이 판정에 결정적이었는지** 시각화합니다."
                            )
                            try:
                                import tensorflow as _tf_gc
                                import matplotlib.pyplot as _plt_gc
                                import matplotlib.cm as _cm_gc
                                import io as _io_gc
                                from PIL import Image as _PIL_gc
                                import numpy as _np_gc

                                # 최고위험 베어링 자동 선택
                                if "bearing" in result_df.columns:
                                    _gc_risk = result_df.groupby("bearing")["ML_열화확률(%)"].mean().sort_values(ascending=False)
                                    _gc_target = _gc_risk.index[0]
                                    _gc_df_sel = up_df[up_df["bearing"] == _gc_target].copy()
                                    st.info(f"📍 분석 대상: **베어링 {_gc_target}** (평균 열화확률 {_gc_risk.iloc[0]:.1f}%)")
                                else:
                                    _gc_target = "전체"
                                    _gc_df_sel = up_df.copy()

                                # 진동 RMS 시계열 → matplotlib 이미지
                                _gc_time  = _gc_df_sel["minute"].values if "minute" in _gc_df_sel.columns else range(len(_gc_df_sel))
                                _gc_hrms  = _gc_df_sel["h_rms"].values  if "h_rms"  in _gc_df_sel.columns else _gc_df_sel.iloc[:, 0].values
                                _gc_vrms  = _gc_df_sel["v_rms"].values  if "v_rms"  in _gc_df_sel.columns else None
                                _avg_prob = (_gc_risk.iloc[0] if "bearing" in result_df.columns else probas.mean() * 100)
                                _vib_col  = "red" if _avg_prob >= ml_threshold * 100 else "green"

                                _fig_vib, _ax_vib = _plt_gc.subplots(figsize=(8, 3), facecolor="white")
                                _ax_vib.plot(_gc_time, _gc_hrms, color=_vib_col, linewidth=1.5, label="h_rms (수평)")
                                if _gc_vrms is not None:
                                    _ax_vib.plot(_gc_time, _gc_vrms, color="orange", linewidth=1.0, alpha=0.7, linestyle="--", label="v_rms (수직)")
                                _ax_vib.set_xlabel("경과 시간 (분)"); _ax_vib.set_ylabel("RMS 가속도 (g)")
                                _ax_vib.set_title(f"베어링 {_gc_target} — 진동 RMS 시계열")
                                _ax_vib.legend(fontsize=8); _ax_vib.tick_params(labelsize=7)
                                _plt_gc.tight_layout()
                                _vib_buf = _io_gc.BytesIO()
                                _fig_vib.savefig(_vib_buf, format="png", dpi=96, bbox_inches="tight")
                                _plt_gc.close(_fig_vib); _vib_buf.seek(0)
                                _vib_pil = _PIL_gc.open(_vib_buf).convert("RGB")

                                # CNN 모델 로드 & 예측
                                if _cnn_model_path and _cnn_model_path.exists():
                                    @st.cache_resource
                                    def _csv_cnn(p): return _tf_gc.keras.models.load_model(p)

                                    _gc_cnn = _csv_cnn(str(_cnn_model_path))
                                    _gc_iH  = _gc_cnn.input_shape[1] or 64
                                    _gc_iW  = _gc_cnn.input_shape[2] or 96
                                    _vib_r  = _vib_pil.resize((_gc_iW, _gc_iH), _PIL_gc.BILINEAR)
                                    _gc_arr = _np_gc.expand_dims(_np_gc.array(_vib_r, dtype=_np_gc.float32) / 255.0, 0)
                                    _gc_p   = _gc_cnn.predict(_gc_arr, verbose=0)[0]
                                    _gc_idx = int(_np_gc.argmax(_gc_p))
                                    _gc_lbl = ["정상(OK)", "결함(Defect)"][_gc_idx]

                                    (st.error if _gc_idx > 0 else st.success)(
                                        f"{'🔴' if _gc_idx>0 else '🟢'} CNN 판정: **{_gc_lbl}** ({_gc_p[_gc_idx]*100:.1f}%)"
                                    )

                                    # Grad-CAM 계산
                                    @st.cache_resource
                                    def _csv_gm(_m):
                                        _lc = None
                                        for _l in _m.layers:
                                            if isinstance(_l, _tf_gc.keras.layers.Conv2D): _lc = _l.name
                                        return _tf_gc.keras.Model(inputs=_m.inputs, outputs=[_m.get_layer(_lc).output, _m.output]) if _lc else None

                                    _gm = _csv_gm(_gc_cnn)
                                    if _gm:
                                        with _tf_gc.GradientTape() as _tp:
                                            _co, _pr = _gm(_tf_gc.cast(_gc_arr, _tf_gc.float32))
                                            _ls = _pr[:, _gc_idx]
                                        _gr  = _tp.gradient(_ls, _co)
                                        _pw  = _tf_gc.reduce_mean(_gr, axis=(0, 1, 2)).numpy()
                                        _cam = _np_gc.einsum("hwc,c->hw", _co[0].numpy(), _pw)
                                        _cam = _np_gc.maximum(_cam, 0)
                                        if _cam.max() > 0: _cam /= _cam.max()

                                        _cam_pil = _PIL_gc.fromarray((_cam * 255).astype(_np_gc.uint8)).resize((_gc_iW, _gc_iH), _PIL_gc.BILINEAR)
                                        _cam_n   = _np_gc.array(_cam_pil) / 255.0
                                        _heat    = _cm_gc.get_cmap("jet")(_cam_n)[:, :, :3]
                                        _orig    = _np_gc.array(_vib_r).astype(float) / 255.0
                                        _ovl     = _np_gc.clip(0.55 * _orig + 0.45 * _heat, 0, 1)

                                        _fig_gc, _ax_gc = _plt_gc.subplots(1, 3, figsize=(13, 4))
                                        for _a, _im, _ti, _cmp in zip(
                                            _ax_gc,
                                            [_np_gc.array(_vib_r), _cam_n, (_ovl * 255).astype(_np_gc.uint8)],
                                            ["① 원본 진동 이미지", "② Grad-CAM 히트맵", "③ 오버레이 결과"],
                                            [None, "jet", None],
                                        ):
                                            _a.imshow(_im, cmap=_cmp); _a.set_title(_ti); _a.axis("off")
                                        _plt_gc.colorbar(_plt_gc.cm.ScalarMappable(cmap="jet"), ax=_ax_gc[1], fraction=0.046, pad=0.04, label="중요도")
                                        _plt_gc.tight_layout()
                                        st.pyplot(_fig_gc); _plt_gc.close(_fig_gc)
                                        st.caption(
                                            f"※ 베어링 {_gc_target} 진동 RMS → 이미지 변환 → CNN → Grad-CAM  |  "
                                            f"빨간 영역 = CNN이 '{_gc_lbl}' 판정 시 집중한 시간 구간"
                                        )
                                else:
                                    st.warning("CNN 모델 없음 — Grad-CAM 생략 (`models/casting_defect_cnn.keras` 필요)")
                            except Exception as _gc_csv_err:
                                st.warning(f"Grad-CAM 생성 실패: {_gc_csv_err}")

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
                styled = cmp_df.style.highlight_min(subset=["OOS RMSE (분)"], color="#2d7a2d", axis=0
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
# Tab 6: CNN 이미지 분류
# ════════════════════════════════════════════════════════
with tab6:
    st.header("🖼️ CNN 이미지 분류 — 결함 판정")
    st.caption("이미지 파일을 업로드하면 CNN 모델이 결함 여부를 판정합니다.")

    # 모델 경로 후보
    _cnn_candidates = [
        MODEL_DIR / "casting_defect_cnn.keras",
        MODEL_DIR / "casting_defect_cnn.h5",
        MODEL_DIR / "image_cnn.keras",
        MODEL_DIR / "image_cnn.h5",
    ]
    _cnn_model_path = next((p for p in _cnn_candidates if p.exists()), None)

    _gradcam_data = None  # (model, arr, pred_idx, img_resized, h, w) — Grad-CAM용
    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.subheader("📂 이미지 파일 선택")

        _img_mode = st.radio(
            "입력 방법",
            ["📁 파일 업로드", "📦 레포 샘플 불러오기"],
            horizontal=True,
            key="cnn_input_mode",
        )

        _IMG_DIR = ROOT / "demo data" / "Bearing_image_file"
        _IMG_SAMPLES = {
            "bearing_normal.png  (정상 베어링 진동)": _IMG_DIR / "bearing_normal.png",
            "bearing_defect.png  (불량 베어링 진동)": _IMG_DIR / "bearing_defect.png",
        }

        # UploadedFile 호환 래퍼
        class _RepoImageFile:
            def __init__(self, path):
                self._data = Path(path).read_bytes()
                self.name = Path(path).name
            def read(self): return self._data

        uploaded = None

        if _img_mode == "📁 파일 업로드":
            st.session_state.pop("_cnn_repo_img", None)
            uploaded = st.file_uploader(
                "JPG / PNG / BMP 파일을 업로드하세요",
                type=["jpg", "jpeg", "png", "bmp"],
                help="제조 공정 이미지 파일 (예: 주조 결함 탐지용)",
            )
        else:
            _avail_imgs = {k: v for k, v in _IMG_SAMPLES.items() if v.exists()}
            if not _avail_imgs:
                st.warning("레포에 샘플 이미지가 없습니다.")
            else:
                _sel_img = st.selectbox("샘플 이미지 선택", list(_avail_imgs.keys()), key="cnn_sample_sel")
                if st.button("📦 레포 샘플 로드", key="load_sample_img"):
                    st.session_state["_cnn_repo_img"] = str(_avail_imgs[_sel_img])
                if "_cnn_repo_img" in st.session_state:
                    uploaded = _RepoImageFile(st.session_state["_cnn_repo_img"])
                else:
                    st.info("위에서 샘플을 선택하고 [레포 샘플 로드] 버튼을 누르세요.")

        if uploaded:
            # _RepoImageFile: read() 항상 전체 반환 / UploadedFile: 직접 전달(버퍼 소모 방지)
            _disp = uploaded.read() if isinstance(uploaded, _RepoImageFile) else uploaded
            st.image(_disp, caption=f"선택: {uploaded.name}", use_container_width=True)

    with col_result:
        st.subheader("🔍 CNN 판정 결과")

        if uploaded is None:
            st.info("왼쪽에서 이미지를 업로드하면 CNN 판정이 시작됩니다.")
        elif _cnn_model_path is None:
            st.warning(
                "CNN 이미지 분류 모델이 없습니다.  \n"
                "아래 경로 중 하나에 모델을 저장하세요:  \n"
                "- `models/casting_defect_cnn.keras`  \n"
                "- `models/image_cnn.keras`  \n\n"
                "**학습 방법**: `python -m src.femto_image_cnn` 실행  \n"
                "(Casting Defect Dataset 필요)"
            )
            # 기본 픽셀 통계 표시 (모델 없어도 이미지 분석)
            st.divider()
            st.caption("기본 이미지 통계 (모델 없음 — 참고용)")
            try:
                from PIL import Image as PILImage
                import numpy as np
                import io
                img_bytes = uploaded.read()
                img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                arr = np.array(img).astype(float)
                st.metric("평균 밝기", f"{arr.mean():.1f}")
                st.metric("표준편차", f"{arr.std():.1f}")
                st.metric("이미지 크기", f"{img.width} × {img.height} px")
                dark_ratio = float((arr.mean(axis=2) < 80).mean())
                st.metric("어두운 영역 비율", f"{dark_ratio*100:.1f}%",
                          help="어두운 영역이 많으면 결함 가능성 높음 (단순 추정)")
                if dark_ratio > 0.3:
                    st.error("⚠️ 어두운 영역 비율 높음 — 결함 의심 (CNN 모델로 정밀 판정 필요)")
                else:
                    st.success("✅ 이미지 밝기 정상 범위")
            except Exception as e:
                st.error(f"이미지 분석 실패: {e}")
        else:
            # CNN 모델 로드 & 예측
            try:
                import numpy as np
                from PIL import Image as PILImage
                import io
                import tensorflow as tf

                @st.cache_resource
                def _load_cnn(path: str):
                    return tf.keras.models.load_model(path)

                cnn_model = _load_cnn(str(_cnn_model_path))
                inp_shape = cnn_model.input_shape  # (None, H, W, C)
                target_h  = inp_shape[1] or 224
                target_w  = inp_shape[2] or 224
                n_classes = cnn_model.output_shape[-1]
                CLASS_NAMES = (
                    ["정상(OK)", "결함(Defect)"] if n_classes == 2
                    else [f"Class {i}" for i in range(n_classes)]
                )

                img_bytes = uploaded.read()
                img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                img_resized = img.resize((target_w, target_h))
                arr = np.array(img_resized, dtype=np.float32) / 255.0
                arr = np.expand_dims(arr, axis=0)

                preds = cnn_model.predict(arr, verbose=0)[0]
                pred_idx = int(np.argmax(preds))
                confidence = float(preds[pred_idx])
                pred_label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Class {pred_idx}"

                if "결함" in pred_label or pred_idx > 0:
                    st.error(f"🔴 판정: **{pred_label}**  ({confidence*100:.1f}%)")
                else:
                    st.success(f"🟢 판정: **{pred_label}**  ({confidence*100:.1f}%)")

                st.divider()
                st.caption("클래스별 확률")
                prob_data = {CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}": float(preds[i])
                             for i in range(len(preds))}
                import pandas as pd
                prob_df = pd.DataFrame({"클래스": list(prob_data.keys()),
                                        "확률": list(prob_data.values())})
                st.dataframe(prob_df.style.format({"확률": "{:.4f}"}), use_container_width=True)
                st.metric("모델", _cnn_model_path.name)
                st.metric("입력 크기", f"{target_h}×{target_w} px")
                _gradcam_data = (cnn_model, arr, pred_idx, img_resized, target_h, target_w)

            except Exception as e:
                st.error(f"CNN 판정 실패: {e}")
                st.code(str(e))

    # ── Grad-CAM 전체 폭 섹션 ──────────────────────────────────
    if _gradcam_data is not None:
        _gc_model, _gc_arr, _gc_pred_idx, _gc_img, _gc_h, _gc_w = _gradcam_data
        st.divider()
        st.subheader("🔥 Grad-CAM 시각화 — CNN이 '어디를 보았는지'")
        st.caption(
            "마지막 Conv 레이어의 Feature Map 기울기(Gradient)를 역전파하여 "
            "예측에 영향을 준 영역을 히트맵으로 표시합니다.  "
            "**빨간색 = 판정에 가장 중요한 영역 / 파란색 = 덜 중요한 영역**"
        )
        try:
            import tensorflow as tf
            import numpy as np
            import matplotlib.pyplot as _plt
            import matplotlib.cm as _cm
            from PIL import Image as _PIL

            @st.cache_resource
            def _build_grad_model(_m):
                last_conv = None
                for layer in _m.layers:
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        last_conv = layer.name
                if last_conv is None:
                    return None
                return tf.keras.Model(
                    inputs=_m.inputs,
                    outputs=[_m.get_layer(last_conv).output, _m.output],
                )

            _grad_model = _build_grad_model(_gc_model)

            if _grad_model is None:
                st.warning("Conv2D 레이어를 찾을 수 없어 Grad-CAM을 생성할 수 없습니다.")
            else:
                # ① 순전파 + 역전파로 기울기 계산
                _gc_tensor = tf.cast(_gc_arr, tf.float32)
                with tf.GradientTape() as _tape:
                    _conv_out, _preds = _grad_model(_gc_tensor)
                    _loss = _preds[:, _gc_pred_idx]
                _grads = _tape.gradient(_loss, _conv_out)          # (1, fH, fW, C)

                # ② 채널별 중요도(가중치) = 공간 평균 기울기
                _pooled = tf.reduce_mean(_grads, axis=(0, 1, 2)).numpy()  # (C,)

                # ③ Feature Map 가중합 + ReLU
                _feat = _conv_out[0].numpy()                               # (fH, fW, C)
                _cam  = np.einsum("hwc,c->hw", _feat, _pooled)            # (fH, fW)
                _cam  = np.maximum(_cam, 0)
                if _cam.max() > 0:
                    _cam /= _cam.max()

                # ④ 원본 크기로 리사이즈
                _cam_pil    = _PIL.fromarray((_cam * 255).astype(np.uint8)).resize(
                    (_gc_w, _gc_h), _PIL.BILINEAR
                )
                _cam_norm   = np.array(_cam_pil) / 255.0

                # ⑤ jet 컬러맵 적용 + 원본과 오버레이
                _heatmap_rgb = _cm.get_cmap("jet")(_cam_norm)[:, :, :3]
                _orig_arr    = np.array(_gc_img).astype(float) / 255.0
                _overlay     = np.clip(0.55 * _orig_arr + 0.45 * _heatmap_rgb, 0, 1)

                # ⑥ 시각화 — 원본 / 히트맵 / 오버레이
                _fig, _axes = _plt.subplots(1, 3, figsize=(13, 4))
                _titles = ["① 원본 이미지", "② Grad-CAM 히트맵", "③ 오버레이 결과"]
                _imgs   = [
                    np.array(_gc_img),
                    _cam_norm,
                    (_overlay * 255).astype(np.uint8),
                ]
                _cmaps  = [None, "jet", None]
                for _ax, _im, _ti, _cmp in zip(_axes, _imgs, _titles, _cmaps):
                    _ax.imshow(_im, cmap=_cmp)
                    _ax.set_title(_ti, fontsize=11, pad=6)
                    _ax.axis("off")
                _plt.colorbar(
                    _plt.cm.ScalarMappable(cmap="jet"), ax=_axes[1],
                    fraction=0.046, pad=0.04, label="중요도"
                )
                _plt.tight_layout()
                st.pyplot(_fig)
                _plt.close(_fig)

                # ⑦ 채널 중요도 상위 5개 표시
                with st.expander("📊 채널별 중요도 상세 (상위 5개)"):
                    _top_idx = np.argsort(_pooled)[::-1][:5]
                    import pandas as _pd2
                    st.dataframe(
                        _pd2.DataFrame({
                            "채널 번호": _top_idx,
                            "중요도(기울기 평균)": _pooled[_top_idx].round(5),
                        }),
                        use_container_width=True,
                    )
                    st.caption("※ 마지막 Conv 레이어 기준 | 양수=예측 클래스 강화, 음수=억제")

        except Exception as _gc_err:
            st.warning(f"Grad-CAM 생성 실패: {_gc_err}")

    st.divider()
    st.caption(
        "**CNN 모델 학습 방법**: `python -m src.femto_image_cnn`  ·  "
        "결함 이미지 데이터셋(ok/defect 폴더 구조) 필요  ·  "
        "학습 완료 후 `models/casting_defect_cnn.keras` 자동 저장"
    )