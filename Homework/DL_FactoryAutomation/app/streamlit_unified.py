"""
CNC 설비 고장 예측 - 3-Source 통합 Streamlit 앱
Source 1: ML (XGBoost, AI4I 정형 데이터)
Source 2: DL (CNN+LSTM, 합성 시계열)
Source 3: DL (1D-CNN, NASA Milling 원시신호)
"""
import streamlit as st
import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="CNC 설비 고장 예측 - ML+DL 통합",
    page_icon="??",
    layout="wide",
)

# ── 경로 설정 ──────────────────────────────────────────────────────────────
DL_ROOT = Path(__file__).parent.parent
ML_ROOT = DL_ROOT.parent / "ML_FactoryAutomation"
MDL_DIR = DL_ROOT / "models"
DATA_DIR = DL_ROOT / "data"

# ── 캐시 로더 ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_ml_model():
    import pickle
    scaler = pickle.load(open(ML_ROOT / "model/scaler.pkl", "rb"))
    xgb    = pickle.load(open(ML_ROOT / "model/xgb_model.pkl", "rb"))
    info   = json.load(open(ML_ROOT / "model/model_info.json"))
    return scaler, xgb, info

@st.cache_resource
def load_dl_ts_model():
    import tensorflow as tf
    model = tf.keras.models.load_model(str(MDL_DIR / "cnn_lstm_binary.keras"))
    norm  = np.load(str(MDL_DIR / "cnn_lstm_binary_norm.npz"))
    return model, norm["arr_0"], norm["arr_1"]

@st.cache_resource
def load_dl_milling_model():
    import tensorflow as tf
    keras_path = MDL_DIR / "milling_cnn1d.keras"
    if not keras_path.exists():
        return None, None, None
    model = tf.keras.models.load_model(str(keras_path))
    norm  = np.load(str(DATA_DIR / "milling_norm.npz"))
    cmp   = json.load(open(MDL_DIR / "milling_comparison.json", encoding="utf-8")) if (MDL_DIR / "milling_comparison.json").exists() else {}
    return model, norm, cmp

@st.cache_resource
def load_demo_ts():
    demo_dir = DATA_DIR / "demo"
    files = sorted(demo_dir.glob("demo_*.csv"))[:20]
    return files

def load_milling_demo():
    d = np.load(DATA_DIR / "milling_test.npz")
    return d["X"], d["y"]

# ── 유틸 ──────────────────────────────────────────────────────────────────
def risk_level(prob):
    if prob < 0.30: return "?? 정상", "green"
    elif prob < 0.55: return "?? 주의", "orange"
    elif prob < 0.75: return "?? 위험", "red"
    else: return "?? 긴급", "darkred"

# ──────────────────────────────────────────────────────────────────────────
# 헤더
st.title("?? CNC 설비 고장 예측 통합 플랫폼")
st.markdown("""
> **DL은 ML이 처리하지 못하는 시계열·원시신호 데이터를 추가로 활용하여
> 고장 탐지 범위와 정확도를 함께 개선합니다**  
> *(Source 1: ML/정형 | Source 2: DL/시계열 | Source 3: DL/원시신호)*
""")

tabs = st.tabs(["?? ML 진단 (Source 1)", "?? DL 시계열 (Source 2)", "?? DL 원시신호 (Source 3)", "?? 통합 비교"])

# ══════════════════════════════════════════════════════════════════════════
# 탭 1: ML 진단 (XGBoost + AI4I 정형)
# ══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Source 1 — ML 진단 (XGBoost, AI4I 정형 스냅샷)")
    st.caption("단일 시점 센서값으로 고장 여부를 즉시 판단 (특성공학 불필요)")
    
    col1, col2 = st.columns(2)
    with col1:
        equip_type = st.selectbox("설비 유형", ["L (경량)", "M (중형)", "H (중장비)"])
        rpm   = st.slider("회전속도 (rpm)", 1000, 3000, 1540)
        torque = st.slider("토크 (Nm)", 3.0, 77.0, 40.0, 0.5)
        tool_wear = st.slider("공구 마모 (min)", 0, 250, 100)
    with col2:
        air_temp  = st.slider("공기 온도 (K)", 295.0, 305.0, 300.0, 0.1)
        proc_temp = st.slider("프로세스 온도 (K)", 305.0, 315.0, 310.0, 0.1)
    
    if st.button("ML 고장 예측", type="primary", key="ml_btn"):
        try:
            scaler, xgb, info = load_ml_model()
            type_map = {"L (경량)": [0,1,0], "M (중형)": [0,0,1], "H (중장비)": [1,0,0]}
            th, tl, tm = type_map[equip_type]
            omega = rpm * 2 * 3.14159 / 60
            power = torque * omega
            overstrain = tool_wear * torque
            temp_diff  = proc_temp - air_temp
            feat = np.array([[air_temp, proc_temp, rpm, torque, tool_wear,
                               power, overstrain, temp_diff, th, tl, tm]])
            feat_s = scaler.transform(feat)
            prob = float(xgb.predict_proba(feat_s)[0][1])
            pred = 1 if prob >= 0.75 else 0
            
            level, color = risk_level(prob)
            st.metric("고장 확률", f"{prob*100:.1f}%")
            st.markdown(f"### 판정: <span style='color:{color}'>{level}</span>", unsafe_allow_html=True)
            st.info("ML(XGBoost): 단일 시점 11개 피처 기반 즉시 판단. 시간 패턴 미반영.")
        except Exception as e:
            st.error(f"ML 모델 로드 실패: {e}")

# ══════════════════════════════════════════════════════════════════════════
# 탭 2: DL 시계열 (CNN+LSTM)
# ══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Source 2 — DL 시계열 진단 (CNN+LSTM, 50 타임스텝)")
    st.caption("50스텝 열화 궤적을 학습 — ML이 놓치는 '조기 고장 신호'를 탐지")
    
    demo_files = load_demo_ts()
    col1, col2 = st.columns([1, 2])
    with col1:
        sel = st.selectbox("데모 시퀀스 선택", [f.name for f in demo_files])
        if st.button("DL 시계열 예측", type="primary", key="dl_ts_btn"):
            try:
                from src.config import FEATURE_COLS
                df = pd.read_csv(DATA_DIR / "demo" / sel)
                label = int(df["failure"].iloc[-1])
                model, mean, std = load_dl_ts_model()
                X = df[FEATURE_COLS].values.astype("float32")
                mean2 = np.load(str(MDL_DIR / "cnn_lstm_binary_norm.npz"))
                # normalize
                X_n = (X - mean2["arr_0"].squeeze()) / (mean2["arr_1"].squeeze() + 1e-8)
                prob = float(model.predict(X_n[np.newaxis], verbose=0)[0][0])
                level, color = risk_level(prob)
                st.metric("고장 확률", f"{prob*100:.1f}%")
                st.markdown(f"### <span style='color:{color}'>{level}</span>", unsafe_allow_html=True)
                st.caption(f"실제 레이블: {'고장' if label else '정상'}")
                st.info("CNN+LSTM: 50 타임스텝 전체 패턴 학습 → 열화 궤적으로 조기 감지")
            except Exception as e:
                st.error(f"DL 모델 오류: {e}")
    with col2:
        if demo_files:
            try:
                from src.config import SENSOR_COLS
                df_demo = pd.read_csv(DATA_DIR / "demo" / (sel if "sel" in dir() else demo_files[0].name))
                st.line_chart(df_demo[SENSOR_COLS].reset_index(drop=True),
                              use_container_width=True, height=300)
                st.caption("센서 열화 궤적 (50 타임스텝)")
            except: pass

# ══════════════════════════════════════════════════════════════════════════
# 탭 3: DL 원시신호 (1D-CNN + NASA Milling)
# ══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Source 3 — DL 원시신호 진단 (1D-CNN, NASA Milling)")
    st.caption("CNC 절삭 중 진동·절삭력·AE 신호 512샘플 → 공구 마모 판정")
    
    dl_milling, norm_m, cmp_m = load_dl_milling_model()
    
    if dl_milling is None:
        st.warning("Source 3 모델 학습 중... 잠시 후 다시 확인해 주세요.")
    else:
        X_te_m, y_te_m = load_milling_demo()
        idx = st.slider("테스트 세그먼트 선택", 0, len(X_te_m)-1, 0)
        
        col1, col2 = st.columns([1,2])
        with col1:
            if st.button("1D-CNN 공구마모 예측", type="primary", key="mill_btn"):
                seg = X_te_m[idx:idx+1]
                prob = float(dl_milling.predict(seg, verbose=0)[0][0])
                level, color = risk_level(prob)
                st.metric("마모 확률", f"{prob*100:.1f}%")
                st.markdown(f"### <span style='color:{color}'>{level}</span>", unsafe_allow_html=True)
                st.caption(f"실제: {'마모' if y_te_m[idx] else '정상'}")
        with col2:
            ch_names = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
            seg_df = pd.DataFrame(X_te_m[idx], columns=ch_names)
            st.line_chart(seg_df[["vib_table", "AE_spindle", "smcAC"]], height=280)
            st.caption("원시 신호 (진동·AE·전류)")

# ══════════════════════════════════════════════════════════════════════════
# 탭 4: 통합 비교 대시보드
# ══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("ML vs DL 통합 성능 비교")
    
    cmp_path = MDL_DIR / "milling_comparison.json"
    s2_path  = MDL_DIR / "cnn_lstm_binary_cv_results.json"
    
    rows = []
    # Source 1 (ML)
    try:
        mi = json.load(open(ML_ROOT / "model/model_info.json"))
        xgb_m = mi["models"]["XGBoost"]["metrics"]
        rows.append({"소스": "Source 1 - ML", "모델": "XGBoost",
                     "데이터": "AI4I 정형 (10K행)",
                     "Accuracy": f"{xgb_m['accuracy']:.3f}",
                     "F1(고장)": f"{xgb_m['f1']:.3f}",
                     "ROC-AUC": f"{xgb_m['roc_auc']:.3f}",
                     "특징": "빠른 단일값 판단"})
    except: pass
    
    # Source 2 (DL 시계열)
    try:
        cv = json.load(open(s2_path))
        rows.append({"소스": "Source 2 - DL", "모델": "CNN+LSTM",
                     "데이터": "합성 시계열 (60K 시퀀스)",
                     "Accuracy": f"{cv['test']['compile_metrics']:.3f}",
                     "F1(고장)": "학습 중",
                     "ROC-AUC": "-",
                     "특징": "50스텝 열화 궤적 감지"})
    except: pass
    
    # Source 3 (DL 원시신호)
    if cmp_path.exists():
        try:
            cmp = json.load(open(cmp_path, encoding="utf-8"))
            rows.append({"소스": "Source 3a - ML", "모델": "RandomForest + FFT",
                         "데이터": "NASA Milling (원시신호)",
                         "Accuracy": f"{cmp['ML_RandomForest']['accuracy']:.3f}",
                         "F1(고장)": f"{cmp['ML_RandomForest']['f1']:.3f}",
                         "ROC-AUC": f"{cmp['ML_RandomForest']['roc_auc']:.3f}",
                         "특징": "수동 FFT 피처 필요"})
            rows.append({"소스": "Source 3b - DL", "모델": "1D-CNN",
                         "데이터": "NASA Milling (원시신호)",
                         "Accuracy": f"{cmp['DL_1DCNN']['accuracy']:.3f}",
                         "F1(고장)": f"{cmp['DL_1DCNN']['f1']:.3f}",
                         "ROC-AUC": f"{cmp['DL_1DCNN']['roc_auc']:.3f}",
                         "특징": "원시신호 직접 처리"})
        except: pass
    
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("소스"), use_container_width=True)
    else:
        st.info("모델 학습 완료 후 비교 결과가 표시됩니다.")
    
    st.divider()
    st.markdown("""
    ### ?? 핵심 주장
    > **"DL은 ML이 처리하지 못하는 시계열·원시신호 데이터를 추가로 활용하여  
    > 고장 탐지 범위와 정확도를 함께 개선한다"**
    
    | 계층 | 모델 | ML 대비 강점 | 현장 적용 |
    |------|------|------------|---------|
    | Layer 1 | XGBoost | 빠른 판단, 해석 용이 | SCADA 실시간 알람 |
    | Layer 2 | CNN+LSTM | 열화 궤적 50스텝 패턴 | 조기 경보 (예방 정비) |
    | Layer 3 | 1D-CNN | 원시신호 직접 처리, 수동 피처 불필요 | 절삭 품질 모니터링 |
    """)
    st.divider()
    st.markdown("### Insight — 이번 실험에서 얻은 핵심 교훈")

    col_a, col_b = st.columns(2)
    with col_a:
        st.error("""
**Insight 1: 데이터 품질 > 모델 선택**

Source 3 초기 실험: RF와 1D-CNN 모두 ROC-AUC = 0.50 (랜덤 수준)  
원인: 전역 채널 정규화 후 RF 피처(RMS/std)가 모두 ~1.0 으로 수렴  
→ 아무리 복잡한 모델도 피처 정보가 없으면 학습 불가

> **전처리 실패 = ML 실패 = DL 실패 (동일)**
""")

        st.success("""
**Insight 2: DL의 실질적 강점은 '피처 엔지니어링 자동화'**

ML(RF): 도메인 전문가가 FFT·RMS 피처를 직접 설계해야 함  
DL(1D-CNN): 원시신호에서 컨볼루션으로 자동 피처 추출  
→ 전처리가 올바르면 DL이 수동 피처보다 미세 패턴 포착에 유리

단, 데이터 < 1,000개 구간에서는 ML이 여전히 우위
""")

    with col_b:
        st.info("""
**Insight 3: 3-계층 구조가 Reasonable한 이유**

| 계층 | 데이터 | ML 탐지 | DL 추가 탐지 |
|------|--------|---------|------------|
| L1 | SCADA 정형 | 순간 이상 O | - |
| L2 | 시계열 추이 | 제한적 | 열화 궤적 O |
| L3 | 진동 원시 | FFT 필요 | 자동 패턴 O |

각 계층이 이전 계층의 탐지 사각지대를 보완  
→ 전체 Recall 확장 = 미탐지(FN) 감소 = 비계획 정지 절감
""")

        st.warning("""
**Insight 4: 공장 현장 도입 현실 체크**

- L1(ML/SCADA): 즉시 배포 가능, 추가 하드웨어 없음
- L2(DL/시계열): IoT 센서 + 엣지 컴퓨팅 필요, 3~6개월
- L3(DL/원시신호): 고주파 DAQ + 전처리 튜닝, 6~12개월

**DL은 ML을 대체하는 것이 아니라 ML 위에 쌓는 것**  
Siemens MindSphere · GE Predix · Bosch AI Factory 동일 구조 채택
""")

