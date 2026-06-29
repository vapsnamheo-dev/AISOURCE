# -*- coding: utf-8 -*-
"""
FEMTO-ST 베어링 SHAP 예측 설명 분석 (로컬 CPU, GPU 불필요)

실행:
    pip install shap
    python -m src.femto_shap

출력:
    산출물/_imgs/S1_shap_clf.png   — 분류 모델(XGB/RF) SHAP summary beeswarm
    산출물/_imgs/S2_shap_gru.png   — GRU v2 RUL 모델 SHAP summary beeswarm
    산출물/_imgs/S3_shap_bar.png   — 두 모델 피처 중요도 비교 막대

소요 시간: TreeExplainer < 1분 / DeepExplainer 5~15분 (CPU)
"""
from __future__ import annotations

import sys
import time
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT          = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "FEMTO_processed"
MODEL_DIR     = ROOT / "models"
OUT_DIR       = ROOT / "산출물" / "_imgs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── SHAP 설치 확인 ─────────────────────────────────────────────────────────────
try:
    import shap
    print(f"[OK] shap {shap.__version__}")
except ImportError:
    print("[ERROR] shap 미설치. 먼저 실행: pip install shap")
    sys.exit(1)

# ── 피처 목록 ──────────────────────────────────────────────────────────────────
SEL_PATH = PROCESSED_DIR / "selected_features.csv"
CLF_FEATURES = (pd.read_csv(SEL_PATH)["feature"].tolist() if SEL_PATH.exists()
                else ["h_rms","h_kurt","h_skew","h_crest",
                      "v_rms","v_kurt","v_skew","v_crest","temp_mean"])

GRU_FEATURES = None  # 실행 시 gru_model.input_shape[-1]로 결정
WINDOW = 20

print(f"[피처] 분류 {len(CLF_FEATURES)}개: {CLF_FEATURES}")
print("[피처] GRU  → 모델 로드 후 input_shape에서 자동 결정")

# ══════════════════════════════════════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] 데이터 로드...")
df = pd.read_csv(PROCESSED_DIR / "femto_features.csv")
df_test  = df[df["split"] == "test"].copy()
df_train = df[df["split"] == "train"].copy()

for col in CLF_FEATURES:
    if col in df.columns:
        df_test[col]  = df_test[col].fillna(df_test[col].median()).fillna(0.0)
        df_train[col] = df_train[col].fillna(df_train[col].median()).fillna(0.0)

print(f"    테스트 {len(df_test)}행  라벨: {df_test['label'].value_counts().to_dict()}")

clf_shap_ok = False
gru_shap_ok = False
shap_clf = None
shap_gru_mean = None


# ══════════════════════════════════════════════════════════════════════════════
# PART 1. 분류 모델 SHAP — TreeExplainer
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] 분류 모델 SHAP (TreeExplainer) ─ 트리 구조 직접 계산, 배경 불필요")
clf_path = MODEL_DIR / "femto_best_clf.pkl"

if not clf_path.exists():
    print(f"    [SKIP] {clf_path.name} 없음")
else:
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    print(f"    모델: {type(clf).__name__}")

    # 모델이 실제로 학습한 피처 수 자동 감지
    n_clf = getattr(clf, "n_features_in_", None)
    if n_clf is None:
        coef = getattr(clf, "coef_", None)
        n_clf = coef.shape[-1] if coef is not None else len(CLF_FEATURES)
    USE_FEATURES = CLF_FEATURES[:n_clf]
    print(f"    모델 학습 피처 수: {n_clf}개 → {USE_FEATURES}")

    X_clf = df_test[USE_FEATURES].fillna(0).values
    y_clf = df_test["label"].values

    t0 = time.time()
    model_type = type(clf).__name__

    if hasattr(clf, "feature_importances_"):
        # 트리 계열(RF, XGBoost, GBM) → TreeExplainer
        explainer_clf = shap.TreeExplainer(clf)
        shap_clf_raw  = explainer_clf.shap_values(X_clf)
        shap_clf = shap_clf_raw[1] if isinstance(shap_clf_raw, list) else shap_clf_raw
    elif hasattr(clf, "coef_"):
        # 선형 모델(LogisticRegression, Ridge 등) → LinearExplainer
        X_tr_use = df_train[USE_FEATURES].fillna(0).values
        bg_masker = shap.maskers.Independent(X_tr_use, max_samples=200)
        explainer_clf = shap.LinearExplainer(clf, bg_masker)
        shap_clf_raw  = explainer_clf.shap_values(X_clf)
        shap_clf = shap_clf_raw[1] if isinstance(shap_clf_raw, list) else shap_clf_raw
    else:
        # 폴백: KernelExplainer (느림, 200샘플만)
        X_bg_use = shap.sample(df_train[USE_FEATURES].fillna(0).values, 50)
        explainer_clf = shap.KernelExplainer(clf.predict_proba, X_bg_use)
        shap_clf_raw  = explainer_clf.shap_values(X_clf[:200])
        shap_clf = shap_clf_raw[1] if isinstance(shap_clf_raw, list) else shap_clf_raw

    elapsed = time.time() - t0
    print(f"    완료: {elapsed:.1f}초  모델={model_type}  shape={shap_clf.shape}")
    clf_shap_ok = True

    fig = plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_clf, X_clf, feature_names=USE_FEATURES,
                      plot_type="dot", show=False, max_display=len(USE_FEATURES))
    plt.title("분류 모델 SHAP Summary — 열화 감지 (Class 1 기여도)\n"
              "빨강=열화 확률↑, 파랑=열화 확률↓",
              fontsize=11, fontweight="bold")
    plt.tight_layout()
    out1 = OUT_DIR / "S1_shap_clf.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    저장: {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2. GRU v2 RUL 모델 SHAP — DeepExplainer (배경 20, 설명 100)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] GRU RUL 모델 SHAP (DeepExplainer) ─ background=20, explain=100")
gru_path = MODEL_DIR / "femto_best_dl_GRU_v2.keras"

if not gru_path.exists():
    print(f"    [SKIP] {gru_path.name} 없음")
else:
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras.models import load_model
    except ImportError:
        print("    [SKIP] TensorFlow 미설치")
    else:
        gru_model = load_model(str(gru_path))
        n_gru = gru_model.input_shape[-1]
        GRU_FEATURES = CLF_FEATURES[:n_gru]
        print(f"    입력 shape: {gru_model.input_shape}  피처: {GRU_FEATURES}")

        # 스케일러 로드 (피처 수 불일치 시 새로 피팅)
        seq_sc_path = MODEL_DIR / "femto_seq_scaler.pkl"
        seq_scaler  = None
        if seq_sc_path.exists():
            _sc_raw = pickle.load(open(seq_sc_path, "rb"))
            if getattr(_sc_raw, "n_features_in_", n_gru) == n_gru:
                seq_scaler = _sc_raw

        # 슬라이딩 윈도우 시퀀스 생성
        def _make_seq(data_df, features, window):
            X_list, y_list = [], []
            for _, bdf in data_df.groupby("bearing"):
                bdf = bdf.sort_values("minute").reset_index(drop=True)
                feat = bdf[features].copy()
                for c in features:
                    feat[c] = feat[c].fillna(feat[c].median()).fillna(0.0)
                fv = feat.values.astype(np.float32)
                rv = bdf["rul"].values
                for i in range(len(bdf) - window):
                    X_list.append(fv[i:i+window])
                    y_list.append(rv[i+window])
            return np.array(X_list, dtype=np.float32), np.array(y_list)

        X_tr_raw, _ = _make_seq(df_train, GRU_FEATURES, WINDOW)
        X_te_raw, _ = _make_seq(df_test,  GRU_FEATURES, WINDOW)
        n_feat = len(GRU_FEATURES)

        if seq_scaler is not None:
            X_tr_sc = seq_scaler.transform(
                X_tr_raw.reshape(-1, n_feat)).reshape(X_tr_raw.shape)
            X_te_sc = seq_scaler.transform(
                X_te_raw.reshape(-1, n_feat)).reshape(X_te_raw.shape)
        else:
            from sklearn.preprocessing import MinMaxScaler
            _sc = MinMaxScaler()
            X_tr_sc = _sc.fit_transform(
                X_tr_raw.reshape(-1, n_feat)).reshape(X_tr_raw.shape)
            X_te_sc = _sc.transform(
                X_te_raw.reshape(-1, n_feat)).reshape(X_te_raw.shape)

        rng   = np.random.default_rng(42)
        X_bg  = X_tr_sc[rng.choice(len(X_tr_sc), size=min(20, len(X_tr_sc)), replace=False)]
        X_ex  = X_te_sc[rng.choice(len(X_te_sc), size=min(100, len(X_te_sc)), replace=False)]
        print(f"    background={X_bg.shape}  explain={X_ex.shape}")

        t0 = time.time()
        method = None

        # 1차: DeepExplainer / GradientExplainer 시도
        for cls_name, ExplainerClass in [("DeepExplainer", shap.DeepExplainer),
                                          ("GradientExplainer", shap.GradientExplainer)]:
            try:
                exp = ExplainerClass(gru_model, X_bg)
                sv  = exp.shap_values(X_ex)
                sv  = sv[0] if isinstance(sv, list) else sv
                shap_gru_mean = sv.mean(axis=1)   # (n, 20, F) → (n, F)
                X_ex_mean     = X_ex.mean(axis=1)
                method = cls_name
                break
            except Exception as e:
                msg = str(e).encode("ascii", errors="replace").decode("ascii")
                print(f"    [{cls_name}] 실패: {msg}")

        # 2차: KernelExplainer 폴백 (시퀀스 → 시간축 평균 → 1D 래퍼)
        if method is None:
            print("    [KernelExplainer] 폴백: 시간축 평균 근사 계산 중...")
            n_bg_k, n_ex_k = min(20, len(X_bg)), min(50, len(X_ex))
            X_bg_flat = X_bg[:n_bg_k].mean(axis=1)   # (n_bg, F)
            X_ex_flat = X_ex[:n_ex_k].mean(axis=1)   # (n_ex, F)

            def _gru_pred_flat(X_2d: np.ndarray) -> np.ndarray:
                X_3d = np.repeat(X_2d[:, np.newaxis, :], WINDOW, axis=1).astype(np.float32)
                return gru_model.predict(X_3d, verbose=0).flatten()

            exp_k = shap.KernelExplainer(_gru_pred_flat, X_bg_flat)
            shap_gru_mean = exp_k.shap_values(X_ex_flat, nsamples=50, silent=True)
            if isinstance(shap_gru_mean, list):
                shap_gru_mean = shap_gru_mean[0]
            X_ex_mean = X_ex_flat
            method = "KernelExplainer(시간평균 근사)"

        if method:
            elapsed = time.time() - t0
            print(f"    {method} 완료: {elapsed:.1f}초  shape={shap_gru_mean.shape}")
            gru_shap_ok = True

            fig = plt.figure(figsize=(9, 5))
            shap.summary_plot(shap_gru_mean, X_ex_mean,
                              feature_names=GRU_FEATURES,
                              plot_type="dot", show=False, max_display=9)
            plt.title(f"GRU v2 RUL 예측 SHAP Summary ({method})\n"
                      "빨강=RUL↑(오래 남음), 파랑=RUL↓(곧 고장)",
                      fontsize=11, fontweight="bold")
            plt.tight_layout()
            out2 = OUT_DIR / "S2_shap_gru.png"
            fig.savefig(out2, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    저장: {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3. 피처 중요도 비교 막대 차트
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] 피처 중요도 비교 차트...")
panels = []
if clf_shap_ok and shap_clf is not None:
    panels.append(("분류 모델\n열화 감지", USE_FEATURES, shap_clf, "#4C72B0"))
if gru_shap_ok and shap_gru_mean is not None:
    panels.append(("GRU v2\nRUL 회귀", GRU_FEATURES, shap_gru_mean, "#DD8452"))

if panels:
    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 6))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, feats, sv, color) in zip(axes, panels):
        mean_abs = np.abs(sv).mean(axis=0)
        order    = np.argsort(mean_abs)   # 오름차순
        bars = ax.barh([feats[i] for i in order], mean_abs[order],
                       color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("평균 |SHAP value|")
    plt.suptitle("FEMTO-ST — 피처 중요도 비교 (SHAP)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out3 = OUT_DIR / "S3_shap_bar.png"
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    저장: {out3}")

print("\n=== FEMTO SHAP 분석 완료 ===")
print(f"출력: {OUT_DIR}")
