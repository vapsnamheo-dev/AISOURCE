"""설비 고장 예측 (PdM-Guard) — Streamlit 웹앱.

3개 탭으로 구성:
  ① 단건 예측      : 학습된 XGBoost 모델(joblib)을 불러와 센서값 1건 예측 + (선택)물리 실제라벨 매칭
  ② CSV 일괄검증   : 업로드한 CSV를 일괄 예측, 실제라벨이 있으면 정확도·혼동행렬 표시
  ③ 인터랙티브 학습 : 강사 예시(Iris 분류기) 패턴 — 사이드바 슬라이더(트리 개수/최대 깊이)로
                     실시간 재학습 → 모델 정확도 + 특징 중요도(st.bar_chart) 시각화

설계 원칙
  - 딥러닝 미사용, 고전 ML(RandomForest / XGBoost)만 사용
  - ①②는 대표님 기존 모듈(src.*)에 의존하되, 모듈이 없거나 시그니처가 달라도
    앱 전체가 죽지 않도록 try/except로 방어. ③은 src.* 없이도 단독 동작.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── 프로젝트 루트 경로 추가 (src 패키지 import 용) ──────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="설비 고장 예측 (PdM-Guard)", page_icon="🛠️", layout="wide")

# ── 데이터셋 표준 컬럼 정의 (AI4I / shivamb 버전 기준) ──────────────────────
# 컬럼명이 다른 CSV도 허용하기 위해 후보를 둔다.
NUM_CANDIDATES = {
    "air_temp": ["Air temperature [K]", "Air temperature", "air_temp"],
    "proc_temp": ["Process temperature [K]", "Process temperature", "proc_temp"],
    "rot_speed": ["Rotational speed [rpm]", "Rotational speed", "rot_speed"],
    "torque": ["Torque [Nm]", "Torque", "torque"],
    "tool_wear": ["Tool wear [min]", "Tool wear", "tool_wear"],
}
TYPE_CANDIDATES = ["Type", "type"]
TARGET_CANDIDATES = ["Target", "Machine failure", "target", "actual", "label"]
TYPE_DUMMIES = ["Type_H", "Type_L", "Type_M"]  # one-hot 후 8 feature = 5 numeric + 3 type


def _resolve(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """후보 컬럼명 중 df에 실제 존재하는 첫 컬럼명을 반환."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """원본 DataFrame → 8개 feature(X) + (있으면)target(y)으로 전처리.

    - 숫자 5개 + Type 원-핫(Type_H/L/M)
    - 컬럼명이 표준과 달라도 후보 매핑으로 흡수
    """
    out = pd.DataFrame(index=df.index)
    for key, cands in NUM_CANDIDATES.items():
        col = _resolve(df, cands)
        if col is None:
            raise ValueError(f"필수 숫자 컬럼을 찾지 못했습니다: {cands[0]}")
        out[key] = pd.to_numeric(df[col], errors="coerce")

    type_col = _resolve(df, TYPE_CANDIDATES)
    if type_col is None:
        raise ValueError("Type 컬럼을 찾지 못했습니다.")
    dummies = pd.get_dummies(df[type_col].astype(str).str.upper(), prefix="Type")
    for d in TYPE_DUMMIES:
        out[d] = dummies[d] if d in dummies.columns else 0
    out = out[[*NUM_CANDIDATES.keys(), *TYPE_DUMMIES]].astype(float)

    tgt_col = _resolve(df, TARGET_CANDIDATES)
    y = pd.to_numeric(df[tgt_col], errors="coerce").astype(int) if tgt_col else None
    return out, y


# ── 학습용 CSV 자동 탐색 (③ 탭에서 사용) ──────────────────────────────────
@st.cache_data(show_spinner=False)
def _find_training_csv() -> Path | None:
    candidates = [
        ROOT / "data" / "predictive_maintenance.csv",
        ROOT / "data" / "ai4i2020.csv",
        ROOT / "demo data" / "demo_1000.csv",
    ]
    candidates += sorted(ROOT.glob("**/*.csv"))
    for p in candidates:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def _load_dataframe(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


# ── 학습된 아티팩트 로드 (① 단건 예측) ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_artifacts():
    """models/ 아래의 학습된 XGBoost 아티팩트(joblib) 로드.

    대표님 src.predict.load_artifacts가 있으면 우선 사용하고,
    없으면 joblib 파일을 직접 읽는다(방어적).
    """
    # 1) 기존 모듈 우선
    try:
        from src import predict  # type: ignore

        art = predict.load_artifacts("xgb")
        return {"source": "src.predict", "art": art}
    except Exception:
        pass
    # 2) joblib 직접 로드
    try:
        import joblib

        mdir_candidates = [ROOT / "models", Path("models")]
        for mdir in mdir_candidates:
            mfile = mdir / "model_xgb.joblib"
            if mfile.exists():
                model = joblib.load(mfile)
                scaler = joblib.load(mdir / "scaler.joblib") if (mdir / "scaler.joblib").exists() else None
                fcols = (
                    joblib.load(mdir / "feature_columns.joblib")
                    if (mdir / "feature_columns.joblib").exists()
                    else [*NUM_CANDIDATES.keys(), *TYPE_DUMMIES]
                )
                return {"source": "joblib", "model": model, "scaler": scaler, "feature_columns": fcols}
    except Exception as e:  # noqa: BLE001
        return {"source": "error", "error": str(e)}
    return {"source": "missing"}


def _physics_actual_label(type_, air, proc, rot, torque, wear):
    """src.synth_ai4i.actual_label이 있으면 물리 실제라벨을 계산(없으면 None)."""
    try:
        from src import synth_ai4i  # type: ignore

        return int(synth_ai4i.actual_label(type_, air, proc, rot, torque, wear))
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════
# 사이드바 — ③ 인터랙티브 학습용 하이퍼파라미터 (강사 예시 패턴)
# ════════════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ 사용자 입력 파라미터")
st.sidebar.caption("③ 인터랙티브 학습 탭에 적용됩니다.")
algo = st.sidebar.selectbox("학습 알고리즘", ["RandomForest", "XGBoost"], index=0)
n_estimators = st.sidebar.slider("트리 개수 (n_estimators)", 1, 200, 100)
max_depth = st.sidebar.slider("최대 깊이 (max_depth)", 1, 20, 6)
test_size = st.sidebar.slider("테스트 비율 (test_size)", 0.1, 0.5, 0.3, step=0.05)

st.title("🛠️ 설비 고장 예측 (PdM-Guard)")
st.caption("센서값으로 고장 발생 여부(정상=0 / 고장=1)를 예측합니다. · XGBoost 주모델 · 딥러닝 미사용")

tab1, tab2, tab3 = st.tabs(["① 단건 예측", "② CSV 일괄검증", "③ 인터랙티브 학습"])

# ────────────────────────────────────────────────────────────────────────
# ① 단건 예측
# ────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("센서값 1건 입력 → 고장 확률 예측")
    arts = _load_artifacts()

    c1, c2, c3 = st.columns(3)
    with c1:
        ptype = st.selectbox("제품 등급 (Type)", ["L", "M", "H"], index=0)
        air = st.number_input("공기 온도 [K]", value=300.0, step=0.1)
    with c2:
        proc = st.number_input("공정 온도 [K]", value=310.0, step=0.1)
        rot = st.number_input("회전 속도 [rpm]", value=1500.0, step=10.0)
    with c3:
        torque = st.number_input("토크 [Nm]", value=40.0, step=1.0)
        wear = st.number_input("공구 마모 [min]", value=100.0, step=1.0)

    if st.button("🔮 고장 예측", type="primary"):
        if arts["source"] in ("missing", "error"):
            st.error(
                "학습된 모델을 찾지 못했습니다. models/model_xgb.joblib 등을 확인하세요.\n"
                f"세부: {arts.get('error', 'models/ 폴더 없음')}"
            )
        else:
            raw = pd.DataFrame(
                [{
                    "Air temperature [K]": air, "Process temperature [K]": proc,
                    "Rotational speed [rpm]": rot, "Torque [Nm]": torque,
                    "Tool wear [min]": wear, "Type": ptype,
                }]
            )
            X, _ = _build_features(raw)

            # src.predict 경로면 그 함수에 위임, 아니면 직접 스케일+예측
            try:
                if arts["source"] == "src.predict":
                    from src import predict  # type: ignore

                    proba = float(predict.predict_proba(arts["art"], X)[0])  # type: ignore[attr-defined]
                else:
                    model, scaler, fcols = arts["model"], arts["scaler"], arts["feature_columns"]
                    Xa = X.reindex(columns=fcols, fill_value=0)
                    Xs = scaler.transform(Xa) if scaler is not None else Xa.values
                    proba = float(model.predict_proba(Xs)[0, 1])
            except Exception as e:  # noqa: BLE001
                st.error(f"예측 중 오류: {e}. 모듈 시그니처를 확인해 주세요.")
                proba = None

            if proba is not None:
                pred = int(proba >= 0.5)
                m1, m2 = st.columns(2)
                m1.metric("고장 확률", f"{proba * 100:.1f}%")
                if pred == 1:
                    m2.error("⚠️ 예측: **고장(1)** — 점검 권장")
                else:
                    m2.success("✅ 예측: **정상(0)**")
                st.progress(min(max(proba, 0.0), 1.0))

                actual = _physics_actual_label(ptype, air, proc, rot, torque, wear)
                if actual is not None:
                    ok = (actual == pred)
                    st.write(
                        f"물리 실제라벨: **{actual}** ({'정상' if actual == 0 else '고장'}) → "
                        f"{'🎯 일치' if ok else '❌ 불일치'}"
                    )

# ────────────────────────────────────────────────────────────────────────
# ② CSV 일괄검증
# ────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("CSV 업로드 → 일괄 예측 · 실제라벨 있으면 정확도/혼동행렬")
    up = st.file_uploader("예측할 CSV 파일", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        st.write(f"업로드: {len(df):,}행 · {df.shape[1]}열")
        arts = _load_artifacts()
        if arts["source"] in ("missing", "error"):
            st.error("학습된 모델을 찾지 못했습니다. models/ 폴더를 확인하세요.")
        else:
            try:
                X, y = _build_features(df)
                # 기존 model_store 일괄 함수가 있으면 우선 사용
                used = None
                try:
                    from src import model_store  # type: ignore

                    model_store.ensure_model_in_db("xgb")
                    out, _, has_actual = model_store.predict_dataframe_from_db(df)
                    preds = out["pred_label"].to_numpy()
                    used = "src.model_store"
                except Exception:
                    model, scaler, fcols = (
                        arts.get("model"), arts.get("scaler"),
                        arts.get("feature_columns", [*NUM_CANDIDATES.keys(), *TYPE_DUMMIES]),
                    )
                    Xa = X.reindex(columns=fcols, fill_value=0)
                    Xs = scaler.transform(Xa) if scaler is not None else Xa.values
                    preds = model.predict(Xs)
                    used = "joblib"

                res = df.copy()
                res["pred_label"] = preds
                st.caption(f"예측 경로: {used}")
                st.dataframe(res.head(50), width="stretch")

                if y is not None:
                    from sklearn.metrics import accuracy_score, confusion_matrix

                    acc = accuracy_score(y, preds)
                    cm = confusion_matrix(y, preds)
                    st.metric("일괄 정확도", f"{acc * 100:.2f}%")
                    cm_df = pd.DataFrame(
                        cm,
                        index=["실제 정상", "실제 고장"],
                        columns=["예측 정상", "예측 고장"],
                    )
                    st.write("혼동행렬")
                    st.dataframe(cm_df, width="stretch")
                else:
                    st.info("실제라벨(Target) 컬럼이 없어 정확도는 계산하지 않았습니다.")
            except Exception as e:  # noqa: BLE001
                st.error(f"일괄 예측 오류: {e}")

# ────────────────────────────────────────────────────────────────────────
# ③ 인터랙티브 학습 (강사 예시 Iris 분류기 패턴)
# ────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("사이드바 슬라이더로 실시간 재학습 → 정확도 · 특징 중요도")
    st.caption("강사 예시(Iris 분류기)와 동일한 패턴: 하이퍼파라미터 조정 → 학습 → 결과 시각화")

    csv_path = _find_training_csv()
    if csv_path is None:
        st.error("학습용 CSV를 찾지 못했습니다. data/ 폴더에 데이터셋을 두세요.")
    else:
        st.write(f"학습 데이터: `{csv_path.relative_to(ROOT)}`")
        df = _load_dataframe(str(csv_path))
        try:
            X, y = _build_features(df)
            if y is None:
                st.error("Target(정상/고장) 컬럼이 없어 학습할 수 없습니다.")
            else:
                from sklearn.metrics import accuracy_score
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                scaler = StandardScaler().fit(X_train)
                Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

                if algo == "XGBoost":
                    try:
                        from xgboost import XGBClassifier

                        pos = max(int((y_train == 0).sum()), 1) / max(int((y_train == 1).sum()), 1)
                        clf = XGBClassifier(
                            n_estimators=n_estimators, max_depth=max_depth,
                            scale_pos_weight=pos, eval_metric="logloss",
                            random_state=42, n_jobs=2,
                        )
                    except Exception:
                        st.warning("xgboost 미설치 → RandomForest로 대체합니다.")
                        from sklearn.ensemble import RandomForestClassifier

                        clf = RandomForestClassifier(
                            n_estimators=n_estimators, max_depth=max_depth,
                            class_weight="balanced", random_state=42, n_jobs=2,
                        )
                else:
                    from sklearn.ensemble import RandomForestClassifier

                    clf = RandomForestClassifier(
                        n_estimators=n_estimators, max_depth=max_depth,
                        class_weight="balanced", random_state=42, n_jobs=2,
                    )

                clf.fit(Xtr, y_train)
                acc = accuracy_score(y_test, clf.predict(Xte))

                st.write(f"### 모델 정확도: {acc * 100:.2f}%")
                st.caption(
                    f"{algo} · n_estimators={n_estimators} · max_depth={max_depth} "
                    f"· test_size={test_size} · 고장 비율={y.mean() * 100:.2f}%"
                )

                st.subheader("특징 중요도")
                importances = getattr(clf, "feature_importances_", None)
                if importances is not None:
                    fi = (
                        pd.DataFrame({"특징": X.columns, "중요도": importances})
                        .sort_values("중요도", ascending=False)
                        .set_index("특징")
                    )
                    st.bar_chart(fi)
                else:
                    st.info("이 모델은 feature_importances_를 제공하지 않습니다.")
        except Exception as e:  # noqa: BLE001
            st.error(f"학습 오류: {e}")
