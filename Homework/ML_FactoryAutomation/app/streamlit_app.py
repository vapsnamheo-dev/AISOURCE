# ════════════════════════════════════════════════════════════════════
# [역할] Streamlit 프로토타입 — 단건 예측(물리 실제라벨 일치 표시) + CSV 일괄검증.
# [단계] 프로토타입 (보고서 8장)
# [작업 메모] 작업: st.tabs 구성·일괄검증 탭. (별도 제공한 3탭 버전엔 인터랙티브 학습 탭 추가본 있음)
# (이 헤더는 이전/이번 작업 맥락을 요약한 주석입니다 — 기능에는 영향 없음)
# ════════════════════════════════════════════════════════════════════
"""Streamlit 예측 웹앱: 단건 예측(실제 일치 표시) + CSV 일괄 검증(실제 vs 예측)."""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from src import db, predict, model_store, synth_ai4i, config

st.set_page_config(page_title="설비 고장 예측 (PdM-Guard)", page_icon="🛠️", layout="wide")

# ──────────────────────────────────────────────────────────────
# AI 어시스턴트 헬퍼 (Gemini). 키 없으면 규칙 기반으로 폴백.
#   키 설정: 환경변수 GEMINI_API_KEY(또는 GOOGLE_API_KEY) 또는 .streamlit/secrets.toml
# ──────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-1.5-flash"  # AI Studio 키로 사용 · 필요시 모델명 변경
PDM_SYSTEM = ("당신은 설비 예지보전(PdM) 전문가입니다. AI4I 고장모드(TWF 공구마모, "
              "HDF 방열, PWF 전력부족, OSF 과부하)를 근거로, 한국어로 간결하고 "
              "실무적으로(원인·조치 중심) 답하세요.")


def _get_api_key():
    import os
    try:
        k = st.secrets.get("GEMINI_API_KEY")
        if k:
            return k
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def ai_generate(prompt: str):
    """Gemini 호출. 키/라이브러리 없으면 None(폴백 신호)."""
    key = _get_api_key()
    if not key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=PDM_SYSTEM)
        return model.generate_content(prompt).text
    except Exception as e:  # 네트워크/쿼터/모델명 오류 등
        return f"(AI 호출 오류: {e})"


def rule_based_summary(ctx: str) -> str:
    """API 키가 없을 때의 규칙 기반 해설."""
    return ("자동 요약(규칙 기반):\n"
            f"- {ctx}\n"
            "- 놓친 고장(FN)이 있으면 판정 임계값을 낮춰 재현율을 높이는 것을 검토하세요.\n"
            "- 오경보(FP)가 많으면 임계값 상향으로 불필요 점검을 줄일 수 있습니다.\n"
            "- 우선 조치: 토크·공구마모가 높은 설비를 점검하고 과부하(OSF)·방열(HDF) 조건을 확인하세요.")


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

    # ── 판정 임계값 (운영 조정 가능) ──
    st.divider()
    st.subheader("⚙️ 판정 임계값")
    # 기본값 0.85 = 현 프로젝트 F1 최적(T*≈0.85): 재현율(0.809) 유지 + 정밀도 0.833→0.932 (오경보↓)
    DEFAULT_THRESHOLD = config.DECISION_THRESHOLD  # 단일 출처(config) = 0.85 (F1 최적)
    threshold = st.slider(
        "고장 판정 임계값", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01,
        help="고장 확률이 이 값 이상이면 '고장'으로 판정. 낮추면 재현율↑(놓침↓), "
             "높이면 정밀도↑(오경보↓). 기본 0.85는 PR곡선상 F1 최적값.")
    # 변경 이력 로깅 (운영 권장 흐름) — 데모는 세션 기록, 운영은 DB/로그 저장 권장
    if "thr_log" not in st.session_state:
        st.session_state.thr_log = []
    if (not st.session_state.thr_log) or st.session_state.thr_log[-1][1] != threshold:
        st.session_state.thr_log.append((datetime.now().strftime("%H:%M:%S"), threshold))
    with st.expander("임계값 변경 이력"):
        for ts, thv in st.session_state.thr_log[-10:]:
            st.write(f"- {ts} → {thv:.2f}")
        st.caption("※ 운영 환경에서는 이 이력을 DB/로그 저장소에 영구 기록하세요.")

    # ── QA 시험결과 다운로드 ──
    st.divider()
    _qa_path = Path(__file__).resolve().parent.parent / "산출물" / "QA_시험결과_20260623.md"
    if _qa_path.exists():
        st.download_button(
            "📥 QA 시험결과 다운로드",
            data=_qa_path.read_text(encoding="utf-8"),
            file_name="QA_시험결과_20260623.md",
            mime="text/markdown",
        )

tab1, tab2, tab3 = st.tabs(["🔮 단건 예측", "📁 CSV 일괄 검증", "📊 성능 대시보드"])

with tab1:
    c1, c2 = st.columns([1, 1])
    if go:
        inp = {"type": type_, "air_temperature": air, "process_temperature": proc,
               "rotational_speed": rpm, "torque": torque, "tool_wear": wear}
        with db.get_session(engine) as s:
            res = predict.predict_and_log(inp, s, artifacts=artifacts)
        res["pred_label"] = int(res["pred_proba"] >= threshold)  # 사이드바 임계값 적용
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
    st.caption("demo data/demo_1000.csv 처럼 Type·센서값(+Target 실제라벨)이 든 CSV를 업로드하거나 repo 샘플을 불러오세요.")

    input_mode = st.radio(
        "입력 방법",
        ["파일 업로드", "레포 샘플 불러오기"],
        horizontal=True,
        key="batch_input_mode"
    )

    df = None
    if input_mode == "파일 업로드":
        up = st.file_uploader("CSV 파일 선택", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            st.success("업로드된 CSV를 불러왔습니다.")
        if "batch_df" in st.session_state:
            st.session_state.pop("batch_df", None)
            st.session_state.pop("batch_source", None)
    else:
        sample_paths = {
            "demo data/demo_1000.csv": Path(__file__).resolve().parent.parent / "demo data" / "demo_1000.csv",
            "predictive_maintenance.csv": Path(__file__).resolve().parent.parent / "predictive_maintenance.csv",
        }
        sample_choice = st.selectbox("샘플 데이터 선택", list(sample_paths.keys()))
        if st.button("📂 레포 샘플 로드"):
            sample_path = sample_paths[sample_choice]
            if not sample_path.exists():
                st.error(f"샘플 파일을 찾을 수 없습니다: {sample_path}")
            else:
                df = pd.read_csv(sample_path)
                st.session_state.batch_df = df.to_dict("records")
                st.session_state.batch_source = sample_choice
                st.success(f"샘플 '{sample_choice}' 로드 완료 ({len(df):,}건)")

        if df is None and st.session_state.get("batch_df") is not None:
            try:
                df = pd.DataFrame(st.session_state.batch_df)
            except Exception:
                st.session_state.pop("batch_df", None)
                st.session_state.pop("batch_source", None)

        if df is None:
            st.info("레포 샘플을 불러오려면 위에서 샘플을 선택하고 버튼을 누르세요.")

    if df is not None:

        # ── 업로드 CSV 방어 가드 (필수컬럼 / 컬럼명·값 공백 / 필수 센서 결측) ──
        from src import preprocess as _ppg
        try:
            df, _rep = _ppg.validate_and_clean(df, require_target=False)
        except ValueError as _ve:
            st.error(f"❌ 업로드 데이터 오류 — {_ve}")
            st.stop()
        _msgs = []
        if _rep["stripped_cells"]:
            _msgs.append(f"문자열 앞뒤 공백 {_rep['stripped_cells']}건 정리")
        if _rep["dropped_na"]:
            _msgs.append(f"필수 센서 결측 {_rep['dropped_na']}행 제외")
        if _msgs:
            st.warning("⚠️ 전처리 가드: " + " · ".join(_msgs)
                       + f"  (입력 {_rep['n_in']:,} → 검증 {_rep['n_out']:,}건)")
        else:
            st.caption(f"✅ 전처리 가드 통과 — 결측·공백 없음 ({_rep['n_out']:,}건)")
        if _rep["n_out"] == 0:
            st.error("유효한 행이 없습니다. CSV 내용을 확인하세요.")
            st.stop()

        out, model_id, has_actual = model_store.predict_dataframe_from_db(df)
        out["pred_label"] = (out["pred_proba"] >= threshold).astype(int)  # 임계값 적용
        if has_actual and "actual" in out.columns:
            out["match"] = (out["pred_label"] == out["actual"])
        st.write(f"총 **{len(out):,}건** 예측 완료 (DB 활성 모델 id={model_id}, 임계값 {threshold:.2f}).")
        result_ctx = ""
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
            # AI 컨텍스트
            result_ctx = (f"총 {len(out)}건, 정확도 {acc*100:.1f}%. 혼동행렬 — 정상→정상 {tn}, "
                          f"오경보(FP) {fp}, 놓침(FN) {fn}, 탐지(TP) {tp}. 고장 {tp+fn}건 중 {tp}건 탐지.")
            missed = out[(out["pred_label"] == 0) & (out["actual"] == 1)]
            fts = [c for c in ["Torque [Nm]", "Tool wear [min]", "Rotational speed [rpm]"] if c in missed.columns]
            if len(missed) and fts:
                result_ctx += " 놓친 고장 평균 — " + ", ".join(
                    f"{c.split('[')[0].strip()} {missed[c].mean():.1f}" for c in fts) + "."
        else:
            fail = int((out["pred_label"] == 1).sum())
            st.info("Target(실제라벨) 열이 없어 예측 결과만 표시합니다.")
            st.dataframe(out.head(100), width="stretch")
            result_ctx = f"총 {len(out)}건 예측, 고장 예측 {fail}건({fail/max(len(out),1)*100:.1f}%). 실제 라벨 없음."

        # ───────────── AI 분석 & 조치 어시스턴트 (결과 하단) ─────────────
        st.divider()
        st.subheader("🤖 AI 분석 & 조치 어시스턴트")
        if _get_api_key() is None:
            st.caption("⚙️ GEMINI_API_KEY(또는 GOOGLE_API_KEY)를 환경변수/secrets에 설정하면 AI 해설·챗봇이 활성화됩니다. "
                       "미설정 시 규칙 기반 요약으로 동작합니다.")
        st.session_state.setdefault("ai_log", [])

        # ① AI 자동 해설
        if st.button("📊 AI 자동 해설 생성"):
            with st.spinner("결과 분석 중..."):
                txt = ai_generate("다음 설비 고장 일괄 검증 결과를 3~5줄로 분석·해설하고 "
                                  f"핵심 조치를 제안하세요.\n[결과] {result_ctx}")
            if txt is None:
                txt = rule_based_summary(result_ctx)
            st.session_state["ai_log"].append(("📊 AI 해설", txt))
        for _, txt in [e for e in st.session_state["ai_log"] if e[0] == "📊 AI 해설"][-1:]:
            st.markdown(txt)

        # ② 조치 Q&A 챗봇
        st.markdown("**💬 조치 Q&A 챗봇** — 결과에 대한 원인·조치를 질문하세요.")
        q = st.text_input("질문 입력", key="chat_q",
                          placeholder="예: 놓친 고장을 줄이려면 어떤 조치를 해야 하나요?")
        if st.button("질문하기") and q:
            with st.spinner("답변 생성 중..."):
                ans = ai_generate(f"[검증 결과] {result_ctx}\n[질문] {q}\n"
                                  "위 결과를 근거로 원인과 구체적 조치를 답하세요.")
            if ans is None:
                ans = "AI 키 미설정 — " + rule_based_summary(result_ctx)
            st.session_state["ai_log"].append((f"🙋 {q}", ans))
        for kind, ans in [e for e in st.session_state["ai_log"] if e[0].startswith("🙋")][-5:]:
            with st.chat_message("user"):
                st.markdown(kind[2:])
            with st.chat_message("assistant"):
                st.markdown(ans)

        # ③ 히스토리 저장
        if st.session_state["ai_log"]:
            md = f"# PdM AI 분석·조치 히스토리\n\n- 검증 요약: {result_ctx}\n\n"
            for kind, txt in st.session_state["ai_log"]:
                md += f"## {kind}\n{txt}\n\n"
            cc1, cc2 = st.columns(2)
            cc1.download_button("💾 히스토리 저장 (.md)", md, "pdm_ai_history.md", "text/markdown")
            if cc2.button("🗑️ 히스토리 비우기"):
                st.session_state["ai_log"] = []
                st.rerun()

# ══════════════════════════════════════════════════════════════
# TAB 3 — 성능 대시보드 (4중첩 서브탭)
# ══════════════════════════════════════════════════════════════
with tab3:
    import json
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    import seaborn as sns

    # ── 공통 데이터 로드 ──
    @st.cache_data
    def _load_model_info():
        p = config.BASE_DIR / "model" / "model_info.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return {}

    @st.cache_data
    def _load_train_df():
        p = config.DATA_PATH
        if p.exists():
            return pd.read_csv(p)
        return pd.DataFrame()

    model_info = _load_model_info()
    train_df = _load_train_df()

    d1, d2, d3, d4 = st.tabs([
        "🎯 모델 성능 요약",
        "🔥 고장유형 파레토",
        "🔬 상관·분포 분석",
        "📌 특성 중요도",
    ])

    # ── 서브탭 1: 모델 성능 요약 ──────────────────────────────
    with d1:
        st.subheader("🎯 모델 성능 요약")
        models_meta = model_info.get("models", {})
        if not models_meta:
            st.warning("model/model_info.json 파일을 찾을 수 없습니다.")
        else:
            # KPI 카드 (XGBoost 기준)
            xgb_m = models_meta.get("XGBoost", {}).get("metrics", {})
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("정확도", f"{xgb_m.get('accuracy', 0)*100:.1f}%")
            k2.metric("정밀도", f"{xgb_m.get('precision', 0)*100:.1f}%")
            k3.metric("재현율", f"{xgb_m.get('recall', 0)*100:.1f}%")
            k4.metric("F1", f"{xgb_m.get('f1', 0)*100:.1f}%")
            k5.metric("ROC-AUC", f"{xgb_m.get('roc_auc', 0):.4f}")
            st.caption("※ XGBoost (임계값 0.5 기준 평가지표 · 운영 임계값 T*=0.85)")

            col_a, col_b = st.columns(2)

            # 정확도 도넛 차트
            with col_a:
                st.markdown("**정확도 도넛 (3모델 비교)**")
                names, accs = [], []
                for name, meta in models_meta.items():
                    names.append(name)
                    accs.append(meta.get("metrics", {}).get("accuracy", 0) * 100)
                fig_d, ax_d = plt.subplots(figsize=(4, 4))
                wedges, texts, autotexts = ax_d.pie(
                    accs, labels=names, autopct="%1.1f%%",
                    startangle=90, pctdistance=0.75,
                    wedgeprops={"width": 0.5})
                ax_d.set_title("Accuracy (%)")
                st.pyplot(fig_d)
                plt.close(fig_d)

            # 혼동행렬 6지표 테이블
            with col_b:
                st.markdown("**혼동행렬 6지표 (3모델)**")
                rows = []
                metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
                metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
                for name, meta in models_meta.items():
                    m = meta.get("metrics", {})
                    row = {"모델": name}
                    for k, lbl in zip(metric_keys, metric_labels):
                        row[lbl] = f"{m.get(k, 0):.4f}"
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows).set_index("모델"), use_container_width=True)

            # 요약 해설
            st.divider()
            st.markdown("**📝 자동 해설**")
            best = max(models_meta.items(), key=lambda x: x[1].get("metrics", {}).get("f1", 0))
            best_name, best_f1 = best[0], best[1].get("metrics", {}).get("f1", 0)
            xgb_rec = xgb_m.get("recall", 0)
            st.info(
                f"- **최고 F1 모델**: {best_name} (F1={best_f1:.4f})\n"
                f"- XGBoost 재현율 {xgb_rec*100:.1f}% — 고장 10건 중 약 {xgb_rec*10:.0f}건 탐지\n"
                f"- 운영 임계값 T*=0.85 적용 시 정밀도 상승(오경보↓), 재현율은 0.5 기준 대비 유지\n"
                f"- 불균형 데이터(정상:{train_df[config.TARGET].value_counts().get(0,'?')} vs 고장:{train_df[config.TARGET].value_counts().get(1,'?')})에 scale_pos_weight 적용"
                if not train_df.empty else
                f"- **최고 F1 모델**: {best_name} (F1={best_f1:.4f})\n"
                f"- XGBoost 재현율 {xgb_rec*100:.1f}% — 고장 10건 중 약 {xgb_rec*10:.0f}건 탐지\n"
                f"- 운영 임계값 T*=0.85 적용 시 정밀도 상승(오경보↓)")

    # ── 서브탭 2: 고장유형 파레토 ─────────────────────────────
    with d2:
        st.subheader("🔥 고장유형 파레토")
        if train_df.empty:
            st.warning("학습 데이터를 찾을 수 없습니다.")
        else:
            if "Failure Type" in train_df.columns:
                fail_counts = (train_df[train_df["Failure Type"] != "No Failure"]
                               ["Failure Type"].value_counts())
            else:
                fail_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
                avail = [c for c in fail_cols if c in train_df.columns]
                if avail:
                    type_map = {v: k for k, v in config.FAILURE_TYPES.items()}
                    fail_counts = pd.Series({config.FAILURE_TYPES.get(c, c): int(train_df[c].sum())
                                             for c in avail}).sort_values(ascending=False)
                else:
                    fail_counts = pd.Series(dtype=int)

            if fail_counts.empty:
                st.info("고장유형 데이터가 없습니다.")
            else:
                labels = list(fail_counts.index)
                vals = list(fail_counts.values)
                cumsum = [sum(vals[:i+1]) / max(sum(vals), 1) * 100 for i in range(len(vals))]

                fig_p, ax1 = plt.subplots(figsize=(8, 4))
                ax1.bar(labels, vals, color="#e05c5c", alpha=0.85)
                ax1.set_ylabel("Count")
                ax1.set_xlabel("Failure Type")
                ax2 = ax1.twinx()
                ax2.plot(labels, cumsum, "o-", color="#2563eb", linewidth=2, label="Cumulative %")
                ax2.axhline(80, color="gray", linestyle="--", linewidth=1)
                ax2.set_ylabel("Cumulative (%)")
                ax2.set_ylim(0, 110)
                ax2.legend(loc="lower right")
                plt.title("Failure Type Pareto Chart")
                plt.tight_layout()
                st.pyplot(fig_p)
                plt.close(fig_p)

                top80 = [labels[i] for i, v in enumerate(cumsum) if (i == 0 or cumsum[i-1] < 80)]
                st.info(
                    f"**상위 {len(top80)}개 유형**({', '.join(top80)})이 "
                    f"전체 고장의 80% 이상을 차지합니다.\n"
                    "→ 해당 유형 예방 점검을 우선 조치하면 고장 발생을 집중 억제할 수 있습니다.")

    # ── 서브탭 3: 상관·분포 분석 (EDA) ────────────────────────
    with d3:
        st.subheader("🔬 상관·분포 분석 (EDA)")
        if train_df.empty:
            st.warning("학습 데이터를 찾을 수 없습니다.")
        else:
            num_cols = [c for c in config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES
                        if c in train_df.columns]
            target_col = config.TARGET

            col_h, col_b = st.columns(2)

            # 상관관계 히트맵
            with col_h:
                st.markdown("**상관관계 히트맵**")
                heat_cols = num_cols + ([target_col] if target_col in train_df.columns else [])
                corr = train_df[heat_cols].corr()
                fig_h, ax_h = plt.subplots(figsize=(6, 5))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                            center=0, ax=ax_h, annot_kws={"size": 7})
                ax_h.set_title("Feature Correlation")
                plt.tight_layout()
                st.pyplot(fig_h)
                plt.close(fig_h)

            # 정상 vs 고장 박스플롯
            with col_b:
                st.markdown("**정상 vs 고장 박스플롯**")
                top_feats = ["Torque [Nm]", "Tool wear [min]", "Rotational speed [rpm]"]
                avail_f = [f for f in top_feats if f in train_df.columns]
                if avail_f and target_col in train_df.columns:
                    fig_box, axes = plt.subplots(1, len(avail_f), figsize=(5, 3))
                    if len(avail_f) == 1:
                        axes = [axes]
                    for ax, feat in zip(axes, avail_f):
                        groups = [train_df.loc[train_df[target_col] == v, feat].dropna()
                                  for v in [0, 1]]
                        ax.boxplot(groups, labels=["Normal", "Failure"])
                        ax.set_title(feat.split("[")[0].strip(), fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig_box)
                    plt.close(fig_box)

            # 자동 해설
            st.divider()
            if target_col in train_df.columns and num_cols:
                corr_with_target = train_df[num_cols + [target_col]].corr()[target_col].drop(target_col)
                top_corr = corr_with_target.abs().sort_values(ascending=False).head(3)
                st.info(
                    "**고장 상관 상위 동적 계산:**\n"
                    + "\n".join(f"- {feat}: r={corr_with_target[feat]:.3f}"
                                for feat in top_corr.index)
                    + "\n→ 상관이 높은 피처일수록 고장 예측에 핵심 역할을 합니다."
                )

    # ── 서브탭 4: 특성 중요도 ─────────────────────────────────
    with d4:
        st.subheader("📌 특성 중요도 (XGBoost)")

        @st.cache_resource
        def _load_xgb_model():
            try:
                m, _, _ = predict.load_artifacts("xgb")
                return m
            except Exception:
                return None

        xgb_model = _load_xgb_model()
        feat_names = model_info.get("features", config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES
                                    + [f"Type_{t}" for t in ["H", "L", "M"]])

        if xgb_model is None:
            st.warning("XGBoost 모델을 로드할 수 없습니다.")
        else:
            try:
                imp = xgb_model.feature_importances_
                imp_df = pd.DataFrame({"feature": feat_names[:len(imp)], "importance": imp})
                imp_df = imp_df.sort_values("importance", ascending=True)

                fig_fi, ax_fi = plt.subplots(figsize=(6, 5))
                ax_fi.barh(imp_df["feature"], imp_df["importance"], color="#2563eb", alpha=0.85)
                ax_fi.set_xlabel("Feature Importance (gain)")
                ax_fi.set_title("XGBoost Feature Importance")
                plt.tight_layout()
                st.pyplot(fig_fi)
                plt.close(fig_fi)

                top3 = imp_df.sort_values("importance", ascending=False).head(3)
                st.info(
                    "**상위 3대 특성:**\n"
                    + "\n".join(f"- {r['feature']}: {r['importance']:.4f}" for _, r in top3.iterrows())
                    + "\n\n**EDA 교차검증 해설:**\n"
                    "- 토크·회전속도·전력은 박스플롯·상관계수에서도 고장 구분력이 가장 높게 확인됨\n"
                    "- 공구마모(Tool wear)는 TWF 고장유형과 직결되어 파레토·중요도 모두 상위권\n"
                    "- 모델 중요도와 EDA 상관계수가 일치할수록 피처 신뢰도가 높다고 판단"
                )
            except Exception as e:
                st.error(f"특성 중요도 계산 오류: {e}")
