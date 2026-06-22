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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

# ── [PostgreSQL 영구저장 시 활성화] ─────────────────────────────────────────
# Supabase 등 외부 DB 연동 시 아래 블록의 주석을 해제하고
# Streamlit Cloud Secrets 에 DATABASE_URL 을 설정하세요.
# secrets.toml.example 참고.
#
# import os
# try:
#     _secret_db = st.secrets.get("DATABASE_URL")
#     if _secret_db:
#         os.environ.setdefault("DATABASE_URL", _secret_db)
# except Exception:
#     pass
# ────────────────────────────────────────────────────────────────────────────

from src import db, predict, model_store, synth_ai4i

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


@st.cache_data
def _dashboard_metrics(data_path: str):
    """학습/테스트 분할 동일 조건으로 테스트셋 예측 — model_store 파이프라인 재사용."""
    from sklearn.model_selection import train_test_split
    from src import config

    df = pd.read_csv(data_path)

    failure_map = {
        "TWF": "TWF\n(공구마모)", "HDF": "HDF\n(방열)",
        "PWF": "PWF\n(전력부족)", "OSF": "OSF\n(과부하)", "RNF": "RNF\n(무작위)",
    }
    failure_counts = {v: int(df[k].values.sum()) for k, v in failure_map.items() if k in df.columns}

    # 동일 random_state로 테스트셋 분리
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    test_df = test_df.reset_index(drop=True)

    # 이미 검증된 predict 파이프라인 재사용
    out, _, has_actual = model_store.predict_dataframe_from_db(test_df)

    tp = tn = fp = fn = 0
    if has_actual:
        tp = int(((out["pred_label"] == 1) & (out["actual"] == 1)).sum())
        tn = int(((out["pred_label"] == 0) & (out["actual"] == 0)).sum())
        fp = int(((out["pred_label"] == 1) & (out["actual"] == 0)).sum())
        fn = int(((out["pred_label"] == 0) & (out["actual"] == 1)).sum())

    total_n = max(len(out), 1)
    acc = (tp + tn) / total_n
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    y = df["Target"].values if "Target" in df.columns else []
    return {
        "total": len(df),
        "train_n": len(df) - len(test_df), "test_n": len(test_df),
        "total_failure": int(sum(y)), "total_normal": int(sum(1 for v in y if v == 0)),
        "failure_counts": failure_counts,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "acc": acc, "precision": precision, "recall": recall, "f1": f1,
    }

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

    st.divider()
    st.subheader("⚙️ 판정 임계값")

    # 세션 첫 로드 시에만 DB에서 기본값 읽기
    if "thr_db_loaded" not in st.session_state:
        try:
            _db_thr = model_store.load_threshold_from_db()
            st.session_state["thr_default"] = _db_thr if _db_thr is not None else 0.85
        except Exception:
            st.session_state["thr_default"] = 0.85
        st.session_state["thr_db_loaded"] = True

    threshold = st.slider(
        "고장 판정 임계값", 0.0, 1.0,
        float(st.session_state["thr_default"]), 0.01,
        help="이 값 이상의 고장 확률을 '고장'으로 판정합니다."
    )
    if st.button("💾 이 임계값 저장 (DB)", help="저장 시 다음 실행에서도 이 값이 기본값으로 유지됩니다."):
        try:
            model_store.save_threshold_to_db(threshold)
            st.session_state["thr_default"] = threshold
            st.success(f"임계값 {threshold:.2f} 저장 완료!")
        except Exception as _e:
            st.error(f"저장 실패: {_e}")

    with st.expander("임계값 변경 이력 (DB)"):
        try:
            _hist = model_store.load_threshold_history(limit=20)
            if _hist:
                for _h in _hist:
                    _ts = _h["changed_at"].strftime("%Y-%m-%d %H:%M")
                    st.write(f"{_ts}  {_h['old']:.2f} → **{_h['new']:.2f}**")
            else:
                st.caption("저장된 이력 없음 (저장 버튼을 눌러야 기록됩니다)")
        except Exception:
            st.caption("이력 로드 실패")

tab1, tab2, tab3 = st.tabs(["🔮 단건 예측", "📁 CSV 일괄 검증", "📊 성능 대시보드"])

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
            pred_by_thr = 1 if res["pred_proba"] >= threshold else 0
            if pred_by_thr == 1:
                st.error("⚠️ 예측: 고장 위험 (점검 권장)")
            else:
                st.success("✅ 예측: 정상 범위")
            st.markdown("**실제(물리규칙) 판정 vs 예측**")
            st.write(f"- 실제 결과(물리규칙): **{'고장' if actual == 1 else '정상'}**")
            st.write(f"- 모델 예측: **{'고장' if pred_by_thr == 1 else '정상'}** (임계값 {threshold:.2f})")
            if pred_by_thr == actual:
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

with tab3:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    st.subheader("📊 설비 고장 예측 모델 — 성능 대시보드")
    st.caption("데이터: predictive_maintenance.csv · 모델: XGBoost · 테스트셋 20% (random_state=42)")

    try:
        from src import config as _cfg
        d = _dashboard_metrics(str(_cfg.DATA_PATH))
    except Exception as e:
        st.error(f"대시보드 데이터 로드 실패: {e}")
        d = None

    if d:
        # ── KPI 카드 ──────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("전체 데이터", f"{d['total']:,} 건",
                  f"학습 {d['train_n']:,} / 테스트 {d['test_n']:,}")
        k2.metric("정상 건수 (전체)", f"{d['total_normal']:,} 건",
                  f"정상률 {d['total_normal']/d['total']*100:.1f}%")
        k3.metric("고장 건수 (전체)", f"{d['total_failure']:,} 건",
                  f"불량률 {d['total_failure']/d['total']*100:.1f}%")
        k4.metric("테스트 정확도", f"{d['acc']*100:.1f} %",
                  f"F1 {d['f1']*100:.1f}%")

        st.divider()
        col_l, col_r = st.columns([3, 2])

        # ── 파레토 차트 ───────────────────────────────────────────
        with col_l:
            st.markdown("**고장 유형별 파레토 분석 (전체 데이터)**")
            fc = d["failure_counts"]
            sorted_fc = sorted(fc.items(), key=lambda x: x[1], reverse=True)
            labels_s = [k for k, _ in sorted_fc]
            values_s = [v for _, v in sorted_fc]
            total_f = max(sum(values_s), 1)
            cumulative = [sum(values_s[:i+1]) / total_f * 100 for i in range(len(values_s))]

            fig, ax1 = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor("#0e1117")
            ax1.set_facecolor("#0e1117")

            bar_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]
            bars = ax1.bar(range(len(labels_s)), values_s,
                           color=bar_colors[:len(labels_s)], edgecolor="none")
            for i, v in enumerate(values_s):
                ax1.text(i, v + 0.5, str(v), ha="center", va="bottom",
                         color="white", fontsize=10, fontweight="bold")
            ax1.set_xticks(range(len(labels_s)))
            ax1.set_xticklabels(labels_s, color="white", fontsize=9)
            ax1.set_ylabel("Count", color="white")
            ax1.tick_params(axis="y", colors="white")
            for spine in ax1.spines.values():
                spine.set_color("#333")

            ax2 = ax1.twinx()
            ax2.plot(range(len(labels_s)), cumulative, "o-",
                     color="#00d4ff", lw=2, ms=6, label="누적%")
            ax2.axhline(80, color="#ffcc00", linestyle="--", lw=1, alpha=0.7)
            ax2.set_ylabel("Cumulative %", color="#00d4ff")
            ax2.tick_params(colors="#00d4ff")
            ax2.set_ylim(0, 115)
            for s in ["top", "left", "bottom"]:
                ax2.spines[s].set_visible(False)
            ax2.spines["right"].set_color("#00d4ff")

            ax1.set_title("Failure Type Pareto", color="white", fontsize=12, pad=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ── 도넛 차트 (정확도) ────────────────────────────────────
        with col_r:
            st.markdown("**테스트셋 예측 정확도**")
            fig2, ax = plt.subplots(figsize=(4, 4))
            fig2.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")

            acc_v = d["acc"] * 100
            ax.pie([acc_v, 100 - acc_v],
                   colors=["#2ecc71", "#e74c3c"],
                   startangle=90,
                   wedgeprops=dict(width=0.45, edgecolor="#0e1117", linewidth=3))
            ax.text(0, 0.1, f"{acc_v:.1f}%", ha="center", va="center",
                    fontsize=26, fontweight="bold", color="white")
            ax.text(0, -0.28, "Accuracy", ha="center", va="center",
                    fontsize=11, color="#aaaaaa")
            ax.legend(handles=[Patch(color="#2ecc71", label="Correct"),
                                Patch(color="#e74c3c", label="Error")],
                      loc="lower center", ncol=2, frameon=False,
                      labelcolor="white", fontsize=9)
            ax.set_title("Test Set Performance", color="white", fontsize=12, pad=8)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        # ── 혼동행렬 + 상세 지표 ──────────────────────────────────
        st.divider()
        st.markdown("**혼동행렬 · 상세 지표 (테스트셋)**")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("TP (탐지성공)", d["tp"])
        m2.metric("TN (정상확인)", d["tn"])
        m3.metric("FP (오경보)", d["fp"])
        m4.metric("FN (고장놓침)", d["fn"])
        m5.metric("Precision", f"{d['precision']*100:.1f}%")
        m6.metric("Recall", f"{d['recall']*100:.1f}%")
