# -*- coding: utf-8 -*-
"""FEMTO-ST RAG 유사 사례 검색 Streamlit 앱 — Level 1 (FAISS + 12-dim 벡터).

실행:
    C:\\AISOURCE\\.venv\\Scripts\\streamlit.exe run app/streamlit_rag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src.femto_rag_search import (
    INDEX_PATH, META_PATH, FEAT_PATH, PROCESSED_DIR,
    build_index, search_and_estimate_rul, _load_feature_list, _fill_nan,
)

# ── 한국어 폰트 설정 (matplotlib + Plotly) ───────────────────────────────────
import matplotlib
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

try:
    import plotly.io as pio
    _korean_template = pio.templates["plotly_white"]
    _korean_template.layout.font.family = "Malgun Gothic, Apple Gothic, sans-serif"
    pio.templates["korean"] = _korean_template
    pio.templates.default = "korean"
except Exception:
    pass

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FEMTO RAG — 유사 사례 검색",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 FEMTO-ST 베어링 유사 사례 검색 (RAG Level 1)")
st.caption("현재 센서 측정값 → FAISS 코사인 유사도 검색 → 과거 유사 사례 Top-K 및 RUL 추정")

# ── 사이드바: 설정 ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 검색 설정")
    k = st.slider("유사 사례 수 (Top-K)", 1, 20, 5)
    exclude_same = st.checkbox("동일 베어링 제외", value=True,
                               help="쿼리와 같은 베어링의 이력을 결과에서 제외")
    st.divider()

    # 인덱스 빌드 버튼
    st.subheader("🗄️ 인덱스 관리")
    index_exists = INDEX_PATH.exists() and META_PATH.exists()
    st.caption(f"인덱스 상태: {'✅ 존재' if index_exists else '❌ 없음'}")
    if st.button("🔨 인덱스 빌드 / 재빌드", type="primary"):
        with st.spinner("FAISS 인덱스 빌드 중…"):
            try:
                build_index(verbose=False)
                st.success("인덱스 빌드 완료!")
                st.rerun()
            except Exception as e:
                st.error(f"빌드 실패: {e}")

    st.divider()
    st.info(
        "**사용 방법**\n"
        "1. 인덱스가 없으면 위 버튼 클릭\n"
        "2. 메인 화면에서 센서값 입력\n"
        "3. '유사 사례 검색' 버튼 클릭\n"
        "4. Top-K 결과 + RUL 추정 확인"
    )

# ── 캐시: 기준 통계 로드 ──────────────────────────────────────────────────────
@st.cache_data
def _load_stats():
    if not FEAT_PATH.exists():
        return {}, []
    features = _load_feature_list()
    df = pd.read_csv(FEAT_PATH)
    df = _fill_nan(df, features)
    stats = {f: {"min": float(df[f].min()), "max": float(df[f].max()),
                 "mean": float(df[f].mean()), "med": float(df[f].median())}
             for f in features}
    return stats, features


@st.cache_data
def _load_all_data():
    if not FEAT_PATH.exists():
        return pd.DataFrame(), []
    features = _load_feature_list()
    df = pd.read_csv(FEAT_PATH)
    df = _fill_nan(df, features)
    return df, features


stats, features = _load_stats()
df_all, _ = _load_all_data()

# ── 인덱스 없을 때 안내 ───────────────────────────────────────────────────────
if not index_exists:
    st.warning("⚠️ FAISS 인덱스가 없습니다. 좌측 사이드바에서 '인덱스 빌드'를 먼저 실행하세요.")
    st.stop()

# ── 탭 구성 ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 유사 사례 검색", "📊 베어링 이력 탐색", "ℹ️ RAG 개념 설명"])

# ════════════════════════════════════════════════════════════════════════════
# Tab 1: 유사 사례 검색 (핵심 기능)
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📥 현재 센서 측정값 입력")

    input_mode = st.radio(
        "입력 방법",
        ["슬라이더 직접 입력", "베어링 이력에서 선택"],
        horizontal=True,
    )

    query: dict = {}

    if input_mode == "슬라이더 직접 입력":
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**수평 진동**")
            for f in ["h_rms", "h_kurt", "h_skew", "h_crest"]:
                if f in stats:
                    s = stats[f]
                    query[f] = st.slider(f, float(s["min"]), float(s["max"]),
                                         float(s["med"]), key=f"q_{f}",
                                         format="%.4f")
        with col2:
            st.markdown("**수직 진동**")
            for f in ["v_rms", "v_kurt", "v_skew", "v_crest"]:
                if f in stats:
                    s = stats[f]
                    query[f] = st.slider(f, float(s["min"]), float(s["max"]),
                                         float(s["med"]), key=f"q_{f}",
                                         format="%.4f")
        with col3:
            st.markdown("**파생 특성**")
            for f in ["temp_mean", "energy", "health_idx", "rms_ratio"]:
                if f in stats:
                    s = stats[f]
                    query[f] = st.slider(f, float(s["min"]), float(s["max"]),
                                         float(s["med"]), key=f"q_{f}",
                                         format="%.4f")

    else:  # 베어링 이력에서 선택
        if df_all.empty:
            st.error("데이터 없음")
        else:
            col_s1, col_s2 = st.columns(2)
            bearing_list = sorted(df_all["bearing"].unique().tolist())
            sel_bearing = col_s1.selectbox("베어링 선택", bearing_list)
            bdf = df_all[df_all["bearing"] == sel_bearing].sort_values("minute")
            sel_minute = col_s2.selectbox(
                "시각(분) 선택",
                bdf["minute"].tolist(),
                index=len(bdf) // 2,
            )
            row = bdf[bdf["minute"] == sel_minute].iloc[0]
            query = {f: float(row[f]) for f in features if f in row.index}
            query["bearing"] = sel_bearing

            # 선택 시점 정보 표시
            st.info(
                f"선택: **{sel_bearing}**  t={sel_minute}분  "
                f"실제 RUL={row['rul']:.0f}분  "
                f"상태={'🔴 열화' if row['label'] == 1 else '🟢 정상'}"
            )

    st.divider()

    # ── 검색 실행 ──────────────────────────────────────────────────────────
    run_search = st.button("🔍 유사 사례 검색", type="primary", use_container_width=True)

    if run_search and features:
        with st.spinner(f"FAISS 코사인 유사도 검색 중… (Top-{k})"):
            try:
                result = search_and_estimate_rul(
                    query, k=k, exclude_same_bearing=exclude_same
                )
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()

        cases = result["similar_cases"]

        # ── 결과 헤더 ──────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("검색된 사례 수", f"{len(cases)}개")
        m2.metric("평균 유사도", f"{result['avg_similarity']}%")
        est = result["estimated_rul"]
        if est is not None:
            risk = ("🔴 긴급" if est < 50 else
                    "🟠 주의" if est < 150 else "🟢 양호")
            m3.metric("RAG 추정 RUL", f"{est:.0f} 분", help="유사 사례 가중평균")
            st.markdown(f"### {risk}  RAG 추정 잔여수명: **{est:.0f}분**")
        else:
            m3.metric("RAG 추정 RUL", "N/A")

        st.divider()

        # ── 유사 사례 테이블 ───────────────────────────────────────────────
        st.subheader(f"유사 사례 Top-{len(cases)}")
        tbl = pd.DataFrame([{
            "순위":       r["rank"],
            "유사도(%)":  r["similarity"],
            "베어링":     r["bearing"],
            "시각(분)":   r["minute"],
            "RUL(분)":    int(r["rul"]) if r["rul"] else "-",
            "RUL(%)":     f"{r['rul_pct']*100:.1f}" if r.get("rul_pct") is not None else "-",
            "상태":       "🔴 열화" if r["label"] == 1 else "🟢 정상",
            "데이터셋":   r["split"],
        } for r in cases])

        st.dataframe(
            tbl.style.background_gradient(subset=["유사도(%)"], cmap="Blues"),
            use_container_width=True,
            hide_index=True,
        )

        # ── 유사도 시각화 ──────────────────────────────────────────────────
        st.subheader("유사도 비교")
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            colors = ["#C00000" if r["label"] == 1 else "#4472C4" for r in cases]
            labels = [f"{r['bearing']}<br>t={r['minute']}분" for r in cases]
            sims   = [r["similarity"] for r in cases]

            fig.add_trace(go.Bar(
                x=labels, y=sims,
                marker_color=colors,
                text=[f"{s:.1f}%" for s in sims],
                textposition="outside",
                name="",
                showlegend=False,
            ))
            fig.add_hline(y=result["avg_similarity"],
                          line_dash="dash", line_color="gray",
                          annotation_text=f"평균 {result['avg_similarity']}%")
            fig.add_trace(go.Bar(x=[None], y=[None], marker_color="#C00000", name="열화"))
            fig.add_trace(go.Bar(x=[None], y=[None], marker_color="#4472C4", name="정상"))
            fig.update_layout(
                yaxis_title="코사인 유사도 (%)",
                xaxis_title="유사 사례",
                height=350,
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(pd.DataFrame({"유사도(%)": [r["similarity"] for r in cases],
                                       "베어링": [r["bearing"] for r in cases]}).set_index("베어링"))

        # ── RUL 분포 비교 ──────────────────────────────────────────────────
        rul_cases = [r for r in cases if r["rul"] is not None]
        if rul_cases and est is not None:
            st.subheader("RUL 분포 — 유사 사례 vs 추정값")
            try:
                import plotly.graph_objects as go
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=[r["rank"] for r in rul_cases],
                    y=[r["rul"] for r in rul_cases],
                    mode="markers+text",
                    marker=dict(size=14, color=[r["similarity"] for r in rul_cases],
                                colorscale="Blues", showscale=True,
                                colorbar=dict(title="유사도%")),
                    text=[f"{r['rul']:.0f}분" for r in rul_cases],
                    textposition="top center",
                    name="유사 사례 RUL",
                ))
                fig2.add_hline(y=est, line_dash="dash", line_color="#C00000",
                               annotation_text=f"RAG 추정 RUL={est:.0f}분",
                               annotation_font_color="#C00000")
                fig2.update_layout(
                    xaxis_title="유사 사례 순위",
                    yaxis_title="RUL (분)",
                    height=320,
                )
                st.plotly_chart(fig2, use_container_width=True)
            except ImportError:
                pass

        # ── 특성값 비교 ────────────────────────────────────────────────────
        with st.expander("📋 특성값 상세 비교 (쿼리 vs 유사 사례)"):
            cmp_rows = {"쿼리(현재)": {f: round(query.get(f, 0.0), 4) for f in features}}
            for r in cases[:3]:
                label = f"#{r['rank']} {r['bearing']} t={r['minute']}"
                cmp_rows[label] = {f: round(r["features"].get(f, 0.0), 4) for f in features}
            st.dataframe(pd.DataFrame(cmp_rows).T, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# Tab 2: 베어링 이력 탐색
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 베어링별 h_rms 진동 추이")
    if df_all.empty:
        st.info("데이터 없음. femto_preprocess 먼저 실행하세요.")
    else:
        try:
            import plotly.graph_objects as go
            fig3 = go.Figure()
            colors_list = ["#4472C4","#ED7D31","#A9D18E","#FF0000","#7030A0","#00B0F0","#FFC000"]
            for i, (bearing, bdf) in enumerate(df_all.groupby("bearing")):
                bdf = bdf.sort_values("minute")
                fig3.add_trace(go.Scatter(
                    x=bdf["minute"], y=bdf["h_rms"],
                    name=bearing, line=dict(color=colors_list[i % len(colors_list)], width=1.5),
                ))
            fig3.update_layout(xaxis_title="시간 (분)", yaxis_title="h_rms", height=420)
            st.plotly_chart(fig3, use_container_width=True)
        except ImportError:
            pivot = df_all.pivot_table(index="minute", columns="bearing", values="h_rms")
            st.line_chart(pivot)

        # 베어링별 요약
        st.subheader("베어링별 요약 통계")
        summary = df_all.groupby("bearing").agg(
            총분=("minute", "count"),
            최대시간=("minute", "max"),
            열화비율=("label", "mean"),
            최소RUL=("rul", "min"),
            최대RUL=("rul", "max"),
        ).reset_index()
        summary["열화비율"] = (summary["열화비율"] * 100).round(1).astype(str) + "%"
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# Tab 3: RAG 개념 설명
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("ℹ️ RAG Level 1 — 동작 원리")
    st.markdown("""
### 시스템 구성
```
현재 측정값 (12개 특성)
        ↓
  MinMaxScaler 정규화
        ↓
  L2 정규화 (코사인 유사도용)
        ↓
 FAISS IndexFlatIP 검색
        ↓
 유사 사례 Top-K 반환
        ↓
 가중평균 RUL 추정
```

### 유사도 계산 공식
- **코사인 유사도** = (쿼리벡터 · DB벡터) / (|쿼리| × |DB|)
- L2 정규화 후 내적(Inner Product) = 코사인 유사도
- 100% = 완전 동일, 0% = 완전 무관

### RUL 추정 방법 (가중 평균)
```
추정 RUL = Σ(유사도ᵢ × RULᵢ) / Σ(유사도ᵢ)
```

### Level 1 → Level 2 업그레이드 경로
| | Level 1 (현재) | Level 2 |
|---|---|---|
| 임베딩 | 12-dim 특성 벡터 | CNN/GRU 인코더 32-dim |
| 의미론적 유사도 | 수치 유사도 | 패턴 의미 유사도 |
| LLM 연동 | ❌ | ✅ 자동 진단 보고서 |
""")
