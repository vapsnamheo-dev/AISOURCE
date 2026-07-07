# %% [markdown]
# 허남 LLM 프로젝트 — FEMTO-ST 베어링 예지보전 ML+DL+RAG+LLM 통합 진단 노트북

# %% [markdown]
# # FEMTO-ST 베어링 예지보전 — ML+DL+RAG+LLM 통합 진단 노트북
# **PRONOSTIA IEEE PHM 2012 · ML(RandomForest 열화분류) · DL(LSTM/GRU RUL) ·
# RAG 2단계(FAISS 수치유사도 + Chroma Hybrid 문서RAG) · LLM(Claude API 자연어 진단, v0.6)**
#
# 이 노트북은 `models/` 폴더의 사전 학습된 아티팩트(RF 분류기, LSTM/GRU RUL 모델,
# FAISS 인덱스, Chroma 벡터DB)를 그대로 로드하여, 센서 측정값 하나로부터
# **ML 판정 → DL RUL 예측 → RAG(Level1 수치 + Level2 Hybrid 문서) 근거 검색 →
# LLM 자연어 진단 보고서**까지 이어지는 end-to-end 파이프라인을 위에서 아래로
# 한 번에 실행하며 보여준다. LLM 보고서 생성은 기본적으로 **Mock 모드**를 사용해
# ANTHROPIC_API_KEY 없이도 전체 파이프라인을 무료로 재현할 수 있다.

# %% [markdown]
# ## 0. 환경 설정 및 모듈 임포트
# %%
import warnings; warnings.filterwarnings("ignore")
import sys
import pickle
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

# LLM_FactoryAutomation 프로젝트 루트를 sys.path에 추가 (src 모듈 임포트용)
CANDS = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
ROOT = next((c for c in CANDS if (c / "src").exists()), Path.cwd())
sys.path.insert(0, str(ROOT))

from src import femto_rag_search
from src import femto_doc_rag
from src import femto_llm_report

MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "FEMTO_processed" / "femto_features.csv"

BASE_ML_FEATURES = [
    "h_rms", "h_kurt", "h_skew", "h_crest",
    "v_rms", "v_kurt", "v_skew", "v_crest", "temp_mean",
]
WINDOW_SIZE = 30

print("프로젝트 루트:", ROOT)
print("모듈 임포트 완료")

# %% [markdown]
# ## 1. ML — 열화 분류 (RandomForest)
# 사전 학습된 `models/femto_best_clf.pkl`(RandomForest)과 `models/femto_scaler.pkl`을
# 로드해, 샘플 센서 측정값의 열화 확률을 예측한다.
# %%
with open(MODEL_DIR / "femto_best_clf.pkl", "rb") as f:
    clf = pickle.load(f)
with open(MODEL_DIR / "femto_scaler.pkl", "rb") as f:
    ml_scaler = pickle.load(f)

# 열화 후반부를 흉내낸 샘플 센서값 (진동 RMS·첨도 급등 + 온도 상승, 데모용 합성 값)
sample_sensor = {
    "h_rms": 1.8, "h_kurt": 8.5, "h_skew": 0.9, "h_crest": 4.2,
    "v_rms": 1.5, "v_kurt": 7.2, "v_skew": 0.8, "v_crest": 3.9,
    "temp_mean": 42.0,
}

X_ml = ml_scaler.transform([[sample_sensor[f] for f in BASE_ML_FEATURES]])
ml_prob = float(clf.predict_proba(X_ml)[0, 1])
ml_label = int(ml_prob >= 0.5)

print(f"ML 열화 확률: {ml_prob:.2%}")
print(f"ML 판정: {'열화' if ml_label else '정상'}")

# %% [markdown]
# ## 2. DL — 잔여수명(RUL) 예측 (LSTM)
# 사전 학습된 `models/femto_lstm_rul.keras` + `femto_seq_scaler.pkl` + `femto_y_scaler.pkl`을
# 로드한다. 단일 시점 입력은 앱과 동일한 방식으로 윈도우 길이(30)만큼 복제(tile)해
# 시계열 입력 형태로 변환한 뒤 예측한다.
# %%
import tensorflow as tf  # noqa: E402

lstm_rul = tf.keras.models.load_model(MODEL_DIR / "femto_lstm_rul.keras")
with open(MODEL_DIR / "femto_seq_scaler.pkl", "rb") as f:
    seq_scaler = pickle.load(f)
with open(MODEL_DIR / "femto_y_scaler.pkl", "rb") as f:
    y_scaler = pickle.load(f)

input_vals = np.array([[sample_sensor[f] for f in BASE_ML_FEATURES]])
seq_input = np.tile(input_vals, (WINDOW_SIZE, 1))
seq_scaled = seq_scaler.transform(seq_input)[np.newaxis, :, :]

rul_raw = lstm_rul.predict(seq_scaled, verbose=0)[0][0]
rul_min = max(0.0, float(y_scaler.inverse_transform([[rul_raw]])[0][0]))

print(f"DL 예측 잔여수명(RUL): {rul_min:.1f}분")

# %% [markdown]
# ## 3. RAG Level 1 — 수치 유사도 검색 (FAISS)
# `src/femto_rag_search.py`의 FAISS 인덱스로, 현재 센서값과 유사한 과거 베어링
# 사례 Top-k를 검색하고 유사도 가중 평균으로 RUL을 보조 추정한다.
# %%
rag1_query = dict(sample_sensor)
rag1_result = femto_rag_search.search_and_estimate_rul(rag1_query, k=3)

print(f"RAG-Level1 유사 사례 {rag1_result['k']}건, 평균 유사도 {rag1_result['avg_similarity']}%")
print(f"RAG-Level1 유사도 가중 RUL 추정: {rag1_result['estimated_rul']}분")
for case in rag1_result["similar_cases"]:
    print(f"  - Bearing {case['bearing']} (유사도 {case['similarity']}%, RUL {case['rul']}분)")

# %% [markdown]
# ## 4. RAG Level 2 — 문서 RAG: 일반(벡터) vs Hybrid(BM25+벡터) 비교
# `src/femto_doc_rag.py`는 정비 지식 문서(`bearing_maintenance_guide.txt`)를 Chroma
# 벡터DB에 인덱싱해 검색한다. v0.6에서 BM25(키워드)+벡터 검색을 EnsembleRetriever로
# 결합한 **Hybrid RAG**가 추가되었다. 아래에서 "온도만 상승하면 어떻게 해야 하나요?"라는
# 정확한 수치/조건 표현이 포함된 질문으로 일반 RAG(벡터만)와 Hybrid RAG를 비교한다.
#
# 참고: RAG-Level2의 `ask()`(LLM 답변 생성)는 로컬 Ollama 서버(gemma)가 필요해
# Streamlit Cloud에서는 동작하지 않는다. 아래 `retrieve_docs()`는 문서 청크만
# 검색하므로 Ollama 없이 어디서나 동작한다.
# %%
doc_index = femto_doc_rag.load_index()
question = "온도만 상승하면 어떻게 해야 하나요?"

docs_vector_only = femto_doc_rag.retrieve_docs(question, k=2, vectorstore=doc_index, use_hybrid=False)
docs_hybrid = femto_doc_rag.retrieve_docs(question, k=2, vectorstore=doc_index, use_hybrid=True)

print("=== 질문:", question, "===")
print("\n[일반 RAG — 벡터 검색만]")
for d in docs_vector_only:
    print(" -", d[:120].replace("\n", " "))

print("\n[Hybrid RAG — BM25(키워드) + 벡터, v0.6]")
for d in docs_hybrid:
    print(" -", d[:120].replace("\n", " "))

# %% [markdown]
# ### 참고: GraphRAG는 현재 미구현
# 이 프로젝트의 RAG는 Level1(FAISS 수치 유사도) + Level2(Chroma Hybrid 문서 검색)
# 2단계까지만 구현되어 있다. 설비 간 인과관계·계통도처럼 개체(entity)-관계(relation)가
# 복잡하게 얽힌 지식에 강점이 있는 **GraphRAG(지식그래프 기반 검색)는 아직 구현되지
# 않았고**, 로드맵상 우선순위가 가장 낮은 4순위로 남아있다(networkx/neo4j 등 그래프
# 라이브러리 미사용). 현재 데이터(센서 수치 + 정비 가이드 텍스트)는 벡터/Hybrid 검색만
# 으로 충분히 커버되기 때문이다.

# %% [markdown]
# ## 5. LLM — 자연어 진단 보고서 생성 (Proposal A)
# `src/femto_llm_report.py`가 위에서 얻은 ML 판정 + DL RUL + RAG Level1/2 근거를
# 종합해 자연어 보고서를 생성한다. 아래는 **Mock 모드**(API 키 불필요, 과금 없음)로
# 실행한다 — 실제 Claude API를 쓰려면 `generate_report(...)` /
# `generate_report_structured(...)`를 사용하면 된다(환경변수 `ANTHROPIC_API_KEY` 필요).
# %%
doc_snippets = docs_hybrid  # Level2 Hybrid RAG 근거를 그대로 사용

report_text = femto_llm_report.generate_report_mock(
    sensor=sample_sensor,
    ml_prob=ml_prob,
    ml_label=ml_label,
    rul_min=rul_min,
    rul_alarm_min=60.0,
    doc_snippets=doc_snippets,
)
print("=== LLM 진단 보고서 (자유 텍스트, Mock) ===")
print(report_text)

# %% [markdown]
# ### Structured Output (JSON Schema 강제 출력)
# 대시보드 배지 표시·알람 연동·DB 저장에 바로 쓸 수 있도록 정형 데이터로도 생성한다.
# %%
report_structured = femto_llm_report.generate_report_structured_mock(
    sensor=sample_sensor,
    ml_prob=ml_prob,
    ml_label=ml_label,
    rul_min=rul_min,
    rul_alarm_min=60.0,
    doc_snippets=doc_snippets,
)
print("=== LLM 진단 보고서 (Structured Output, Mock) ===")
for k, v in report_structured.items():
    print(f"{k}: {v}")

# %% [markdown]
# ## 6. 결론
# ML(RandomForest 열화 분류) → DL(LSTM RUL 예측) → RAG(Level1 FAISS 수치 유사도 +
# Level2 Chroma Hybrid 문서 검색) → LLM(Claude API 자연어 진단, v0.6 기준 Hybrid RAG
# 적용) 로 이어지는 전체 파이프라인을 사전 학습된 아티팩트만으로 재현했다.
# 실제 배포본은 https://llmfactoryautomation.streamlit.app/ (app/streamlit_femto.py)
# 에서 동일한 파이프라인을 Streamlit UI로 제공한다.
# %%
print("파이프라인 요약")
print(f"  ML 판정        : {'열화' if ml_label else '정상'} (확률 {ml_prob:.2%})")
print(f"  DL RUL 예측    : {rul_min:.1f}분")
print(f"  RAG-Level1     : 유사 사례 {rag1_result['k']}건, 평균 유사도 {rag1_result['avg_similarity']}%")
print(f"  RAG-Level2     : Hybrid RAG(BM25+벡터) 근거 {len(docs_hybrid)}건")
print(f"  LLM 진단 상태  : {report_structured.get('status')}")
