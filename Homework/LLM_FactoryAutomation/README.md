# LLM_FactoryAutomation — LLM 기반 예지보전(PdM) 진단 시스템

FEMTO-ST 베어링 PdM에 **2단계 RAG + 이중 LLM**을 통합하여, ML/DL 수치 예측 결과를 현장 담당자가 즉시 이해할 수 있는 **자연어 정비 권고 보고서**로 자동 생성하는 시스템입니다.

ML_FactoryAutomation(분류) → DL_FactoryAutomation(RUL 예측)의 ML+DL 기반 위에, LLM(Proposal A: RAG+ML/DL → 자연어 보고서)을 결합해 end-to-end PdM 파이프라인을 완성했습니다. (v0.7)

---

## 핵심 기능 — Proposal A + 2단계 RAG + 이중 LLM

```text
센서 측정값
    |
    +-- ML 모델 (RandomForest/XGBoost) ---- 열화 확률 + 정상/열화 판정
    +-- DL 모델 (LSTM+RF 앙상블, 30분 윈도우) -- 잔여수명(RUL) 예측 (분)
    +-- RAG Level 1 (FAISS, 수치 유사도) ---- 유사 사례 Top-k 검색
    +-- RAG Level 2 (Chroma, Hybrid RAG) ---- 정비 지식 문서 검색 (BM25+벡터 50:50)
             |
    +-------------------------------------------+
    |  이중 LLM 구조                              |
    |  · Claude API (claude-haiku-4-5) -> 진단 보고서 생성 (Structured Output)
    |  · 로컬 Ollama (gemma4:e2b)      -> RAG-Level2 자유 질의응답
    +-------------------------------------------+
             |
    환각(Hallucination) 방어 3층 게이트 (신뢰도 톤 분기 · 입력 OOD 판정 · 클래스 한정 자기인지)
             |
    자연어 진단 보고서 (5섹션 + 출처근거 Citation)
    [현재상태] [이상신호] [정비권고] [유사사례] [문서근거]
```

---

## 시스템 구성

| 레이어 | 모듈 | 역할 |
|---|---|---|
| LLM 보고서 | `src/femto_llm_report.py` | Claude API 호출, Structured Output(JSON Schema), 자연어 5섹션 보고서 |
| 환각 방어 | `src/femto_llm_guard.py` | 신뢰도 톤 분기 · 입력 OOD 판정 · 클래스 한정 자기인지 3층 게이트 |
| RAG Level 1 | `src/femto_rag_search.py` | FAISS IndexFlatIP(코사인), 유사 사례 Top-k 검색 |
| RAG Level 2 | `src/femto_doc_rag.py` | Chroma 벡터DB + BM25 Hybrid RAG, 정비 지식 문서 Q&A + 출처근거(Citation) 태깅 |
| ML | `src/femto_ml.py` | RandomForest/XGBoost 열화 분류 |
| DL | `src/femto_dl_rul.py` | LSTM+RandomForest 앙상블 RUL 회귀 (30분 슬라이딩 윈도우, 9피처) |
| UI | `app/streamlit_femto.py` (외 3종) | 7탭 통합 진단 대시보드 (Tab: AI 정비 권고) |

> `src/femto_gru_rul.py`(GRU v2, BN+LN)는 "아키텍처 비교(5종)" 탭에서 LSTM·BiLSTM·1D-CNN·CNN-LSTM과 성능을 비교하기 위한 모델이며, LLM 정비 권고 탭의 RUL 예측에는 연결돼 있지 않습니다.

---

## Streamlit 앱 구성 (4종)

| 앱 파일 | 역할 | 사용 LLM |
|---|---|---|
| `app/streamlit_app.py` | DL 메인 — 시계열 고장 예측 | - |
| `app/streamlit_rag.py` | LLM/RAG 전용 — Level1 유사 사례 데모 | - |
| `app/streamlit_femto.py` | **ML+DL+LLM 통합 진단 (7탭)** | Ollama(gemma4:e2b) + Claude Haiku 4.5 병행 |
| `app/streamlit_unified.py` | 3-Source 통합 (ML+DL+DL Milling) | - |

## Tab — AI 정비 권고 (`streamlit_femto.py`)

7탭 구성: 데이터 탐색 · ML 성능 · DL RUL 예측 · 통합 진단(실시간) · 아키텍처 비교(5종) · CNN 이미지 분류 · **LLM 정비 권고(AI 진단)**

1. 센서 슬라이더 입력 (h_rms, h_kurt, v_rms 등 8개 + 온도)
2. "AI 정비 권고 보고서 생성" 버튼 클릭
3. ML 확률 / DL RUL / RAG Level1·2 근거 중간 메트릭 표시
4. Claude API 자연어 5섹션 보고서 + 출처근거(Citation) 출력

**ANTHROPIC_API_KEY 미설정 시**: 규칙 기반 Mock 모드로 자동 전환 (API 비용 없이 시연 가능, on/off 스위치 기본 ON)
**로컬 Ollama 미설치 시**: RAG-Level2 자유 질의응답(ask())만 제한, 나머지 기능은 정상 동작 (Streamlit Cloud 배포판도 동일 — 버그 아닌 의도된 제약)

---

## 빠른 시작

```bash
# 1. 환경 설정
pip install -r requirements.txt

# 2. API 키 설정 (선택 -- 없으면 Mock 모드)
set ANTHROPIC_API_KEY=sk-ant-...

# 3. (선택) 로컬 LLM Ollama 설치 — RAG-Level2 자유 질의응답용
ollama pull gemma4:e2b

# 4. 데이터 전처리 (최초 1회)
python -m src.femto_preprocess

# 5. 모델 학습
python -m src.femto_ml
python -m src.femto_dl_rul
python -m src.femto_rag_search       # RAG Level1 - FAISS 인덱스 빌드
python -m src.femto_doc_rag          # RAG Level2 - Chroma 인덱스 빌드

# 6. Streamlit 실행
streamlit run app/streamlit_femto.py
```

---

## 모델 성능

| 모델 | 유형 | 성능 |
|---|---|---|
| RandomForest/XGBoost | ML 열화 분류 | AUC 0.99, Recall 0.91 (GroupKFold, 베어링 단위 분리로 검증) |
| LSTM+RF 앙상블 | DL RUL 예측 | LSTM RMSE 934분 / RF RMSE 955분 (30분 윈도우, femto_rul_results.json) |
| Autoencoder | 비지도 이상탐지 | AUC 0.968 (라벨 불필요) |
| FAISS RAG (Level1) | 유사 사례 검색 | 코사인 유사도, 12-dim |
| Hybrid RAG (Level2) | 문서 검색 | BM25(키워드)+Chroma(벡터) 50:50 EnsembleRetriever |
| Claude Haiku 4.5 | 진단 보고서 생성 | Structured Output(JSON Schema), 5섹션 |

---

## 기술 스택

- **언어**: Python 3.11
- **LLM(클라우드)**: Anthropic Claude API (claude-haiku-4-5-20251001) — 진단 보고서 생성, Structured Output
- **LLM(로컬)**: Ollama (gemma4:e2b) — RAG-Level2 자유 질의응답
- **딥러닝**: TensorFlow 2.x · Keras (GRU, LSTM, Autoencoder)
- **ML**: scikit-learn (RandomForest), XGBoost
- **RAG**: FAISS(IndexFlatIP, Level1 수치유사도) · LangChain + Chroma + BM25Retriever(Level2 Hybrid 문서RAG)
- **임베딩**: all-MiniLM-L6-v2 (chromadb 내장 ONNX 런타임 — PyTorch·TensorFlow 프로세스 충돌 방지를 위해 sentence-transformers 패키지에서 교체, 2026-09-07)
- **UI**: Streamlit (4개 앱)
- **설명가능성**: SHAP (LinearExplainer, KernelExplainer)

---

## 릴리스 노트

| 버전 | 내용 |
|---|---|
| v0.5 | RAG(2-Level) + LLM 진단보고서 통합, Structured Output(JSON Schema) |
| v0.6 | Hybrid RAG(BM25+벡터) 적용, ANTHROPIC_API_KEY on/off 스위치, 배포 안정성 버그 수정 |
| v0.7 | 환각(Hallucination) 방어 3층 게이트(`femto_llm_guard.py`) 신규, RAG 출처근거(Citation) 표기 개선(`femto_doc_rag.py`) |

---

## 프로젝트 구조

```text
LLM_FactoryAutomation/
├── src/
│   ├── femto_llm_report.py    # Claude API + Structured Output + 자연어 보고서 생성
│   ├── femto_llm_guard.py     # 환각 방어 3층 게이트
│   ├── femto_rag_search.py    # RAG Level1 - FAISS 유사 사례 검색
│   ├── femto_doc_rag.py       # RAG Level2 - Chroma+BM25 Hybrid 문서 RAG + 출처근거 태깅
│   ├── femto_ml.py            # ML 열화 분류 (RF/XGBoost)
│   ├── femto_dl_rul.py        # DL RUL 예측 (LSTM+RF 앙상블, 30분 윈도우)
│   └── femto_preprocess.py    # 데이터 전처리
├── app/
│   ├── streamlit_app.py       # DL 메인
│   ├── streamlit_rag.py       # RAG Level1 데모
│   ├── streamlit_femto.py     # ML+DL+LLM 통합 진단 (7탭, Tab: AI 정비 권고)
│   └── streamlit_unified.py   # 3-Source 통합
├── tests/
│   ├── test_femto_llm_guard.py
│   └── test_femto_doc_rag_citations.py
├── 산출물/
│   └── LLM-RAG 프로젝트/
│       ├── LLM_산출물_20260629_FEMTO_LLM진단.docx
│       └── LLM_프로젝트_발표용.pptx
└── requirements.txt
```

> Streamlit Cloud 배포(`llmfactoryautomation.streamlit.app`)는 `app/requirements.txt`를 사용합니다.

---

## 참고 — DL_FactoryAutomation 계승 파일

이 폴더는 DL_FactoryAutomation에서 복제해 시작했기 때문에, AI4I 합성 시계열 파이프라인 파일이 함께 남아 있습니다: `src/config.py` · `generate_ts_data.py` · `data_loader.py` · `dl_model.py` · `train.py` · `predict.py`, `app/streamlit_app.py`, `tests/test_generate.py`, `data/train·test·demo`, `.github/workflows/ci.yml`.
이 파이프라인의 설명은 [DL_FactoryAutomation/README.md](../DL_FactoryAutomation/README.md)를 참고하세요.

---

*데이터: FEMTO-ST PRONOSTIA IEEE PHM 2012 베어링 가속열화 데이터셋*
