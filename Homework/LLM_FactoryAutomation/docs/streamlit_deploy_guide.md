# Streamlit Community Cloud 배포 가이드

FEMTO-ST 베어링 예지보전 프로젝트를 share.streamlit.io에 배포하는 절차입니다.

## 사전 준비 (완료됨)

- GitHub 리포지토리: https://github.com/vapsnamheo-dev/AISOURCE (main 브랜치)
- 릴리스: `llm-factory-automation-v0.5` (draft)
- 모델 아티팩트: `models/` 대부분 추적됨 (단, `femto_rf_rul.pkl` 72MB는 GitHub 용량 문제로 제외).
  `models/chroma_store/`(RAG-Level2 벡터DB 바이너리)도 git에 커밋되어 있음 — 향후 용량 문제 발생 시 확인 대상

## 배포 절차

1. https://share.streamlit.io 접속 → GitHub 계정(vapsnamheo-dev)으로 로그인
2. **New app** 클릭
   - Repository: `vapsnamheo-dev/AISOURCE`
   - Branch: `main`
   - Main file path: `Homework/LLM_FactoryAutomation/app/streamlit_femto.py`
     (다른 앱을 배포하려면 `streamlit_rag.py` / `streamlit_unified.py` / `streamlit_app.py` 중 선택)
3. **Advanced settings**
   - Python version: 3.11
   - Secrets에 추가:
     ```toml
     ANTHROPIC_API_KEY = "sk-ant-..."
     ```
     (LLM 진단 보고서 기능용. 미설정 시 Mock 모드로 자동 폴백)
4. **Deploy** 클릭 → 수 분 후 `https://[앱이름].streamlit.app` 형태의 영구 URL 발급

**재배포가 안 반영될 때**: main에 push하면 보통 자동으로 재배포되지만, 가끔 캐시가 남아 반영이 늦는 경우 앱 우측 하단 **⋮** → **Manage app** → **Reboot app**으로 강제 재시작할 수 있습니다.

## Secrets 설정 방법 (2가지)

`ANTHROPIC_API_KEY` 같은 값은 아래 두 경로 중 상황에 맞는 쪽으로 넣습니다. 두 방법 모두 최종적으로 Python 코드에서는 `os.environ.get("ANTHROPIC_API_KEY", "")`로 동일하게 읽힙니다.

| 방법 | 대상 | 절차 |
| --- | --- | --- |
| **① Cloud 대시보드에서 직접 입력** (배포/운영용) | 실제 배포된 앱 | share.streamlit.io → 앱 선택 → 우측 하단 **⋮(점 3개)** → **Settings** → **Secrets** 탭 → 텍스트박스에 TOML 형식으로 붙여넣기(`ANTHROPIC_API_KEY = "sk-ant-..."`) → **Save**. 저장 즉시 앱이 자동 재시작되며 값이 바로 반영됨 |
| **② 로컬 `.streamlit/secrets.toml` 파일** (로컬 개발/테스트용) | 내 PC에서 `streamlit run`으로 실행할 때 | 프로젝트 루트에 `.streamlit/secrets.toml` 파일을 만들고 같은 내용을 저장. Streamlit이 로컬 실행 시 자동으로 읽어 `os.environ`처럼 노출됨 |

**차이**: ①은 Cloud에 배포된 앱 자체에 주입되는 값(팀/공유 배포용), ②는 로컬 실행 시에만 쓰이는 값(개인 PC별로 각자 보유).

**주의**: `.streamlit/secrets.toml`은 실제 키 값을 담고 있으므로 **절대 git에 커밋하면 안 됩니다** — `.gitignore`에 `.streamlit/secrets.toml`이 이미 추가되어 있습니다.

## ANTHROPIC_API_KEY 사용 on/off 스위치 (과금 방지)

`app/streamlit_femto.py`의 "AI 정비 권고 보고서" 섹션(사이드바)에 **"ANTHROPIC_API_KEY 사용"** 체크박스가 있습니다.

| 상태 | 동작 | 과금 |
| --- | --- | --- |
| **ON**(기본값) | `ANTHROPIC_API_KEY`가 설정돼 있으면 실제 Claude API(`generate_report()`)를 호출해 진짜 AI 보고서 생성 | 발생 |
| **OFF** | 키가 설정돼 있어도 항상 Mock 모드(`generate_report_mock()`, 규칙 기반)로 동작 | 없음 |

- 실제 호출 시(ON) 화면에 "AI 비용 정보 화면에 표시" 토글(기본 ON)로 토큰 수·예상 비용을 바로 확인할 수 있고, 같은 정보가 Python `logging`으로도 기록됩니다(`src/femto_llm_report.py`의 `logger.info(...)`).
- Secrets에 키를 넣어도 즉시 과금되는 게 아니라, **이 스위치가 ON이고 실제로 보고서 생성 버튼을 눌러야만** API가 호출됩니다.
- 데모·시연처럼 과금을 원치 않는 상황에서는 이 스위치를 OFF로 두면 키를 지우지 않고도 안전하게 Mock 모드로 동작합니다.

## 알려진 제약사항

- `femto_rf_rul.pkl`(72MB, RUL 회귀 모델 중 하나)은 리포지토리에서 제외되어 있어
  해당 모델을 직접 참조하는 기능은 Cloud에서 동작하지 않습니다.
  (`femto_lstm_rul.keras`는 포함되어 있어 LSTM 기반 RUL 예측은 정상 동작)
- RAG Level 2(`femto_doc_rag.py`)의 `ask()` 함수(`streamlit_femto.py`의 "🤖 AI 답변 생성" 버튼)는
  로컬 Ollama 서버(`http://localhost:11434`)가 필요해 Cloud 환경에서는
  `HTTPConnectionPool ... Connection refused` 에러가 정상적으로 발생합니다(버그 아님, 의도된
  동작 — Streamlit Community Cloud는 Ollama 같은 상시 백그라운드 프로세스를 실행할 수 없는
  샌드박스 환경이라 근본적으로 Ollama 자체를 Cloud에 띄울 수 없습니다). 문서 청크 검색만 하는
  `retrieve_docs()` 경로는 Ollama 없이도 정상 동작합니다.
  - **Cloud에서도 AI 답변을 받고 싶다면**: Ollama를 Cloud에서 직접 실행하는 대신, 이미 LLM
    진단 보고서용으로 쓰는 `ANTHROPIC_API_KEY`(Claude API)를 `ask()`의 대체 백엔드로 써서
    "Ollama 연결 실패 시 Claude API로 자동 폴백"하는 방식이 현실적입니다(외부에 별도 Ollama
    서버를 상시 띄워 두는 방법도 있으나 운영 부담·비용이 커서 비권장). 필요 시 별도 작업으로
    진행하세요.
- `data/` 폴더는 Git에서 제외되어 있으나, 앱이 최초 실행 시 `femto_preprocess`를 자동 실행하여
  `data/FEMTO_processed/*.csv`를 재생성합니다 (최초 1회 1~2분 소요).
  **주의**: 이때 VIF 분석으로 `selected_features.csv`(피처 목록)도 함께 재생성되는데, Cloud의
  데모 데이터가 로컬과 달라 VIF 계산 결과(선택되는 피처 개수·목록)가 로컬과 다르게 나올 수
  있습니다. `models/femto_scaler.pkl` 등 사전 학습된 아티팩트는 **고정된 9개 기본 센서 피처**로
  학습되어 있으므로, ML/DL 예측 코드가 이 동적 목록을 그대로 신뢰하면 `"X has N features,
  expecting 9"` 같은 개수 불일치 에러가 날 수 있습니다(2026-07-03 발생·수정 이력:
  `app/streamlit_femto.py`의 `BASE_ML_FEATURES` 고정 목록 사용으로 해결). 비슷한 에러가 다시
  보이면 이 문서의 이 항목을 먼저 확인하세요.
- `requirements.txt`/`app/requirements.txt`의 `pandas>=2.0`처럼 **상한 없는 버전 지정**이 많아,
  Cloud가 배포 시점에 로컬보다 최신 버전을 설치하면서 로컬-Cloud 간 동작 차이(예: pandas
  `Styler.applymap` 제거로 인한 `AttributeError`)가 발생할 수 있습니다. 재현이 안 되는 Cloud
  전용 에러가 나면 라이브러리 버전 차이부터 의심하세요.

## 로컬 실행 (배포 전 확인용)

```bash
cd C:\AISOURCE\Homework\LLM_FactoryAutomation
pip install -r requirements.txt -r app/requirements.txt
streamlit run app/streamlit_femto.py
```
