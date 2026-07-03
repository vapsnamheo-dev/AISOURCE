# Streamlit Community Cloud 배포 가이드

FEMTO-ST 베어링 예지보전 프로젝트를 share.streamlit.io에 배포하는 절차입니다.

## 사전 준비 (완료됨)

- GitHub 리포지토리: https://github.com/vapsnamheo-dev/AISOURCE (main 브랜치)
- 릴리스: `llm-factory-automation-v0.5` (draft)
- 모델 아티팩트: `models/` 대부분 추적됨 (단, `femto_rf_rul.pkl` 72MB는 GitHub 용량 문제로 제외)

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

## Secrets 설정 방법 (2가지)

`ANTHROPIC_API_KEY` 같은 값은 아래 두 경로 중 상황에 맞는 쪽으로 넣습니다. 두 방법 모두 최종적으로 Python 코드에서는 `os.environ.get("ANTHROPIC_API_KEY", "")`로 동일하게 읽힙니다.

| 방법 | 대상 | 절차 |
| --- | --- | --- |
| **① Cloud 대시보드에서 직접 입력** (배포/운영용) | 실제 배포된 앱 | share.streamlit.io → 앱 선택 → 우측 하단 **⋮(점 3개)** → **Settings** → **Secrets** 탭 → 텍스트박스에 TOML 형식으로 붙여넣기(`ANTHROPIC_API_KEY = "sk-ant-..."`) → **Save**. 저장 즉시 앱이 자동 재시작되며 값이 바로 반영됨 |
| **② 로컬 `.streamlit/secrets.toml` 파일** (로컬 개발/테스트용) | 내 PC에서 `streamlit run`으로 실행할 때 | 프로젝트 루트에 `.streamlit/secrets.toml` 파일을 만들고 같은 내용을 저장. Streamlit이 로컬 실행 시 자동으로 읽어 `os.environ`처럼 노출됨 |

**차이**: ①은 Cloud에 배포된 앱 자체에 주입되는 값(팀/공유 배포용), ②는 로컬 실행 시에만 쓰이는 값(개인 PC별로 각자 보유).

**주의**: `.streamlit/secrets.toml`은 실제 키 값을 담고 있으므로 **절대 git에 커밋하면 안 됩니다** — 사용하려면 `.gitignore`에 `.streamlit/secrets.toml`을 추가한 뒤 로컬에만 두세요(현재 프로젝트 `.gitignore`에는 아직 이 항목이 없습니다).

## 알려진 제약사항

- `femto_rf_rul.pkl`(72MB, RUL 회귀 모델 중 하나)은 리포지토리에서 제외되어 있어
  해당 모델을 직접 참조하는 기능은 Cloud에서 동작하지 않습니다.
  (`femto_lstm_rul.keras`는 포함되어 있어 LSTM 기반 RUL 예측은 정상 동작)
- RAG Level 2(`femto_doc_rag.py`)의 `ask()` 함수는 로컬 Ollama 서버가 필요해
  Cloud 환경에서는 동작하지 않습니다. Claude API 기반 `retrieve_docs()` 경로(문서 청크 검색만)는
  Ollama 없이도 정상 동작합니다.
- `data/` 폴더는 Git에서 제외되어 있으나, 앱이 최초 실행 시 `femto_preprocess`를 자동 실행하여
  `data/FEMTO_processed/*.csv`를 재생성합니다 (최초 1회 1~2분 소요).

## 로컬 실행 (배포 전 확인용)

```bash
cd C:\AISOURCE\Homework\LLM_FactoryAutomation
pip install -r requirements.txt -r app/requirements.txt
streamlit run app/streamlit_femto.py
```
