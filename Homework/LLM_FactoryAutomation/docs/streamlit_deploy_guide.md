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
