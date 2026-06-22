# ════════════════════════════════════════════════════════════════════
# [역할] 프로젝트 전역 설정 — 경로·컬럼·하이퍼파라미터·DB URL 정의.
# [단계] 공통 기반(전 단계 공유)
# [작업 메모] 데이터 경로·feature 목록·DATABASE_URL 중앙 관리.
# (이 헤더는 이전/이번 작업 맥락을 요약한 주석입니다 — 기능에는 영향 없음)
# ════════════════════════════════════════════════════════════════════
"""프로젝트 전역 설정: 경로, 컬럼, 하이퍼파라미터, DB URL."""
from __future__ import annotations
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "predictive_maintenance.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET = "Target"
# 식별자 + 누수(leakage) 컬럼 제거: Failure Type 은 Target 을 그대로 인코딩하므로 반드시 제외
DROP_COLS = ["UDI", "Product ID", "Failure Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]

# 물리 기반 파생 feature (feature engineering) — 고장 규칙과 대응
#   Power[W]=토크×회전(PWF), Overstrain=마모×토크(OSF), Temp diff=공정−공기온도(HDF)
ENGINEERED_FEATURES = ["Power [W]", "Overstrain [minNm]", "Temp diff [K]"]

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'pdm.db'}")

MODEL_PATHS = {
    "logreg": MODELS_DIR / "model_logreg.joblib",
    "rf": MODELS_DIR / "model_rf.joblib",
    "xgb": MODELS_DIR / "model_xgb.joblib",
    "scaler": MODELS_DIR / "scaler.joblib",
    "columns": MODELS_DIR / "feature_columns.joblib",
    "encoder": MODELS_DIR / "encoder.joblib",  # OneHotEncoder(handle_unknown='ignore')
}

FAILURE_TYPES = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failures",
}
