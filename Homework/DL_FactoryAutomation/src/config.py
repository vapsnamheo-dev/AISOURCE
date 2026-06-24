"""DL_FactoryAutomation - 설정값 (시계열 1D-CNN/LSTM 기반 설비 고장 예측)"""
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── 피처 (ML 프로젝트와 동일) ───────────────────────────────────────────────
SENSOR_COLS = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
TYPE_COL = "type_encoded"  # L=0, M=1, H=2
DERIVED_COLS = ["power_w", "overstrain_minnm", "temp_diff_k"]
FEATURE_COLS = SENSOR_COLS + [TYPE_COL] + DERIVED_COLS   # 9 features

TARGET_COL = "failure"          # 이진: 0=정상, 1=고장
FAILURE_TYPE_COL = "failure_type"  # 다중: 0=정상 1=TWF 2=HDF 3=PWF 4=OSF 5=RNF

# ── 시계열 설정 ───────────────────────────────────────────────────────────────
SEQ_LEN = 50          # 시퀀스 길이 (타임스텝 수)
N_FEATURES = len(FEATURE_COLS)  # 9

# ── 학습 설정 ────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
CV_FOLDS = 3          # 교차검증 fold 수
TEST_RATIO = 0.20     # 80:20 분할

# ── 데이터 경로 ──────────────────────────────────────────────────────────────
DATA_DIR   = ROOT / "data"
RAW_DIR    = DATA_DIR / "raw"
TRAIN_DIR  = DATA_DIR / "train"
TEST_DIR   = DATA_DIR / "test"
DEMO_DIR   = DATA_DIR / "demo"
MODEL_DIR  = ROOT / "models"

# ── 합성 데이터 버전 관리 ─────────────────────────────────────────────────────
DATA_VERSION      = "v2"    # 현재 사용 중인 데이터 버전
NOISE_LEVEL       = 0.075   # 가우시안 노이즈 표준편차 비율 (0=없음, 0.075=7.5%)
SYNTH_TOTAL_RUNS  = 60000  # 전체 장비 가동 시퀀스 수 (train+test), v1=6000
DEMO_RUNS         = 500    # 데모 전용 시퀀스 (train/test에 미포함), v1=200
SYNTH_FAILURE_RATE = 0.35  # 고장 발생 비율

# ── 분류 임계값 ───────────────────────────────────────────────────────────────
PRED_THRESHOLD = 0.75
