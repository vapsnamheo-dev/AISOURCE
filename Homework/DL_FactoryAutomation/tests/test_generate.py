"""tests/test_generate.py — 데이터 생성 및 로더 단위 테스트."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_generate_run_shape():
    from src.generate_ts_data import generate_run
    from src.config import SEQ_LEN, FEATURE_COLS
    df = generate_run(equip_type="M", force_failure=False)
    assert df.shape[0] == SEQ_LEN
    for col in FEATURE_COLS:
        assert col in df.columns, f"피처 누락: {col}"


def test_generate_failure_run():
    from src.generate_ts_data import generate_run
    df = generate_run(equip_type="L", force_failure=True)
    assert "failure" in df.columns
    assert "failure_type" in df.columns


def test_generate_dataset_count():
    from src.generate_ts_data import generate_dataset
    runs = generate_dataset(20)
    assert len(runs) == 20


def test_save_and_load(tmp_path):
    from src.generate_ts_data import generate_dataset, save_sequences
    from src.config import FEATURE_COLS, SEQ_LEN
    import numpy as np

    runs = generate_dataset(10)
    save_sequences(runs, tmp_path)

    files = list(tmp_path.glob("*.csv"))
    assert len(files) == 10

    df = pd.read_csv(files[0])
    assert df.shape[0] == SEQ_LEN
    for col in FEATURE_COLS:
        assert col in df.columns


def test_normalize():
    from src.data_loader import normalize
    X_train = np.random.randn(100, 50, 9).astype(np.float32)
    X_test  = np.random.randn(20, 50, 9).astype(np.float32)
    Xtr, Xte, mean, std = normalize(X_train, X_test)
    assert Xtr.shape == X_train.shape
    assert Xte.shape == X_test.shape
