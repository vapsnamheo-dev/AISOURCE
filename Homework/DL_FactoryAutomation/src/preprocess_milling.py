"""
Source 3: NASA UC Berkeley Milling 데이터 전처리
입력: data/raw/milling_data/mill.mat
출력: data/milling_train.npz, data/milling_test.npz, data/milling_norm.npz
6채널 원시신호 → 슬라이딩 윈도우 → 정규화 → 저장
"""
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

MAT_PATH = Path("c:/AISOURCE/Homework/DL_FactoryAutomation/data/raw/milling_data/mill.mat")
OUT_DIR   = Path("c:/AISOURCE/Homework/DL_FactoryAutomation/data")
CHANNELS  = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
WIN_SIZE  = 512    # 약 51ms (10kHz 기준)
STEP      = 128    # 75% 오버랩
RANDOM    = 2026

mat = sio.loadmat(str(MAT_PATH))["mill"]
X_segs, y_segs = [], []

for i in range(mat.shape[1]):
    r = mat[0][i]
    vb = float(r["VB"].flat[0])
    if np.isnan(vb):
        continue

    # 6채널 → (9000, 6)
    sig = np.column_stack([r[ch].flatten() for ch in CHANNELS]).astype("float32")

    # 슬라이딩 윈도우
    n = sig.shape[0]
    starts = range(0, n - WIN_SIZE + 1, STEP)
    for s in starts:
        X_segs.append(sig[s:s+WIN_SIZE])
        # 이진 레이블: 0=정상(VB<0.2), 1=마모(VB>=0.2)
        y_segs.append(0 if vb < 0.2 else 1)

X = np.stack(X_segs).astype("float32")   # (N, 512, 6)
y = np.array(y_segs, dtype="int32")

print(f"총 세그먼트: {len(X)}")
print(f"  - 정상(0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
print(f"  - 마모(1): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

# 80:20 분할 (cut 단위 누수 없게 stratify)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM, stratify=y
)

# per-channel 표준화 (train 기준)
mean = X_tr.mean(axis=(0, 1), keepdims=True)
std  = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
X_tr_n = (X_tr - mean) / std
X_te_n = (X_te - mean) / std

np.savez_compressed(str(OUT_DIR / "milling_train.npz"), X=X_tr_n, y=y_tr)
np.savez_compressed(str(OUT_DIR / "milling_test.npz"),  X=X_te_n, y=y_te)
np.savez_compressed(str(OUT_DIR / "milling_norm.npz"),  mean=mean, std=std)

print(f"Train: {X_tr_n.shape} | Test: {X_te_n.shape}")
print("저장 완료: data/milling_train.npz, milling_test.npz, milling_norm.npz")
