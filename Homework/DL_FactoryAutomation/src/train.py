"""
train.py — DL 학습 파이프라인
  1) 데이터 로드 (train/test 80:20)
  2) 정규화
  3) CV=3 Stratified K-Fold 교차검증
  4) 최종 모델 전체 train으로 재학습
  5) 모델 저장
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    BATCH_SIZE, EPOCHS, MODEL_DIR, CV_FOLDS, PRED_THRESHOLD
)
from src.data_loader import load_train, load_test, normalize, kfold_splits
from src.dl_model import MODEL_REGISTRY


def run_training(model_name: str = "cnn_lstm", n_classes: int = 1, verbose: int = 1):
    """
    Args:
        model_name: 'cnn1d' | 'lstm' | 'cnn_lstm'
        n_classes: 1=이진분류, 6=다중분류(고장유형)
        verbose: 0=silent, 1=progress
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow 미설치: pip install tensorflow")
        return

    print(f"\n{'='*55}")
    print(f"모델: {model_name} | 클래스: {n_classes} | CV: {CV_FOLDS}-Fold")
    print(f"{'='*55}")

    # ── 1. 데이터 로드 ──────────────────────────────────────────
    print("데이터 로드 중...")
    X_train_raw, y_train_bin, y_train_multi = load_train()
    X_test_raw, y_test_bin, y_test_multi = load_test()

    y_train = y_train_bin if n_classes == 1 else y_train_multi
    y_test = y_test_bin if n_classes == 1 else y_test_multi

    print(f"  train: {X_train_raw.shape}, test: {X_test_raw.shape}")
    print(f"  train 고장률: {y_train_bin.mean()*100:.1f}%")

    # ── 2. 정규화 (train 기준) ──────────────────────────────────
    X_train, X_test, mean, std = normalize(X_train_raw, X_test_raw)

    # ── 3. Cross-Validation ────────────────────────────────────
    cv_results = []
    build_fn = MODEL_REGISTRY[model_name]

    for fold, Xf_tr, Xf_val, yf_tr, yf_val in kfold_splits(X_train, y_train):
        print(f"\n  [Fold {fold+1}/{CV_FOLDS}]")
        model = build_fn(n_classes=n_classes)
        cb = [
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, verbose=0),
        ]
        # 불균형 가중치
        if n_classes == 1:
            pos = yf_tr.sum(); neg = len(yf_tr) - pos
            cw = {0: 1.0, 1: neg / (pos + 1e-8)}
        else:
            cw = None

        model.fit(
            Xf_tr, yf_tr,
            validation_data=(Xf_val, yf_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            class_weight=cw,
            callbacks=cb, verbose=0,
        )
        val_metrics = model.evaluate(Xf_val, yf_val, verbose=0)
        names = model.metrics_names
        fold_result = dict(zip(names, val_metrics))
        cv_results.append(fold_result)
        print(f"    val_loss={fold_result['loss']:.4f}", end="")
        for k, v in fold_result.items():
            if k != "loss":
                print(f" | {k}={v:.4f}", end="")
        print()

    # ── 4. CV 결과 요약 ────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"CV {CV_FOLDS}-Fold 평균:")
    all_keys = list(cv_results[0].keys())
    cv_mean = {}
    for k in all_keys:
        vals = [r[k] for r in cv_results]
        cv_mean[k] = float(np.mean(vals))
        print(f"  {k}: {cv_mean[k]:.4f} ± {np.std(vals):.4f}")

    # ── 5. 전체 train으로 최종 모델 학습 ────────────────────────
    print("\n최종 모델 전체 학습 데이터로 재학습...")
    final_model = build_fn(n_classes=n_classes)
    if n_classes == 1:
        pos = y_train.sum(); neg = len(y_train) - pos
        cw_final = {0: 1.0, 1: neg / (pos + 1e-8)}
    else:
        cw_final = None

    final_model.fit(
        X_train, y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        class_weight=cw_final,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0,
    )

    # ── 6. 테스트셋 최종 평가 ──────────────────────────────────
    print("\n[Test Set 최종 평가]")
    test_metrics = final_model.evaluate(X_test, y_test, verbose=0)
    test_result = dict(zip(final_model.metrics_names, test_metrics))
    for k, v in test_result.items():
        print(f"  {k}: {v:.4f}")

    if n_classes == 1:
        from sklearn.metrics import classification_report, confusion_matrix
        y_pred_prob = final_model.predict(X_test, verbose=0).ravel()
        y_pred = (y_pred_prob >= PRED_THRESHOLD).astype(int)
        print(f"\n혼동행렬 (임계값={PRED_THRESHOLD}):")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=["Normal", "Failure"]))

    # ── 7. 모델 저장 ───────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "binary" if n_classes == 1 else "multiclass"
    model_path = MODEL_DIR / f"{model_name}_{suffix}.keras"
    final_model.save(str(model_path))
    print(f"\n모델 저장: {model_path}")

    # 정규화 파라미터 저장
    norm_path = MODEL_DIR / f"{model_name}_{suffix}_norm.npz"
    np.savez(norm_path, mean=mean, std=std)

    # CV 결과 저장
    result_path = MODEL_DIR / f"{model_name}_{suffix}_cv_results.json"
    result_path.write_text(json.dumps({"cv_mean": cv_mean, "test": test_result}, indent=2), encoding="utf-8")

    return final_model, cv_mean, test_result


if __name__ == "__main__":
    # 기본: CNN+LSTM 이진분류
    run_training(model_name="cnn_lstm", n_classes=1)
