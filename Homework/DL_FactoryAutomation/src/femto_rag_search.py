# -*- coding: utf-8 -*-
"""FEMTO-ST 베어링 유사 사례 검색 — Level 1 RAG (FAISS + 12-dim 특성 벡터).

실행:
    python -m src.femto_rag_search          # 인덱스 빌드
    python -m src.femto_rag_search --demo   # 빌드 + 샘플 쿼리

출력:
    models/femto_faiss.index     (FAISS 코사인 유사도 인덱스)
    models/femto_faiss_meta.pkl  (벡터별 메타데이터)
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT          = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "FEMTO_processed"
MODEL_DIR     = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH  = MODEL_DIR / "femto_faiss.index"
META_PATH   = MODEL_DIR / "femto_faiss_meta.pkl"
SCALER_PATH = MODEL_DIR / "femto_scaler.pkl"
SEL_PATH    = PROCESSED_DIR / "selected_features.csv"
FEAT_PATH   = PROCESSED_DIR / "femto_features.csv"

FALLBACK_FEATURES = [
    "h_rms", "h_kurt", "h_skew", "h_crest",
    "v_rms", "v_kurt", "v_skew", "v_crest",
    "temp_mean", "energy", "health_idx", "rms_ratio",
]


def _load_feature_list() -> list[str]:
    if SEL_PATH.exists():
        return pd.read_csv(SEL_PATH)["feature"].tolist()
    return FALLBACK_FEATURES


def _fill_nan(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in features:
        if c in df.columns:
            med = df[c].median()
            df[c] = df[c].fillna(med if np.isfinite(med) else 0.0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 인덱스 빌드
# ─────────────────────────────────────────────────────────────────────────────

def build_index(verbose: bool = True) -> None:
    """femto_features.csv 전체를 벡터화하여 FAISS 코사인 인덱스를 구축·저장한다."""
    import faiss

    if not FEAT_PATH.exists():
        raise FileNotFoundError(f"피처 없음: {FEAT_PATH} — femto_preprocess 먼저 실행")

    features = _load_feature_list()
    df = pd.read_csv(FEAT_PATH)
    df = _fill_nan(df, features)

    X = df[features].values.astype("float32")

    # 스케일러 적용
    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X).astype("float32")
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X).astype("float32")
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

    # L2 정규화 → 내적 = 코사인 유사도
    faiss.normalize_L2(X_scaled)

    dim   = X_scaled.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X_scaled)
    faiss.write_index(index, str(INDEX_PATH))

    # 메타데이터 저장 (인덱스 i → 베어링 이력 정보)
    meta_cols = ["minute", "bearing", "split", "label", "rul", "rul_pct"]
    available = [c for c in meta_cols if c in df.columns]
    meta = df[available].reset_index(drop=True).to_dict("records")
    for i, rec in enumerate(meta):
        rec["features"] = {f: float(df[features].iloc[i][f]) for f in features}

    with open(META_PATH, "wb") as f:
        pickle.dump({"meta": meta, "features": features}, f)

    if verbose:
        print(f"[RAG] 인덱스 빌드 완료  벡터={index.ntotal:,}개  dim={dim}")
        print(f"[RAG] 베어링: {sorted(df['bearing'].unique().tolist())}")
        print(f"[RAG] 저장 → {INDEX_PATH.name} / {META_PATH.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 인덱스 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_index():
    """저장된 FAISS 인덱스 + 메타 + 스케일러를 반환한다."""
    import faiss

    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "인덱스 없음. 먼저 실행하세요:\n  python -m src.femto_rag_search"
        )

    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        store = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return index, store["meta"], store["features"], scaler


# ─────────────────────────────────────────────────────────────────────────────
# 유사 사례 검색
# ─────────────────────────────────────────────────────────────────────────────

def search(
    query: dict,
    index=None,
    meta: list | None = None,
    features: list | None = None,
    scaler=None,
    k: int = 5,
    exclude_same_bearing: bool = False,
) -> list[dict]:
    """
    현재 측정값(dict)과 유사한 과거 사례 Top-k 반환.

    Parameters
    ----------
    query                : {"h_rms": 1.2, "h_kurt": 3.1, ...}
    k                    : 반환할 사례 수
    exclude_same_bearing : True이면 쿼리와 동일 베어링 제외

    Returns
    -------
    [{"rank", "similarity", "bearing", "minute", "rul", "label", "features"}, ...]
    """
    import faiss

    if index is None:
        index, meta, features, scaler = load_index()

    vec = np.array([[query.get(f, 0.0) for f in features]], dtype="float32")
    vec_sc = scaler.transform(vec).astype("float32")
    faiss.normalize_L2(vec_sc)

    search_k = k * 10 if exclude_same_bearing else k
    D, I = index.search(vec_sc, search_k)

    results = []
    qbearing = query.get("bearing")

    for sim, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        rec = meta[idx]
        if exclude_same_bearing and rec.get("bearing") == qbearing:
            continue
        results.append({
            "rank":       len(results) + 1,
            "similarity": round(float(sim) * 100, 1),
            "bearing":    rec.get("bearing", "?"),
            "minute":     int(rec.get("minute", 0)),
            "rul":        float(rec["rul"]) if rec.get("rul") is not None else None,
            "rul_pct":    float(rec["rul_pct"]) if rec.get("rul_pct") is not None else None,
            "label":      int(rec.get("label", 0)),
            "split":      rec.get("split", "?"),
            "features":   rec.get("features", {}),
        })
        if len(results) >= k:
            break

    return results


def search_and_estimate_rul(
    query: dict,
    k: int = 5,
    exclude_same_bearing: bool = False,
) -> dict:
    """유사 사례 검색 + 유사도 가중 평균 RUL 추정."""
    index, meta, features, scaler = load_index()
    cases = search(query, index=index, meta=meta, features=features,
                   scaler=scaler, k=k,
                   exclude_same_bearing=exclude_same_bearing)

    rul_sims = [(r["rul"], r["similarity"]) for r in cases
                if r["rul"] is not None and r["similarity"] > 0]

    estimated_rul = None
    if rul_sims:
        total_sim = sum(s for _, s in rul_sims)
        estimated_rul = round(sum(r * s for r, s in rul_sims) / total_sim, 1)

    sims = [r["similarity"] for r in cases]
    return {
        "similar_cases":  cases,
        "estimated_rul":  estimated_rul,
        "k":              len(cases),
        "avg_similarity": round(float(np.mean(sims)), 1) if sims else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI 데모
# ─────────────────────────────────────────────────────────────────────────────

def _demo(k: int = 5) -> None:
    df       = pd.read_csv(FEAT_PATH)
    features = _load_feature_list()
    df       = _fill_nan(df, features)

    # OOS 베어링 열화 구간 중간 시점 샘플
    oos = df[df["bearing"].isin(["Bearing1_7", "Bearing3_3"]) & (df["label"] == 1)]
    if oos.empty:
        oos = df[df["label"] == 1]
    sample = oos.iloc[len(oos) // 2]

    query             = {f: float(sample[f]) for f in features}
    query["bearing"]  = sample["bearing"]
    actual_rul        = float(sample["rul"])

    print(f"\n{'='*60}")
    print(f"[쿼리 베어링] {sample['bearing']}  t={int(sample['minute'])}분")
    print(f"  실제 RUL={actual_rul:.0f}분  상태={'열화' if sample['label'] else '정상'}")
    print(f"  h_rms={query['h_rms']:.4f}  v_rms={query['v_rms']:.4f}  "
          f"temp={query.get('temp_mean', float('nan')):.1f}°C")

    result = search_and_estimate_rul(query, k=k, exclude_same_bearing=True)

    print(f"\n[유사 사례 Top-{result['k']}]  평균 유사도 {result['avg_similarity']}%")
    print(f"{'순위':<4} {'유사도':>7}  {'베어링':<14} {'시각(분)':>8}  "
          f"{'RUL(분)':>8}  {'상태':<6}")
    print("-" * 60)
    for r in result["similar_cases"]:
        st = "열화" if r["label"] == 1 else "정상"
        print(f"{r['rank']:<4} {r['similarity']:>6.1f}%  "
              f"{r['bearing']:<14} {r['minute']:>8}  "
              f"{r['rul']:>8.0f}  {st}")

    est = result["estimated_rul"]
    if est is not None:
        err = abs(est - actual_rul)
        print(f"\n[RAG 추정 RUL] {est:.0f}분  (실제 {actual_rul:.0f}분  오차 {err:.0f}분)")


def run() -> None:
    parser = argparse.ArgumentParser(description="FEMTO RAG Level1 — FAISS 유사 사례 검색")
    parser.add_argument("--demo", action="store_true", help="빌드 후 샘플 쿼리 실행")
    parser.add_argument("--k",    type=int, default=5,  help="검색 결과 수 (기본 5)")
    args = parser.parse_args()

    print("=" * 60)
    print("FEMTO-ST RAG 유사 사례 검색  (Level 1 — FAISS + 12-dim)")
    print("=" * 60)
    build_index(verbose=True)
    if args.demo:
        _demo(k=args.k)


if __name__ == "__main__":
    run()
