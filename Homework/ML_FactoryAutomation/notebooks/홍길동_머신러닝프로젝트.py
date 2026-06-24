# %% [markdown]
# 홍길동 머신러닝 프로젝트 — 설비 고장 예측 (PdM-Guard)

# %% [markdown]
# # 설비 고장 예측 (PdM-Guard) — 머신러닝 통합 노트북
# **AI4I 2020 예지보전 데이터 · 분류(XGBoost) · 불균형 대응 · 물리 특성공학**
#
# 이 노트북은 `data/predictive_maintenance.csv` 한 파일로 데이터 로드 → EDA → 전처리 → 학습 → 평가 → 시각화까지
# **위에서 아래로 한 번에 실행**되도록 구성한 자립형 파이프라인입니다. (src 패키지 import 없이 단독 실행)

# %% [markdown]
# ## 0. 라이브러리 임포트
# %%
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    f1_score, recall_score, precision_score)
from xgboost import XGBClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.2
np.random.seed(RANDOM_STATE)

# %% [markdown]
# ## 1. 데이터 로드
# 식별자(UDI·Product ID)와 **누수 컬럼 Failure Type**(Target을 그대로 인코딩)을 제거한다.
# %%
# 데이터 경로(노트북 위치에 무관하게 탐색)
CANDS = [Path("data/predictive_maintenance.csv"),
         Path("../data/predictive_maintenance.csv"),
         Path("predictive_maintenance.csv")]
DATA_PATH = next((p for p in CANDS if p.exists()), CANDS[0])
df = pd.read_csv(DATA_PATH)
print("shape:", df.shape)
print("컬럼:", list(df.columns))
df.head()

# %%
TARGET = "Target"
DROP_COLS = ["UDI", "Product ID", "Failure Type"]   # 식별자 + 누수 컬럼 제거
NUMERIC_FEATURES = ["Air temperature [K]", "Process temperature [K]",
                    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
CATEGORICAL_FEATURES = ["Type"]

n = len(df); pos = int(df[TARGET].sum())
print(f"전체 {n:,}건 · 고장 {pos:,}건 ({pos/n*100:.2f}%) · 정상 {n-pos:,}건  → 심한 불균형")

# %% [markdown]
# ## 2. 탐색적 데이터 분석 (EDA)
# %%
# 2-1. 기술통계
df[NUMERIC_FEATURES].describe().T[["mean","std","min","max"]]

# %%
# 2-2. 제품 등급(Type)별 고장률
g = df.groupby("Type")[TARGET].mean().mul(100).round(2).sort_values(ascending=False)
print("등급별 고장률(%):"); print(g.to_string())

# %%
# 2-3. 수치형 상관 히트맵 (범주형·Target 제외)
corr = df[NUMERIC_FEATURES].corr()
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(NUMERIC_FEATURES))); ax.set_yticks(range(len(NUMERIC_FEATURES)))
ax.set_xticklabels([c.split(" [")[0] for c in NUMERIC_FEATURES], rotation=45, ha="right")
ax.set_yticklabels([c.split(" [")[0] for c in NUMERIC_FEATURES])
for i in range(len(NUMERIC_FEATURES)):
    for j in range(len(NUMERIC_FEATURES)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=9)
fig.colorbar(im); ax.set_title("Numeric Feature Correlation"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. 전처리 — 물리 특성공학 · 원-핫 · 분할 · 표준화
# - **물리 파생 3종**: Power(토크×회전, PWF), Overstrain(마모×토크, OSF), Temp diff(공정−공기온, HDF)
# - **Type 원-핫**(OneHotEncoder, handle_unknown='ignore') → 미정의 등급도 전부 0으로 안전 처리
# - **8:2 stratify 분할** + **StandardScaler**(거리·기울기 기반 모델 공정 비교)
# %%
def engineer(d):
    d = d.copy()
    d["Power [W]"] = d["Torque [Nm]"] * d["Rotational speed [rpm]"] * (2*np.pi/60)
    d["Overstrain [minNm]"] = d["Tool wear [min]"] * d["Torque [Nm]"]
    d["Temp diff [K]"] = d["Process temperature [K]"] - d["Air temperature [K]"]
    return d
ENGINEERED = ["Power [W]", "Overstrain [minNm]", "Temp diff [K]"]

work = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
work = engineer(work)

enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe = pd.DataFrame(enc.fit_transform(work[CATEGORICAL_FEATURES]).astype(int),
                   columns=enc.get_feature_names_out(CATEGORICAL_FEATURES), index=work.index)
y = work[TARGET].astype(int).reset_index(drop=True)
X = pd.concat([work[NUMERIC_FEATURES + ENGINEERED].reset_index(drop=True),
               ohe.reset_index(drop=True)], axis=1)
print("최종 피처(%d개):" % X.shape[1], list(X.columns))

# %%
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                          random_state=RANDOM_STATE, stratify=y)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)
print(f"학습 {len(y_tr):,} / 테스트 {len(y_te):,}  (테스트 고장 {int(y_te.sum())}건)")

# %% [markdown]
# ## 4. 모델 학습 — 베이스라인 LogReg + RandomForest + 최종 XGBoost
# 불균형 보정: LogReg·RF는 `class_weight`, XGBoost는 `scale_pos_weight = 음성/양성 비`.
# %%
spw = int((y_tr==0).sum()) / max(int((y_tr==1).sum()), 1)
logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=2)
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=2)
for m in (logreg, rf, xgb): m.fit(X_tr_s, y_tr)
print(f"scale_pos_weight ≈ {spw:.1f}  · 학습 완료(LogReg·RF·XGBoost)")

# %% [markdown]
# ## 5. 성능 평가 — 불균형 지표(Recall·F1·PR-AUC) 중심
# %%
def metrics(model, name):
    proba = model.predict_proba(X_te_s)[:,1]; pred = (proba>=0.5).astype(int)
    return {"모델":name, "Recall":recall_score(y_te,pred), "Precision":precision_score(y_te,pred),
            "F1":f1_score(y_te,pred), "PR-AUC":average_precision_score(y_te,proba),
            "ROC-AUC":roc_auc_score(y_te,proba)}
res = pd.DataFrame([metrics(logreg,"LogReg"), metrics(rf,"RandomForest"), metrics(xgb,"XGBoost")])
res.set_index("모델").round(3)

# %%
# XGBoost 분류 보고서 + 혼동행렬
proba = xgb.predict_proba(X_te_s)[:,1]; pred = (proba>=0.5).astype(int)
print(classification_report(y_te, pred, target_names=["정상(0)","고장(1)"], digits=3))
cm = confusion_matrix(y_te, pred)
print("혼동행렬 [TN FP / FN TP]:\n", cm)

# %% [markdown]
# ## 6. 시각화
# %% [markdown]
# ### 6-3. ROC Curve (3개 모델 비교)
# %%
plt.figure(figsize=(6,5))
for m,name in [(logreg,"LogReg"),(rf,"RandomForest"),(xgb,"XGBoost")]:
    p = m.predict_proba(X_te_s)[:,1]; fpr,tpr,_ = roc_curve(y_te,p)
    plt.plot(fpr,tpr,label=f"{name} (AUC={roc_auc_score(y_te,p):.3f})")
plt.plot([0,1],[0,1],"k--",lw=1); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve"); plt.legend(loc="lower right"); plt.tight_layout(); plt.show()

# %% [markdown]
# ### 6-4. Precision-Recall Curve (불균형에 더 적합)
# %%
plt.figure(figsize=(6,5))
for m,name in [(logreg,"LogReg"),(rf,"RandomForest"),(xgb,"XGBoost")]:
    p = m.predict_proba(X_te_s)[:,1]; pr,rc,_ = precision_recall_curve(y_te,p)
    plt.plot(rc,pr,label=f"{name} (AP={average_precision_score(y_te,p):.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve"); plt.legend(loc="lower left"); plt.tight_layout(); plt.show()

# %% [markdown]
# ### 6-5. 임계값(Threshold)별 Precision·Recall·F1 — 운영 임계값 선택
# %%
proba = xgb.predict_proba(X_te_s)[:,1]
ths = np.linspace(0.1,0.95,18); rows=[]
for t in ths:
    pr = (proba>=t).astype(int)
    rows.append((t, precision_score(y_te,pr,zero_division=0),
                 recall_score(y_te,pr), f1_score(y_te,pr,zero_division=0)))
T = pd.DataFrame(rows, columns=["threshold","Precision","Recall","F1"])
best = T.loc[T["F1"].idxmax()]
plt.figure(figsize=(7,4))
for c in ["Precision","Recall","F1"]: plt.plot(T["threshold"],T[c],marker=".",label=c)
plt.axvline(best["threshold"],color="gray",ls="--")
plt.title(f"Threshold sweep (F1 best @ {best['threshold']:.2f})")
plt.xlabel("threshold"); plt.legend(); plt.tight_layout(); plt.show()
print(f"F1 최대 임계값 ≈ {best['threshold']:.2f}  (P={best['Precision']:.3f}, R={best['Recall']:.3f}, F1={best['F1']:.3f})")

# %% [markdown]
# ## 7. 특성 중요도 (XGBoost)
# %%
imp = pd.Series(xgb.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(7,5)); plt.barh(imp.index, imp.values)
plt.title("XGBoost Feature Importance"); plt.tight_layout(); plt.show()
imp.sort_values(ascending=False).round(4)

# %% [markdown]
# ## 8. 결론
# - 불균형(고장 ≈3.4%) 분류를 **가중치 학습 + 물리 특성공학(8→11 feature)**으로 해결.
# - 최종 **XGBoost**: ROC-AUC ~0.97 / Recall ~0.81 / F1 ~0.72 (테스트셋 기준, 실행 환경에 따라 소수점 변동).
# - **재현율 우선**(놓친 고장 FN 비용↑) + 오경보 최소화 균형. 평가지표는 accuracy 대신 Recall·F1·PR-AUC 중심.
# - 향후: 고장유형 다중분류 · 임계값 경보 연계 · 시계열 롤링통계 · 물리규칙 데이터 증강.
