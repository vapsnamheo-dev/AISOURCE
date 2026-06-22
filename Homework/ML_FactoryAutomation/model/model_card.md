# 모델 카드 (Model Card) — 설비 고장 예측

- 생성: 2026-06-22T02:04:18
- 데이터: AI4I 2020 Predictive Maintenance (10,000 rows, 8:2 train/test)
- 타깃: 0=정상(normal), 1=고장(failure)
- Feature(11개): Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Power [W], Overstrain [minNm], Temp diff [K], Type_H, Type_L, Type_M
- 스케일러: StandardScaler

## 모델별 하이퍼파라미터·지표

### XGBoost (XGBClassifier) — `xgb_model.pkl`
- 지표: accuracy 0.979, precision 0.6548, recall 0.8088, f1 0.7237, roc_auc 0.9706
- 주요 하이퍼파라미터: objective=binary:logistic, colsample_bytree=0.9, enable_categorical=True, eval_metric=logloss, learning_rate=0.1, max_depth=4, missing=nan, n_estimators=300, n_jobs=2, random_state=42, scale_pos_weight=28.52029520295203, subsample=0.9

### RandomForest (RandomForestClassifier) — `rf_model.pkl`
- 지표: accuracy 0.983, precision 0.8269, recall 0.6324, f1 0.7167, roc_auc 0.9611
- 주요 하이퍼파라미터: bootstrap=True, ccp_alpha=0.0, class_weight=balanced_subsample, criterion=gini, max_features=sqrt, min_impurity_decrease=0.0, min_samples_leaf=2, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=2, oob_score=False

### LogisticRegression (LogisticRegression) — `logreg_model.pkl`
- 지표: accuracy 0.825, precision 0.1421, recall 0.8235, f1 0.2424, roc_auc 0.9069
- 주요 하이퍼파라미터: C=1.0, class_weight=balanced, dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=0.0, max_iter=1000, penalty=deprecated, random_state=42, solver=lbfgs, tol=0.0001, verbose=0
- 절편(표준화): -1.8981
- 가중치(표준화 공간):
    - Air temperature [K]: 0.2577
    - Process temperature [K]: -0.0922
    - Rotational speed [rpm]: 2.4726
    - Torque [Nm]: 7.8935
    - Tool wear [min]: 2.3403
    - Power [W]: -4.2176
    - Overstrain [minNm]: -1.2883
    - Temp diff [K]: -0.6511
    - Type_H: -0.0422
    - Type_L: 0.1115
    - Type_M: -0.0917