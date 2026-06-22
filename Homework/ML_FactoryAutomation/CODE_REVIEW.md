# 코드 리뷰 & 점검 결과 — PdM-Guard

## 1. 실행/검증 결과 요약
| 항목 | 결과 |
|---|---|
| 데이터 로드 | 10,000행 / 결측 0 / Target 0:9661, 1:339 (고장 3.4%) |
| 단위·통합 테스트 (pytest) | **6 passed** |
| 린트 (ruff) | **All checks passed** |
| 학습/평가 파이프라인 | 정상 (모델·리포트 생성) |
| Streamlit 앱 | 문법·의존성 점검 통과 |

## 2. 모델 성능 (목표 대비)
| 지표 | 목표 | LogReg(베이스라인) | **XGBoost** | 달성 |
|---|---|---|---|---|
| ROC-AUC | ≥0.95 | 0.907 | **0.971** | ✅ |
| Recall(고장) | ≥0.70 | 0.824 | **0.809** | ✅ |
| F1(고장) | ≥0.70 | 0.242 | **0.724** | ✅ |
| Accuracy | (참고) | 0.825 | 0.979 | - |

- **혼동행렬(XGBoost)**: 실제 고장 68건 중 55건 탐지(미탐 13), 오경보 29건.
- **[8] 특성 제거 실험**: 중요도 최하위 `Type_M, Type_H` 제거 후 F1 0.2424→0.2419로 거의 동일 → 해당 변수 기여도 미미함을 정량 확인.
- **[9] 모델 비교 서사**: 선형 모델(LogReg)은 계수 해석 용이(해석력), XGBoost는 비선형 학습으로 F1·정밀도 대폭 향상(성능) — 목적별 모델 선택 근거 확보.

## 3. 리뷰에서 수정한 사항
1. **타깃 누수 차단**: `Failure Type` 컬럼을 X에서 제외(Target을 직접 인코딩하므로 누수). `test_preprocess`로 검증.
2. **불균형 대응**: `stratify=y` 분할, LogReg `class_weight='balanced'`, XGBoost `scale_pos_weight=neg/pos`.
3. **그래프 한글 폰트 경고 제거**: 혼동행렬 라벨 영문화(Normal/Failure).
4. **deprecation 제거**: `datetime.utcnow()` → timezone-aware `_utcnow()`.
5. **스타일 정리**: ruff 기준 세미콜론 다중문 분리, 미사용 import 제거.

## 4. 잔여 개선 포인트 (선택)
- LogReg 정밀도 낮음(0.14) → 임계값 튜닝/SHAP 분석으로 보완 가능.
- `notebooks/eda.ipynb` 미생성(현재 `src/eda.py`로 대체) — 필요 시 노트북화.
- 배포 시 SQLite는 휘발성 → 영구 로깅이 필요하면 `DATABASE_URL`로 Supabase 연결.

## 5. 배포 점검 체크리스트 (Streamlit Cloud)
- [ ] GitHub 저장소 push (models/ 포함 또는 최초 1회 학습 스크립트 실행)
- [ ] share.streamlit.io → 엔트리포인트 `app/streamlit_app.py`
- [ ] Secrets에 `DATABASE_URL` 등록(외부 DB 사용 시)
- [ ] 배포 URL에서 입력→예측→이력 동작 확인
