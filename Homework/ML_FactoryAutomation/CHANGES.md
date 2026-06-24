# PdM-Guard 풀번들 — 변경 요약 (CHANGES)

기준 레포: `vapsnamheo-dev/AISOURCE` · 경로: `Homework/ML_FactoryAutomation/`
각 파일을 아래 위치에 그대로 덮어쓰면 됩니다.

## 소스 (7)
| 파일 | 위치 | 변경 |
|---|---|---|
| `app/streamlit_app.py` | `app/streamlit_app.py` | 대시보드 **중첩 서브탭 4개**(성능/파레토/EDA 히트맵·박스플롯/특성중요도) · 배치탭 **업로드 방어 가드** · 배치탭 **레포 샘플 바로 불러오기**(demo_1000.csv/전체) |
| `src/preprocess.py` | `src/preprocess.py` | `validate_and_clean()` 방어 유틸 · build_features 컬럼 정규화 · **[3]X/y·[4]분할·[5]표준화** 주석 |
| `src/train.py` | `src/train.py` | XGBoost 하이퍼파라미터 **주석** |
| `src/improve.py` | `src/improve.py` | 튜닝 탐색공간 **주석**(7개만 튜닝, L1/L2 제외 사유) |
| `.github/workflows/ci.yml` | `.github/workflows/ci.yml` | Python **3.12 → 3.10** |
| `Dockerfile` | `Dockerfile` | `python:3.12-slim → 3.10-slim` |
| `README.md` | `README.md` | 3.11 → **3.10** |

## 문서 (4)
| 파일 | 변경 |
|---|---|
| `산출물/…상세본_갱신.docx` | **3.5.1** 왜도·첨도/로그변환 검토 · **4.5.1** XGBoost 튜닝 항목·기준 · **8.3** 배포 설정 |
| `산출물/…수행내역서_갱신.docx` | 상세본과 동일하게 **3.5.1 · 4.5.1 · 8.3** 반영 |
| `산출물/…발표용_갱신.docx` | **「보강. 분포 진단·XGBoost 튜닝 근거」** 섹션 1개 추가(6장과 7장 사이) |
| `산출물/…발표자료_갱신.pptx` | **「ANALYSIS & TUNING」 요약 슬라이드 1장** 추가(결론 앞, 총 22장) |

## Python 버전 (3.10 통일)
로컬 3.10.6 · Streamlit Cloud 드롭다운 3.10(UI, 필요시 삭제 후 재배포) · CI/Dockerfile/README 3.10.
※ Streamlit Cloud는 `runtime.txt`를 무시하므로 생성하지 않음 — 드롭다운으로만 제어.

## ⚠️ 미포함: 통합 노트북 (홍길동_머신러닝프로젝트.ipynb / .py)
GitHub 레포·zip 어디에도 없어(로컬 전용) 본 번들에 포함하지 못했습니다.
- 옵션 A: 현재 파일을 업로드해 주시면 그대로 동봉
- 옵션 B: 현재 src(전처리·학습·평가)로 통합 노트북을 새로 생성해 동봉

## Git 반영 예시
```bash
cd /c/AISOURCE/Homework/ML_FactoryAutomation
# (번들을 이 폴더에 풀어 덮어쓴 뒤)
git add app/streamlit_app.py src/preprocess.py src/train.py src/improve.py \
        .github/workflows/ci.yml Dockerfile README.md 산출물/
git commit -m "feat: 대시보드·배치가드·샘플로드 + 보고서3종(3.5.1·4.5.1·8.3) + 발표용/PPT 요약 + Python 3.10"
git push origin main
```

## [추가] App URL 변경 (mlfactoryautomation.streamlit.app)
- 상세본·수행내역서 8.3 표의 App URL: `aisource-…` → `mlfactoryautomation.streamlit.app`
- `Streamlit_클라우드_배포절차.md`: 배포 URL 예시 갱신
- 소스(.py)에는 App URL 하드코딩이 없어 코드 변경 불필요(앱이 자기 URL을 참조하지 않음)

## [추가] 메타 피클 + 상세본 8.3
- `src/train.py`: 학습 시 `models/features_meta.joblib` 저장(numeric/categorical/engineered + 최종 피처순서·target·n_features)
- `app/streamlit_app.py`: features_meta 로드 → 일괄검증 탭에 "모델 학습 피처 메타" 표시(없으면 config 폴백)
- 상세본 8.3 신설: 성능·EDA 대시보드(4서브탭)·업로드 방어 가드·레포 샘플·피처 메타 / 기존 배포설정 → 8.4
