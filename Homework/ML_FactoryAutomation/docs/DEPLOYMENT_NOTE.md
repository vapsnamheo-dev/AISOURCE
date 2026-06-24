배포된 Streamlit 앱에서 `CSV 일괄 검증` 탭의 `입력 방법` 라디오 버튼이 보이지 않는다면, 현재 배포 버전이 로컬 코드와 일치하지 않을 수 있습니다.

로컬 코드의 해당 UI 위치:
- 앱 파일: `app/streamlit_app.py`
- 줄 위치: `st.radio("입력 방법", ["파일 업로드", "레포 샘플 불러오기"], horizontal=True, key="batch_input_mode")`

해결 방법:
1. 로컬 코드로 앱을 빌드/배포 다시 실행
2. 배포 스펙이 최신인지 확인
3. 필요시 `demo data/demo_1000.csv`를 포함한 최신 repo로 배포하도록 설정 변경
