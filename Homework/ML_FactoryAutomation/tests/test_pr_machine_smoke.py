# PR머신 동작 검증용 스모크 테스트 (안전한 트리비얼 변경 — sensitive/opus 경로 아님)
def test_pr_machine_smoke():
    assert True


def test_pr_machine_smoke_v2():
    # 배치 승격 라벨(awaiting-promotion) + P2 이슈 묶음 방식 재검증용
    value = 1 + 1
    assert value == 2
