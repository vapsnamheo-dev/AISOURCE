from src import data_loader, preprocess, config


def test_no_leakage_columns():
    df = data_loader.load_data()
    X, y = preprocess.build_features(df)
    for col in config.DROP_COLS + [config.TARGET]:
        assert col not in X.columns
    assert "Failure Type" not in X.columns  # 누수 방지 핵심


def test_shapes_and_stratify():
    df = data_loader.load_data()
    X, y = preprocess.build_features(df)
    Xtr, Xte, ytr, yte, scaler, cols = preprocess.split_and_scale(X, y)
    assert Xtr.shape[0] == len(ytr)
    assert Xte.shape[0] == len(yte)
    assert Xtr.shape[1] == len(cols)
    # stratify: train/test 고장비율 유사
    assert abs(ytr.mean() - yte.mean()) < 0.01
