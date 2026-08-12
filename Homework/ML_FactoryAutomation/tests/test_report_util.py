import os

from src.report_util import summarize_csv, save_summary


def test_summarize_csv_returns_na_for_header_only(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("machine_id,risk\n", encoding="utf-8")

    result = summarize_csv(str(p))

    assert result["count"] == 0
    assert result["grade"] == "N/A"


def test_summarize_csv_grades_high_when_mean_above_threshold(tmp_path):
    p = tmp_path / "risk.csv"
    p.write_text("machine_id,risk\nM-1,0.9\nM-2,0.8\n", encoding="utf-8")

    result = summarize_csv(str(p))

    assert result["count"] == 2
    assert result["grade"] == "HIGH"


def test_save_summary_writes_expected_file(tmp_path):
    out_dir = tmp_path / "reports"

    out_path = save_summary({"count": 2, "mean": 0.85, "grade": "HIGH"}, out_dir=str(out_dir))

    assert os.path.exists(out_path)
    assert "grade=HIGH" in open(out_path).read()
