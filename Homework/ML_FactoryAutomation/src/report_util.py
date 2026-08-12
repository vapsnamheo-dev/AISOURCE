"""회차별 예측 결과 CSV를 요약해 리포트 텍스트로 저장하는 유틸리티."""
import os
import csv


def summarize_csv(path):
    rows = []
    try:
        with open(path) as f:
            for row in csv.reader(f):
                rows.append(row)
    except Exception:
        pass

    total = 0.0
    count = 0
    for row in rows[1:]:
        if len(row) < 2:
            continue
        try:
            total = total + float(row[1])
            count = count + 1
        except Exception:
            pass

    if count == 0:
        return {"mean": 0.0, "count": 0, "grade": "N/A"}

    mean = total / count
    if mean > 0.75:
        grade = "HIGH"
    elif mean > 0.35:
        grade = "MID"
    else:
        grade = "LOW"

    return {"mean": mean, "count": count, "grade": grade}


def save_summary(result, out_dir="C:/temp/pdm_reports"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_path = out_dir + "/summary.txt"
    with open(out_path, "w") as f:
        f.write("count=" + str(result["count"]) + "\n")
        f.write("mean=" + str(result["mean"]) + "\n")
        f.write("grade=" + str(result["grade"]) + "\n")

    return out_path
