"""로컬 연습 채점기. 실제 SSAFY 채점기·리더보드가 아닙니다.

사용: python -I score_local.py [제출 폴더]   (기본: 모의문제 루트의 제출물/)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SUBMIT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / "제출물"

TASKS = {
    "q1": ("delivery_min", "RMSE(낮을수록 좋음)"),
    "q2": ("label", "macro-F1(높을수록 좋음)"),
    "q3": ("label", "accuracy(높을수록 좋음)"),
}


def macro_f1(y, p):
    scores = []
    for c in sorted(set(y)):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


for q, (col, metric) in TASKS.items():
    path = SUBMIT / f"{q}_submission.csv"
    if not path.exists():
        print(f"{q}: 제출 파일 없음 ({path.name})")
        continue
    sub = pd.read_csv(path)
    truth = pd.read_csv(HERE / f"{q}_truth.csv")
    problems = []
    if list(sub.columns) != ["id", col]:
        problems.append(f"열 {list(sub.columns)}")
    if sub["id"].duplicated().any() or set(sub["id"]) != set(truth["id"]):
        problems.append("id 누락/중복/불일치")
    if sub[col].isna().any() if col in sub else True:
        problems.append("결측값")
    if problems:
        print(f"{q}: 형식 오류 → {', '.join(problems)}")
        continue
    merged = truth.merge(sub, on="id", suffixes=("_true", "_pred"))
    y, p = merged[f"{col}_true"].to_numpy(), merged[f"{col}_pred"].to_numpy()
    if q == "q1":
        score = float(np.sqrt(np.mean((y - p.astype(float)) ** 2)))
    elif q == "q2":
        score = macro_f1(y, p)
    else:
        score = float(np.mean(y == p))
    print(f"{q}: {metric} = {score:.4f}")
