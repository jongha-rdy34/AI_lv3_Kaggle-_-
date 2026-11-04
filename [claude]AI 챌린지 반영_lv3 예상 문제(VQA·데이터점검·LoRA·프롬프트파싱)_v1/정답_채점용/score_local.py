"""로컬 연습 채점기 (Kaggle 제출 대신). 실제 SSAFY 채점기가 아닙니다.

사용: python -I score_local.py <submission.csv>
지표(가정): 정답률(accuracy). 질문 유형별 정답률도 함께 보여 줍니다.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if len(sys.argv) < 2:
    sys.exit("사용: python score_local.py <submission.csv>")
path = Path(sys.argv[1])
sub = pd.read_csv(path)
truth = pd.read_csv(HERE / "test_truth.csv")
problems = []
if list(sub.columns) != ["id", "answer"]:
    problems.append(f"열 이름 {list(sub.columns)} (id, answer 필요)")
elif set(sub["id"]) != set(truth["id"]) or sub["id"].duplicated().any():
    problems.append("id 누락/중복/불일치")
elif not sub["answer"].isin(list("abcd")).all():
    problems.append("answer는 a, b, c, d 중 하나여야 함")
if problems:
    sys.exit(f"{path.name}: 형식 오류 → {'; '.join(problems)}")
m = truth.merge(sub, on="id", suffixes=("_true", "_pred"))
m["ok"] = m["answer_true"] == m["answer_pred"]
by_type = m.groupby("qtype")["ok"].mean().round(3).to_dict()
print(f"{path.name}: accuracy = {m['ok'].mean():.4f} | 유형별 {by_type} | 예측 분포 {sub['answer'].value_counts().to_dict()}")
