"""로컬 연습 채점기. 실제 SSAFY 채점·코드 리뷰가 아닙니다.

사용: python -I score_local.py [제출 폴더]   (기본: 모의문제 루트의 제출물/)
지표(가정): Q3 macro-F1, Q4 QWK, Q5 필드 정확도 + 스키마 위반 수
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SUBMIT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / "제출물"
KEYS = ["age", "sex", "chief_complaint", "symptom_days", "temperature_c", "medications", "allergy", "follow_up_days"]


def macro_f1(y, p):
    out = []
    for c in sorted(set(y)):
        tp, fp, fn = np.sum((p == c) & (y == c)), np.sum((p == c) & (y != c)), np.sum((p != c) & (y == c))
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out))


def qwk(a, b, n=5):
    O = np.zeros((n, n)); np.add.at(O, (a, b), 1)
    i, j = np.indices((n, n)); W = (i - j) ** 2 / (n - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return float(1 - (W * O).sum() / (W * E).sum())


def load(name, col, truth_name):
    path = SUBMIT / name
    if not path.exists():
        print(f"{name}: 없음"); return None
    sub, truth = pd.read_csv(path), pd.read_csv(HERE / truth_name)
    if list(sub.columns) != ["id", col] or set(sub.id) != set(truth.id) or sub[col].isna().any():
        print(f"{name}: 형식 오류"); return None
    m = truth.merge(sub, on="id", suffixes=("_t", "_p"))
    return m[f"{col}_t"].to_numpy(), m[f"{col}_p"].to_numpy()


r = load("q3_submission.csv", "label", "q3_truth.csv")
if r:
    y, p = r
    print(f"q3: macro-F1 = {macro_f1(y, p):.4f} | multiple 재현율 = {np.mean(p[y == 'multiple'] == 'multiple'):.3f}")
r = load("q4_submission.csv", "grade", "q4_truth.csv")
if r:
    print(f"q4: QWK = {qwk(r[0].astype(int), r[1].astype(int)):.4f}")

path = SUBMIT / "q5_submission.jsonl"
if path.exists():
    truth = json.loads((HERE / "q5_truth.json").read_text(encoding="utf-8"))
    sub = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line); sub[row["id"]] = row["record"]
    try:
        import jsonschema
        sys.path.insert(0, str(HERE.parent))
        schema_errors = None
    except ImportError:
        jsonschema = None
    med_key = lambda ms: sorted([(m["name"], None if m["dose_mg"] is None else float(m["dose_mg"]), m["times_per_day"]) for m in ms], key=str)
    hits, bad_keys = [], 0
    for nid, gold in truth.items():
        rec = sub.get(nid, {})
        if set(rec) != set(KEYS):
            bad_keys += 1
        for k in KEYS:
            if k == "medications":
                try:
                    hits.append(med_key(rec.get(k) or []) == med_key(gold[k]))
                except (KeyError, TypeError):
                    hits.append(False)
            else:
                v = rec.get(k)
                hits.append(v == gold[k] and type(v) is type(gold[k]) or (isinstance(v, float) and isinstance(gold[k], (int, float)) and v == gold[k]))
    print(f"q5: 필드 정확도 = {np.mean(hits):.4f} | 키 집합 불일치 레코드 = {bad_keys} | 제출 {len(sub)}/{len(truth)}")
else:
    print("q5_submission.jsonl: 없음")
