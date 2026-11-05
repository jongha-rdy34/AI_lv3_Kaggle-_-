"""로컬 연습 채점기 (Kaggle 리더보드 대신). 실제 SSAFY 채점·코드 리뷰가 아닙니다.

사용: python -I -X utf8 정답_채점용/score_local.py [제출 폴더] [--json]
      (기본 폴더: 모의문제 루트의 제출물/. --json이면 마지막 줄에 'SCORES_JSON {...}'를 출력)

FAQ 반영: 제출 형식이 틀리면 그 문항은 0점으로 처리합니다(형식 검사는 submission_check.check_submission).
성능 기준(이 모의의 가정): Q3 macro-F1 ≥ 0.88, Q4 QWK ≥ 0.90(심화 q4_ens도 같은 기준),
                          Q5 필드 정확도 ≥ 0.95 + 스키마 위반 0
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import submission_check as ps  # noqa: E402  (torch 없이 동작)

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
AS_JSON = "--json" in sys.argv
SUBMIT = Path(ARGS[0]).resolve() if ARGS else ROOT / "제출물"
THRESH = {"q3": 0.88, "q4": 0.90, "q4_ens": 0.90, "q5": 0.95}
KEYS = ["age", "sex", "chief_complaint", "symptom_days", "temperature_c", "medications", "allergy", "follow_up_days"]
COMPLAINTS = ["fever", "cough", "headache", "abdominal_pain", "rash"]
GENERICS = ["acetaminophen", "ibuprofen", "amoxicillin", "cetirizine"]


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


def schema_violation(rec):
    try:
        import jsonschema
    except ImportError:
        return set(rec) != set(KEYS)
    schema = {
        "type": "object", "required": KEYS, "additionalProperties": False,
        "properties": {
            "age": {"type": ["integer", "null"]}, "sex": {"enum": ["M", "F", None]},
            "chief_complaint": {"enum": COMPLAINTS}, "symptom_days": {"type": ["integer", "null"]},
            "temperature_c": {"type": ["number", "null"], "minimum": 30, "maximum": 45},
            "medications": {"type": "array", "items": {
                "type": "object", "required": ["name", "dose_mg", "times_per_day"], "additionalProperties": False,
                "properties": {"name": {"enum": GENERICS}, "dose_mg": {"type": ["number", "null"]},
                               "times_per_day": {"type": ["integer", "null"]}}}},
            "allergy": {"type": ["boolean", "null"]}, "follow_up_days": {"type": ["integer", "null"]},
        },
    }
    return bool(list(jsonschema.Draft202012Validator(schema).iter_errors(rec)))


# 결과: kind → {"score": float|None, "pass": bool, "note": str}. 제출하지 않은 선택 문항은 빠집니다.
RESULTS = {}


def record(kind, score, ok, note=""):
    RESULTS[kind] = {"score": None if score is None else round(float(score), 4), "pass": bool(ok), "note": note}
    shown = "0점(형식 오류)" if score is None else f"{score:.4f}"
    print(f"{kind}: {shown} (기준 {THRESH[kind]}) → {'PASS' if ok else 'FAIL'}{(' | ' + note) if note else ''}")


for kind, col, truth_name in [("q3", "label", "q3_truth.csv"), ("q4", "grade", "q4_truth.csv"),
                              ("q4_ens", "grade", "q4_truth.csv")]:
    if kind == "q4_ens" and not (SUBMIT / ps.GUIDE[kind]["file"]).exists():
        print("q4_ens: 제출 없음 (Q4 심화는 선택 문항)")
        continue
    problems = ps.check_submission(kind, SUBMIT)
    if problems:
        record(kind, None, False, f"형식 오류 {problems}")
        continue
    sub = pd.read_csv(SUBMIT / ps.GUIDE[kind]["file"], dtype={"id": str})
    m = pd.read_csv(HERE / truth_name, dtype={"id": str}).merge(sub, on="id", suffixes=("_t", "_p"))
    y, p = m[f"{col}_t"].to_numpy(), m[f"{col}_p"].to_numpy()
    if kind == "q3":
        score = macro_f1(y, p)
        record(kind, score, score >= THRESH[kind], f"multiple 재현율 {np.mean(p[y == 'multiple'] == 'multiple'):.3f}")
    else:
        score = qwk(y.astype(int), p.astype(int))
        record(kind, score, score >= THRESH[kind])

problems = ps.check_submission("q5", SUBMIT)
if problems:
    record("q5", None, False, f"형식 오류 {problems}")
else:
    truth = json.loads((HERE / "q5_truth.json").read_text(encoding="utf-8"))
    sub = {}
    for line in (SUBMIT / "q5_submission.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line); sub[row["id"]] = row["record"]
    med_key = lambda ms: sorted([(m["name"], None if m["dose_mg"] is None else float(m["dose_mg"]), m["times_per_day"]) for m in ms], key=str)
    hits, bad = [], 0
    for nid, gold in truth.items():
        rec = sub.get(nid, {})
        bad += schema_violation(rec)
        for k in KEYS:
            if k == "medications":
                try:
                    hits.append(med_key(rec.get(k) or []) == med_key(gold[k]))
                except (KeyError, TypeError):
                    hits.append(False)
            else:
                v, g = rec.get(k), gold[k]
                same_type = (type(v) is type(g)) or (isinstance(v, (int, float)) and isinstance(g, (int, float))
                                                     and not isinstance(v, bool) and not isinstance(g, bool))
                hits.append(v == g and same_type)
    acc = float(np.mean(hits))
    record("q5", acc, acc >= THRESH["q5"] and bad == 0, f"스키마 위반 레코드 {bad} (기준 0)")

print("\n요약:", " | ".join(f"{k} {'PASS' if v['pass'] else 'FAIL'}" for k, v in RESULTS.items()))
print("※ 성능 기준은 이 모의의 가정입니다. 서술·코드 리뷰 항목은 채점기준.md로 스스로 점검하세요.")
if AS_JSON:
    print("SCORES_JSON " + json.dumps({"results": RESULTS, "thresholds": THRESH}, ensure_ascii=False))
