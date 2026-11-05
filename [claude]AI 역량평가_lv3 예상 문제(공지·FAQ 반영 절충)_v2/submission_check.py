"""제출 형식 점검 (torch 없이 동작). practice_support와 채점기가 함께 씁니다."""
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SUBMIT = ROOT / "제출물"
CLASSES = ["healthy", "rust", "scab", "multiple"]


def csv(folder, name):
    return pd.read_csv(DATA / folder / name)


GUIDE = {
    "q3": dict(file="q3_submission.csv", sample=("leaf", "sample_submission.csv"), col="label",
               allowed=set(CLASSES)),
    "q4": dict(file="q4_submission.csv", sample=("grade", "sample_submission.csv"), col="grade",
               allowed={0, 1, 2, 3, 4}),
}


def check_submission(kind, folder=None):
    """제출가이드.md의 규칙으로 제출 파일을 검사합니다. 문제 목록(빈 리스트면 통과)을 반환.

    Kaggle에 올렸을 때 형식 오류로 거절되는 경우를 미리 잡는 용도입니다.
    """
    folder = SUBMIT if folder is None else Path(folder)
    problems = []
    if kind == "q5":
        path = folder / "q5_submission.jsonl"
        if not path.exists():
            return [f"{path.name} 없음"]
        test_ids = csv("notes", "notes.csv").query("split == 'test'")["id"].tolist()
        seen = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"{n}번째 줄이 JSON이 아님")
                continue
            if set(row) != {"id", "record"}:
                problems.append(f"{n}번째 줄 키가 id, record가 아님: {sorted(row)}")
                continue
            if not isinstance(row["record"], dict):
                problems.append(f"{n}번째 줄 record가 객체가 아님")
            seen.append(row["id"])
        if seen != test_ids:
            problems.append(f"id 목록·순서가 notes.csv의 test와 다름 (제출 {len(seen)}줄, 기대 {len(test_ids)}줄)")
        return problems

    g = GUIDE[kind]
    path = folder / g["file"]
    if not path.exists():
        return [f"{g['file']} 없음"]
    sample = csv(*g["sample"])
    try:
        sub = pd.read_csv(path, dtype={"id": str})
    except Exception as exc:
        return [f"CSV로 읽을 수 없음: {exc}"]
    if list(sub.columns) != list(sample.columns):
        problems.append(f"열 이름·순서가 다름: {list(sub.columns)} (기대 {list(sample.columns)})")
        return problems
    if len(sub) != len(sample):
        problems.append(f"행 수가 다름: {len(sub)} (기대 {len(sample)})")
    if sub["id"].duplicated().any():
        problems.append("id 중복")
    if sub["id"].tolist() != sample["id"].astype(str).tolist():
        problems.append("id 순서가 sample_submission과 다름")
    if sub[g["col"]].isna().any():
        problems.append("빈 값 있음")
    if kind == "q4":
        col = sub[g["col"]]
        if not pd.api.types.is_integer_dtype(col):
            problems.append(f"grade가 정수가 아님 (dtype {col.dtype})")
        elif not set(col.unique()) <= g["allowed"]:
            problems.append(f"허용되지 않은 grade 값: {sorted(set(col.unique()) - g['allowed'])}")
    else:
        bad = set(sub[g["col"]].astype(str).unique()) - g["allowed"]
        if bad:
            problems.append(f"허용되지 않은 label 값: {sorted(bad)}")
    return problems


