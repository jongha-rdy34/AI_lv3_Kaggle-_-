"""[claude] Lv3 공지 후 모의 v1 데이터 생성기. 모든 데이터는 합성 연습 자료입니다.

실행: python -I make_data.py [leaf|grade|pretrain|notes ...]  (인자 없으면 전부)
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRUTH = ROOT / "정답_채점용"
sys.path.insert(0, str(ROOT))
CLASSES = ["healthy", "rust", "scab", "multiple"]
S = 48


# ======================================================================
# 잎 이미지 렌더러
# ======================================================================
def _ellipse(xx, yy, cx, cy, a, b, ang):
    c, s = np.cos(ang), np.sin(ang)
    u = (xx - cx) * c + (yy - cy) * s
    v = -(xx - cx) * s + (yy - cy) * c
    return (u / a) ** 2 + (v / b) ** 2, u, v


def render_leaf(rng, rust_n=0, scab_n=0, dark=1.0):
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    r0 = rng.uniform(110, 160)
    img = np.empty((S, S, 3))
    img[..., 0] = r0
    img[..., 1] = r0 * rng.uniform(0.6, 0.75)
    img[..., 2] = r0 * rng.uniform(0.3, 0.45)
    img += rng.normal(0, 10, (S, S, 3))
    cx, cy = rng.uniform(20, 28, 2)
    a, b, ang = rng.uniform(14, 20), rng.uniform(8, 12), rng.uniform(0, np.pi)
    d, u, v = _ellipse(xx, yy, cx, cy, a, b, ang)
    leaf = d <= 1
    col = np.array([rng.uniform(40, 90), rng.uniform(120, 180), rng.uniform(25, 70)])
    shade = 1 - 0.18 * np.sqrt(np.clip(d, 0, 1))
    img[leaf] = col * shade[leaf, None]
    vein = leaf & (np.abs(v) < 0.8)
    img[vein] *= 0.75
    pts = np.argwhere(d <= 0.75)
    for _ in range(rust_n):
        py, px = pts[rng.integers(len(pts))]
        rad = rng.uniform(1.2, 2.3)
        m = (xx - px) ** 2 + (yy - py) ** 2
        img[(m <= (rad + 0.8) ** 2) & leaf] = img[(m <= (rad + 0.8) ** 2) & leaf] * 0.5 + np.array([200, 190, 60]) * 0.5
        img[m <= rad ** 2] = [rng.uniform(190, 230), rng.uniform(100, 140), rng.uniform(20, 50)]
    for _ in range(scab_n):
        py, px = pts[rng.integers(len(pts))]
        blob = np.zeros((S, S), bool)
        for _ in range(rng.integers(2, 4)):
            oy, ox = rng.normal(0, 1.5, 2)
            rad = rng.uniform(2.0, 3.8)
            blob |= (xx - px - ox) ** 2 + (yy - py - oy) ** 2 <= rad ** 2
        img[blob & leaf] = [rng.uniform(50, 80), rng.uniform(50, 75), rng.uniform(25, 40)]
    img += rng.normal(0, 7, (S, S, 3))
    return np.clip(img * dark, 0, 255).astype(np.uint8)


def leaf_counts(rng, cls):
    rust = rng.integers(3, 11) if cls in ("rust", "multiple") else 0
    scab = rng.integers(2, 5) if cls in ("scab", "multiple") else 0
    return rust, scab


# ======================================================================
# Q1~Q3: 잎 질병 분류 데이터 (품질 문제 포함)
# ======================================================================
def make_leaf():
    rng = np.random.default_rng(1107)
    p = [0.30, 0.33, 0.32, 0.05]  # multiple이 소수 클래스(불균형)

    def build(n, dark_p):
        labs = rng.choice(CLASSES, n, p=p)
        imgs = np.stack([render_leaf(rng, *leaf_counts(rng, c),
                                     dark=rng.uniform(0.35, 0.6) if rng.random() < dark_p else 1.0) for c in labs])
        return imgs, labs

    base_x, base_y = build(1600, 0.08)   # train: 어두운 사진 8%
    test_x, test_y = build(600, 0.35)    # test : 어두운 사진 35% (분포 차이)

    # BGR로 저장된 사진 30장
    bgr = rng.choice(1600, 30, replace=False)
    base_x[bgr] = base_x[bgr][..., ::-1]
    # 깨진 사진 12장(검정 6, 회색 6)
    broken = np.concatenate([np.zeros((6, S, S, 3), np.uint8), np.full((6, S, S, 3), 128, np.uint8)])
    broken_y = rng.choice(CLASSES, 12, p=p)
    # 중복 20장: 15장은 같은 레이블, 5장은 다른 레이블(신뢰 불가)
    dup_src = rng.choice(np.setdiff1d(np.arange(1600), bgr), 20, replace=False)
    dup_x = base_x[dup_src].copy()
    dup_y = base_y[dup_src].copy()
    for i in range(15, 20):
        dup_y[i] = rng.choice([c for c in CLASSES if c != dup_y[i]])

    train_x = np.concatenate([base_x, broken, dup_x])
    train_y = np.concatenate([base_y, broken_y, dup_y])
    n_tr = len(train_x)
    ids = np.array([f"L{i:05d}" for i in rng.permutation(n_tr + 600 + 6) + 1])
    tr_ids, te_ids, orphan_ids = ids[:n_tr], ids[n_tr:n_tr + 600], ids[n_tr + 600:]
    perm = rng.permutation(n_tr)  # 이미지 파일 안의 순서 섞기
    train_x, train_y, tr_ids = train_x[perm], train_y[perm], tr_ids[perm]

    # 정상 이미지가 BGR 판별 규칙에 걸리지 않는지 생성 단계에서 확인
    is_bgr = np.zeros(n_tr, bool); is_bgr[np.isin(perm, bgr)] = True
    diff = train_x[..., 2].mean((1, 2)) - train_x[..., 0].mean((1, 2))
    assert (diff[is_bgr] > 10).all() and (diff[~is_bgr & (train_x.std((1, 2, 3)) > 1)] < 10).all(), "BGR 규칙 분리 실패"

    out = ROOT / "data/leaf"
    np.savez_compressed(out / "train_images.npz", ids=tr_ids.astype("U8"), images=train_x)
    np.savez_compressed(out / "test_images.npz", ids=te_ids.astype("U8"), images=test_x)
    # 레이블 CSV: 8장은 레이블 누락, 이미지 없는 id 6개 추가, 표기 흔들림 4%
    lab = pd.DataFrame({"id": tr_ids, "label": train_y})
    lab = lab.drop(index=rng.choice(n_tr, 8, replace=False))
    lab = pd.concat([lab, pd.DataFrame({"id": orphan_ids, "label": rng.choice(CLASSES, 6)})])
    messy = rng.random(len(lab)) < 0.04
    lab.loc[messy, "label"] = [rng.choice([s.upper(), " " + s, s + " ", s.capitalize()]) for s in lab.loc[messy, "label"]]
    lab = lab.sample(frac=1, random_state=3).reset_index(drop=True)
    lab.to_csv(out / "train_labels.csv", index=False)
    pd.DataFrame({"id": te_ids, "label": "healthy"}).to_csv(out / "sample_submission.csv", index=False)
    pd.DataFrame({"id": te_ids, "label": test_y}).to_csv(TRUTH / "q3_truth.csv", index=False)

    # 정답 기준 정제 결과(독립 진행용 준비 입력)
    ids_c, imgs_c, labs_c = clean_reference(tr_ids, train_x, lab)
    np.savez_compressed(out / "_ready_clean.npz", ids=ids_c.astype("U8"), images=imgs_c, labels=labs_c.astype("U10"))
    print("leaf train", train_x.shape, "test", test_x.shape, "clean", imgs_c.shape,
          pd.Series(labs_c).value_counts().to_dict())


def clean_reference(ids, imgs, lab):
    """Q1 정답과 같은 정제 규칙."""
    lab = lab.copy()
    lab["label"] = lab["label"].str.strip().str.lower()
    lab = lab[lab["id"].isin(ids)]                      # 이미지 없는 id 제거
    m = pd.Series(lab["label"].values, index=lab["id"])
    keep = np.isin(ids, m.index)                        # 레이블 없는 이미지 제거
    keep &= imgs.std(axis=(1, 2, 3)) >= 1.0            # 깨진 이미지 제거
    hashes = np.array([hashlib.md5(x.tobytes()).hexdigest() for x in imgs])
    df = pd.DataFrame({"i": np.arange(len(ids)), "h": hashes, "y": m.reindex(ids).values, "keep": keep})
    df = df[df.keep]
    bad_groups = df.groupby("h")["y"].nunique()
    df = df[~df.h.isin(bad_groups[bad_groups > 1].index)]   # 레이블이 엇갈린 중복은 모두 제거
    df = df.drop_duplicates("h", keep="first")               # 같은 레이블 중복은 하나만
    sel = df.i.to_numpy()
    out = imgs[sel].copy()
    bgr = out[..., 2].mean((1, 2)) - out[..., 0].mean((1, 2)) > 10
    out[bgr] = out[bgr][..., ::-1]                           # BGR → RGB
    return ids[sel], out, df.y.to_numpy()


# ======================================================================
# Q4: 녹병 심각도 등급(0~4)
# ======================================================================
GRADE_SPOTS = [(0, 0), (2, 4), (6, 9), (11, 15), (18, 26)]


def make_grade():
    rng = np.random.default_rng(2026)
    p = [0.40, 0.20, 0.20, 0.12, 0.08]

    def build(n):
        g = rng.choice(5, n, p=p)
        imgs = np.stack([render_leaf(rng, rust_n=int(rng.integers(lo, hi + 1)),
                                     dark=rng.uniform(0.5, 0.8) if rng.random() < 0.15 else 1.0)
                         for lo, hi in (GRADE_SPOTS[k] for k in g)])
        return imgs, g

    tx, ty = build(1000)
    sx, sy = build(400)
    ids = np.array([f"G{i:05d}" for i in rng.permutation(1400) + 1])
    out = ROOT / "data/grade"
    np.savez_compressed(out / "train_images.npz", ids=ids[:1000].astype("U8"), images=tx)
    np.savez_compressed(out / "test_images.npz", ids=ids[1000:].astype("U8"), images=sx)
    pd.DataFrame({"id": ids[:1000], "grade": ty}).to_csv(out / "train_labels.csv", index=False)
    pd.DataFrame({"id": ids[1000:], "grade": 0}).to_csv(out / "sample_submission.csv", index=False)
    pd.DataFrame({"id": ids[1000:], "grade": sy}).to_csv(TRUTH / "q4_truth.csv", index=False)
    print("grade", tx.shape, np.bincount(ty))


# ======================================================================
# '지정 모델' 사전학습: 무늬 6종 (질병·등급 데이터는 쓰지 않음)
# ======================================================================
def render_pattern(rng, k):
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    bg, fg = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
    while np.abs(bg - fg).sum() < 150:
        fg = rng.uniform(0, 255, 3)
    m = np.zeros((S, S), bool)
    if k == 0:    # 작은 점
        for _ in range(rng.integers(5, 20)):
            px, py = rng.uniform(3, 45, 2); m |= (xx - px) ** 2 + (yy - py) ** 2 <= rng.uniform(1, 2.5) ** 2
    elif k == 1:  # 줄무늬
        ang, w = rng.uniform(0, np.pi), rng.uniform(3, 7)
        m = np.sin((xx * np.cos(ang) + yy * np.sin(ang)) / w * np.pi) > 0
    elif k == 2:  # 체커
        w = rng.integers(4, 9); m = ((xx // w + yy // w) % 2) == 0
    elif k == 3:  # 고리
        px, py = rng.uniform(16, 32, 2); d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2); r = rng.uniform(8, 16)
        m = (d <= r) & (d >= r * 0.6)
    elif k == 4:  # 큰 얼룩
        for _ in range(rng.integers(2, 5)):
            px, py = rng.uniform(8, 40, 2); m |= (xx - px) ** 2 + (yy - py) ** 2 <= rng.uniform(4, 8) ** 2
    else:         # 무늬 없는 타원
        d, _, _ = _ellipse(xx, yy, *rng.uniform(18, 30, 2), rng.uniform(10, 20), rng.uniform(6, 12), rng.uniform(0, np.pi))
        m = d <= 1
    img = np.where(m[..., None], fg, bg) + rng.normal(0, 12, (S, S, 3))
    return np.clip(img * (rng.uniform(0.4, 1.0) if rng.random() < 0.3 else 1.0), 0, 255).astype(np.uint8)


def pretrain():
    import torch
    from torch import nn
    from practice_support import make_tinyresnet, PRETRAIN_MEAN, PRETRAIN_STD
    rng = np.random.default_rng(7)
    torch.manual_seed(0); torch.set_num_threads(4)
    y = rng.integers(0, 6, 6000)
    x = torch.tensor(np.stack([render_pattern(rng, k) for k in y])).permute(0, 3, 1, 2).float() / 255
    x = (x - torch.tensor(PRETRAIN_MEAN).view(1, 3, 1, 1)) / torch.tensor(PRETRAIN_STD).view(1, 3, 1, 1)
    y = torch.tensor(y)
    model = make_tinyresnet(6)
    opt = torch.optim.Adam(model.parameters(), 2e-3, weight_decay=1e-4)
    import time
    for ep in range(8):
        t0 = time.time(); model.train(); perm = torch.randperm(5000)
        for s in range(0, 5000, 64):
            b = perm[s:s + 64]
            opt.zero_grad(); nn.functional.cross_entropy(model(x[b]), y[b]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(x[5000:]).argmax(1) == y[5000:]).float().mean().item()
        print(f"  pretrain ep{ep} held-out acc {acc:.3f} ({time.time() - t0:.0f}s)")
    torch.save(model.state_dict(), ROOT / "assets/pretrained_tinyresnet.pt")


# ======================================================================
# Q5: 가상 진료 메모 → JSON (모의 LLM 응답 포함)
# ======================================================================
COMPLAINT_KO = {"fever": ["발열", "열이 남"], "cough": ["기침", "마른기침"], "headache": ["두통", "머리가 아픔"],
                "abdominal_pain": ["복통", "배가 아픔"], "rash": ["발진", "두드러기"]}
DRUGS = {"타이레놀": ("acetaminophen", [500, 650]), "애드빌": ("ibuprofen", [200, 400]),
         "이부프로펜": ("ibuprofen", [200, 400]), "아목시실린": ("amoxicillin", [250, 500]), "세티리진": ("cetirizine", [10])}
KOR_NUM = {1: "한", 2: "두", 3: "세"}


def make_note(rng):
    rec, seg = {}, []
    age = int(rng.integers(3, 86)) if rng.random() > 0.12 else None
    sex = rng.choice(["M", "F"]) if rng.random() > 0.1 else None
    rec["age"], rec["sex"] = age, (str(sex) if sex else None)
    if age is not None and sex is not None:
        word = {"M": ["남", "남성", "M"], "F": ["여", "여성", "F"]}[sex]
        if age < 13:
            word = ["남아"] if sex == "M" else ["여아"]
        w = str(rng.choice(word))
        seg.append(rng.choice([f"{age}세 {w}", f"{w}/{age}세", f"{w}, {age}세"]))
    elif sex is not None:
        seg.append(f"{'남' if sex == 'M' else '여'}, 나이 미상")
    elif age is not None:
        seg.append(f"{age}세, 성별 미기재")
    cc = str(rng.choice(list(COMPLAINT_KO)))
    rec["chief_complaint"] = cc
    cc_ko = str(rng.choice(COMPLAINT_KO[cc]))
    r = rng.random()
    if r < 0.15:
        rec["symptom_days"] = None
        seg.append(rng.choice([f"며칠 전부터 {cc_ko}", f"주 증상 {cc_ko}"]))
    else:
        dd = int(rng.choice([1, 2, 3, 4, 5, 7, 10, 14]))
        rec["symptom_days"] = dd
        ph = {7: "일주일째", 14: "2주 전부터"}.get(dd, rng.choice([f"{dd}일 전부터", f"{dd}일째"]))
        seg.append(f"{ph} {cc_ko}")
    r = rng.random()
    temp_f = False
    if r < 0.55:
        c = round(float(rng.uniform(36.0, 40.0)), 1); rec["temperature_c"] = c; seg.append(f"체온 {c}도")
    elif r < 0.78:
        f = round(float(rng.uniform(36.0, 40.0)) * 9 / 5 + 32, 1); rec["temperature_c"] = round((f - 32) * 5 / 9, 1)
        seg.append(f"체온 {f}F"); temp_f = True
    else:
        rec["temperature_c"] = None
        if rng.random() < 0.5:
            seg.append("미열 있음")
    meds = []
    for brand in rng.choice(list(DRUGS), int(rng.choice([0, 1, 1, 2])), replace=False):
        generic, doses = DRUGS[brand]
        dose = int(rng.choice(doses)) if rng.random() > 0.1 else None
        times = int(rng.integers(1, 4)) if rng.random() > 0.1 else None
        dtxt = "(용량 미기재)" if dose is None else str(rng.choice(
            [f"{dose}mg", f"{dose} mg", f"{dose}밀리그램"] + ([f"{dose / 1000:g}g"] if dose >= 250 and dose % 250 == 0 else [])))
        ttxt = "필요시 복용" if times is None else str(rng.choice([f"하루 {times}회", f"1일 {times}회", f"하루 {KOR_NUM[times]} 번"]))
        seg.append(f"{brand} {dtxt} {ttxt} 처방")
        meds.append({"name": generic, "dose_mg": dose, "times_per_day": times})
    rec["medications"] = meds
    r = rng.random()
    if r < 0.3:
        rec["allergy"] = True; seg.append(rng.choice(["페니실린 알레르기 있음", "약물 알레르기 있음"]))
    elif r < 0.65:
        rec["allergy"] = False; seg.append("알레르기 없음")
    else:
        rec["allergy"] = None
    r = rng.random()
    if r < 0.7:
        fu = int(rng.choice([3, 5, 7, 14])); rec["follow_up_days"] = fu
        seg.append({7: "1주 뒤 재방문", 14: "2주 후 경과 관찰"}.get(fu, f"{fu}일 후 재진"))
    else:
        rec["follow_up_days"] = None
        if rng.random() < 0.5:
            seg.append("호전 없으면 재방문")
    head, rest = seg[:1], seg[1:]
    rng.shuffle(rest)
    sep = str(rng.choice([". ", ", ", " / "]))
    return sep.join(head + rest) + ".", rec, temp_f


def mock_response(rng, rec, note, temp_f):
    """정답 레코드에 LLM이 흔히 내는 형식·표기 오류를 섞은 문자열. 실제 LLM 출력이 아님."""
    o = copy.deepcopy(rec)
    unk = lambda: str(rng.choice(["unknown", "N/A", "", "미상"]))
    if o["age"] is not None and rng.random() < 0.2:
        o["age"] = rng.choice([f"{o['age']}", f"{o['age']}세"])
    elif o["age"] is None and rng.random() < 0.4:
        o["age"] = unk()
    if o["sex"] is not None and rng.random() < 0.3:
        o["sex"] = str(rng.choice({"M": ["남성", "male", "남"], "F": ["여성", "female", "여"]}[o["sex"]]))
    if rng.random() < 0.25:
        o["chief_complaint"] = str(rng.choice([COMPLAINT_KO[o["chief_complaint"]][0], o["chief_complaint"].capitalize()]))
    if o["symptom_days"] is None:
        r = rng.random()
        if r < 0.4:
            o["symptom_days"] = str(rng.choice(["며칠", "unknown", ""]))
        elif r < 0.5:
            o["symptom_days"] = 3   # 근거 없는 추측(복구 불가 오류)
    elif rng.random() < 0.2:
        d = o["symptom_days"]
        o["symptom_days"] = {7: "1주", 14: "2주"}.get(d, f"{d}일")
    if temp_f and rng.random() < 0.5:
        o["temperature_c"] = round(rec["temperature_c"] * 9 / 5 + 32, 1)   # 화씨 그대로
    elif o["temperature_c"] is not None and rng.random() < 0.2:
        o["temperature_c"] = f"{o['temperature_c']}"
    elif o["temperature_c"] is None and rng.random() < 0.4:
        o["temperature_c"] = str(rng.choice(["N/A", "미열"]))
    for m in o["medications"]:
        if rng.random() < 0.3:
            m["name"] = next(k for k, v in DRUGS.items() if v[0] == m["name"] and k in note) if any(
                v[0] == m["name"] and k in note for k, v in DRUGS.items()) else m["name"]
        if m["dose_mg"] is not None and rng.random() < 0.25:
            m["dose_mg"] = str(rng.choice([f"{m['dose_mg']}mg", f"{m['dose_mg']} mg"] + (
                [f"{m['dose_mg'] / 1000:g}g"] if m["dose_mg"] >= 250 else [])))
        if m["times_per_day"] is not None and rng.random() < 0.2:
            m["times_per_day"] = f"{m['times_per_day']}회"
    if o["allergy"] is not None and rng.random() < 0.25:
        o["allergy"] = str(rng.choice(["있음", "yes"] if o["allergy"] else ["없음", "no"]))
    elif o["allergy"] is None and rng.random() < 0.3:
        o["allergy"] = "unknown"
    if o["follow_up_days"] is not None and rng.random() < 0.2:
        o["follow_up_days"] = {7: "1주", 14: "2주"}.get(o["follow_up_days"], f"{o['follow_up_days']}일")
    if rng.random() < 0.15:
        o["comment"] = "추가 확인 필요"
    if rng.random() < 0.08:
        o.pop("follow_up_days")
    r = rng.random()
    if r < 0.05:
        text = repr(o)                                   # 파이썬 dict 표기(작은따옴표, None)
    else:
        text = json.dumps(o, ensure_ascii=False, indent=2 if rng.random() < 0.5 else None)
        if r < 0.12:
            text = text[:-1].rstrip() + ",\n}"           # 끝 쉼표 → JSON 오류
        elif r < 0.15:
            text = text[: int(len(text) * 0.6)]          # 응답이 중간에 끊김(복구 불가)
    w = rng.random()
    if w < 0.3:
        text = "```json\n" + text + "\n```"
    elif w < 0.5:
        text = "추출 결과는 다음과 같습니다.\n" + text + "\n확인 부탁드립니다."
    return text


def make_notes():
    rng = np.random.default_rng(55)
    rows, gold, mock = [], {}, {}
    for i in range(70):
        note, rec, tf = make_note(rng)
        nid = f"N{i + 1:03d}"
        rows.append({"id": nid, "split": "dev" if i < 40 else "test", "note": note})
        gold[nid] = rec
        mock[nid] = mock_response(rng, rec, note, tf)
    df = pd.DataFrame(rows)
    out = ROOT / "data/notes"
    df.to_csv(out / "notes.csv", index=False)
    (out / "dev_labels.json").write_text(json.dumps({k: gold[k] for k in df[df.split == "dev"].id}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "mock_llm_outputs.json").write_text(json.dumps(mock, ensure_ascii=False, indent=1), encoding="utf-8")
    (TRUTH / "q5_truth.json").write_text(json.dumps({k: gold[k] for k in df[df.split == "test"].id}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("notes", df.split.value_counts().to_dict())
    print(df.note.head(4).tolist())


if __name__ == "__main__":
    jobs = sys.argv[1:] or ["leaf", "grade", "pretrain", "notes"]
    for j in jobs:
        {"leaf": make_leaf, "grade": make_grade, "pretrain": pretrain, "notes": make_notes}[j]()
