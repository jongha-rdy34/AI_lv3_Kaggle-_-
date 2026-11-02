"""[claude] Lv3 모의 v1 데이터 생성기. 모든 데이터는 합성 연습 자료입니다.

실행: python -I make_data.py   (모의문제 루트는 이 파일 기준 두 단계 위)
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRUTH = ROOT / "정답_채점용"
sys.path.insert(0, str(ROOT))


# ======================================================================
# Q1 회귀: 배달 소요시간(분)
# ======================================================================
def make_q1():
    rng = np.random.default_rng(20261107)
    n_train, n_test = 2400, 800
    n = n_train + n_test
    regions = np.array(["강남", "마포", "송파", "노원", "관악"])
    region_eff = {"강남": 4.0, "마포": 1.0, "송파": 2.0, "노원": -1.0, "관악": 0.0, "은평": 2.5}
    region = rng.choice(regions, n, p=[0.26, 0.2, 0.22, 0.16, 0.16]).astype(object)
    # test에만 등장하는 지역: 학습 때 본 적 없는 범주 처리 연습
    test_new = rng.random(n_test) < 0.04
    region[n_train:][test_new] = "은평"

    weather = rng.choice(["맑음", "흐림", "비", "눈"], n, p=[0.55, 0.25, 0.15, 0.05]).astype(object)
    weather_eff = {"맑음": 0.0, "흐림": 1.0, "비": 5.0, "눈": 12.0}

    day = rng.integers(0, 61, n)
    hour_pool = np.r_[[11, 12, 13, 17, 18, 19, 20], np.arange(9, 23)]
    hour = rng.choice(hour_pool, n)
    minute = rng.integers(0, 60, n)
    ts = pd.Timestamp("2026-09-01") + pd.to_timedelta(day, "D") + pd.to_timedelta(hour, "h") + pd.to_timedelta(minute, "min")
    order_time = ts.strftime("%Y-%m-%d %H:%M")
    weekend = (ts.dayofweek >= 5).astype(float)
    peak = np.isin(hour, [11, 12, 13, 17, 18, 19, 20]).astype(float)

    dist = np.clip(rng.gamma(2.2, 1.4, n), 0.3, 15).round(2)
    items = rng.poisson(1.6, n) + 1
    prep = np.clip(rng.normal(14, 5, n), 3, 40).round(1)
    rating = np.clip(rng.normal(4.5, 0.3, n), 3.0, 5.0).round(1)

    w_eff = np.array([weather_eff[w] for w in weather])
    r_eff = np.array([region_eff[r] for r in region])
    bad = np.isin(weather, ["비", "눈"]).astype(float)
    y = (8 + 3.0 * dist + 0.15 * dist ** 2 + 0.75 * prep + 1.2 * items + 7 * peak
         + w_eff + r_eff + 4 * peak * bad - 4 * (rating - 4.5) + 2 * weekend
         + rng.normal(0, 3.5, n))
    accident = rng.random(n) < 0.008
    y[accident] += rng.uniform(40, 90, accident.sum())
    y = np.clip(y, 5, None).round(1)

    df = pd.DataFrame({
        "id": [f"D{i:05d}" for i in rng.permutation(n) + 1],
        "order_time": order_time, "region": region, "weather": weather,
        "distance_km": dist, "n_items": items, "prep_min": prep, "courier_rating": rating,
    })
    # 목표값을 만든 뒤 결측을 넣는다: 대치가 필요하지만 정보는 사라짐
    for col, p in [("weather", 0.06), ("distance_km", 0.05), ("prep_min", 0.08), ("courier_rating", 0.04)]:
        df.loc[rng.random(n) < p, col] = np.nan
    df["delivery_min"] = y

    train, test = df.iloc[:n_train].copy(), df.iloc[n_train:].copy()
    out = ROOT / "data/q1"
    train.to_csv(out / "train.csv", index=False)
    test.drop(columns="delivery_min").to_csv(out / "test.csv", index=False)
    # sample_submission은 id 오름차순: test.csv 행 순서와 다르다(위치로 맞추면 오답)
    sample = pd.DataFrame({"id": sorted(test["id"]), "delivery_min": 30.0})
    sample.to_csv(out / "sample_submission.csv", index=False)
    test[["id", "delivery_min"]].to_csv(TRUTH / "q1_truth.csv", index=False)
    print("Q1", train.shape, test.shape, "unseen region rows:", int(test_new.sum()))


# ======================================================================
# Q2 텍스트 분류 + 모의 임베딩
# ======================================================================
# 각 개념: (train에서 쓰는 표현, test에서만 쓰는 표현)
LEXICON = {
    "delivery": {
        "SHIP": (["배송이 아직 시작되지 않았어요", "택배가 출발을 안 했어요", "배달이 안 와요"],
                 ["물류 출고가 안 됐어요", "운송이 시작되지 않았네요"]),
        "ARRIVE": (["상품을 아직 못 받았어요", "도착 예정일이 지났어요", "수령을 못 했어요"],
                   ["감감무소식이에요", "물건이 손에 안 들어왔어요"]),
        "DELAY": (["배송이 너무 늦어요", "며칠째 배송 준비 중이에요"],
                  ["하세월이네요", "한참을 기다려도 소식이 없어요"]),
        "TRACK": (["송장 조회가 안 돼요", "운송장 번호가 안 떠요"],
                  ["트래킹이 멈춰 있어요", "위치 추적이 안 돼요"]),
    },
    "refund": {
        "REFUND": (["환불해 주세요", "환불 처리가 언제 되나요", "돈을 돌려받고 싶어요"],
                   ["반환금은 언제 들어오나요", "대금 회수를 원해요"]),
        "RETURN": (["반품 신청하고 싶어요", "반송 접수 부탁드려요", "회수 신청을 했어요"],
                   ["물건을 되돌려 보내고 싶어요", "도로 보내려고 해요"]),
        "CANCEL": (["주문을 취소하고 싶어요", "구매 취소가 되나요"],
                   ["구매를 철회할게요", "청약 철회 가능할까요"]),
    },
    "payment": {
        "PAY": (["결제가 안 돼요", "카드 결제가 실패했어요", "계산이 이상해요"],
                ["지불이 막혔어요", "대금 납부가 안 돼요"]),
        "DOUBLE": (["두 번 결제됐어요", "중복 결제가 됐어요", "돈이 이중으로 빠졌어요"],
                   ["같은 금액이 2회 청구됐어요", "승인이 겹쳐서 났어요"]),
        "COUPON": (["쿠폰이 적용되지 않아요", "할인 코드가 안 먹혀요", "적립금이 안 쓰여요"],
                   ["프로모션 코드가 무효래요", "마일리지 차감이 안 돼요"]),
        "FAIL": (["결제 오류가 떠요", "카드 승인이 거절됐어요"],
                 ["페이 실패 알림이 와요", "한도 초과 메시지가 나와요"]),
    },
    "account": {
        "LOGIN": (["로그인이 안 돼요", "앱 접속이 안 돼요"],
                  ["사인인이 안 됩니다", "입장 자체가 막혀요"]),
        "PASSWORD": (["비밀번호를 잊어버렸어요", "패스워드 변경이 안 돼요"],
                     ["암호가 기억이 안 나요", "비번 재설정 메일이 안 와요"]),
        "PROFILE": (["회원정보를 바꾸고 싶어요", "아이디를 찾고 싶어요", "계정을 삭제하고 싶어요"],
                    ["프로필 수정이 안 돼요", "가입 정보를 변경하려고요"]),
        "LOCK": (["계정이 잠겼어요", "회원 계정이 정지됐어요"],
                 ["휴면 처리됐다고 나와요", "아이디가 막혀 버렸어요"]),
    },
    "defect": {
        "BROKEN": (["고장이 났어요", "작동이 안 돼요", "전원이 안 켜져요"],
                   ["완전 먹통이에요", "맛이 갔어요"]),
        "DAMAGE": (["파손된 채로 왔어요", "화면이 깨져서 왔어요", "금이 가 있어요"],
                   ["박살 나서 왔어요", "찌그러진 상태예요"]),
        "FAULT": (["불량품 같아요", "하자가 있어요", "제품 결함이 있어요"],
                  ["초기 이상 증상이 있어요", "제조 문제로 보여요"]),
    },
}
HEADS = ["어제 주문한 상품인데", "지난주에 산 이어폰이요.", "오늘 받은 노트북 건입니다.", "선물용으로 산 건데", "안녕하세요."]
TAILS = ["확인 부탁드립니다.", "빠른 답변 부탁드려요.", "어떻게 해야 하나요?", "주문번호는 메일로 보냈어요."]
LABEL_DESC = {
    "delivery": "배송 지연, 도착, 송장 조회 등 상품 배송 관련 문의",
    "refund": "환불, 반품, 주문 취소 관련 문의",
    "payment": "결제 오류, 중복 결제, 쿠폰과 적립금 관련 문의",
    "account": "로그인, 비밀번호, 회원정보 등 계정 관련 문의",
    "defect": "상품 고장, 파손, 불량 등 품질 관련 문의",
}


def _sentence(rng, label, test_mode, mix_label=None):
    concepts = list(LEXICON[label])
    picks = rng.choice(concepts, size=rng.integers(1, 3), replace=False)
    parts = []
    for c in picks:
        train_words, test_words = LEXICON[label][c]
        use_new = test_mode and rng.random() < 0.55
        parts.append(rng.choice(test_words if use_new else train_words))
    if mix_label is not None:  # 다른 범주 표현이 섞인 애매한 문장(레이블은 주된 범주)
        c = rng.choice(list(LEXICON[mix_label]))
        parts.append("그리고 " + rng.choice(LEXICON[mix_label][c][0]))
    head = rng.choice(HEADS) if rng.random() < 0.6 else ""
    return " ".join(x for x in [head, ", ".join(parts) + ".", rng.choice(TAILS)] if x)


class MockEncoder:
    """**가짜** 의미 인코더. 실제 LLM/문장임베딩 모델이 아니다.

    동의어 사전으로 같은 개념의 표현을 같은 개념 벡터로 보낸다.
    사전에 없는 글자 조각은 해시 벡터(작은 가중치)를 더한다.
    문장마다 크기를 무작위로 키우거나 줄여 정규화되지 않은 벡터를 만든다.
    """

    def __init__(self, dim=64, seed=7):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.phrases = []
        concept_vec = {}
        label_axis = {lab: rng.normal(0, 1, dim) for lab in LEXICON}
        for lab, concepts in LEXICON.items():
            for c, (a, b) in concepts.items():
                concept_vec[c] = label_axis[lab] * 0.8 + rng.normal(0, 1, dim)
                for p in a + b:
                    self.phrases.append((p, concept_vec[c]))
        self.phrases.sort(key=lambda t: -len(t[0]))
        self.rng = np.random.default_rng(seed + 1)

    def _hash_vec(self, token):
        h = zlib.crc32(token.encode("utf-8"))
        return np.random.default_rng(h).normal(0, 1, self.dim)

    def encode(self, text):
        v = np.zeros(self.dim)
        rest = text
        for p, vec in self.phrases:
            if p in rest:
                v += vec
                rest = rest.replace(p, " ")
        for tok in rest.split():
            v += 0.35 * self._hash_vec(tok)
        v += self.rng.normal(0, 0.6, self.dim)
        return v * self.rng.uniform(0.5, 3.0)  # 일부러 노름을 제각각으로


def make_q2():
    rng = np.random.default_rng(1107)
    labels = list(LEXICON)

    def build(n, test_mode):
        rows = []
        for _ in range(n):
            lab = rng.choice(labels, p=[0.26, 0.22, 0.2, 0.14, 0.18])
            mix = rng.choice([l for l in labels if l != lab]) if rng.random() < 0.08 else None
            rows.append((_sentence(rng, lab, test_mode, mix), lab))
        return rows

    train_rows, test_rows = build(1200, False), build(600, True)
    enc = MockEncoder()
    n_all = len(train_rows) + len(test_rows)
    ids = [f"T{i:05d}" for i in rng.permutation(n_all) + 1]
    train = pd.DataFrame(train_rows, columns=["text", "label"]); train.insert(0, "id", ids[:1200])
    test = pd.DataFrame(test_rows, columns=["text", "label"]); test.insert(0, "id", ids[1200:])
    out = ROOT / "data/q2"
    train.to_csv(out / "train.csv", index=False)
    test.drop(columns="label").to_csv(out / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "label": "delivery"}).to_csv(out / "sample_submission.csv", index=False)
    test[["id", "label"]].to_csv(TRUTH / "q2_truth.csv", index=False)

    # 임베딩 캐시는 id와 함께 저장하되 CSV와 다른 순서로 섞는다(id로 맞추는 연습)
    tr_perm, te_perm = rng.permutation(len(train)), rng.permutation(len(test))
    np.savez_compressed(
        out / "mock_embeddings.npz",
        train_ids=train["id"].to_numpy()[tr_perm].astype("U8"),
        train_emb=np.stack([enc.encode(t) for t in train["text"]])[tr_perm].astype(np.float32),
        test_ids=test["id"].to_numpy()[te_perm].astype("U8"),
        test_emb=np.stack([enc.encode(t) for t in test["text"]])[te_perm].astype(np.float32),
        label_names=np.array(labels, dtype="U10"),
        label_desc_emb=np.stack([enc.encode(LABEL_DESC[l]) for l in labels]).astype(np.float32),
    )
    (out / "labels.json").write_text(json.dumps(LABEL_DESC, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Q2", train.shape, test.shape, train.label.value_counts().to_dict())


# ======================================================================
# Q3 이미지 분류: 도형 (사전학습 백본 + 새 분류 헤드)
# ======================================================================
def _convex(xx, yy, pts):
    m = np.ones_like(xx, dtype=bool)
    k = len(pts)
    area = sum(pts[i][0] * pts[(i + 1) % k][1] - pts[(i + 1) % k][0] * pts[i][1] for i in range(k))
    s = 1 if area > 0 else -1
    for i in range(k):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % k]
        m &= s * ((x2 - x1) * (yy - y1) - (y2 - y1) * (xx - x1)) >= 0
    return m


def _rot(cx, cy, pts, a):
    c, s = np.cos(a), np.sin(a)
    return [(cx + c * x - s * y, cy + s * x + c * y) for x, y in pts]


def _shape_mask(kind, size, rng):
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    cx, cy = rng.uniform(size * 0.32, size * 0.68, 2)
    r = rng.uniform(size * 0.2, size * 0.32)
    a = rng.uniform(0, 2 * np.pi)
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    if kind == "circle":
        return d2 <= r ** 2
    if kind == "square":
        return _convex(xx, yy, _rot(cx, cy, [(-r, -r), (r, -r), (r, r), (-r, r)], a * 0.25))
    if kind == "triangle":
        return _convex(xx, yy, _rot(cx, cy, [(0, -r * 1.2), (r * 1.1, r * 0.8), (-r * 1.1, r * 0.8)], a * 0.3))
    if kind == "cross":
        w = r * 0.35
        return (_convex(xx, yy, _rot(cx, cy, [(-r, -w), (r, -w), (r, w), (-r, w)], a * 0.2))
                | _convex(xx, yy, _rot(cx, cy, [(-w, -r), (w, -r), (w, r), (-w, r)], a * 0.2)))
    # 사전학습 전용 도형
    if kind == "ring":
        return (d2 <= r ** 2) & (d2 >= (r * 0.6) ** 2)
    if kind == "ellipse":
        return ((xx - cx) / r) ** 2 + ((yy - cy) / (r * 0.55)) ** 2 <= 1
    if kind == "rect":
        return _convex(xx, yy, _rot(cx, cy, [(-r, -r * 0.5), (r, -r * 0.5), (r, r * 0.5), (-r, r * 0.5)], a * 0.25))
    if kind == "hbar":
        return _convex(xx, yy, _rot(cx, cy, [(-r, -r * 0.3), (r, -r * 0.3), (r, r * 0.3), (-r, r * 0.3)], a * 0.15))
    if kind == "vbar":
        return _convex(xx, yy, _rot(cx, cy, [(-r * 0.3, -r), (r * 0.3, -r), (r * 0.3, r), (-r * 0.3, r)], a * 0.15))
    if kind == "diamond":
        return _convex(xx, yy, _rot(cx, cy, [(0, -r * 1.2), (r * 0.8, 0), (0, r * 1.2), (-r * 0.8, 0)], 0))
    if kind == "hollow_square":
        outer = _convex(xx, yy, [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)])
        inner = _convex(xx, yy, [(cx - r * .55, cy - r * .55), (cx + r * .55, cy - r * .55), (cx + r * .55, cy + r * .55), (cx - r * .55, cy + r * .55)])
        return outer & ~inner
    if kind == "xmark":
        w = r * 0.3
        return (_convex(xx, yy, _rot(cx, cy, [(-r, -w), (r, -w), (r, w), (-r, w)], np.pi / 4))
                | _convex(xx, yy, _rot(cx, cy, [(-r, -w), (r, -w), (r, w), (-r, w)], -np.pi / 4)))
    raise ValueError(kind)


def render(kinds, n, rng, size=32):
    imgs = np.empty((n, size, size, 3), np.uint8)
    labels = rng.integers(0, len(kinds), n)
    for i, lab in enumerate(labels):
        m = _shape_mask(kinds[lab], size, rng)
        bg = rng.uniform(0, 255, 3)
        fg = rng.uniform(0, 255, 3)
        while np.abs(fg - bg).sum() < 180:
            fg = rng.uniform(0, 255, 3)
        img = np.where(m[..., None], fg, bg) + rng.normal(0, 12, (size, size, 3))
        # 방해용 가는 선 하나
        if rng.random() < 0.3:
            row = rng.integers(0, size)
            img[row, :, :] = img[row, :, :] * 0.5 + rng.uniform(0, 255, 3) * 0.5
        imgs[i] = np.clip(img, 0, 255).astype(np.uint8)
    return imgs, labels


TARGET = ["circle", "square", "triangle", "cross"]
PRETRAIN = ["ring", "ellipse", "rect", "diamond", "hollow_square", "xmark"]


def make_q3():
    import torch
    from torch import nn
    from practice_support import make_backbone
    rng = np.random.default_rng(3107)
    torch.manual_seed(0); torch.set_num_threads(4)

    # 1) 다른 도형 6종으로 백본 사전학습(학생은 이 과정을 보지 않는다)
    px, py = render(PRETRAIN, 5000, rng)
    x = torch.tensor(px).permute(0, 3, 1, 2).float() / 255
    x = (x - x.mean((0, 2, 3), keepdim=True)) / x.std((0, 2, 3), keepdim=True)
    y = torch.tensor(py)
    backbone = make_backbone()
    model = nn.Sequential(backbone, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, len(PRETRAIN)))
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    for ep in range(20):
        perm = torch.randperm(len(y))
        for s in range(0, len(y), 64):
            b = perm[s:s + 64]
            opt.zero_grad(); loss = nn.functional.cross_entropy(model(x[b]), y[b]); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (model(x[:1000]).argmax(1) == y[:1000]).float().mean().item()
        print(f"  pretrain ep{ep} loss {loss.item():.3f} acc {acc:.3f}")
    torch.save(backbone.state_dict(), ROOT / "assets/q3_backbone.pt")

    # 2) 목표 과제: 원/사각형/삼각형/십자
    tx, ty = render(TARGET, 1200, rng)
    sx, sy = render(TARGET, 800, rng)
    n_all = 2000
    ids = np.array([f"I{i:05d}" for i in rng.permutation(n_all) + 1])
    out = ROOT / "data/q3"
    np.savez_compressed(out / "train_images.npz", ids=ids[:1200].astype("U8"), images=tx)
    np.savez_compressed(out / "test_images.npz", ids=ids[1200:].astype("U8"), images=sx)
    # 학습 레이블은 CSV로 따로, 행 순서도 섞어서 제공(id로 맞추는 연습)
    perm = rng.permutation(1200)
    pd.DataFrame({"id": ids[:1200][perm], "label": np.array(TARGET)[ty][perm]}).to_csv(out / "train_labels.csv", index=False)
    pd.DataFrame({"id": ids[1200:], "label": "circle"}).to_csv(out / "sample_submission.csv", index=False)
    pd.DataFrame({"id": ids[1200:], "label": np.array(TARGET)[sy]}).to_csv(TRUTH / "q3_truth.csv", index=False)
    print("Q3", tx.shape, sx.shape, np.bincount(ty))


if __name__ == "__main__":
    make_q1(); make_q2(); make_q3()
