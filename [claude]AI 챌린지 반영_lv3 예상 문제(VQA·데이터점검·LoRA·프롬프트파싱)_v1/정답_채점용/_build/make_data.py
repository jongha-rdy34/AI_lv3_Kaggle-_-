"""[claude] AI 챌린지 반영 VQA 모의 데이터 생성기 + 로컬 MiniVLM 사전학습.

실행: python -I make_data.py [data|pretrain|zip]   (인자 없으면 전부)
모든 이미지·질문은 합성 연습 자료입니다.
"""
import shutil
import string
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LOCAL = ROOT / "로컬_CPU판"
TRUTH = ROOT / "정답_채점용"
sys.path.insert(0, str(LOCAL))

S = 64
COLORS = {"빨강": (220, 40, 40), "파랑": (40, 80, 220), "초록": (40, 170, 60),
          "노랑": (235, 205, 30), "보라": (140, 60, 190), "주황": (240, 140, 30)}
SHAPES = ["원", "사각형", "삼각형", "십자"]
LETTERS = ["a", "b", "c", "d"]
JOSA = {"원": "은", "사각형": "은", "삼각형": "은", "십자": "는"}

# baseline과 같은 지시문·프롬프트 형식 (사전학습도 이 형식으로 했음)
SYSTEM_INSTRUCT = ("You are a helpful visual question answering assistant. "
                   "Answer using exactly one letter among a, b, c, or d. No explanation.")


def build_mc_prompt(question, a, b, c, d):
    return (f"{question}\n(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n"
            "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요.")


# 목표 과제(시험 데이터) 질문 문장
TARGET_Q = {
    "color": lambda s: f"그림 속 {s}{JOSA[s]} 무슨 색인가요?",
    "count": lambda: "그림에 도형이 모두 몇 개 있나요?",
    "shape": lambda c: f"{c} 도형의 모양은 무엇인가요?",
    "left": lambda: "가장 왼쪽에 있는 도형의 모양은 무엇인가요?",
    "right": lambda: "가장 오른쪽에 있는 도형의 모양은 무엇인가요?",
}
# 사전학습 질문 문장 (표현이 다름 → 목표 과제에 맞게 미세조정할 여지)
PRETRAIN_Q = {
    "color": lambda s: f"{s}의 색깔은?",
    "count": lambda: "도형 개수는?",
    "shape": lambda c: f"{c} 도형은 어떤 모양?",
    "left": lambda: "맨 왼쪽 도형은?",
    "right": lambda: "맨 오른쪽 도형은?",
}


# ----------------------------------------------------------------------
def draw_scene(rng):
    """도형 1~4개를 서로 다른 열(x 위치)에 그린다. 모양·색은 그림 안에서 겹치지 않음."""
    k = int(rng.integers(1, 5))
    cols = np.sort(rng.choice(4, k, replace=False))
    shapes = list(rng.choice(SHAPES, k, replace=False))
    colors = list(rng.choice(list(COLORS), k, replace=False))
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    bg = rng.uniform(225, 250)
    img = np.full((S, S, 3), bg) + rng.normal(0, 6, (S, S, 3))
    objs = []
    for col, shp, colr in zip(cols, shapes, colors):
        cx = 8 + 16 * col + rng.uniform(-1.5, 1.5)
        cy = rng.uniform(12, 52)
        r = rng.uniform(5.5, 7.0)
        dx, dy = xx - cx, yy - cy
        if shp == "원":
            m = dx ** 2 + dy ** 2 <= r ** 2
        elif shp == "사각형":
            m = (np.abs(dx) <= r * 0.85) & (np.abs(dy) <= r * 0.85)
        elif shp == "삼각형":
            m = (dy <= r * 0.8) & (dy >= -r) & (np.abs(dx) <= (dy + r) * 0.6)
        else:
            w = r * 0.35
            m = ((np.abs(dx) <= w) & (np.abs(dy) <= r)) | ((np.abs(dy) <= w) & (np.abs(dx) <= r))
        img[m] = np.array(COLORS[colr]) + rng.normal(0, 8, 3)
        objs.append({"shape": shp, "color": colr, "x": cx})
    return np.clip(img, 0, 255).astype(np.uint8), objs


def make_question(rng, objs, templates, letter_p):
    qtype = rng.choice(["color", "count", "shape", "left", "right"], p=[0.25, 0.2, 0.25, 0.15, 0.15])
    if qtype == "color":
        o = objs[rng.integers(len(objs))]
        q, ans, pool = templates["color"](o["shape"]), o["color"], list(COLORS)
    elif qtype == "count":
        q, ans, pool = templates["count"](), f"{len(objs)}개", ["1개", "2개", "3개", "4개"]
    elif qtype == "shape":
        o = objs[rng.integers(len(objs))]
        q, ans, pool = templates["shape"](o["color"]), o["shape"], SHAPES
    else:
        o = min(objs, key=lambda t: t["x"]) if qtype == "left" else max(objs, key=lambda t: t["x"])
        q, ans, pool = templates[qtype](), o["shape"], SHAPES
    distract = list(rng.choice([p for p in pool if p != ans], 3, replace=False))
    pos = int(rng.choice(4, p=letter_p))
    opts = distract[:pos] + [ans] + distract[pos:]
    return qtype, q, opts, LETTERS[pos]


def build_rows(rng, n, templates, letter_p, prefix, img_dir=None):
    rows, images = [], []
    for i in range(n):
        img, objs = draw_scene(rng)
        qtype, q, opts, ans = make_question(rng, objs, templates, letter_p)
        rid = f"{prefix}_{i + 1:04d}"
        rows.append({"id": rid, "path": f"{prefix.lower()}/{rid}.png", "question": q,
                     "a": opts[0], "b": opts[1], "c": opts[2], "d": opts[3], "answer": ans, "qtype": qtype})
        images.append(img)
        if img_dir is not None:
            Image.fromarray(img).save(img_dir / f"{rid}.png")
    return pd.DataFrame(rows), images


def make_data():
    rng = np.random.default_rng(1107)
    for sub in ["train", "test"]:
        shutil.rmtree(DATA / sub, ignore_errors=True)
        (DATA / sub).mkdir(parents=True)
    # train: 정답 글자가 a에 쏠림(45%) / test: 고르게
    train, _ = build_rows(rng, 1000, TARGET_Q, [0.45, 0.55 / 3, 0.55 / 3, 0.55 / 3], "TRAIN", DATA / "train")
    test, _ = build_rows(rng, 400, TARGET_Q, [0.25] * 4, "TEST", DATA / "test")
    # ---- 품질 문제 주입 (train만) ----
    broken = rng.choice(len(train), 8, replace=False)
    for i in broken:                                   # 깨진 사진: 검은 화면
        Image.fromarray(np.zeros((S, S, 3), np.uint8)).save(DATA / train.loc[i, "path"])
    rest = np.setdiff1d(np.arange(len(train)), broken)
    missing = rng.choice(rest, 8, replace=False)
    for i in missing:                                  # 사진 파일이 없음
        (DATA / train.loc[i, "path"]).unlink()
    messy = rng.random(len(train)) < 0.06              # 정답 표기 흔들림
    train.loc[messy, "answer"] = [str(rng.choice([a.upper(), f" {a}", f"({a})", f"{a}."])) for a in train.loc[messy, "answer"]]
    cols = ["id", "path", "question", "a", "b", "c", "d"]
    train[cols + ["answer"]].to_csv(DATA / "train.csv", index=False)
    test[cols].to_csv(DATA / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "answer": "a"}).to_csv(DATA / "sample_submission.csv", index=False)
    test[["id", "answer", "qtype"]].to_csv(TRUTH / "test_truth.csv", index=False)
    print("train", train.shape, "test", test.shape, "| train 정답 분포", train.answer.str.strip().str.lower().str.strip("().").value_counts().to_dict())


# ----------------------------------------------------------------------
def vocab_tokens():
    """토크나이저 사전: 모든 질문·보기·지시문의 단어 + 출력 가능한 ASCII 글자 + 자주 쓸 한글 글자(모르는 단어용)."""
    from mini_vlm import split_tokens
    texts = [SYSTEM_INSTRUCT, build_mc_prompt("", "", "", "", ""), "assistant\n system user", '{"answer": "a"}']
    for t in (TARGET_Q, PRETRAIN_Q):
        for s in SHAPES:
            texts.append(t["color"](s))
        for c in COLORS:
            texts.append(t["shape"](c))
        texts += [t["count"](), t["left"](), t["right"]()]
    texts += list(COLORS) + SHAPES + ["1개", "2개", "3개", "4개"]
    tokens = [tok for t in texts for tok in split_tokens(t)]
    extra = ("가나다라마바사아자차카타파하는은이가을를의에서으로와과도만한하고하여해서그리고또는만약정답보기선택지문제질문"
             "그림사진이미지도형모양색깔개수왼오른쪽위아래가운데중앙몇무엇어떤어느입니다세요습니다합니다출력반드시하나소문자글자"
             "만으로설명없이당신은시각답변도우미예시참고규칙형식")
    return tokens + list(string.printable) + list(extra)


def pretrain(epochs=10, n_total=6000):
    import torch
    from mini_vlm import MiniTokenizer, MiniProcessor, MiniVLM
    rng = np.random.default_rng(7)
    torch.manual_seed(0); torch.set_num_threads(4)
    df, imgs = build_rows(rng, n_total, PRETRAIN_Q, [0.25] * 4, "PRE")
    tok = MiniTokenizer(vocab_tokens())
    proc = MiniProcessor(tok)
    model = MiniVLM(len(tok))
    model.set_special_ids(tok)
    # 대화 문자열과 정답 위치(assistant 답 토큰만 손실)를 미리 만든다
    texts = []
    for r in df.itertuples():
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": build_mc_prompt(r.question, r.a, r.b, r.c, r.d)}]},
                {"role": "assistant", "content": [{"type": "text", "text": r.answer}]}]
        texts.append(proc.apply_chat_template(msgs))
    pix = torch.stack([proc.image_to_tensor(Image.fromarray(x)) for x in imgs])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    n_tr = n_total - 500
    total_steps = epochs * (n_tr // 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=total_steps, pct_start=0.1)
    start_id = tok.stoi["<|im_start|>"]
    prefix_len = len(tok.encode("assistant\n"))
    for ep in range(epochs):
        t0 = time.time(); model.train(); perm = rng.permutation(n_tr)
        for s in range(0, n_tr - 31, 32):
            b = perm[s:s + 32]
            enc = proc(text=[texts[i] for i in b])
            labels = torch.full_like(enc["input_ids"], -100)
            for j in range(len(b)):                     # 마지막 <|im_start|>assistant\n 다음 토큰부터 정답
                st = (enc["input_ids"][j] == start_id).nonzero()[-1].item() + prefix_len + 1
                end = enc["attention_mask"][j].sum().item()
                labels[j, st:end] = enc["input_ids"][j, st:end]
            out = model(enc["input_ids"], enc["attention_mask"], pix[b], labels)
            # 언어모델 경로도 함께 학습(프롬프트 토큰 다음 토큰 예측) → 실제 사전학습 LM처럼 프롬프트를 어느 정도 '안다'
            lm_labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
            lm_loss = torch.nn.functional.cross_entropy(out.logits[:, :-1].reshape(-1, out.logits.shape[-1]),
                                                        lm_labels[:, 1:].reshape(-1), ignore_index=-100)
            loss = out.loss + 0.5 * lm_loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        # 확인: held-out 500개 정답률 (정답 글자 위치의 argmax)
        model.eval(); correct = 0
        with torch.no_grad():
            for s in range(n_tr, n_total, 50):
                idx = list(range(s, min(s + 50, n_total)))
                enc = proc(text=[texts[i] for i in idx])
                logits = model(enc["input_ids"], enc["attention_mask"], pix[idx]).logits
                for j, i in enumerate(idx):
                    st = (enc["input_ids"][j] == start_id).nonzero()[-1].item() + prefix_len
                    pred = tok.itos[logits[j, st].argmax().item()]
                    correct += pred == df.answer[i]
        print(f"  pretrain ep{ep} loss {out.loss.item():.3f} held-out acc {correct / 500:.3f} ({time.time() - t0:.0f}s)", flush=True)
    model.save_pretrained(LOCAL / "assets")
    proc.save_pretrained(LOCAL / "assets")


def make_zip():
    """Colab용 데이터 zip (구글 드라이브에 올려 압축 해제)."""
    out = ROOT / "Colab_실제판" / "vqa_mock_data.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(DATA.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(DATA).as_posix())
    print("zip", out.name, round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    jobs = sys.argv[1:] or ["data", "pretrain", "zip"]
    for j in jobs:
        if j.startswith("pretrain:"):          # 예: pretrain:3:3000 (epochs:샘플 수) 빠른 확인용
            _, e, n = j.split(":"); pretrain(int(e), int(n)); continue
        {"data": make_data, "pretrain": pretrain, "zip": make_zip}[j]()
