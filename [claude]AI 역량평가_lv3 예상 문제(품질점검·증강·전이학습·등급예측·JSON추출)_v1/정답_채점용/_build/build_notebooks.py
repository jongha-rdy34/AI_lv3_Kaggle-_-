"""문제/정답 노트북 생성기. 실행: python -I build_notebooks.py

코드 셀마다 (문제용, 정답용) 두 버전. 정답용은 줄마다 주석을 단 풀이 예시.
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
CELLS = []


def md(text):
    CELLS.append(("md", text.strip("\n"), None))


def run(code):
    CELLS.append(("code", "# [실행만 하세요]\n" + code.strip("\n"), None))


def write(problem, answer):
    CELLS.append(("code", "# [작성하세요]\n" + problem.strip("\n"), "# [정답 예시]\n" + answer.strip("\n")))


# ======================================================================
md(r"""
# [claude] Lv3 모의평가 (공지 반영) v1

**2026-11-04에 받은 Level 3 상세 공지를 반영한 창작 모의문제입니다. 실제 시험 문제가 아닙니다.** 먼저 `README.md`를 읽으세요.

| 문항 | 공지 분야 | 내용 |
|---|---|---|
| Q1 | 데이터 분석/전처리 (이미지) | 데이터셋 구성 파악, 품질·분포 점검, 육안 검사, 정제 |
| Q2 | 데이터 분석/전처리 (이미지) | 변환·증강, 불균형을 고려한 학습 데이터 재구성 |
| Q3 | AI 모델링 및 최적화 (이미지) | 전이학습용 구조 변경, 학습 범위 제어, 손실 함수 구현, 불균형 대응 |
| Q4 | AI 모델링 및 최적화 (이미지) | 등급 예측(회귀 헤드), QWK, 임계값, 하이퍼파라미터 탐색, 성능 분석 |
| Q5 | 프롬프트 엔지니어링 (텍스트) | 정보 추출 프롬프트, JSON 스키마, 정규화, 미확정 정보 처리, 자료형 검사 |

- 권장 4시간: Q1 40분 · Q2 35분 · Q3 55분 · Q4 50분 · Q5 50분 · 점검 10분. CPU 전체 실행은 약 6~8분입니다.
- `[작성하세요]` 셀의 `None`·빈칸을 채웁니다. **함수 이름·매개변수·반환 형식은 그대로** 둡니다.
- 공지의 평가 방식은 "요구사항 충족 + 풀이 과정 + 최종 코드 리뷰"입니다. **읽기 쉬운 코드와 주석, 서술 답안**도 평가 대상이라고 생각하고 작성하세요.
- 앞 소문항을 못 풀면 확인 셀이 **진행용 준비 입력**으로 대신 이어 갑니다(해당 소문항 평가와 무관).
- test 정답은 쓰지 않습니다. `정답_채점용/`은 풀이가 끝날 때까지 열지 마세요.
""")

run(r"""
from pathlib import Path
import sys, time, copy, json, re, hashlib, itertools
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "practice_support.py").exists()), None)
if ROOT is None:
    raise FileNotFoundError("모의문제 폴더(또는 그 하위 폴더)에서 노트북을 실행하세요.")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, confusion_matrix
import practice_support as ps
from practice_support import attempt, csv, CLASSES

ps.seed_everything(42)
print("준비 완료:", ROOT.name)
""")

# ======================================================================  Q1
md(r"""
---
## Q1 데이터셋 구성·품질·분포 점검 (데이터 분석/전처리)

사과 잎 사진으로 질병을 분류하는 데이터입니다(합성 자료, 48×48 RGB).
- `data/leaf/train_images.npz`: `ids`, `images` (N,48,48,3) uint8
- `data/leaf/train_labels.csv`: `id,label` — 레이블은 `healthy, rust, scab, multiple`
- `data/leaf/test_images.npz`: 레이블 없는 평가용 사진

**이 데이터에는 품질 문제가 섞여 있을 수 있습니다.** 어떤 문제가 있는지는 알려 주지 않습니다. 직접 찾아내세요.
""")

run(r"""
q1_ids, q1_img = ps.load_images("leaf", "train")
q1_test_ids, q1_test_img = ps.load_images("leaf", "test")
q1_lab = csv("leaf", "train_labels.csv")
print(q1_img.shape, q1_img.dtype, "| 레이블 행", len(q1_lab), "| test", q1_test_img.shape)
q1_lab.head()
""")

md(r"""
### 1-1 데이터셋 구성 파악
`summarize_dataset(ids, images, labels)`가 다음 키를 가진 dict를 반환하게 하세요.
- `n_images`, `n_label_rows`, `image_shape`(한 장의 모양 튜플), `dtype`(문자열)
- `ids_without_label`: 레이블 행이 없는 이미지 수 / `labels_without_image`: 이미지가 없는 레이블 행 수
- `raw_label_values`: 원래 레이블 문자열의 고유값(정렬된 리스트)
- `class_counts`: 레이블을 `strip().lower()`로 정리한 뒤 클래스별 개수(dict)
- `imbalance_ratio`: 가장 많은 클래스 수 / 가장 적은 클래스 수
""")

write(r"""
def summarize_dataset(ids, images, labels):
    info = {}
    # 작성하세요
    return info
""", r"""
def summarize_dataset(ids, images, labels):
    info = {}
    # --- 기본 구성: 개수, 한 장의 모양, 자료형 ---
    info["n_images"] = len(ids)
    info["n_label_rows"] = len(labels)
    info["image_shape"] = tuple(images.shape[1:])        # (48, 48, 3)
    info["dtype"] = str(images.dtype)                     # 'uint8'

    # --- 이미지와 레이블의 id 대응 확인 (집합 차집합) ---
    image_ids, label_ids = set(ids), set(labels["id"])
    info["ids_without_label"] = len(image_ids - label_ids)    # 사진은 있는데 레이블이 없음
    info["labels_without_image"] = len(label_ids - image_ids) # 레이블은 있는데 사진이 없음

    # --- 레이블 표기 흔들림 확인: 정리 전 고유값을 그대로 본다 ---
    info["raw_label_values"] = sorted(labels["label"].unique())

    # --- 정리(공백 제거 + 소문자) 후 클래스 분포와 불균형 비율 ---
    clean = labels["label"].str.strip().str.lower()
    counts = clean.value_counts()
    info["class_counts"] = counts.to_dict()
    info["imbalance_ratio"] = float(counts.max() / counts.min())
    return info
""")

run(r"""
q1_info = attempt("1-1", summarize_dataset, q1_ids, q1_img, q1_lab)
if q1_info:
    for k, v in q1_info.items():
        print(f"{k:22s}: {v}")
""")

md(r"""
### 1-2 품질 점검: 깨진 사진과 중복 사진
- `find_broken(images, std_thr=1.0)`: 픽셀 표준편차가 `std_thr` 미만인 사진(검은 화면, 단색 화면)을 `True`로 표시한 bool 배열 (N,)
- `find_duplicates(images)`: **바이트가 완전히 같은** 사진 묶음 리스트. 각 원소는 2개 이상의 위치를 담은 정렬된 리스트 (예: `[[3, 917], [40, 1201]]`). `hashlib.md5(x.tobytes())`를 쓰세요.
""")

write(r"""
def find_broken(images, std_thr=1.0):
    return None  # 작성하세요

def find_duplicates(images):
    groups = []
    # 작성하세요
    return groups
""", r"""
def find_broken(images, std_thr=1.0):
    # 한 장 안의 모든 픽셀(높이·너비·채널)의 표준편차.
    # 검은 화면·단색 화면은 값이 거의 변하지 않으므로 std가 0에 가깝다.
    stds = images.reshape(len(images), -1).std(axis=1)
    return stds < std_thr

def find_duplicates(images):
    # 같은 바이트 → 같은 해시. 해시별로 위치를 모은다.
    by_hash = {}
    for i, x in enumerate(images):
        h = hashlib.md5(x.tobytes()).hexdigest()
        by_hash.setdefault(h, []).append(i)
    # 2장 이상 모인 해시만 '중복 묶음'
    groups = [sorted(pos) for pos in by_hash.values() if len(pos) > 1]
    return sorted(groups)
""")

run(r"""
def _check_1_2():
    toy = np.zeros((4, 2, 2, 3), np.uint8); toy[1] = np.arange(12).reshape(2, 2, 3); toy[3] = toy[1]
    assert find_broken(toy).tolist() == [True, False, True, False], "find_broken"
    assert find_duplicates(toy) == [[0, 2], [1, 3]], "find_duplicates"
    b, d = find_broken(q1_img), find_duplicates(q1_img)
    lab_map = q1_lab.assign(label=q1_lab["label"].str.strip().str.lower()).set_index("id")["label"]
    conflict = [g for g in d if lab_map.reindex(q1_ids[g]).nunique() > 1]
    print(f"깨진 사진 {int(b.sum())}장 | 중복 묶음 {len(d)}개 (레이블이 엇갈린 묶음 {len(conflict)}개)")
    return b, d
q1_qc = attempt("1-2", _check_1_2)
""")

md(r"""
### 1-3 육안 검사와 분포 점검
아래 셀은 클래스별 무작위 사진, 깨진 사진, 중복 사진을 보여 줍니다. **눈으로 확인하고 이상한 점을 찾으세요.** (힌트: 잎과 흙의 색을 보세요.)

그다음 작성하세요.
- `find_bgr(images, margin=10)`: 채널 순서가 뒤바뀐(BGR로 저장된) 사진을 찾는 bool 배열. 정상 사진은 흙색 때문에 **R 평균이 B 평균보다 큽니다.** `B 평균 − R 평균 > margin`이면 `True`.
- `mean_brightness(images)`: 사진별 전체 픽셀 평균(0~255) 배열 (N,)
- 서술 `q1_visual_note`: 육안 검사와 아래 분포 점검에서 발견한 점, 그리고 그것이 이후 학습에 주는 의미를 2~3문장으로.
""")

run(r"""
rng = np.random.default_rng(0)
_lab = q1_lab.assign(label=q1_lab["label"].str.strip().str.lower()).set_index("id")["label"].reindex(q1_ids)
for c in CLASSES:
    pos = np.flatnonzero(_lab.values == c)
    ps.show_images(q1_img[rng.choice(pos, 8, replace=False)], [c] * 8)
if q1_qc:
    ps.show_images(q1_img[np.flatnonzero(q1_qc[0])][:8], ["broken"] * 8)
# 색이 이상해 보이는 사진이 섞여 있는지 무작위 32장을 더 봅니다.
pick = rng.choice(len(q1_img), 32, replace=False)
ps.show_images(q1_img[pick], q1_ids[pick])
""")

write(r"""
def find_bgr(images, margin=10):
    return None  # 작성하세요

def mean_brightness(images):
    return None  # 작성하세요

q1_visual_note = ""  # 작성하세요
""", r"""
def find_bgr(images, margin=10):
    # 채널 축(마지막 축) 기준: 0=R, 1=G, 2=B. 사진마다 채널 평균을 구한다.
    r_mean = images[..., 0].mean(axis=(1, 2))
    b_mean = images[..., 2].mean(axis=(1, 2))
    # 정상: 흙(갈색) 때문에 R > B. BGR로 저장되면 R과 B가 바뀌어 B가 크게 나온다.
    return (b_mean - r_mean) > margin

def mean_brightness(images):
    # 사진 한 장의 모든 픽셀 평균 → 밝기 지표 (0=검정, 255=흰색)
    return images.reshape(len(images), -1).mean(axis=1)

q1_visual_note = (
    "일부 사진은 흙이 파랗게 보여 채널 순서가 BGR로 저장된 것으로 판단했고, 채널을 뒤집어 RGB로 고친다. "
    "검은 화면·회색 화면은 정보가 없어 제거한다. "
    "test에는 어두운 사진 비율이 train보다 훨씬 높아, 학습 때 밝기 증강을 넣어 이 차이에 대비해야 한다."
)
""")

run(r"""
def _check_1_3():
    toy = np.zeros((2, 2, 2, 3), np.uint8); toy[0, ..., 0] = 150; toy[1, ..., 2] = 150
    assert find_bgr(toy).tolist() == [False, True]
    assert np.allclose(mean_brightness(toy), [50, 50])
    bgr = find_bgr(q1_img)
    print("BGR 의심:", int(bgr.sum()), "장")
    ps.show_images(q1_img[bgr][:8], ["BGR?"] * 8)
    dark = lambda imgs: float((mean_brightness(imgs) < 60).mean())
    print(f"어두운 사진(평균 밝기<60) 비율 | train {dark(q1_img):.3f} | test {dark(q1_test_img):.3f}")
    print("서술 작성:", bool(q1_visual_note.strip()))
attempt("1-3", _check_1_3)
""")

md(r"""
### 1-4 정제된 학습 데이터 만들기
`build_clean_dataset(ids, images, labels)` → `(clean_ids, clean_images, clean_labels)` (모두 numpy 배열, 같은 순서). 규칙:
1. 레이블은 `strip().lower()`로 정리
2. 이미지와 레이블 **둘 다 있는** id만 사용
3. 깨진 사진 제거 (`find_broken`)
4. 중복 묶음: 레이블이 **모두 같으면 첫 번째 하나만** 남기고, **엇갈리면 묶음 전체를 제거**
5. BGR 사진은 채널을 뒤집어 RGB로 고침 (`find_bgr`) — 제거하지 않음
""")

write(r"""
def build_clean_dataset(ids, images, labels):
    # 작성하세요
    return None, None, None
""", r"""
def build_clean_dataset(ids, images, labels):
    # (1) 레이블 문자열 정리 → id로 찾을 수 있는 Series
    lab = labels.assign(label=labels["label"].str.strip().str.lower())
    label_of = lab.set_index("id")["label"]

    # (2) 레이블이 있는 이미지만 남길 표시 (이미지 없는 레이블 행은 자연히 무시됨)
    keep = np.isin(ids, label_of.index)
    # (3) 깨진 사진 제외
    keep &= ~find_broken(images)

    # (4) 중복 처리: 묶음마다 레이블이 같은지 확인
    for group in find_duplicates(images):
        group_labels = label_of.reindex(ids[group])
        if group_labels.nunique(dropna=True) > 1:
            keep[group] = False          # 레이블이 엇갈림 → 어느 쪽도 믿을 수 없어 전부 제거
        else:
            keep[group[1:]] = False      # 같은 레이블 → 첫 장만 남기고 나머지 제거

    sel = np.flatnonzero(keep)
    clean_ids = ids[sel]
    clean_images = images[sel].copy()    # 원본을 바꾸지 않도록 복사
    clean_labels = label_of.reindex(clean_ids).to_numpy()

    # (5) BGR 사진은 채널 축을 뒤집어 RGB로 복원 (제거하지 않음: 내용은 정상)
    bgr = find_bgr(clean_images)
    clean_images[bgr] = clean_images[bgr][..., ::-1]
    return clean_ids, clean_images, clean_labels
""")

run(r"""
def _check_1_4():
    cid, cimg, clab = build_clean_dataset(q1_ids, q1_img, q1_lab)
    assert len(cid) == len(cimg) == len(clab) and len(set(cid)) == len(cid)
    assert not find_broken(cimg).any() and not find_bgr(cimg).any() and not find_duplicates(cimg)
    print("정제 후", len(cid), "장 |", pd.Series(clab).value_counts().to_dict())
    return cid, cimg, clab
q1_clean = attempt("1-4", _check_1_4)
if q1_clean is None:
    print("→ 이후 진행용 준비 데이터를 사용합니다(1-4 평가와 무관).")
    q1_clean = ps.ready_clean_leaf()
clean_ids, clean_img, clean_lab = q1_clean
""")

# ======================================================================  Q2
md(r"""
---
## Q2 변환·증강과 학습 데이터 재구성 (데이터 분석/전처리)

제공 모델의 모델 카드: **입력은 [0,1]로 바꾼 뒤 채널별 `mean=PRETRAIN_MEAN`, `std=PRETRAIN_STD`로 정규화**해 사전학습했습니다(`ps.PRETRAIN_MEAN`, `ps.PRETRAIN_STD`).
""")

md(r"""
### 2-1 텐서 변환과 정규화
- `to_tensor(images)`: (N,H,W,3) uint8 → `torch.float32` (N,3,H,W), 값 [0,1]
- `normalize(x)`: 모델 카드의 mean/std로 채널별 정규화 (브로드캐스팅)
""")

write(r"""
def to_tensor(images):
    return None  # 작성하세요

def normalize(x):
    return None  # 작성하세요
""", r"""
MEAN = torch.tensor(ps.PRETRAIN_MEAN).view(1, 3, 1, 1)   # (1,C,1,1) 모양 → (N,C,H,W)에 브로드캐스팅
STD = torch.tensor(ps.PRETRAIN_STD).view(1, 3, 1, 1)

def to_tensor(images):
    x = torch.from_numpy(np.ascontiguousarray(images))     # (N,H,W,C) uint8
    x = x.permute(0, 3, 1, 2)                               # 채널을 앞으로 → (N,C,H,W)
    return x.float() / 255.0                                # [0,255] → [0,1]

def normalize(x):
    # 사전학습 때와 같은 통계를 써야 모델이 익숙한 입력 분포가 된다.
    # (train에서 새로 계산한 통계로 바꾸지 않는다)
    return (x - MEAN) / STD
""")

run(r"""
def _check_2_1():
    x = to_tensor(clean_img[:5])
    assert x.shape == (5, 3, 48, 48) and x.dtype == torch.float32 and 0 <= float(x.min()) and float(x.max()) <= 1
    assert torch.allclose(x[0, :, 3, 7], torch.tensor(clean_img[0, 3, 7] / 255.0, dtype=torch.float32)), "축 순서"
    z = normalize(torch.full((1, 3, 2, 2), 0.5))
    assert torch.allclose(z, torch.zeros_like(z)), "mean 0.5인 입력은 0이 되어야 함"
attempt("2-1", _check_2_1)
""")

md(r"""
### 2-2 데이터 증강
`augment(xb)`: [0,1] 텐서 배치 (B,3,H,W)를 받아 **샘플마다** 다음을 무작위로 적용한 새 텐서를 반환합니다(입력은 바꾸지 않음).
- 좌우 반전 50%, 상하 반전 50% (잎 방향은 질병과 무관)
- 밝기 배율 `U(0.4, 1.1)`을 곱한 뒤 [0,1]로 자르기

서술 `q2_aug_note`: (1) 밝기 증강을 넣는 근거, (2) **색조(hue) 변환을 쓰면 안 되는 이유**를 쓰세요.
""")

write(r"""
def augment(xb):
    xb = xb.clone()
    # 작성하세요
    return xb

q2_aug_note = ""  # 작성하세요
""", r"""
def augment(xb):
    xb = xb.clone()                                   # 입력 텐서를 직접 바꾸지 않기 위해 복사
    B = len(xb)
    # 샘플마다 동전 던지기 → True인 샘플만 너비 축(dim=3)으로 뒤집기 = 좌우 반전
    flip_lr = torch.rand(B) < 0.5
    xb[flip_lr] = xb[flip_lr].flip(3)
    # 높이 축(dim=2)으로 뒤집기 = 상하 반전
    flip_ud = torch.rand(B) < 0.5
    xb[flip_ud] = xb[flip_ud].flip(2)
    # 샘플별 밝기 배율 (B,1,1,1) → 모든 채널·픽셀에 같은 배율
    scale = torch.empty(B, 1, 1, 1).uniform_(0.4, 1.1)
    return (xb * scale).clamp(0.0, 1.0)               # 범위를 벗어난 값은 [0,1]로 자르기

q2_aug_note = (
    "(1) 1-3에서 test의 어두운 사진 비율이 train보다 훨씬 높았으므로, 밝기를 낮춘 사진을 학습에 섞어 분포 차이를 줄인다. "
    "(2) rust는 주황색, scab은 검은색 반점처럼 색이 곧 질병의 근거이므로, 색조를 바꾸면 레이블이 바뀐 사진을 만들 수 있다."
)
""")

run(r"""
def _check_2_2():
    torch.manual_seed(0)
    x = to_tensor(clean_img[:64]); before = x.clone()
    y = augment(x)
    assert torch.equal(x, before), "입력이 바뀌었습니다"
    assert y.shape == x.shape and float(y.min()) >= 0 and float(y.max()) <= 1
    assert not torch.equal(y, x), "증강이 적용되지 않았습니다"
    print(f"평균 밝기 원본 {float(x.mean()):.3f} → 증강 {float(y.mean()):.3f} | 서술 작성: {bool(q2_aug_note.strip())}")
    ps.show_images((y[:8].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8), ["aug"] * 8)
attempt("2-2", _check_2_2)
""")

md(r"""
### 2-3 불균형을 고려한 학습 데이터 재구성
- `stratified_split(labels, val_ratio=0.2, seed=42)`: 클래스 비율을 유지한 `(train_idx, val_idx)` 정수 배열
- `balanced_order(label_ids, generator)`: 정수 레이블 텐서를 받아, **클래스별 표본 수의 역수**를 가중치로 하는 복원추출로 같은 길이의 위치 텐서를 반환 (`torch.multinomial`). 한 epoch에 각 클래스가 비슷하게 뽑히게 됩니다.
""")

write(r"""
def stratified_split(labels, val_ratio=0.2, seed=42):
    return None, None  # 작성하세요

def balanced_order(label_ids, generator):
    return None  # 작성하세요
""", r"""
def stratified_split(labels, val_ratio=0.2, seed=42):
    idx = np.arange(len(labels))
    # stratify=labels → train/val 모두 원래 클래스 비율 유지 (소수 클래스가 val에서 사라지지 않게)
    return train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=labels)

def balanced_order(label_ids, generator):
    label_ids = torch.as_tensor(label_ids)
    counts = torch.bincount(label_ids).float()     # 클래스별 표본 수
    weights = 1.0 / counts[label_ids]              # 표본마다 '자기 클래스 수의 역수'
    # 가중치에 비례해 복원추출: 소수 클래스는 여러 번, 다수 클래스는 덜 뽑힌다
    return torch.multinomial(weights, num_samples=len(label_ids), replacement=True, generator=generator)
""")

run(r"""
def _check_2_3():
    tr, va = stratified_split(clean_lab)
    assert len(set(tr) & set(va)) == 0 and len(tr) + len(va) == len(clean_lab)
    y = torch.tensor([CLASSES.index(c) for c in clean_lab[tr]])
    order = balanced_order(y, torch.Generator().manual_seed(0))
    assert order.shape == y.shape
    share = torch.bincount(y[order], minlength=4).float() / len(order)
    print("원래 비율", (torch.bincount(y).float() / len(y)).numpy().round(3), "→ 재구성 비율", share.numpy().round(3))
    assert (share - 0.25).abs().max() < 0.05
    return tr, va
q2_split = attempt("2-3", _check_2_3)
# Q3에서 쓸 공통 입력 (실패 시 진행용 준비 함수로 대체)
if q2_split is None:
    q2_split = train_test_split(np.arange(len(clean_lab)), test_size=0.2, random_state=42, stratify=clean_lab)
tr_idx, va_idx = q2_split
_to_tensor = to_tensor if attempt("2-1 재확인", lambda: to_tensor(clean_img[:1]).shape) else ps.ready_to_tensor
X_all = _to_tensor(clean_img)
X_test = _to_tensor(q1_test_img)
Y_all = torch.tensor([CLASSES.index(c) for c in clean_lab])
""")

# ======================================================================  Q3
md(r"""
---
## Q3 전이학습·학습 범위 제어·손실 함수 (AI 모델링 및 최적화)

지정 모델: `ps.load_pretrained_model()` — 무늬 6종 분류로 사전학습한 작은 ResNet. **torchvision ResNet과 같은 속성 이름**(`conv1, bn1, layer1~layer4, avgpool, fc`)을 씁니다. `fc`는 사전학습 과제의 6클래스 출력입니다.
평가지표(가정): **macro-F1**, 특히 소수 클래스 `multiple`의 재현율. 제출: `제출물/q3_submission.csv` (`id,label`).
""")

run(r"""
print(ps.load_pretrained_model())
""")

md(r"""
### 3-1 전이학습을 위한 구조 변경
`build_classifier(num_classes)`: 사전학습 모델을 불러와 `fc`만 **입력 차원은 그대로, 출력은 num_classes**인 새 `nn.Linear`로 바꿉니다. 나머지 층의 사전학습 가중치는 유지합니다.
""")

write(r"""
def build_classifier(num_classes):
    model = ps.load_pretrained_model()
    # 작성하세요
    return model
""", r"""
def build_classifier(num_classes):
    model = ps.load_pretrained_model()           # 사전학습 가중치가 들어 있는 모델
    in_features = model.fc.in_features           # 기존 fc의 입력 차원(96)을 그대로 사용 → 하드코딩 금지
    model.fc = nn.Linear(in_features, num_classes)  # 출력만 우리 과제의 클래스 수로 교체(새로 초기화됨)
    return model
""")

run(r"""
def _check_3_1():
    m, ref = build_classifier(4), ps.load_pretrained_model()
    assert m(torch.zeros(2, 3, 48, 48)).shape == (2, 4)
    assert torch.equal(m.layer4.conv2.weight, ref.layer4.conv2.weight), "사전학습 가중치가 바뀌었습니다"
    print("fc:", m.fc)
attempt("3-1", _check_3_1)
""")

md(r"""
### 3-2 학습 범위 제어
- `set_train_scope(model, prefixes)`: 파라미터 **이름**이 `prefixes` 중 하나로 시작하면 `requires_grad=True`, 아니면 `False`. 학습할 파라미터 리스트를 반환합니다.
- `freeze_bn_outside(model, prefixes)`: 범위 밖의 모든 `nn.BatchNorm2d`를 `eval()` 모드로 둡니다. (`requires_grad=False`만으로는 BN의 `running_mean/var`가 학습 중 계속 바뀝니다.) `model.train()`을 호출한 **뒤에** 써야 합니다.
""")

write(r"""
def set_train_scope(model, prefixes):
    # 작성하세요
    return None

def freeze_bn_outside(model, prefixes):
    # 작성하세요
    pass
""", r"""
def set_train_scope(model, prefixes):
    prefixes = tuple(prefixes)
    for name, p in model.named_parameters():          # name 예: 'layer4.conv1.weight', 'fc.bias'
        p.requires_grad = name.startswith(prefixes)   # str.startswith는 튜플을 받아 '하나라도' 검사
    return [p for p in model.parameters() if p.requires_grad]

def freeze_bn_outside(model, prefixes):
    prefixes = tuple(prefixes)
    for name, module in model.named_modules():        # name 예: 'bn1', 'layer2.bn1'
        if isinstance(module, nn.BatchNorm2d) and not name.startswith(prefixes):
            module.eval()                             # running_mean/var 갱신 중지, 저장된 통계로 정규화
""")

run(r"""
def _check_3_2():
    m = build_classifier(4)
    params = set_train_scope(m, ["layer3", "layer4", "fc"])
    n_train = sum(p.numel() for p in params); n_all = sum(p.numel() for p in m.parameters())
    assert not m.conv1.weight.requires_grad and m.layer4.conv1.weight.requires_grad and m.fc.weight.requires_grad
    before = m.bn1.running_mean.clone()
    m.train(); freeze_bn_outside(m, ["layer3", "layer4", "fc"])
    assert not m.bn1.training and m.layer4.bn1.training
    with torch.no_grad():
        m(torch.randn(8, 3, 48, 48) * 3)
    assert torch.equal(before, m.bn1.running_mean), "범위 밖 BN 통계가 바뀌었습니다"
    print(f"학습 파라미터 {n_train:,} / 전체 {n_all:,}")
attempt("3-2", _check_3_2)
""")

md(r"""
### 3-3 손실 함수 구현 (불균형 대응)
`F.log_softmax`로 직접 구현하세요(`F.cross_entropy` 호출 금지).
- `weighted_cross_entropy(logits, target, weight)`: 표본 i의 손실 `-log p_i[y_i]`에 `weight[y_i]`를 곱해 더한 뒤 **가중치 합으로 나눈** 값 (PyTorch `CrossEntropyLoss(weight=...)`와 같은 정의)
- `focal_loss(logits, target, gamma=2.0)`: 평균 `-(1 - p_t)^gamma · log p_t` (잘 맞힌 쉬운 표본의 손실을 줄임). `gamma=0`이면 CE와 같아야 합니다.
- `class_weights(label_ids, num_classes)`: `전체 수 / (클래스 수 × 클래스별 수)` 텐서
""")

write(r"""
def weighted_cross_entropy(logits, target, weight):
    return None  # 작성하세요

def focal_loss(logits, target, gamma=2.0):
    return None  # 작성하세요

def class_weights(label_ids, num_classes):
    return None  # 작성하세요
""", r"""
def weighted_cross_entropy(logits, target, weight):
    log_p = F.log_softmax(logits, dim=1)                          # (B,C) 수치적으로 안정한 log 확률
    log_p_true = log_p.gather(1, target.unsqueeze(1)).squeeze(1)   # (B,) 정답 클래스의 log 확률만 뽑기
    w = weight[target]                                             # (B,) 표본마다 자기 클래스의 가중치
    return -(w * log_p_true).sum() / w.sum()                       # 가중 합 / 가중치 합 (PyTorch 정의)

def focal_loss(logits, target, gamma=2.0):
    log_p_true = F.log_softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
    p_true = log_p_true.exp()                                      # 정답 클래스 확률 p_t
    # (1-p_t)^gamma: 이미 잘 맞히는(p_t≈1) 표본은 거의 0 → 어려운 표본에 학습 집중
    return (-(1.0 - p_true) ** gamma * log_p_true).mean()

def class_weights(label_ids, num_classes):
    counts = torch.bincount(torch.as_tensor(label_ids), minlength=num_classes).float()
    return counts.sum() / (num_classes * counts)                  # 적은 클래스일수록 큰 가중치
""")

run(r"""
def _check_3_3():
    torch.manual_seed(0)
    lg, t, w = torch.randn(16, 4) * 3, torch.randint(0, 4, (16,)), torch.tensor([1.0, 2.0, 0.5, 4.0])
    assert torch.allclose(weighted_cross_entropy(lg, t, w), F.cross_entropy(lg, t, weight=w), atol=1e-6)
    assert torch.allclose(focal_loss(lg, t, gamma=0.0), F.cross_entropy(lg, t), atol=1e-6)
    assert focal_loss(lg, t, 2.0) < F.cross_entropy(lg, t)
    big = torch.tensor([[1000.0, -1000.0, 0.0, 0.0]])
    assert torch.isfinite(focal_loss(big, torch.tensor([1]))), "큰 logit에서 inf/NaN"
    print("클래스 가중치:", class_weights(Y_all[tr_idx], 4).numpy().round(2))
attempt("3-3", _check_3_3)
""")

md(r"""
### 3-4 학습·예측 함수
- `train_one_epoch(model, x, y, order, optimizer, loss_fn, scope, augment_fn, batch_size=64)`: `order`(위치 텐서) 순서로 배치를 만들고, 배치마다 **증강 → 정규화 → 순전파 → 손실 → 역전파 → 갱신**. 시작할 때 `model.train()` 후 `freeze_bn_outside`를 호출합니다. 평균 손실(float)을 반환합니다.
- `predict_logits(model, x, batch_size=256)`: 평가 모드·기울기 없이 정규화한 입력의 logits `(N, C)`를 반환합니다(증강 없음).

`augment_fn`, 정규화는 Q2 함수를 씁니다(실패 시 아래 셀이 준비 함수로 대체).
""")

write(r"""
def train_one_epoch(model, x, y, order, optimizer, loss_fn, scope, augment_fn, batch_size=64):
    total, n = 0.0, 0
    # 작성하세요
    return total / max(n, 1)

def predict_logits(model, x, batch_size=256):
    return None  # 작성하세요
""", r"""
def train_one_epoch(model, x, y, order, optimizer, loss_fn, scope, augment_fn, batch_size=64):
    model.train()                       # Dropout/BN을 학습 모드로
    freeze_bn_outside(model, scope)     # 단, 학습 범위 밖 BN은 다시 평가 모드로 (순서 중요: train() 다음)
    total, n = 0.0, 0
    for start in range(0, len(order), batch_size):
        b = order[start:start + batch_size]          # 이번 배치의 위치
        xb = _normalize(augment_fn(x[b]))            # 증강은 [0,1]에서, 정규화는 그 다음
        loss = loss_fn(model(xb), y[b])
        optimizer.zero_grad()                        # 이전 배치의 기울기 지우기
        loss.backward()                              # 기울기 계산 (requires_grad=True인 파라미터만)
        optimizer.step()                             # 파라미터 갱신
        total += loss.item() * len(b)                # 배치 크기로 가중 → 마지막 짧은 배치도 공정하게
        n += len(b)
    return total / max(n, 1)

def predict_logits(model, x, batch_size=256):
    model.eval()                                     # 모든 BN·Dropout 평가 모드
    outs = []
    with torch.no_grad():                            # 추론에는 기울기 불필요 → 메모리·시간 절약
        for start in range(0, len(x), batch_size):  # 마지막 짧은 배치까지 포함
            outs.append(model(_normalize(x[start:start + batch_size])))
    return torch.cat(outs)
""")

run(r"""
_normalize = normalize if attempt("2-1 재확인", lambda: normalize(torch.zeros(1, 3, 2, 2)).shape) else ps.ready_normalize
_augment = augment if attempt("2-2 재확인", lambda: augment(torch.zeros(2, 3, 4, 4)).shape) else ps.ready_augment
_balanced = balanced_order if attempt("2-3 재확인", lambda: balanced_order(torch.tensor([0, 1]), torch.Generator()).shape) else ps.ready_balanced_order

def _check_3_4():
    m = build_classifier(4)
    scope = ["layer3", "layer4", "fc"]
    opt = torch.optim.Adam(set_train_scope(m, scope), lr=1e-3)
    order = torch.as_tensor(tr_idx[:130])
    loss = train_one_epoch(m, X_all, Y_all, order, opt, nn.CrossEntropyLoss(), scope, _augment)
    out = predict_logits(m, X_all[:300], batch_size=128)
    assert isinstance(loss, float) and out.shape == (300, 4)
    print(f"짧은 학습 확인: loss {loss:.3f}")
attempt("3-4", _check_3_4)
""")

md(r"""
### 3-5 불균형 대응·분포 차이를 고려한 비교와 분석
아래 셀은 같은 조건(범위 `layer3, layer4, fc`, 10 epoch, lr 1e-3)으로 두 설정을 학습합니다(각 약 30초).
- **A**: 일반 CE + 무작위 순서
- **B**: `Q3_STRATEGY` — `"weighted_ce"`(3-3 가중 CE), `"focal"`(3-3 focal), `"sampler"`(2-3 균형 재구성) 중 하나

각 설정은 두 가지 validation으로 평가합니다.
- `val`: 일반 validation
- `val_dark`: 같은 validation 사진의 밝기를 0.45배로 낮춘 것 — 1-3에서 본 **test의 어두운 사진 비율**을 흉내 낸 분포 점검용 검증

`Q3_SELECT_BY`(`"val"` 또는 `"val_dark"`)를 정해 최종 모델을 고를 기준을 정하세요. 결과를 본 뒤 `q3_analysis`에 (1) 두 설정과 두 검증 결과 비교, (2) 기준과 최종 선택의 이유, (3) 남는 위험을 쓰세요.
""")

write(r"""
Q3_STRATEGY = ""   # 작성하세요
Q3_SELECT_BY = ""  # 작성하세요: "val" 또는 "val_dark"
""", r"""
Q3_STRATEGY = "sampler"     # 2-3에서 만든 균형 재구성: 소수 클래스(multiple)를 epoch마다 고르게 보여 줌
Q3_SELECT_BY = "val_dark"   # test에 어두운 사진이 많으므로(1-3) 그 조건을 흉내 낸 검증으로 고른다
""")

run(r"""
def run_config(strategy, epochs=10, lr=1e-3, scope=("layer3", "layer4", "fc")):
    ps.seed_everything(42)
    m = build_classifier(4)
    opt = torch.optim.Adam(set_train_scope(m, scope), lr=lr)
    y_tr = Y_all[tr_idx]
    if strategy == "weighted_ce":
        w = class_weights(y_tr, 4); loss_fn = lambda lg, t: weighted_cross_entropy(lg, t, w)
    elif strategy == "focal":
        loss_fn = focal_loss
    else:
        loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(42)
    tr_t = torch.as_tensor(tr_idx)
    t0 = time.time()
    for ep in range(epochs):
        order = tr_t[_balanced(y_tr, g)] if strategy == "sampler" else tr_t[torch.randperm(len(tr_t), generator=g)]
        train_one_epoch(m, X_all, Y_all, order, opt, loss_fn, list(scope), _augment)
    yv = Y_all[va_idx].numpy()
    res = dict(strategy=strategy, sec=round(time.time() - t0))
    for name, xv in [("val", X_all[va_idx]), ("val_dark", X_all[va_idx] * 0.45)]:
        pred = predict_logits(m, xv).argmax(1).numpy()
        res[f"{name}_f1"] = round(f1_score(yv, pred, average="macro"), 3)
        res[f"{name}_rec_multiple"] = round(recall_score(yv, pred, labels=[3], average="macro"), 2)
        if name == "val_dark":
            print(pd.DataFrame(confusion_matrix(yv, pred, labels=range(4)), index=CLASSES, columns=CLASSES))
    print(res)
    return m, res

q3_runs = {}
for strat in ["ce", Q3_STRATEGY]:
    if strat in ("ce", "weighted_ce", "focal", "sampler") and strat not in q3_runs:
        out = attempt(f"3-5 {strat}", run_config, strat)
        if out:
            q3_runs[strat] = out
""")

write(r"""
q3_analysis = ""  # 작성하세요
""", r"""
q3_analysis = (
    "(1) 일반 validation에서는 CE 0.960, sampler 0.938로 둘 다 높았지만, 어둡게 만든 validation에서는 CE 0.935(multiple 재현율 0.88), "
    "sampler 0.912(재현율 1.00)로 떨어졌다. 일반 validation은 test의 밝기 차이를 반영하지 못해 점수를 낙관적으로 보여 준다. "
    "(2) test에 어두운 사진이 많으므로 val_dark macro-F1을 기준으로 정했고, 그 기준에서 높은 CE를 최종 선택했다. "
    "(3) sampler는 소수 클래스 multiple을 하나도 놓치지 않았으므로, multiple을 놓치는 비용이 크다면 sampler가 더 나은 선택일 수 있다. "
    "multiple이 validation에 17장뿐이라 두 모델의 차이는 흔들릴 수 있다."
)
""")

run(r"""
def _save_q3():
    crit = (Q3_SELECT_BY if Q3_SELECT_BY in ("val", "val_dark") else "val") + "_f1"
    best = max(q3_runs.values(), key=lambda mr: mr[1][crit])        # validation 계열 기준으로만 선택
    print(pd.DataFrame([r for _, r in q3_runs.values()]).to_string(index=False))
    print("기준:", crit, "→ 최종:", best[1]["strategy"], "| 서술 작성:", bool(q3_analysis.strip()))
    pred = predict_logits(best[0], X_test).argmax(1).numpy()
    sample = csv("leaf", "sample_submission.csv")
    sub = sample[["id"]].copy()
    sub["label"] = pd.Series(np.array(CLASSES)[pred], index=q1_test_ids).reindex(sub["id"]).to_numpy()
    return ps.save_submission(sub, "q3_submission.csv", sample)
_ = attempt("3 제출", _save_q3)
""")

# ======================================================================  Q4
md(r"""
---
## Q4 등급 예측·하이퍼파라미터 탐색·성능 분석 (AI 모델링 및 최적화)

잎의 **녹병 심각도 등급 0~4**를 예측합니다(합성 자료, 반점 수가 많을수록 높은 등급). 등급은 순서가 있으므로 **회귀로 실수를 예측한 뒤 임계값으로 등급을 나눕니다.**
- `data/grade/train_images.npz`(1,000장), `train_labels.csv`(`id,grade`), `test_images.npz`(400장)
- 평가지표(가정): **QWK(quadratic weighted kappa)**. 제출: `제출물/q4_submission.csv` (`id,grade`)
- 학습은 제공 함수 `ps.train_regressor(model, x, y, train_idx, scope, lr, epochs)`가 담당합니다(증강·정규화 포함).
""")

run(r"""
g_ids, g_img = ps.load_images("grade", "train")
g_test_ids, g_test_img = ps.load_images("grade", "test")
g_y = csv("grade", "train_labels.csv").set_index("id").loc[g_ids, "grade"].to_numpy()
gX, gX_test = ps.ready_to_tensor(g_img), ps.ready_to_tensor(g_test_img)
gY = torch.tensor(g_y, dtype=torch.float32)
g_tr, g_va = train_test_split(np.arange(len(g_y)), test_size=0.2, random_state=42, stratify=g_y)
print(gX.shape, "등급 분포", np.bincount(g_y))
""")

md(r"""
### 4-1 회귀 헤드로 구조 변경
`build_regressor(dropout=0.2)`: 사전학습 모델의 `fc`를 `nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))`로 바꿉니다. 출력은 `(B, 1)`입니다.
""")

write(r"""
def build_regressor(dropout=0.2):
    model = ps.load_pretrained_model()
    # 작성하세요
    return model
""", r"""
def build_regressor(dropout=0.2):
    model = ps.load_pretrained_model()
    in_features = model.fc.in_features
    # 분류용 6출력 대신 실수 1개(등급 점수)를 내는 회귀 헤드. Dropout으로 작은 데이터 과적합 완화
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
    return model
""")

run(r"""
def _check_4_1():
    m = build_regressor()
    assert m(torch.zeros(3, 3, 48, 48)).shape == (3, 1)
    assert isinstance(m.fc[0], nn.Dropout) and m.fc[1].out_features == 1
attempt("4-1", _check_4_1)
""")

md(r"""
### 4-2 QWK 구현
`quadratic_weighted_kappa(y_true, y_pred, n_classes=5)`: 정수 등급 두 배열의 QWK. 라이브러리(`cohen_kappa_score`) 호출 없이 구현합니다.
1. 관측 행렬 `O[i, j]` = 실제 i, 예측 j인 개수
2. 가중치 `W[i, j] = (i - j)^2 / (n_classes - 1)^2`
3. 기대 행렬 `E = outer(실제 등급별 개수, 예측 등급별 개수) / 전체 수`
4. `QWK = 1 - Σ(W·O) / Σ(W·E)`
""")

write(r"""
def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    return None  # 작성하세요
""", r"""
def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    # 1) 관측 행렬: (실제, 예측) 쌍의 개수. np.add.at은 같은 칸에 여러 번 더하기를 정확히 처리
    O = np.zeros((n_classes, n_classes))
    np.add.at(O, (y_true, y_pred), 1)
    # 2) 가중치: 등급 차이의 제곱 → 멀리 틀릴수록 큰 벌점 (대각선은 0)
    i, j = np.indices((n_classes, n_classes))
    W = (i - j) ** 2 / (n_classes - 1) ** 2
    # 3) 기대 행렬: 실제·예측 분포가 서로 독립일 때(우연히 맞을 때) 기대되는 개수
    E = np.outer(O.sum(axis=1), O.sum(axis=0)) / O.sum()
    # 4) 1 - (관측 벌점 / 우연 벌점): 1이면 완벽, 0이면 우연 수준
    return float(1.0 - (W * O).sum() / (W * E).sum())
""")

run(r"""
def _check_4_2():
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(0)
    for _ in range(5):
        a, b = rng.integers(0, 5, 50), rng.integers(0, 5, 50)
        assert abs(quadratic_weighted_kappa(a, b) - cohen_kappa_score(a, b, weights="quadratic")) < 1e-9
    assert abs(quadratic_weighted_kappa([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) - 1.0) < 1e-12
    print("sklearn과 일치")
attempt("4-2", _check_4_2)
""")

md(r"""
### 4-3 연속 예측 → 등급: 임계값 최적화
- `to_grade(pred, thresholds)`: 오름차순 임계값 4개로 0~4 정수 등급 배열 (`np.digitize`)
- `fit_thresholds(pred, y, init=(0.5, 1.5, 2.5, 3.5), n_rounds=3, step=0.05)`: 임계값 하나씩 `[현재값-0.5, 현재값+0.5)` 범위를 `step` 간격으로 바꿔 보며 QWK가 가장 높은 값으로 갱신하기를 `n_rounds`번 반복(좌표 탐색). **정렬된 리스트**를 반환합니다.
""")

write(r"""
def to_grade(pred, thresholds):
    return None  # 작성하세요

def fit_thresholds(pred, y, init=(0.5, 1.5, 2.5, 3.5), n_rounds=3, step=0.05):
    th = list(init)
    # 작성하세요
    return sorted(th)
""", r"""
def to_grade(pred, thresholds):
    # digitize: 값이 몇 번째 구간에 있는지 → 임계값 4개면 0~4 다섯 구간
    return np.digitize(np.asarray(pred), np.sort(np.asarray(thresholds)))

_qwk = None  # 아래 실행 셀에서 4-2 함수(또는 sklearn)로 채워짐

def fit_thresholds(pred, y, init=(0.5, 1.5, 2.5, 3.5), n_rounds=3, step=0.05):
    th = list(init)
    for _ in range(n_rounds):                       # 한 바퀴 돌 때마다 임계값 4개를 차례로 개선
        for k in range(len(th)):
            candidates = np.arange(th[k] - 0.5, th[k] + 0.5, step)
            def score(t):                           # k번째만 t로 바꾼 임계값의 QWK
                trial = th[:k] + [t] + th[k + 1:]
                return _qwk(y, to_grade(pred, trial))
            th[k] = float(max(candidates, key=score))  # 가장 QWK가 높은 값 채택
    return sorted(th)
""")

run(r"""
from sklearn.metrics import cohen_kappa_score
_qwk = quadratic_weighted_kappa if attempt("4-2 재확인", lambda: quadratic_weighted_kappa([0, 1], [0, 1])) is not None \
       else (lambda a, b: cohen_kappa_score(a, b, weights="quadratic"))

def _check_4_3():
    assert to_grade([0.2, 0.6, 3.9, 9], [0.5, 1.5, 2.5, 3.5]).tolist() == [0, 1, 4, 4]
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 300); p = y * 0.7 + rng.normal(0, 0.3, 300)       # 일부러 작게 눌린 예측
    th = fit_thresholds(p, y)
    assert th == sorted(th) and len(th) == 4
    print(f"고정 임계값 QWK {_qwk(y, to_grade(p, [0.5, 1.5, 2.5, 3.5])):.3f} → 최적화 {_qwk(y, to_grade(p, th)):.3f} | {np.round(th, 2)}")
attempt("4-3", _check_4_3)
""")

md(r"""
### 4-4 하이퍼파라미터 탐색
`grid_search(grid, trial_fn, budget_sec=240)`: `grid`는 `{"lr": [...], "scope": [...]}` 같은 dict. **모든 조합**(`itertools.product`)마다 `trial_fn(**params)`를 호출해 반환된 validation QWK를 기록합니다. 누적 시간이 `budget_sec`를 넘으면 남은 조합은 건너뜁니다. `(결과 DataFrame(점수 내림차순), 최고 params dict)`를 반환합니다. DataFrame 열: 각 매개변수 이름, `val_qwk`, `sec`.

`trial_fn`(아래 제공)은 5 epoch 학습 → validation 예측 → 4-3 임계값 최적화 → QWK를 반환합니다. 아래 셀은 조합 2개와 비교 실험(fc만 학습) 1개로 약 1분입니다.
""")

write(r"""
def grid_search(grid, trial_fn, budget_sec=240):
    rows = []
    # 작성하세요
    return None, None
""", r"""
def grid_search(grid, trial_fn, budget_sec=240):
    rows = []
    names = list(grid)                               # 매개변수 이름 순서 고정
    spent = 0.0
    for values in itertools.product(*(grid[n] for n in names)):   # 모든 조합
        params = dict(zip(names, values))
        if spent > budget_sec:                       # 시간 예산 초과 → 남은 조합은 건너뜀
            print("예산 초과로 건너뜀:", params)
            continue
        t0 = time.time()
        score = trial_fn(**params)
        sec = time.time() - t0
        spent += sec
        rows.append({**params, "val_qwk": score, "sec": round(sec, 1)})
        print(rows[-1])
    result = pd.DataFrame(rows).sort_values("val_qwk", ascending=False).reset_index(drop=True)
    best = {n: result.loc[0, n] for n in names}      # 1등 행의 매개변수
    return result, best
""")

run(r"""
q4_cache = {}
def trial_fn(lr, scope, epochs=5):
    ps.seed_everything(42)
    m = ps.train_regressor(build_regressor(), gX, gY, g_tr, scope=scope, lr=lr, epochs=epochs)
    pv = ps.predict_regressor(m, gX[g_va])
    th = fit_thresholds(pv, g_y[g_va])
    q4_cache[(lr, scope)] = (m, th)                  # 최종 예측에 다시 쓰기 위해 보관
    return _qwk(g_y[g_va], to_grade(pv, th))

GRID = {"lr": [1e-3, 3e-3], "scope": [("layer3", "layer4", "fc")]}
q4_out = attempt("4-4", grid_search, GRID, trial_fn)
q4_fc_only = attempt("4-4 비교: fc만", trial_fn, 1e-3, ("fc",))
if q4_out and q4_out[0] is not None:
    print(q4_out[0]); print("최고:", q4_out[1], "| fc만 학습 QWK:", q4_fc_only)
""")

md(r"""
### 4-5 성능 분석과 개선안
아래 셀은 최고 설정의 validation 혼동 행렬과 등급별 재현율을 보여 줍니다. `q4_analysis`에 (1) 어떤 등급끼리 주로 혼동하는지와 그 이유, (2) 탐색 결과(학습률·학습 범위·임계값)에서 얻은 결론, (3) **다음에 시도할 개선안 2가지**를 쓰세요.
""")

run(r"""
def _analyze_4():
    best = q4_out[1] if q4_out and q4_out[1] else {"lr": 1e-3, "scope": ("layer3", "layer4", "fc")}
    key = (best["lr"], tuple(best["scope"]))
    if key not in q4_cache:
        trial_fn(*key)
    m, th = q4_cache[key]
    pv = ps.predict_regressor(m, gX[g_va])
    for name, t in [("고정 0.5,1.5,..", [0.5, 1.5, 2.5, 3.5]), ("최적화", th)]:
        print(f"{name:14s} 임계값 QWK {_qwk(g_y[g_va], to_grade(pv, t)):.3f}")
    gv = to_grade(pv, th)
    print(pd.DataFrame(confusion_matrix(g_y[g_va], gv, labels=range(5)), index=[f"실제{k}" for k in range(5)], columns=[f"예측{k}" for k in range(5)]))
    print("등급별 재현율", np.round(recall_score(g_y[g_va], gv, labels=range(5), average=None), 2))
    pred = to_grade(ps.predict_regressor(m, gX_test), th)
    sample = csv("grade", "sample_submission.csv")
    sub = sample[["id"]].copy()
    sub["grade"] = pd.Series(pred, index=g_test_ids).reindex(sub["id"]).to_numpy().astype(int)
    return ps.save_submission(sub, "q4_submission.csv", sample)
_ = attempt("4-5", _analyze_4)
""")

write(r"""
q4_analysis = ""  # 작성하세요
""", r"""
q4_analysis = (
    "(1) 혼동은 거의 모두 인접 등급 사이(실제 2→예측 1, 실제 3→예측 2 등)에서 생기며, 2칸 이상 틀린 경우는 없다. 반점 수 구간이 맞닿아 있고 반점이 겹치거나 어두운 사진에서는 개수를 세기 어렵기 때문이다. "
    "(2) fc만 학습하면 QWK 0.83으로 크게 낮아 layer3·layer4까지 푸는 것이 핵심이었고, 학습률은 3e-3(0.965)이 1e-3(0.949)보다 약간 높았다. "
    "고정 임계값(0.934)보다 validation으로 맞춘 임계값(0.965)이 QWK를 올렸다(같은 validation으로 맞췄으므로 약간 낙관적일 수 있다). "
    "(3) 개선안: ① epoch을 늘리고 전체 층을 낮은 학습률로 미세조정, ② 임계값을 교차검증 OOF 예측으로 맞춰 낙관 편향 줄이기."
)
""")

# ======================================================================  Q5
md(r"""
---
## Q5 정보 추출 프롬프트와 구조화 출력 (프롬프트 엔지니어링)

가상의 진료 메모(`data/notes/notes.csv`: `id, split, note`)에서 정보를 뽑아 **소프트웨어가 바로 쓸 수 있는 JSON**으로 만듭니다. 메모는 모두 합성된 가상 자료입니다.
- `dev` 40개는 정답(`dev_labels.json`)이 있고, `test` 30개는 정답이 없습니다.
- **LLM 호출 대신 `ps.mock_llm(note_id)`가 미리 만든 응답 문자열을 돌려줍니다.** 이것은 실제 LLM이 아니며 여러분의 프롬프트와 무관합니다. 실제 LLM 응답에서 흔한 문제(코드 블록, 앞뒤 설명문, 문자열 숫자, 단위 미변환, 'unknown', 여분의 키, 깨진 JSON)를 일부러 섞었습니다.
- 평가(가정): 필드 정확도와 스키마 위반 0건. 제출: `제출물/q5_submission.jsonl` (줄마다 `{"id":..., "record":{...}}`)

**출력 스키마(필드 8개, 모르면 null)**

| 필드 | 자료형 | 규칙 |
|---|---|---|
| age | 정수 또는 null | 나이(세) |
| sex | `"M"`, `"F"` 또는 null | |
| chief_complaint | 문자열 enum | `fever, cough, headache, abdominal_pain, rash` 중 하나 |
| symptom_days | 정수 또는 null | 증상 기간(일). 일주일=7, 2주=14. "며칠"처럼 불확실하면 null |
| temperature_c | 실수 또는 null | 섭씨, 소수 첫째 자리. 화씨(F)는 섭씨로 변환. "미열"은 null |
| medications | 배열 | 원소 `{"name": 성분명 enum, "dose_mg": 실수|null, "times_per_day": 정수|null}`. g는 mg로. "필요시"는 null |
| allergy | true/false/null | 언급 없으면 null |
| follow_up_days | 정수 또는 null | 재방문까지 일수. 조건부("호전 없으면")는 null |
""")

run(r"""
notes = csv("notes", "notes.csv")
dev_gold = json.loads((ps.DATA / "notes" / "dev_labels.json").read_text(encoding="utf-8"))
dev = notes[notes.split == "dev"].reset_index(drop=True)
test_notes = notes[notes.split == "test"].reset_index(drop=True)

SCHEMA_KEYS = ["age", "sex", "chief_complaint", "symptom_days", "temperature_c", "medications", "allergy", "follow_up_days"]
COMPLAINTS = ["fever", "cough", "headache", "abdominal_pain", "rash"]
GENERICS = ["acetaminophen", "ibuprofen", "amoxicillin", "cetirizine"]
# 정규화 사전(제공)
SEX_MAP = {"m": "M", "male": "M", "남": "M", "남성": "M", "남자": "M", "남아": "M",
           "f": "F", "female": "F", "여": "F", "여성": "F", "여자": "F", "여아": "F"}
COMPLAINT_MAP = {"발열": "fever", "열": "fever", "열이 남": "fever", "기침": "cough", "마른기침": "cough",
                 "두통": "headache", "머리가 아픔": "headache", "복통": "abdominal_pain", "배가 아픔": "abdominal_pain",
                 "발진": "rash", "두드러기": "rash", **{c: c for c in COMPLAINTS}}
DRUG_MAP = {"타이레놀": "acetaminophen", "애드빌": "ibuprofen", "이부프로펜": "ibuprofen",
            "아목시실린": "amoxicillin", "세티리진": "cetirizine", **{g: g for g in GENERICS}}
BOOL_MAP = {"true": True, "yes": True, "y": True, "있음": True, "false": False, "no": False, "n": False, "없음": False}
UNKNOWN = {"", "unknown", "n/a", "na", "none", "null", "미상", "모름", "며칠", "미기재", "미열"}

print(dev.note[0]); print(ps.mock_llm(dev.id[0]))
""")

md(r"""
### 5-1 프롬프트 설계
`build_prompt(note)`: 언어모델에 보낼 **자연어 프롬프트 문자열**을 반환합니다. 다음을 모두 담으세요.
- 역할과 작업, 출력은 **JSON 객체 하나만**(설명문·코드 블록 금지)
- 8개 필드 이름·자료형·허용값(위 표)
- 정규화 규칙: 섭씨·소수 첫째 자리, 화씨 변환, 일 단위, mg 단위, 성분명, 성별 M/F
- **미확정 정보 처리**: 메모에 없거나 불확실하면 추측하지 말고 `null`
- 예시 1개(few-shot), 메모 원문을 구분 기호로 감싸서 넣기

확인 셀의 `ps.check_prompt`는 빠진 요소만 점검하며 프롬프트 품질을 채점하지 않습니다.
""")

write(r"""
def build_prompt(note):
    prompt = ""  # 작성하세요
    return prompt
""", r"""
# 예시 1개(few-shot): 규칙이 실제로 어떻게 적용되는지 보여 주어 형식 오류를 줄인다
FEW_SHOT_NOTE = "여, 34세. 2주 전부터 마른기침. 체온 100.4F. 아목시실린 0.5g 하루 세 번 처방. 호전 없으면 재방문."
FEW_SHOT_JSON = {"age": 34, "sex": "F", "chief_complaint": "cough", "symptom_days": 14, "temperature_c": 38.0,
                 "medications": [{"name": "amoxicillin", "dose_mg": 500, "times_per_day": 3}],
                 "allergy": None, "follow_up_days": None}

def build_prompt(note):
    prompt = f'''당신은 진료 메모에서 정보를 추출해 JSON으로 변환하는 도구입니다.
아래 [메모]를 읽고 JSON 객체 하나만 출력하세요. 설명 문장, 코드 블록(```), 추가 키는 출력하지 마세요.

[출력 스키마] 키 8개를 모두 포함합니다.
- "age": 정수(세) 또는 null
- "sex": "M" 또는 "F" 또는 null (남/남성/남아 → "M", 여/여성/여아 → "F")
- "chief_complaint": "fever", "cough", "headache", "abdominal_pain", "rash" 중 하나
- "symptom_days": 정수(일) 또는 null (일주일=7, 2주=14)
- "temperature_c": 섭씨(°C) 실수, 소수 첫째 자리 또는 null (화씨 F는 (F-32)×5/9로 변환)
- "medications": 배열. 원소는 {{"name": 성분명, "dose_mg": 숫자(mg) 또는 null, "times_per_day": 정수 또는 null}}
  성분명은 "acetaminophen"(타이레놀), "ibuprofen"(애드빌·이부프로펜), "amoxicillin", "cetirizine" 중 하나. g는 mg로 변환(0.5g=500)
- "allergy": true, false 또는 null
- "follow_up_days": 재방문까지 일수(정수) 또는 null

[미확정 정보 규칙]
메모에 없거나 불확실한 정보("며칠 전", "미열", "필요시 복용", "호전 없으면 재방문", "나이 미상")는 추측하지 말고 null로 둡니다.
숫자는 따옴표 없이 숫자 자료형으로, 참/거짓은 true/false로 씁니다.

[예시]
메모: {FEW_SHOT_NOTE}
출력: {json.dumps(FEW_SHOT_JSON, ensure_ascii=False)}

[메모]
<<<
{note}
>>>
출력:'''
    return prompt
""")

run(r"""
def _check_5_1():
    p = build_prompt(dev.note[0])
    missing = ps.check_prompt(p, dev.note[0], SCHEMA_KEYS)
    print(p[:600] + ("..." if len(p) > 600 else ""))
    print("\n빠진 요소:", missing or "없음", "| 길이", len(p))
    assert not missing
attempt("5-1", _check_5_1)
""")

md(r"""
### 5-2 JSON 스키마
`RECORD_SCHEMA`를 **JSON Schema** 형식의 dict로 작성하세요. 모든 키 필수(`required`), 여분 키 금지(`additionalProperties: false`), 위 표의 자료형·enum·null 허용을 반영합니다. `medications`의 원소도 객체 스키마로 정의합니다. (확인 셀은 `jsonschema` 라이브러리가 있으면 dev 정답 40개가 모두 통과하는지, 잘못된 예가 걸러지는지 봅니다.)
""")

write(r"""
RECORD_SCHEMA = {}  # 작성하세요
""", r"""
RECORD_SCHEMA = {
    "type": "object",
    "required": SCHEMA_KEYS,                     # 8개 키 모두 반드시 있어야 함
    "additionalProperties": False,               # 스키마에 없는 키(예: "comment")는 위반
    "properties": {
        "age": {"type": ["integer", "null"], "minimum": 0, "maximum": 120},
        "sex": {"enum": ["M", "F", None]},       # enum에 None(null)을 넣어 null 허용
        "chief_complaint": {"enum": COMPLAINTS},
        "symptom_days": {"type": ["integer", "null"], "minimum": 0},
        "temperature_c": {"type": ["number", "null"], "minimum": 30, "maximum": 45},  # 섭씨 범위로 화씨 실수 차단
        "medications": {
            "type": "array",
            "items": {                           # 배열 원소 하나하나의 스키마
                "type": "object",
                "required": ["name", "dose_mg", "times_per_day"],
                "additionalProperties": False,
                "properties": {
                    "name": {"enum": GENERICS},
                    "dose_mg": {"type": ["number", "null"], "exclusiveMinimum": 0},
                    "times_per_day": {"type": ["integer", "null"], "minimum": 1, "maximum": 6},
                },
            },
        },
        "allergy": {"type": ["boolean", "null"]},
        "follow_up_days": {"type": ["integer", "null"], "minimum": 0},
    },
}
""")

run(r"""
def _check_5_2():
    try:
        import jsonschema
    except ImportError:
        print("jsonschema 없음: 구조만 확인합니다."); assert RECORD_SCHEMA.get("type") == "object"; return
    v = jsonschema.Draft202012Validator(RECORD_SCHEMA)
    bad_gold = [k for k, r in dev_gold.items() if list(v.iter_errors(r))]
    assert not bad_gold, f"정답인데 위반: {bad_gold[:3]}"
    g = copy.deepcopy(next(iter(dev_gold.values())))
    for broken in [{**g, "age": "45"}, {**g, "comment": "x"}, {**g, "temperature_c": 101.3},
                   {k: g[k] for k in SCHEMA_KEYS[:-1]}, {**g, "allergy": "없음"}, {**g, "sex": "남성"}]:
        assert list(v.iter_errors(broken)), f"걸러야 할 예가 통과: {broken}"
    print("dev 정답 40개 통과, 잘못된 예 6개 차단")
attempt("5-2", _check_5_2)
""")

md(r"""
### 5-3 응답에서 JSON 꺼내기
`extract_json(raw)`: LLM 응답 문자열에서 JSON 객체(dict)를 꺼냅니다. 실패하면 `None`. 다음 순서로 시도하세요.
1. 첫 `{`부터 마지막 `}`까지 잘라내기(코드 블록·앞뒤 설명문 제거)
2. `json.loads`
3. 실패하면 `}`나 `]` 앞의 끝 쉼표를 정규식으로 지우고 다시 `json.loads`
4. 그래도 실패하면 `ast.literal_eval`(파이썬 dict 표기 대비). 결과가 dict가 아니면 None
""")

write(r"""
import ast

def extract_json(raw):
    return None  # 작성하세요
""", r"""
import ast

def extract_json(raw):
    if not isinstance(raw, str):
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:                 # 중괄호 쌍이 없으면(응답이 끊김 등) 복구 불가
        return None
    body = raw[start:end + 1]                       # 앞뒤 설명문·```json 표시 제거
    candidates = [body, re.sub(r",\s*([}\]])", r"\1", body)]   # 원본, 끝 쉼표 제거본
    for text in candidates:
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue                                # 다음 후보로
    try:                                            # {'age': 3, 'allergy': None} 같은 파이썬 표기
        obj = ast.literal_eval(body)
        return obj if isinstance(obj, dict) else None
    except (ValueError, SyntaxError):
        return None
""")

run(r"""
def _check_5_3():
    cases = ['```json\n{"a": 1}\n```', '결과입니다.\n{"a": 1}\n감사합니다.', '{"a": [1, 2,], "b": 3,\n}',
             "{'a': None, 'b': True}", '{"a": 1, "b": ', "JSON 없음"]
    want = [{"a": 1}, {"a": 1}, {"a": [1, 2], "b": 3}, {"a": None, "b": True}, None, None]
    for c, w in zip(cases, want):
        assert extract_json(c) == w, f"{c!r} → {extract_json(c)!r}"
    ok = sum(extract_json(ps.mock_llm(i)) is not None for i in dev.id)
    print(f"dev 응답 {len(dev)}개 중 {ok}개 파싱 성공")
attempt("5-3", _check_5_3)
""")

md(r"""
### 5-4 정규화와 미확정 정보 처리
`normalize_record(obj)`: 꺼낸 dict를 **스키마를 지키는 레코드**로 바꿉니다. `obj`가 None이면 모든 필드 null(`medications`는 `[]`)인 레코드를 반환합니다.
- 키: 스키마의 8개만 남기고, 없는 키는 null
- 문자열은 앞뒤 공백 제거·소문자로 비교. `UNKNOWN`에 속하면 null
- `age`, `symptom_days`, `follow_up_days`, `times_per_day`: 문자열에서 숫자를 뽑아 정수. `"1주"`→7, `"2주"`→14, `"일주일"`→7
- `temperature_c`: 실수로 바꾸고 **45보다 크면 화씨로 보고 섭씨로 변환**, 소수 첫째 자리 반올림
- `dose_mg`: `"0.5g"`→500, `"500mg"`/`"500 mg"`→500
- `sex`, `chief_complaint`, 약 이름, `allergy`: 제공한 사전(`SEX_MAP` 등)으로 변환. 사전에 없으면 null (약은 이름이 null이면 그 약을 제외)
- **불리언과 정수 구분**: `True`는 파이썬에서 `int`이기도 하므로 정수 필드에 불리언이 오면 null
""")

write(r"""
def normalize_record(obj):
    rec = {}
    # 작성하세요
    return rec
""", r"""
def _is_unknown(v):
    # None 이거나 '모름'을 뜻하는 문자열이면 미확정
    return v is None or (isinstance(v, str) and v.strip().lower() in UNKNOWN)

def _to_number(v):
    # 숫자 또는 숫자가 들어 있는 문자열 → float. bool은 숫자로 보지 않는다(True==1 함정)
    if isinstance(v, bool) or _is_unknown(v):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d+(\.\d+)?", str(v))
    return float(m.group()) if m else None

def _to_int(v):
    x = _to_number(v)
    return int(round(x)) if x is not None else None

def _to_days(v):
    # '1주'/'일주일'/'2주' 같은 주 단위를 일로 변환, 그 외는 숫자 그대로
    if isinstance(v, str):
        s = v.strip()
        if "일주일" in s:
            return 7
        if "주" in s and _to_number(s) is not None:
            return int(_to_number(s) * 7)
    return _to_int(v)

def _to_temp_c(v):
    x = _to_number(v)
    if x is None:
        return None
    if x > 45:                       # 사람 체온이 45°C를 넘을 수 없으므로 화씨로 판단
        x = (x - 32) * 5 / 9
    return round(x, 1)

def _to_dose_mg(v):
    x = _to_number(v)
    if x is None:
        return None
    if isinstance(v, str) and re.search(r"\d\s*g\b", v.strip().lower()) and "mg" not in v.lower():
        x *= 1000                    # '0.5g' → 500mg (단, 'mg'의 g와 구분)
    return x

def _map(v, table):
    # 사전 변환: 문자열이 아니거나 사전에 없으면 null
    if not isinstance(v, str) or _is_unknown(v):
        return None
    return table.get(v.strip().lower(), table.get(v.strip()))

def _to_bool(v):
    if isinstance(v, bool):
        return v
    return _map(v, BOOL_MAP)

def normalize_record(obj):
    obj = obj if isinstance(obj, dict) else {}       # 파싱 실패(None) → 빈 dict → 전부 null
    rec = {}
    rec["age"] = _to_int(obj.get("age"))
    rec["sex"] = obj.get("sex") if obj.get("sex") in ("M", "F") else _map(obj.get("sex"), SEX_MAP)
    rec["chief_complaint"] = _map(obj.get("chief_complaint"), COMPLAINT_MAP)
    rec["symptom_days"] = _to_days(obj.get("symptom_days"))
    rec["temperature_c"] = _to_temp_c(obj.get("temperature_c"))
    meds = []
    for m in obj.get("medications") or []:            # None이거나 없으면 빈 배열
        if not isinstance(m, dict):
            continue
        name = _map(m.get("name"), DRUG_MAP)
        if name is None:                              # 성분을 알 수 없는 약은 제외
            continue
        meds.append({"name": name, "dose_mg": _to_dose_mg(m.get("dose_mg")),
                     "times_per_day": _to_int(m.get("times_per_day"))})
    rec["medications"] = meds
    rec["allergy"] = _to_bool(obj.get("allergy"))
    rec["follow_up_days"] = _to_days(obj.get("follow_up_days"))
    return {k: rec[k] for k in SCHEMA_KEYS}            # 스키마 키 순서로, 여분 키 없음
""")

run(r"""
def _check_5_4():
    raw = {"age": "45세", "sex": "남성", "chief_complaint": "Cough", "symptom_days": "1주", "temperature_c": 101.3,
           "medications": [{"name": "타이레놀", "dose_mg": "0.5g", "times_per_day": "3회"}, {"name": "모르는약"}],
           "allergy": "없음", "comment": "x"}
    r = normalize_record(raw)
    assert r == {"age": 45, "sex": "M", "chief_complaint": "cough", "symptom_days": 7, "temperature_c": 38.5,
                 "medications": [{"name": "acetaminophen", "dose_mg": 500.0, "times_per_day": 3}],
                 "allergy": False, "follow_up_days": None}, r
    assert normalize_record({"age": True, "symptom_days": "며칠", "temperature_c": "N/A"})["age"] is None
    assert normalize_record(None)["medications"] == [] and list(normalize_record(None)) == SCHEMA_KEYS
    print("정규화 규칙 확인 통과")
attempt("5-4", _check_5_4)
""")

md(r"""
### 5-5 자료형 검사와 평가·제출
`type_errors(rec)`: 라이브러리 없이 레코드의 **자료형 위반 목록**(문자열 리스트)을 반환합니다. 확인할 것: 키 집합이 스키마와 같음, 정수 필드는 `int`(불리언 제외)·null, `temperature_c`와 `dose_mg`는 `int`/`float`(불리언 제외)·null, `sex`·`chief_complaint`·약 이름은 허용값, `allergy`는 `bool`·null, `medications`는 list.

아래 셀은 dev 40개에 `응답 → extract_json → normalize_record → type_errors` 파이프라인을 돌려 **필드 정확도**를 계산하고, test 30개를 제출 파일로 저장합니다.
""")

write(r"""
def type_errors(rec):
    errors = []
    # 작성하세요
    return errors
""", r"""
def type_errors(rec):
    errors = []
    if set(rec) != set(SCHEMA_KEYS):                        # 빠진 키·여분 키 모두 위반
        errors.append(f"키 불일치: {sorted(set(rec) ^ set(SCHEMA_KEYS))}")
        return errors
    is_int = lambda v: isinstance(v, int) and not isinstance(v, bool)        # bool은 int의 하위 클래스!
    is_num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    for k in ["age", "symptom_days", "follow_up_days"]:
        if rec[k] is not None and not is_int(rec[k]):
            errors.append(f"{k}: 정수/null이어야 함 ({rec[k]!r})")
    if rec["temperature_c"] is not None and not is_num(rec["temperature_c"]):
        errors.append(f"temperature_c: 숫자/null이어야 함 ({rec['temperature_c']!r})")
    if rec["sex"] not in ("M", "F", None):
        errors.append(f"sex: 허용값 아님 ({rec['sex']!r})")
    if rec["chief_complaint"] not in COMPLAINTS:
        errors.append(f"chief_complaint: 허용값 아님 ({rec['chief_complaint']!r})")
    if rec["allergy"] is not None and not isinstance(rec["allergy"], bool):
        errors.append(f"allergy: bool/null이어야 함 ({rec['allergy']!r})")
    if not isinstance(rec["medications"], list):
        errors.append("medications: 배열이어야 함")
    else:
        for i, m in enumerate(rec["medications"]):
            if m.get("name") not in GENERICS:
                errors.append(f"medications[{i}].name: 허용값 아님")
            if m.get("dose_mg") is not None and not is_num(m["dose_mg"]):
                errors.append(f"medications[{i}].dose_mg: 숫자/null이어야 함")
            if m.get("times_per_day") is not None and not is_int(m["times_per_day"]):
                errors.append(f"medications[{i}].times_per_day: 정수/null이어야 함")
    return errors
""")

run(r"""
_extract = extract_json if attempt("5-3 재확인", lambda: extract_json('{"a": 1}')) == {"a": 1} else (lambda raw: None)

def _field_match(a, b, k):
    if k == "medications":
        key = lambda m: (m["name"], None if m["dose_mg"] is None else float(m["dose_mg"]), m["times_per_day"])
        return sorted(map(key, a[k]), key=str) == sorted(map(key, b[k]), key=str)
    return a[k] == b[k]

def _run_5_5():
    assert type_errors({**normalize_record(None), "age": True}) != [], "bool을 정수로 통과시킴"
    assert type_errors({**normalize_record(None), "chief_complaint": "cough"}) == []
    rows, n_err, n_parse_fail = [], 0, 0
    for nid in dev.id:
        obj = _extract(ps.mock_llm(nid))
        n_parse_fail += obj is None
        rec = normalize_record(obj)
        if obj is not None and type_errors(rec):    # 파싱 실패 레코드는 chief_complaint가 null이라 제외
            n_err += 1
        rows.append({k: _field_match(rec, dev_gold[nid], k) for k in SCHEMA_KEYS})
    acc = pd.DataFrame(rows).mean()
    print("필드별 정확도\n", acc.round(3).to_string())
    print(f"전체 필드 정확도 {acc.mean():.3f} | 파싱 실패 {n_parse_fail}건 | 자료형 위반 레코드 {n_err}건")
    raw_acc = np.mean([[ (_extract(ps.mock_llm(n)) or {}).get(k) == dev_gold[n][k] for k in SCHEMA_KEYS] for n in dev.id])
    print(f"(참고) 정규화 없이 응답 그대로 쓸 때 필드 정확도 {raw_acc:.3f}")
    out = [{"id": nid, "record": normalize_record(_extract(ps.mock_llm(nid)))} for nid in test_notes.id]
    return ps.save_jsonl(out, "q5_submission.jsonl")
_ = attempt("5-5", _run_5_5)
""")

md(r"""
---
## 제출 전 점검 (코드 리뷰 대비)
- `제출물/`: `q3_submission.csv`, `q4_submission.csv`, `q5_submission.jsonl`
- 서술 5개: `q1_visual_note`, `q2_aug_note`, `q3_analysis`, `q4_analysis` (+ 5-1 프롬프트 자체)
- 코드 리뷰 관점: 함수마다 한 가지 일, 의미 있는 변수 이름, 핵심 결정에 주석, 하드코딩된 크기·개수 없음, seed 고정, test 정답 미사용
- 커널 재시작 후 처음부터 끝까지 실행되는지, 출력이 저장되었는지
- 풀이 후 `python 정답_채점용/score_local.py`로 확인하고 `정답_채점용/채점기준.md`의 요구사항 표와 비교하세요.
""")


# ======================================================================
def build(answer):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, src, ans in CELLS:
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(src))
        else:
            nb.cells.append(nbf.v4.new_code_cell(ans if (answer and ans) else src))
    if answer:
        nb.cells[0].source += ("\n\n> **정답·해설용입니다.** `[정답 예시]` 셀은 줄마다 이유를 주석으로 단 풀이 하나이며, "
                               "같은 요구사항을 만족하는 다른 코드도 정답입니다.")
    return nb


if __name__ == "__main__":
    nbf.write(build(False), ROOT / "Lv3_공지반영_v1_문제.ipynb")
    nbf.write(build(True), ROOT / "정답_채점용" / "Lv3_공지반영_v1_정답.ipynb")
    print("cells:", len(CELLS))
