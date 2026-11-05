"""v2 노트북 생성기. 실행: python -I build_v2.py

작성 셀마다 세 버전을 둡니다.
- A(빈칸): 핵심 자리만 ___ 로 비운 판 (Lv2처럼 한두 단어·한 줄)
- B(직접구현): 함수 이름·인자·설명만 있는 판 (실전에 가까움)
- 정답: 줄마다 이유를 주석으로 단 풀이 예시
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
CELLS = []


def md(text):
    CELLS.append(("md", text.strip("\n"), None, None))


def run(code):
    CELLS.append(("run", "# [실행만 하세요]\n" + code.strip("\n"), None, None))


def write(blank, stub, answer):
    CELLS.append(("write", blank.strip("\n"), stub.strip("\n"), answer.strip("\n")))


# ======================================================================  머리말
md(r"""
# [claude] Lv3 모의평가 v2 — 11/5 공지·FAQ 반영 절충판

**창작 모의문제입니다. 실제 시험 문제가 아닙니다.** 먼저 `README.md`와 `제출가이드.md`를 읽으세요.

| 문항 | 공지 분야 | 유형 (FAQ: 두 방식 혼합) | 권장 시간 |
|---|---|---|---:|
| Q1 | 데이터 분석/전처리 (이미지) | 지시사항 구현 — 구성·품질·분포 점검, 육안 검사, 정제 | 35분 |
| Q2 | 데이터 분석/전처리 (이미지) | 지시사항 구현 — transforms, Dataset, 불균형 샘플러 | 35분 |
| Q3 | AI 모델링 및 최적화 (이미지) | 구현 + **성능 기준** — 구조 변경, 학습 범위, 손실 함수, 학습 루프, 제출 | 60분 |
| Q4 | AI 모델링 및 최적화 (이미지) | **성능 기준** — baseline을 하이퍼파라미터 탐색으로 개선, 등급 예측 | 50분 |
| Q5 | 프롬프트 엔지니어링 (텍스트) | 지시사항 구현 + 기준 — 프롬프트, JSON 스키마, 파싱, 정규화 | 50분 |

**성능 기준(이 모의의 가정):** Q3 test macro-F1 ≥ 0.88 · Q4 test QWK ≥ 0.90 · Q5 dev 필드 정확도 ≥ 0.95, 자료형 위반 0건. test 점수는 풀이가 끝난 뒤 `정답_채점용/score_local.py`로만 확인합니다(시험에서는 Kaggle 리더보드 역할).

진행 규칙
- 시험처럼 **필요한 import는 첫 셀에 이미 있습니다.** 막히면 `Reference/` 폴더(공식 문서 형식의 요약)만 보고 풀어 보세요. 인터넷 검색과 생성형 AI는 쓰지 않습니다.
- GPU가 있으면 `DEVICE`가 `cuda`가 됩니다. 텐서와 모델을 같은 장치에 두는 것도 평가 대상이라고 생각하세요.
- `[작성하세요]` 셀만 고칩니다. 함수 이름·매개변수·반환 형식은 그대로 둡니다. 앞 소문항을 못 풀면 확인 셀이 **진행용 대체물**로 다음을 이어 갑니다(해당 소문항 평가와 무관).
- **제출 파일 형식이 틀리면 그 문항은 0점**(FAQ)입니다. 저장 뒤 `ps.check_submission(...)`으로 확인하세요.
- test 정답은 쓰지 않습니다. `정답_채점용/`은 풀이가 끝날 때까지 열지 마세요.
""")

run(r"""
# 문제 풀이에 필요한 라이브러리 (시험처럼 미리 작성되어 있습니다)
from pathlib import Path
import sys, os, time, copy, json, re, ast, hashlib, itertools, random
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "practice_support.py").exists()), None)
if ROOT is None:
    raise FileNotFoundError("모의문제 폴더(또는 그 하위 폴더)에서 노트북을 여세요.")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, confusion_matrix, cohen_kappa_score
try:
    from torchvision.transforms import v2 as T
    print("torchvision.transforms.v2 사용")
except ImportError:                        # 이 PC에 torchvision이 없을 때만: 같은 이름의 연습용 대체
    from tv_compat import transforms_v2 as T
    print("torchvision 없음 → tv_compat(같은 이름·인자) 사용")
import practice_support as ps
from practice_support import attempt, csv, CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ps.seed_everything(42)
ps.SUBMIT.mkdir(exist_ok=True)
print("장치:", DEVICE, "| torch", torch.__version__, "| 폴더:", ROOT.name)
""")

# ======================================================================  Q1
md(r"""
---
## Q1 데이터셋 구성·품질·분포 점검 (데이터 분석/전처리 · 지시사항 구현)

사과 잎 사진으로 질병을 분류하는 데이터입니다(합성 자료, 48×48 RGB).
- `data/leaf/train_images.npz`: `ids`, `images` (N,48,48,3) uint8
- `data/leaf/train_labels.csv`: `id,label` — 레이블은 `healthy, rust, scab, multiple`
- `data/leaf/test_images.npz`: 레이블 없는 평가용 사진

**이 데이터에는 품질 문제가 섞여 있을 수 있습니다.** 무엇이 문제인지는 알려 주지 않습니다. 직접 찾으세요.
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
    info["n_images"] = len(ids)
    info["n_label_rows"] = ___
    info["image_shape"] = tuple(images.shape[___:])           # 한 장의 모양
    info["dtype"] = str(images.___)
    image_ids, label_ids = set(ids), set(labels["id"])
    info["ids_without_label"] = len(___)                     # 집합 차집합: 사진은 있는데 레이블이 없는 id
    info["labels_without_image"] = len(___)
    info["raw_label_values"] = sorted(labels["label"].___())
    clean = labels["label"].str.___().str.___()
    counts = clean.___()
    info["class_counts"] = counts.to_dict()
    info["imbalance_ratio"] = float(___)
    return info
""", r"""
def summarize_dataset(ids, images, labels):
    '''위 9개 키를 가진 dict를 반환한다.'''
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
    # --- 이미지와 레이블의 id 대응 확인 (집합 차집합, 두 방향 모두) ---
    image_ids, label_ids = set(ids), set(labels["id"])
    info["ids_without_label"] = len(image_ids - label_ids)
    info["labels_without_image"] = len(label_ids - image_ids)
    # --- 표기 흔들림 확인: 정리 전 고유값을 그대로 본다 ---
    info["raw_label_values"] = sorted(labels["label"].unique())
    # --- 정리(공백 제거 + 소문자) 후 분포와 불균형 비율 ---
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
- `find_duplicates(images)`: **바이트가 완전히 같은** 사진 묶음 리스트. 각 원소는 2개 이상의 위치를 담은 정렬된 리스트이고 전체도 정렬합니다(예: `[[3, 917], [40, 1201]]`). `hashlib.md5(x.tobytes())`를 쓰세요.
""")

write(r"""
def find_broken(images, std_thr=1.0):
    stds = images.reshape(len(images), ___).std(axis=___)   # 사진 한 장 전체의 표준편차
    return ___                                              # std_thr 미만이면 True

def find_duplicates(images):
    by_hash = {}
    for i, x in enumerate(images):
        h = hashlib.md5(x.___()).hexdigest()
        by_hash.setdefault(h, []).append(i)
    groups = [sorted(pos) for pos in by_hash.values() if len(pos) > ___]
    return sorted(groups)
""", r"""
def find_broken(images, std_thr=1.0):
    '''픽셀 표준편차 < std_thr 인 사진을 True로 표시한 bool 배열 (N,)'''
    # 작성하세요

def find_duplicates(images):
    '''바이트가 같은 사진의 위치 묶음(2개 이상) 리스트. 묶음 안과 전체 모두 정렬.'''
    # 작성하세요
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
아래 셀은 클래스별 무작위 사진, 깨진 사진, 무작위 32장을 보여 줍니다. **눈으로 보고 이상한 점을 찾으세요.** (힌트: 흙의 색)

그다음 작성하세요.
- `find_bgr(images, margin=10)`: 채널 순서가 뒤바뀐(BGR로 저장된) 사진을 찾는 bool 배열. 정상 사진은 흙색 때문에 **R 평균이 B 평균보다 큽니다.** `B 평균 − R 평균 > margin`이면 `True`.
- `mean_brightness(images)`: 사진별 전체 픽셀 평균(0~255) 배열 (N,)
- 서술 `q1_visual_note`: 육안 검사와 분포 점검에서 발견한 점, 그리고 그것이 이후 학습에 주는 의미 (2~3문장)
""")

run(r"""
rng = np.random.default_rng(0)
_lab = q1_lab.assign(label=q1_lab["label"].str.strip().str.lower()).set_index("id")["label"].reindex(q1_ids)
for c in CLASSES:
    pos = np.flatnonzero(_lab.values == c)
    ps.show_images(q1_img[rng.choice(pos, 8, replace=False)], [c] * 8)
if q1_qc:
    ps.show_images(q1_img[np.flatnonzero(q1_qc[0])][:8], ["broken"] * 8)
pick = rng.choice(len(q1_img), 32, replace=False)
ps.show_images(q1_img[pick], q1_ids[pick])
""")

write(r"""
def find_bgr(images, margin=10):
    r_mean = images[..., ___].mean(axis=(1, 2))   # 채널 축: 0=R, 1=G, 2=B
    b_mean = images[..., ___].mean(axis=(1, 2))
    return (___) > margin

def mean_brightness(images):
    return images.reshape(len(images), -1).___(axis=1)

q1_visual_note = ""  # 작성하세요 (2~3문장)
""", r"""
def find_bgr(images, margin=10):
    '''B 평균 - R 평균 > margin 인 사진을 True로 표시한 bool 배열'''
    # 작성하세요

def mean_brightness(images):
    '''사진별 전체 픽셀 평균 (N,)'''
    # 작성하세요

q1_visual_note = ""  # 작성하세요 (2~3문장)
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
    "일부 사진은 흙이 파랗게 보여 채널 순서가 BGR로 저장된 것으로 판단했고, 내용은 정상이므로 버리지 않고 RGB로 고친다. "
    "검은 화면·회색 화면은 정보가 없어 제거한다. "
    "test에는 어두운 사진 비율이 train보다 훨씬 높으므로, 학습 때 밝기 증강을 넣고 어둡게 만든 검증으로도 확인해야 한다."
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
    lab = labels.assign(label=labels["label"].str.strip().str.lower())
    label_of = lab.set_index("id")["label"]               # id → 정리된 레이블

    keep = np.isin(ids, label_of.___)                       # (2) 레이블이 있는 사진만
    keep &= ___find_broken(images)                          # (3) 깨진 사진 제외 (bool 반전 연산자)

    for group in find_duplicates(images):                   # (4) 중복 묶음
        group_labels = label_of.reindex(ids[group])
        if group_labels.nunique(dropna=True) > 1:
            keep[group] = ___                               # 엇갈림 → 묶음 전체 제거
        else:
            keep[group[___:]] = False                       # 같음 → 첫 장만 남김

    sel = np.flatnonzero(keep)
    clean_ids = ids[sel]
    clean_images = images[sel].___()                        # 원본을 바꾸지 않도록
    clean_labels = label_of.reindex(clean_ids).to_numpy()

    bgr = find_bgr(clean_images)                            # (5) BGR → RGB 복원
    clean_images[bgr] = clean_images[bgr][..., ___]
    return clean_ids, clean_images, clean_labels
""", r"""
def build_clean_dataset(ids, images, labels):
    '''위 1~5 규칙으로 (clean_ids, clean_images, clean_labels)를 반환한다.'''
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
## Q2 변환·증강과 학습 데이터 재구성 (데이터 분석/전처리 · 지시사항 구현)

이 문항은 PyTorch 표준 방식(`transforms` → `Dataset` → `DataLoader`)으로 학습 입력을 만듭니다. 필요한 사용법은 `Reference/transforms.md`, `Reference/torch_data.md`에 있습니다. 정규화 값은 **지정 모델의 모델 카드**(`assets/모델카드.md`)에서 찾으세요.
""")

md(r"""
### 2-1 transforms 파이프라인
`build_transforms(train)`: `T.Compose`를 반환합니다. 입력은 사진 한 장 (48,48,3) uint8 numpy 배열입니다.
- 학습용(`train=True`): `ToImage` → 좌우 반전(p=0.5) → 상하 반전(p=0.5) → `ToDtype(float32, scale=True)` → 밝기 `ColorJitter(brightness=(0.4, 1.1))` → `Normalize(모델 카드의 mean, std)`
- 검증·test용(`train=False`): `ToImage` → `ToDtype(float32, scale=True)` → `Normalize` (무작위 변환 없음)
- `MODEL_MEAN`, `MODEL_STD`에 모델 카드의 값을 리스트로 적으세요.
""")

write(r"""
MODEL_MEAN = ___   # 모델 카드 참고
MODEL_STD = ___

def build_transforms(train):
    steps = [T.ToImage()]
    if train:
        steps += [T.RandomHorizontalFlip(p=0.5), T.___(p=0.5)]
    steps += [T.ToDtype(torch.float32, scale=___)]
    if train:
        steps += [T.ColorJitter(brightness=___)]
    steps += [T.Normalize(mean=MODEL_MEAN, std=___)]
    return T.___(steps)
""", r"""
MODEL_MEAN = None  # 작성하세요: 모델 카드의 mean
MODEL_STD = None   # 작성하세요: 모델 카드의 std

def build_transforms(train):
    '''학습용/검증용 T.Compose를 반환한다.'''
    # 작성하세요
""", r"""
MODEL_MEAN = [0.5, 0.5, 0.5]      # assets/모델카드.md: 사전학습 때 쓴 정규화 값
MODEL_STD = [0.25, 0.25, 0.25]    # (ImageNet 값 0.485… 을 습관처럼 쓰면 안 됨)

def build_transforms(train):
    steps = [T.ToImage()]                                   # (H,W,C) numpy → (C,H,W) 텐서, 아직 uint8
    if train:
        # 잎 방향은 질병과 무관 → 반전은 레이블을 바꾸지 않는 안전한 증강
        steps += [T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)]
    steps += [T.ToDtype(torch.float32, scale=True)]          # [0,255] → [0,1] 실수
    if train:
        # test에 어두운 사진이 많음(1-3) → 밝기를 0.4~1.1배로 흔들어 대비. [0,1]에서 적용해야 의미가 맞다
        steps += [T.ColorJitter(brightness=(0.4, 1.1))]
    steps += [T.Normalize(mean=MODEL_MEAN, std=MODEL_STD)]   # 정규화는 항상 마지막
    return T.Compose(steps)
""")

run(r"""
def _check_2_1():
    assert MODEL_MEAN == ps.PRETRAIN_MEAN and MODEL_STD == ps.PRETRAIN_STD, "모델 카드의 값을 확인하세요"
    tf_v, tf_t = build_transforms(False), build_transforms(True)
    img = clean_img[0]
    x = tf_v(img)
    assert tuple(x.shape) == (3, 48, 48) and x.dtype == torch.float32
    want = (img[3, 7] / 255.0 - 0.5) / 0.25
    assert torch.allclose(torch.as_tensor(x)[:, 3, 7], torch.tensor(want, dtype=torch.float32), atol=1e-5), "축 순서나 정규화"
    assert torch.equal(torch.as_tensor(tf_v(img)), torch.as_tensor(x)), "검증 변환에 무작위성이 있습니다"
    torch.manual_seed(0)
    outs = [torch.as_tensor(tf_t(img)) for _ in range(6)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:]), "학습 변환이 매번 같습니다"
    print(tf_t)
    show = [((torch.as_tensor(tf_t(img)) * 0.25 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8) for _ in range(8)]
    ps.show_images(np.stack(show), ["train tf"] * 8)
attempt("2-1", _check_2_1)
""")

md(r"""
### 2-2 Dataset
`LeafDataset(images, labels, transform)`: `torch.utils.data.Dataset`을 상속합니다.
- `images`: (N,48,48,3) uint8, `labels`: 문자열 레이블 배열 또는 `None`(test), `transform`: 2-1의 변환
- `__len__`: 사진 수
- `__getitem__(i)`: `(transform(images[i]), y)`. `y`는 `CLASSES.index(레이블)`인 `torch.long` 스칼라 텐서이고, `labels`가 `None`이면 `-1`
""")

write(r"""
class LeafDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.transform = transform
        self.y = None if labels is None else [CLASSES.___(c) for c in labels]   # 문자열 → 정수

    def __len__(self):
        return ___

    def __getitem__(self, i):
        x = self.___(self.images[i])
        y = -1 if self.y is None else self.y[i]
        return x, torch.tensor(y, dtype=torch.___)
""", r"""
class LeafDataset(Dataset):
    '''(transform(images[i]), y) 를 돌려주는 Dataset. labels가 None이면 y=-1'''
    def __init__(self, images, labels, transform):
        # 작성하세요
        pass
    # 작성하세요: __len__, __getitem__
""", r"""
class LeafDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images                      # 원본 uint8 배열을 그대로 보관(변환은 꺼낼 때마다)
        self.transform = transform
        # 문자열 레이블을 미리 정수 인덱스로 바꿔 둔다. test는 레이블이 없다.
        self.y = None if labels is None else [CLASSES.index(c) for c in labels]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = self.transform(self.images[i])        # 학습용 변환이면 호출할 때마다 다른 증강
        y = -1 if self.y is None else self.y[i]
        return x, torch.tensor(y, dtype=torch.long)   # CrossEntropyLoss는 long 정답을 받는다
""")

run(r"""
_tf = build_transforms if attempt("2-1 재확인", lambda: build_transforms(False)(clean_img[0]).shape) else ps.ready_transforms
def _check_2_2():
    ds = LeafDataset(clean_img[:5], clean_lab[:5], _tf(False))
    x, y = ds[2]
    assert len(ds) == 5 and tuple(x.shape) == (3, 48, 48) and y.dtype == torch.long and int(y) == CLASSES.index(clean_lab[2])
    xt, yt = LeafDataset(q1_test_img[:3], None, _tf(False))[0]
    assert int(yt) == -1
    xb, yb = next(iter(DataLoader(ds, batch_size=4)))
    print("배치:", tuple(xb.shape), xb.dtype, "| 정답:", yb.tolist())
attempt("2-2", _check_2_2)
""")

md(r"""
### 2-3 분할과 불균형을 고려한 학습 데이터 재구성
- `stratified_split(labels, val_ratio=0.2, seed=42)`: 클래스 비율을 유지한 `(train_idx, val_idx)` 정수 배열 (`train_test_split`의 `stratify`)
- `make_loader(dataset, labels, batch_size=64, balanced=False, seed=42)`: 학습용 `DataLoader`
  - `balanced=False`: `shuffle=True`, `generator=torch.Generator().manual_seed(seed)`
  - `balanced=True`: 표본마다 **(그 표본 클래스의 개수)의 역수**를 가중치로 하는 `WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=...)`를 씁니다(`shuffle`은 주지 않음).
""")

write(r"""
def stratified_split(labels, val_ratio=0.2, seed=42):
    idx = np.arange(len(labels))
    tr, va = train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=___)
    return tr, va

def make_loader(dataset, labels, batch_size=64, balanced=False, seed=42):
    g = torch.Generator().manual_seed(seed)
    if not balanced:
        return DataLoader(dataset, batch_size=batch_size, shuffle=___, generator=g)
    counts = pd.Series(labels).value_counts()                       # 클래스별 개수
    weights = torch.tensor([1.0 / counts[c] for c in ___], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=___, replacement=___, generator=g)
    return DataLoader(dataset, batch_size=batch_size, ___=sampler)
""", r"""
def stratified_split(labels, val_ratio=0.2, seed=42):
    '''층화 분할 (train_idx, val_idx)'''
    # 작성하세요

def make_loader(dataset, labels, batch_size=64, balanced=False, seed=42):
    '''학습용 DataLoader. balanced=True면 역빈도 WeightedRandomSampler'''
    # 작성하세요
""", r"""
def stratified_split(labels, val_ratio=0.2, seed=42):
    idx = np.arange(len(labels))
    # stratify=labels: 소수 클래스(multiple)도 train/val에 같은 비율로 들어가게 한다
    tr, va = train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=labels)
    return tr, va

def make_loader(dataset, labels, batch_size=64, balanced=False, seed=42):
    g = torch.Generator().manual_seed(seed)        # 재현성: 섞는 순서를 고정
    if not balanced:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)
    # 표본 가중치 = 1 / (그 클래스의 개수) → 클래스마다 가중치 합이 같아져 고르게 뽑힌다
    counts = pd.Series(labels).value_counts()
    weights = torch.tensor([1.0 / counts[c] for c in labels], dtype=torch.double)
    # 복원추출(replacement=True)이어야 적은 클래스를 여러 번 뽑을 수 있다
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=g)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)   # sampler와 shuffle은 함께 쓰지 않음
""")

run(r"""
def _check_2_3():
    tr, va = stratified_split(clean_lab)
    assert len(set(tr) & set(va)) == 0 and len(tr) + len(va) == len(clean_lab), "겹침 또는 누락"
    ds = LeafDataset(clean_img[tr], clean_lab[tr], _tf(False))
    seen = torch.cat([yb for _, yb in make_loader(ds, clean_lab[tr], balanced=True)])
    share = torch.bincount(seen, minlength=4).float() / len(seen)
    orig = pd.Series(clean_lab[tr]).value_counts(normalize=True).reindex(CLASSES).to_numpy()
    print("원래 비율", orig.round(3), "→ 균형 샘플러 1 epoch 비율", share.numpy().round(3))
    assert (share - 0.25).abs().max() < 0.06, "클래스가 고르게 뽑히지 않습니다"
    assert len(seen) == len(tr)
    return tr, va
q2_split = attempt("2-3", _check_2_3)
""")

md(r"""
### 2-4 증강 판단 (서술)
`q2_aug_note`에 쓰세요(각 1문장).
1. 밝기 증강을 넣은 근거
2. 이 잎 데이터에서 **색조(hue) 변환**을 쓰면 안 되는 이유
3. Q4의 등급 데이터(반점 수가 많을수록 높은 등급)에서 **크게 잘라 내는 증강**(`RandomResizedCrop` 같은)이 위험한 이유
""")

write(r"""
q2_aug_note = ""  # 작성하세요
""", r"""
q2_aug_note = ""  # 작성하세요
""", r"""
q2_aug_note = (
    "(1) test에 어두운 사진이 train보다 훨씬 많아(1-3), 밝기를 낮춘 사진을 학습 때 보여 주어 분포 차이에 대비한다. "
    "(2) rust는 주황색 반점, scab은 갈색 얼룩처럼 색이 레이블의 근거이므로 색조를 바꾸면 레이블과 맞지 않는 사진이 만들어진다. "
    "(3) 크게 잘라 내면 반점 일부가 잘려 나가 반점 수가 줄어들므로, 사진은 낮은 등급이 되는데 레이블은 그대로 남는다."
)
""")

run(r"""
# Q3에서 쓸 공통 입력 (앞 소문항이 실패하면 진행용 대체물을 씁니다)
_DS = LeafDataset if attempt("2-2 재확인", lambda: LeafDataset(clean_img[:2], clean_lab[:2], _tf(False))[0][0].shape) else ps.ReadyLeafDataset
_loader = make_loader if attempt("2-3 재확인", lambda: len(make_loader(_DS(clean_img[:8], clean_lab[:8], _tf(False)), clean_lab[:8], 4, True))) else ps.ready_train_loader
if q2_split is None:
    q2_split = train_test_split(np.arange(len(clean_lab)), test_size=0.2, random_state=42, stratify=clean_lab)
tr_idx, va_idx = q2_split
y_tr_ids = torch.tensor([CLASSES.index(c) for c in clean_lab[tr_idx]])
y_va = np.array([CLASSES.index(c) for c in clean_lab[va_idx]])
train_ds = _DS(clean_img[tr_idx], clean_lab[tr_idx], _tf(True))

def eval_loader(images, labels=None):
    return DataLoader(_DS(images, labels, _tf(False)), batch_size=256, shuffle=False)   # 평가: 섞지 않음

val_loader = eval_loader(clean_img[va_idx], clean_lab[va_idx])
# val_dark: 같은 검증 사진을 0.45배로 어둡게 → test의 어두운 사진 비율(1-3)을 흉내 낸 분포 점검용 검증
val_dark_loader = eval_loader((clean_img[va_idx] * 0.45).astype(np.uint8), clean_lab[va_idx])
test_loader = eval_loader(q1_test_img)
print("train", len(tr_idx), "| val", len(va_idx), "| test", len(q1_test_img))
""")

# ======================================================================  Q3
md(r"""
---
## Q3 전이학습·학습 범위·손실 함수 (AI 모델링 및 최적화 · 구현 + 성능 기준)

지정 모델: `ps.load_pretrained_model()` — 무늬 6종 분류로 사전학습한 작은 ResNet입니다. `assets/모델카드.md`를 먼저 읽으세요. **torchvision ResNet과 같은 속성 이름**(`conv1, bn1, layer1~layer4, avgpool, fc`)을 쓰므로, 시험의 `torchvision.models.resnet18`에도 같은 코드가 통합니다.

- 지표: **macro-F1**(소수 클래스 `multiple`도 똑같이 반영)
- **성능 기준(가정): test macro-F1 ≥ 0.88** → `제출물/q3_submission.csv` (`제출가이드.md`)
""")

run(r"""
print(ps.load_pretrained_model())
""")

md(r"""
### 3-1 전이학습을 위한 구조 변경
`build_classifier(num_classes)`: 지정 모델을 불러와 `fc`만 **입력 차원은 그대로, 출력은 `num_classes`**인 새 `nn.Linear`로 바꿉니다. 숫자 96을 직접 쓰지 말고 기존 층에서 읽으세요.
""")

write(r"""
def build_classifier(num_classes):
    model = ps.load_pretrained_model()
    in_f = model.fc.___                       # 기존 fc의 입력 차원
    model.fc = nn.___(in_f, num_classes)
    return model
""", r"""
def build_classifier(num_classes):
    '''지정 모델의 fc를 출력 num_classes인 새 Linear로 교체해 반환한다.'''
    model = ps.load_pretrained_model()
    # 작성하세요
    return model
""", r"""
def build_classifier(num_classes):
    model = ps.load_pretrained_model()        # 사전학습 가중치가 들어 있는 몸통 + 6출력 fc
    in_f = model.fc.in_features               # 96을 하드코딩하지 않고 읽는다(resnet18이면 512)
    model.fc = nn.Linear(in_f, num_classes)   # 새 과제의 클래스 수에 맞는 출력층(무작위 초기화)
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
- `set_train_scope(model, prefixes)`: 파라미터 **이름**이 `prefixes` 중 하나로 시작하면 `requires_grad=True`, 아니면 `False`. 학습할 파라미터의 **리스트**를 반환합니다.
- `freeze_bn_outside(model, prefixes)`: 범위 밖의 모든 `nn.BatchNorm2d`를 `eval()` 모드로 둡니다. (`requires_grad=False`만으로는 BN의 `running_mean/var`가 계속 바뀝니다.) `model.train()`을 부른 **뒤에** 씁니다.
""")

write(r"""
def set_train_scope(model, prefixes):
    prefixes = tuple(prefixes)
    for name, p in model.___():                  # (이름, 파라미터) 쌍
        p.requires_grad = name.___(prefixes)
    return [p for p in model.parameters() if p.___]

def freeze_bn_outside(model, prefixes):
    prefixes = tuple(prefixes)
    for name, m in model.named_modules():
        if isinstance(m, nn.___) and not name.startswith(prefixes):
            m.___()
""", r"""
def set_train_scope(model, prefixes):
    '''이름이 prefixes로 시작하는 파라미터만 학습. 학습할 파라미터 리스트를 반환.'''
    # 작성하세요

def freeze_bn_outside(model, prefixes):
    '''범위 밖 BatchNorm2d를 eval()로 둔다. model.train() 다음에 호출.'''
    # 작성하세요
""", r"""
def set_train_scope(model, prefixes):
    prefixes = tuple(prefixes)                   # str.startswith는 튜플을 받아 '이 중 하나로 시작'을 검사
    for name, p in model.named_parameters():     # 예: 'layer4.conv1.weight', 'fc.bias'
        p.requires_grad = name.startswith(prefixes)
    return [p for p in model.parameters() if p.requires_grad]   # optimizer에는 이것만 넘긴다

def freeze_bn_outside(model, prefixes):
    prefixes = tuple(prefixes)
    for name, m in model.named_modules():        # 예: 'bn1', 'layer1.bn2'
        if isinstance(m, nn.BatchNorm2d) and not name.startswith(prefixes):
            m.eval()                             # eval 모드 BN은 running 통계를 갱신하지 않는다
""")

run(r"""
def _check_3_2():
    m = build_classifier(4)
    params = set_train_scope(m, ["layer3", "layer4", "fc"])
    n_train = sum(p.numel() for p in params); n_all = sum(p.numel() for p in m.parameters())
    assert isinstance(params, list)
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
`F.log_softmax`와 `gather`로 직접 구현하세요(`F.cross_entropy`, `nn.CrossEntropyLoss` 호출 금지).
- `weighted_cross_entropy(logits, target, weight)`: 표본 i의 손실 `-log p_i[y_i]`에 `weight[y_i]`를 곱해 더한 뒤 **가중치 합으로 나눈** 값 (`CrossEntropyLoss(weight=...)`와 같은 정의)
- `focal_loss(logits, target, gamma=2.0)`: 평균 `-(1 - p_t)^gamma · log p_t`. `gamma=0`이면 CE와 같아야 합니다.
- `class_weights(label_ids, num_classes)`: `전체 수 / (클래스 수 × 클래스별 수)` 텐서
""")

write(r"""
def weighted_cross_entropy(logits, target, weight):
    log_p = F.log_softmax(logits, dim=___)
    log_p_true = log_p.gather(1, target.___(1)).squeeze(1)      # 정답 클래스의 log 확률 (B,)
    w = weight.to(logits.device)[___]                            # 표본마다 자기 클래스의 가중치
    return -(w * log_p_true).sum() / ___

def focal_loss(logits, target, gamma=2.0):
    log_p_true = F.log_softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
    p_t = log_p_true.___()
    return (-(___) ** gamma * log_p_true).mean()

def class_weights(label_ids, num_classes):
    counts = torch.___(torch.as_tensor(label_ids), minlength=num_classes).float()
    return len(label_ids) / (___ * counts)
""", r"""
def weighted_cross_entropy(logits, target, weight):
    '''가중 CE (가중치 합으로 나눔)'''
    # 작성하세요

def focal_loss(logits, target, gamma=2.0):
    '''focal loss 평균'''
    # 작성하세요

def class_weights(label_ids, num_classes):
    '''N / (C * count_c) 텐서'''
    # 작성하세요
""", r"""
def weighted_cross_entropy(logits, target, weight):
    log_p = F.log_softmax(logits, dim=1)                         # softmax 후 log보다 수치적으로 안정
    log_p_true = log_p.gather(1, target.unsqueeze(1)).squeeze(1) # (B,C)에서 정답 열만 → (B,)
    w = weight.to(logits.device)[target]                          # 표본마다 자기 클래스의 가중치
    return -(w * log_p_true).sum() / w.sum()                      # 평균이 아니라 가중치 합으로 나눔(PyTorch 정의)

def focal_loss(logits, target, gamma=2.0):
    log_p_true = F.log_softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
    p_t = log_p_true.exp()                                        # 정답 확률. 잘 맞힐수록 1에 가까움
    return (-(1 - p_t) ** gamma * log_p_true).mean()              # 쉬운 표본은 (1-p_t)^γ로 손실을 줄임

def class_weights(label_ids, num_classes):
    counts = torch.bincount(torch.as_tensor(label_ids), minlength=num_classes).float()
    return len(label_ids) / (num_classes * counts)                # 적은 클래스일수록 큰 가중치(균형이면 모두 1)
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
    print("클래스 가중치:", class_weights(y_tr_ids, 4).numpy().round(2))
attempt("3-3", _check_3_3)
""")

md(r"""
### 3-4 학습·예측 함수 (장치 포함)
- `train_one_epoch(model, loader, optimizer, loss_fn, scope, device)`: `model.train()` → `freeze_bn_outside` → 배치마다 **장치로 이동 → `zero_grad` → 순전파 → 손실 → `backward` → `step`**. 표본 수로 가중 평균한 손실(float)을 반환합니다.
- `predict_logits(model, loader, device)`: `model.eval()`과 `torch.no_grad()` 안에서 logits를 모아 **CPU** 텐서 `(N, C)`로 반환합니다.
""")

write(r"""
def train_one_epoch(model, loader, optimizer, loss_fn, scope, device):
    model.___()
    freeze_bn_outside(model, scope)
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(___), yb.to(device)
        optimizer.___()
        loss = loss_fn(model(xb), yb)
        loss.___()
        optimizer.___()
        total += loss.item() * len(xb)
        n += len(xb)
    return total / max(n, 1)

def predict_logits(model, loader, device):
    model.___()
    outs = []
    with torch.___():
        for xb, _ in loader:
            outs.append(model(xb.to(device)).___())
    return torch.cat(outs)
""", r"""
def train_one_epoch(model, loader, optimizer, loss_fn, scope, device):
    '''한 epoch 학습 후 표본 가중 평균 손실(float)'''
    # 작성하세요

def predict_logits(model, loader, device):
    '''eval + no_grad 로 logits (N,C) CPU 텐서'''
    # 작성하세요
""", r"""
def train_one_epoch(model, loader, optimizer, loss_fn, scope, device):
    model.train()                                # Dropout·BN을 학습 모드로
    freeze_bn_outside(model, scope)              # 그다음 범위 밖 BN만 다시 eval (순서 중요)
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)    # 모델과 같은 장치로
        optimizer.zero_grad()                    # 이전 배치의 기울기 지우기
        loss = loss_fn(model(xb), yb)            # 순전파 → 손실 (logits를 그대로 넣음)
        loss.backward()                          # 기울기 계산
        optimizer.step()                         # 가중치 갱신
        total += loss.item() * len(xb)           # 마지막 배치가 작아도 정확한 평균이 되도록 표본 수로 가중
        n += len(xb)
    return total / max(n, 1)

def predict_logits(model, loader, device):
    model.eval()                                 # BN은 running 통계, Dropout 끔
    outs = []
    with torch.no_grad():                        # 기울기 기록 안 함 → 메모리·속도 절약
        for xb, _ in loader:
            outs.append(model(xb.to(device)).cpu())   # numpy·sklearn에서 쓰려면 CPU로
    return torch.cat(outs)
""")

run(r"""
def _check_3_4():
    m = build_classifier(4).to(DEVICE)
    scope = ["layer3", "layer4", "fc"]
    opt = torch.optim.AdamW(set_train_scope(m, scope), lr=1e-3)
    small = DataLoader(_DS(clean_img[tr_idx[:130]], clean_lab[tr_idx[:130]], _tf(True)), batch_size=64, shuffle=True)
    loss = train_one_epoch(m, small, opt, nn.CrossEntropyLoss(), scope, DEVICE)
    out = predict_logits(m, eval_loader(clean_img[va_idx[:300]]), DEVICE)
    assert isinstance(loss, float) and tuple(out.shape) == (300, 4) and out.device.type == "cpu"
    print(f"짧은 학습 확인: loss {loss:.3f}")
attempt("3-4", _check_3_4)
""")

md(r"""
### 3-5 최고 성능 가중치 보관 (체크포인트)
`fit(model, train_loader, select_loader, y_select, optimizer, loss_fn, scope, epochs, device)`:
- epoch마다 `train_one_epoch` → `select_loader`로 예측 → macro-F1 계산
- macro-F1이 **지금까지 최고보다 높으면** `copy.deepcopy(model.state_dict())`로 보관
- 끝나면 최고 가중치를 `load_state_dict`로 되돌리고 `(model, best_f1, history DataFrame)`을 반환. `history` 열: `epoch, train_loss, select_f1`
""")

write(r"""
def fit(model, train_loader, select_loader, y_select, optimizer, loss_fn, scope, epochs, device):
    best_f1, best_state, history = -1.0, None, []
    for ep in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scope, device)
        pred = predict_logits(model, select_loader, device).___(1).numpy()     # 가장 큰 logit의 클래스
        f1 = f1_score(y_select, pred, average=___)
        history.append({"epoch": ep, "train_loss": round(loss, 4), "select_f1": round(f1, 4)})
        if f1 > best_f1:
            best_f1, best_state = f1, copy.___(model.state_dict())           # 참조가 아닌 복사본
    model.___(best_state)
    return model, best_f1, pd.DataFrame(history)
""", r"""
def fit(model, train_loader, select_loader, y_select, optimizer, loss_fn, scope, epochs, device):
    '''최고 macro-F1 가중치로 되돌린 (model, best_f1, history) 반환'''
    # 작성하세요
""", r"""
def fit(model, train_loader, select_loader, y_select, optimizer, loss_fn, scope, epochs, device):
    best_f1, best_state, history = -1.0, None, []
    for ep in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scope, device)
        pred = predict_logits(model, select_loader, device).argmax(1).numpy()   # logits → 예측 클래스
        f1 = f1_score(y_select, pred, average="macro")          # 클래스별 F1의 단순 평균 → 소수 클래스도 같은 무게
        history.append({"epoch": ep, "train_loss": round(loss, 4), "select_f1": round(f1, 4)})
        if f1 > best_f1:
            # state_dict()는 살아 있는 텐서를 가리키므로, 그냥 저장하면 다음 학습에서 같이 바뀐다 → deepcopy
            best_f1, best_state = f1, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)                            # 마지막 epoch가 아니라 최고 epoch로 복원
    return model, best_f1, pd.DataFrame(history)
""")

run(r"""
def _check_3_5():
    ps.seed_everything(0)
    m = build_classifier(4).to(DEVICE)
    scope = ["fc"]
    opt = torch.optim.AdamW(set_train_scope(m, scope), lr=1e-2)
    small = DataLoader(_DS(clean_img[tr_idx[:200]], clean_lab[tr_idx[:200]], _tf(True)), batch_size=64, shuffle=True)
    m, best, hist = fit(m, small, val_loader, y_va, opt, nn.CrossEntropyLoss(), scope, 3, DEVICE)
    assert list(hist.columns) == ["epoch", "train_loss", "select_f1"] and len(hist) == 3
    again = f1_score(y_va, predict_logits(m, val_loader, DEVICE).argmax(1).numpy(), average="macro")
    assert abs(again - best) < 1e-9 and abs(best - hist["select_f1"].max()) < 1e-4, "최고 가중치로 복원되지 않았습니다"
    print(hist.to_string(index=False))
attempt("3-5", _check_3_5)
""")

md(r"""
### 3-6 실험 설정과 선택 (성능 기준)
`Q3_CANDIDATES`에 설정 dict를 **2~3개** 적고, `Q3_SELECT_BY`에 최고 epoch와 최종 모델을 고를 검증(`"val"` 또는 `"val_dark"`)을 정하세요. 설정 키:
- `loss`: `"ce"` | `"weighted_ce"` | `"focal"` · `balanced`: 균형 샘플러 사용 여부 · `scope`: 학습 범위 접두어 튜플 · `lr` · `epochs`(합계 30 이하 권장, CPU 기준 epoch당 수 초)

아래 실행 셀이 설정마다 학습하고 `val`과 `val_dark`의 macro-F1, `multiple` 재현율을 표로 보여 준 뒤 `Q3_SELECT_BY` 기준 최고 설정을 고릅니다. **test 점수로 고르지 않습니다.**

자가 기준(이 모의의 경험치): 선택 모델의 `val_dark` macro-F1 ≥ 0.93 **그리고** `val_dark` multiple 재현율 ≥ 0.90이면 test 기준(0.88)을 넘을 가능성이 높습니다. validation의 multiple은 17장뿐이라 macro-F1만으로는 차이가 잘 드러나지 않습니다. 재현율도 함께 보세요.
""")

write(r"""
Q3_CANDIDATES = [
    dict(loss="ce", balanced=False, scope=("layer3", "layer4", "fc"), lr=1e-3, epochs=10),
    dict(loss=___, balanced=___, scope=___, lr=___, epochs=___),
]
Q3_SELECT_BY = ___   # "val" 또는 "val_dark"
""", r"""
Q3_CANDIDATES = []   # 작성하세요: 설정 dict 2~3개
Q3_SELECT_BY = ""    # 작성하세요: "val" 또는 "val_dark"
""", r"""
Q3_CANDIDATES = [
    # A: 기준선 — 일반 CE, 마지막 두 블록 + fc
    dict(loss="ce", balanced=False, scope=("layer3", "layer4", "fc"), lr=1e-3, epochs=10),
    # B: 불균형 대응 — 균형 샘플러로 소수 클래스(multiple)를 epoch마다 고르게 보여 줌
    dict(loss="ce", balanced=True, scope=("layer3", "layer4", "fc"), lr=1e-3, epochs=10),
    # C: 불균형 대응 — 가중 CE (샘플러와 함께 쓰면 보정이 두 번 되므로 따로 비교)
    dict(loss="weighted_ce", balanced=False, scope=("layer3", "layer4", "fc"), lr=1e-3, epochs=10),
]
Q3_SELECT_BY = "val_dark"   # test에 어두운 사진이 많으므로(1-3) 그 조건을 흉내 낸 검증으로 고른다
""")

run(r"""
def run_q3(cfg):
    ps.seed_everything(42)
    scope = list(cfg["scope"])
    model = build_classifier(len(CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(set_train_scope(model, scope), lr=cfg["lr"])
    if cfg["loss"] == "weighted_ce":
        w = class_weights(y_tr_ids, len(CLASSES)).to(DEVICE)
        loss_fn = lambda lg, t: weighted_cross_entropy(lg, t, w)
    elif cfg["loss"] == "focal":
        loss_fn = focal_loss
    else:
        loss_fn = nn.CrossEntropyLoss()
    loader = _loader(train_ds, clean_lab[tr_idx], batch_size=64, balanced=cfg["balanced"])
    select_loader = val_dark_loader if Q3_SELECT_BY == "val_dark" else val_loader
    t0 = time.time()
    model, best, hist = fit(model, loader, select_loader, y_va, opt, loss_fn, scope, cfg["epochs"], DEVICE)
    res = {**cfg, "scope": "+".join(cfg["scope"]), "best_epoch": int(hist.loc[hist.select_f1.idxmax(), "epoch"])}
    for name, ld in [("val", val_loader), ("val_dark", val_dark_loader)]:
        pred = predict_logits(model, ld, DEVICE).argmax(1).numpy()
        res[f"{name}_f1"] = round(f1_score(y_va, pred, average="macro"), 3)
        res[f"{name}_rec_multiple"] = round(recall_score(y_va, pred, labels=[3], average="macro"), 2)
    res["sec"] = round(time.time() - t0)
    return model, res

def _run_3_6():
    assert 1 <= len(Q3_CANDIDATES) <= 3 and Q3_SELECT_BY in ("val", "val_dark"), "설정 2~3개와 선택 기준을 적으세요"
    if sum(c["epochs"] for c in Q3_CANDIDATES) > 30:
        print("주의: epoch 합계가 30을 넘습니다. 시험에서는 시간 예산을 먼저 정하세요.")
    runs = [run_q3(c) for c in Q3_CANDIDATES]
    table = pd.DataFrame([r for _, r in runs])
    print(table.to_string(index=False))
    best_i = int(table[f"{Q3_SELECT_BY}_f1"].idxmax())
    ok = table.loc[best_i, "val_dark_f1"] >= 0.93 and table.loc[best_i, "val_dark_rec_multiple"] >= 0.90
    print(f"선택 기준 {Q3_SELECT_BY}_f1 → 설정 {best_i}", "| 자가 기준(val_dark F1 ≥ 0.93, multiple 재현율 ≥ 0.90):", bool(ok))
    return runs[best_i][0], table

q3_out = attempt("3-6", _run_3_6)
if q3_out:
    q3_best_model, q3_table = q3_out
    q3_test_pred = np.array(CLASSES)[predict_logits(q3_best_model, test_loader, DEVICE).argmax(1).numpy()]
    print("test 예측 분포:", pd.Series(q3_test_pred).value_counts().to_dict())
""")

md(r"""
### 3-7 제출 파일 만들기 (제출가이드 준수)
- `make_submission(sample, ids, values, col)`: `sample`(sample_submission DataFrame)의 **`id` 순서 그대로** `id`와 `col` 두 열을 가진 DataFrame을 반환합니다. `values[k]`는 `ids[k]`의 예측입니다.
- 그 함수로 `q3_sub`를 만들고 `제출물/q3_submission.csv`로 저장하세요(인덱스 열 없이). Q4에서도 같은 함수를 씁니다.
""")

write(r"""
def make_submission(sample, ids, values, col):
    pred = pd.Series(values, index=___)                  # id → 예측
    sub = sample[[___]].copy()                           # sample의 id 순서를 그대로
    sub[col] = pred.___(sub["id"]).to_numpy()            # sample 순서로 다시 줄 세우기
    return sub

q3_sub = make_submission(csv("leaf", "sample_submission.csv"), q1_test_ids, q3_test_pred, "label")
q3_sub.to_csv(ps.SUBMIT / ___, index=___)
""", r"""
def make_submission(sample, ids, values, col):
    '''sample의 id 순서로 [id, col] DataFrame'''
    # 작성하세요

# 작성하세요: q3_sub 만들고 제출물/q3_submission.csv 로 저장
""", r"""
def make_submission(sample, ids, values, col):
    pred = pd.Series(values, index=ids)                  # 예측을 id로 찾을 수 있게
    sub = sample[["id"]].copy()                          # 행 순서는 sample_submission이 기준
    sub[col] = pred.reindex(sub["id"]).to_numpy()        # 위치가 아니라 id로 맞춘다(순서가 달라도 안전)
    return sub

q3_sub = make_submission(csv("leaf", "sample_submission.csv"), q1_test_ids, q3_test_pred, "label")
q3_sub.to_csv(ps.SUBMIT / "q3_submission.csv", index=False)   # index=False: 'Unnamed: 0' 열이 생기지 않게
""")

run(r"""
print("q3 형식 점검:", ps.check_submission("q3") or "통과")
""")

md(r"""
### 3-8 분석 (서술)
`q3_analysis`에 (1) 설정별 `val`/`val_dark` 결과 비교, (2) 선택 기준과 최종 선택의 이유, (3) 남는 위험이나 다음 개선안을 쓰세요(3~5문장, 위 표의 수치를 인용).
""")

write(r"""
q3_analysis = ""  # 작성하세요
""", r"""
q3_analysis = ""  # 작성하세요
""", r"""
q3_analysis = (
    "__Q3_ANALYSIS__"
)
""")

# ======================================================================  Q4
md(r"""
---
## Q4 등급 예측 · 하이퍼파라미터 탐색 (AI 모델링 및 최적화 · 성능 기준)

잎의 **녹병 심각도 등급 0~4**를 예측합니다(합성 자료, 반점 수가 많을수록 높은 등급). 등급은 순서가 있으므로 **회귀로 실수를 예측한 뒤 임계값으로 등급을 나눕니다.**
- `data/grade/train_images.npz`(1,000장), `train_labels.csv`(`id,grade`), `test_images.npz`(400장)
- 지표: **QWK**(`cohen_kappa_score(..., weights="quadratic")`). **성능 기준(가정): test QWK ≥ 0.90** → `제출물/q4_submission.csv`
- 학습은 제공 함수 `ps.train_regressor(model, x, y, train_idx, scope, lr, epochs, augment, device)`가 합니다(`Reference/지정모델_제공함수.md`).

**제공된 baseline은 기준에 못 미칩니다.** 4-1~4-3을 완성한 뒤 4-4에서 설정을 탐색해 기준을 넘기세요.
""")

run(r"""
g_ids, g_img = ps.load_images("grade", "train")
g_test_ids, g_test_img = ps.load_images("grade", "test")
g_y = csv("grade", "train_labels.csv").set_index("id").loc[g_ids, "grade"].to_numpy()
gX, gX_test = ps.to_float_tensor(g_img), ps.to_float_tensor(g_test_img)
gY = torch.tensor(g_y, dtype=torch.float32)
g_tr, g_va = train_test_split(np.arange(len(g_y)), test_size=0.2, random_state=42, stratify=g_y)
qwk = lambda a, b: cohen_kappa_score(a, b, weights="quadratic")
print(gX.shape, "| 등급 분포", np.bincount(g_y))
""")

md(r"""
### 4-1 회귀 헤드로 구조 변경
`build_regressor(dropout=0.2)`: 지정 모델의 `fc`를 `nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))`로 바꿉니다. 출력은 `(B, 1)`입니다.
""")

write(r"""
def build_regressor(dropout=0.2):
    model = ps.load_pretrained_model()
    in_f = model.fc.in_features
    model.fc = nn.Sequential(nn.___(dropout), nn.Linear(in_f, ___))
    return model
""", r"""
def build_regressor(dropout=0.2):
    '''fc → Sequential(Dropout, Linear(in_features, 1))'''
    model = ps.load_pretrained_model()
    # 작성하세요
    return model
""", r"""
def build_regressor(dropout=0.2):
    model = ps.load_pretrained_model()
    in_f = model.fc.in_features
    # 출력 1개 = 연속 점수. 등급은 나중에 임계값으로 나눈다(순서 정보를 살리는 방법)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, 1))
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
### 4-2 연속 예측 → 등급: 임계값 최적화
- `to_grade(pred, thresholds)`: 임계값 4개(오름차순 정렬해서 사용)로 0~4 정수 등급 배열 (`np.digitize`)
- `fit_thresholds(pred, y, init=(0.5, 1.5, 2.5, 3.5), n_rounds=3, step=0.05)`: 임계값 하나씩 `[현재값-0.5, 현재값+0.5)` 범위를 `step` 간격으로 바꿔 보며 QWK가 **더 높아지면** 갱신하기를 `n_rounds`번 반복(좌표 탐색). 정렬된 리스트를 반환합니다. QWK는 위 셀의 `qwk(a, b)`를 쓰세요.
""")

write(r"""
def to_grade(pred, thresholds):
    return np.___(np.asarray(pred), np.sort(thresholds))

def fit_thresholds(pred, y, init=(0.5, 1.5, 2.5, 3.5), n_rounds=3, step=0.05):
    th = list(init)
    best = qwk(y, to_grade(pred, th))
    for _ in range(___):
        for k in range(len(th)):
            for cand in np.arange(th[k] - 0.5, th[k] + 0.5, ___):
                trial = th.copy()
                trial[k] = cand
                score = qwk(y, to_grade(pred, trial))
                if ___:                                  # 더 높아졌을 때만 갱신
                    best, th = score, trial
    return sorted(th)
""", r"""
def to_grade(pred, thresholds):
    '''연속 예측 → 0~4 등급 (np.digitize)'''
    # 작성하세요

def fit_thresholds(pred, y, init=(0.5, 1.5, 2.5, 3.5), n_rounds=3, step=0.05):
    '''좌표 탐색으로 QWK를 높이는 임계값 4개(정렬)를 반환'''
    th = list(init)
    # 작성하세요
    return sorted(th)
""", r"""
def to_grade(pred, thresholds):
    # digitize: 값이 몇 번째 경계 이상인지 → 0~4. 경계 순서가 뒤집혀도 안전하게 정렬
    return np.digitize(np.asarray(pred), np.sort(thresholds))

def fit_thresholds(pred, y, init=(0.5, 1.5, 2.5, 3.5), n_rounds=3, step=0.05):
    th = list(init)
    best = qwk(y, to_grade(pred, th))
    for _ in range(n_rounds):                   # 한 경계를 바꾸면 다른 경계의 최적값도 바뀌므로 여러 번 돈다
        for k in range(len(th)):                # 경계를 하나씩(좌표 탐색)
            for cand in np.arange(th[k] - 0.5, th[k] + 0.5, step):
                trial = th.copy()
                trial[k] = cand
                score = qwk(y, to_grade(pred, trial))
                if score > best:                # 같으면 바꾸지 않음(불필요한 흔들림 방지)
                    best, th = score, trial
    return sorted(th)
""")

run(r"""
def _check_4_2():
    assert to_grade([0.2, 0.6, 3.9, 9], [0.5, 1.5, 2.5, 3.5]).tolist() == [0, 1, 4, 4]
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 300); p = y * 0.7 + rng.normal(0, 0.3, 300)       # 일부러 작게 눌린 예측
    th = fit_thresholds(p, y)
    assert th == sorted(th) and len(th) == 4
    print(f"고정 임계값 QWK {qwk(y, to_grade(p, [0.5, 1.5, 2.5, 3.5])):.3f} → 최적화 {qwk(y, to_grade(p, th)):.3f} | {np.round(th, 2)}")
attempt("4-2", _check_4_2)
""")

md(r"""
### 4-3 하이퍼파라미터 탐색 함수
`grid_search(grid, trial_fn, budget_sec=300)`: `grid`는 `{"lr": [...], "scope": [...], ...}` 같은 dict입니다. **모든 조합**(`itertools.product`)마다 `trial_fn(**params)`를 호출해 반환된 validation QWK를 기록합니다. 시작부터 누적 시간이 `budget_sec`를 넘으면 남은 조합은 건너뜁니다(건너뛴 조합을 출력). `(결과 DataFrame(val_qwk 내림차순), 최고 params dict)`를 반환합니다. DataFrame 열: 각 매개변수 이름, `val_qwk`, `sec`.
""")

write(r"""
def grid_search(grid, trial_fn, budget_sec=300):
    rows, start = [], time.time()
    keys = list(grid)
    for values in itertools.___(*(grid[k] for k in keys)):
        params = dict(zip(keys, ___))
        if time.time() - start > ___:
            print("시간 예산 초과 → 건너뜀:", params)
            continue
        t0 = time.time()
        score = trial_fn(**___)
        rows.append({**params, "val_qwk": score, "sec": round(time.time() - t0, 1)})
    table = pd.DataFrame(rows).sort_values("val_qwk", ascending=___).reset_index(drop=True)
    best_row = max(rows, key=lambda r: r[___])       # 점수가 가장 높은 기록
    best = {k: best_row[k] for k in keys}
    return table, best
""", r"""
def grid_search(grid, trial_fn, budget_sec=300):
    '''모든 조합 시도(시간 예산 내) → (val_qwk 내림차순 DataFrame, 최고 params)'''
    # 작성하세요
""", r"""
def grid_search(grid, trial_fn, budget_sec=300):
    rows, start = [], time.time()
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):    # 모든 조합 (데카르트 곱)
        params = dict(zip(keys, values))
        if time.time() - start > budget_sec:                       # 4시간 시험에서는 시간 예산이 실제 제약
            print("시간 예산 초과 → 건너뜀:", params)
            continue
        t0 = time.time()
        score = trial_fn(**params)                                 # validation QWK
        rows.append({**params, "val_qwk": score, "sec": round(time.time() - t0, 1)})
    table = pd.DataFrame(rows).sort_values("val_qwk", ascending=False).reset_index(drop=True)
    best_row = max(rows, key=lambda r: r["val_qwk"])               # 동점이면 먼저 시도한 조합
    best = {k: best_row[k] for k in keys}                          # DataFrame에서 꺼내면 numpy 자료형이 됨 → rows에서
    return table, best
""")

run(r"""
def _check_4_3():
    calls = []
    fake = lambda a, b: calls.append((a, b)) or (a * 10 + b)
    table, best = grid_search({"a": [1, 2], "b": [0, 5]}, fake)
    assert len(calls) == 4 and best == {"a": 2, "b": 5} and list(table.columns) == ["a", "b", "val_qwk", "sec"]
    assert table["val_qwk"].is_monotonic_decreasing
    t2, _ = grid_search({"a": [1, 2, 3]}, lambda a: time.sleep(0.3) or a, budget_sec=0.4)
    assert len(t2) < 3, "시간 예산을 지키지 않습니다"
    print("grid_search 확인 통과")
attempt("4-3", _check_4_3)
""")

md(r"""
### 4-4 baseline 개선 (성능 기준)
아래 셀은 **baseline 설정**(fc만, lr 1e-3, 3 epoch, 증강 없음)의 validation QWK를 보여 줍니다. `Q4_GRID`를 정해(조합 **4개 이하**, `budget_sec=300`) baseline보다 나은 설정을 찾으세요. 키는 `trial_fn`의 인자(`lr`, `scope`, `epochs`, `augment`)입니다. 자가 기준: 최고 validation QWK ≥ 0.92.

생각할 것: 학습 범위가 `fc`뿐이면 사전학습 특징(무늬)을 반점 개수 세기에 맞게 바꿀 수 없습니다. 학습률이 너무 작거나 epoch가 적으면 과소적합합니다.
""")

run(r"""
q4_cache = {}
def trial_fn(lr, scope, epochs, augment):
    ps.seed_everything(42)
    m = ps.train_regressor(build_regressor(), gX, gY, g_tr, scope=scope, lr=lr, epochs=epochs,
                           augment=augment, device=DEVICE)
    pv = ps.predict_regressor(m, gX[g_va], device=DEVICE)
    th = fit_thresholds(pv, g_y[g_va])                     # 임계값도 validation에서 맞춘다(→ 약간 낙관적)
    q4_cache[(lr, tuple(scope), epochs, augment)] = (m, th)
    return qwk(g_y[g_va], to_grade(pv, th))

BASELINE = dict(lr=1e-3, scope=("fc",), epochs=3, augment=False)
with ps.Timer("baseline"):
    q4_base = attempt("4-4 baseline", trial_fn, **BASELINE)
print("baseline validation QWK:", q4_base)
""")

write(r"""
Q4_GRID = {
    "lr": [___],
    "scope": [___],            # 예: ("layer4", "fc")
    "epochs": [___],
    "augment": [___],
}
""", r"""
Q4_GRID = {}   # 작성하세요: 키 lr, scope, epochs, augment / 조합 4개 이하
""", r"""
Q4_GRID = {
    "lr": [1e-3, 3e-3],                       # 학습 범위를 넓히면 적당한 학습률을 다시 찾아야 함
    "scope": [("layer3", "layer4", "fc")],    # fc만으로는 '반점 세기' 특징을 못 배움 → 마지막 두 블록까지
    "epochs": [8],                            # 3 epoch는 과소적합(예측 범위가 좁게 눌림)
    "augment": [True],                        # 반전·밝기는 반점 수를 바꾸지 않는 안전한 증강
}
""")

run(r"""
def _run_4_4():
    n = int(np.prod([len(v) for v in Q4_GRID.values()]))
    assert set(Q4_GRID) == {"lr", "scope", "epochs", "augment"} and 1 <= n <= 4, "키 4개, 조합 4개 이하"
    table, best = grid_search(Q4_GRID, trial_fn, budget_sec=300)
    print(table.to_string(index=False))
    print("baseline:", round(q4_base or 0, 3), "→ 최고:", round(table.val_qwk[0], 3), best,
          "| 자가 기준(≥ 0.92):", table.val_qwk[0] >= 0.92)
    return table, best
q4_out = attempt("4-4", _run_4_4)
if q4_out:
    q4_best = q4_out[1]
    q4_model, q4_th = q4_cache[(q4_best["lr"], tuple(q4_best["scope"]), q4_best["epochs"], q4_best["augment"])]
    q4_test_pred = to_grade(ps.predict_regressor(q4_model, gX_test, device=DEVICE), q4_th)
    print("test 예측 등급 분포:", np.bincount(q4_test_pred, minlength=5))
""")

md(r"""
### 4-5 제출 파일
3-7의 `make_submission`으로 `q4_sub`(`id,grade`, grade는 **정수**)를 만들고 `제출물/q4_submission.csv`로 저장하세요.
""")

write(r"""
q4_sub = make_submission(csv("grade", "sample_submission.csv"), ___, ___, "grade")
q4_sub["grade"] = q4_sub["grade"].astype(___)
q4_sub.to_csv(ps.SUBMIT / "q4_submission.csv", index=False)
""", r"""
# 작성하세요: q4_sub 만들고 제출물/q4_submission.csv 로 저장 (grade는 정수)
""", r"""
q4_sub = make_submission(csv("grade", "sample_submission.csv"), g_test_ids, q4_test_pred, "grade")
q4_sub["grade"] = q4_sub["grade"].astype(int)     # 2.0 같은 실수로 저장되면 형식 위반
q4_sub.to_csv(ps.SUBMIT / "q4_submission.csv", index=False)
""")

run(r"""
print("q4 형식 점검:", ps.check_submission("q4") or "통과")
""")

md(r"""
### 4-6 성능 분석과 개선안 (서술)
아래 셀은 최고 설정의 validation 혼동 행렬과 등급별 재현율을 보여 줍니다. `q4_analysis`에 (1) 어떤 등급끼리 주로 혼동하는지와 그 이유, (2) baseline 대비 무엇이 효과가 있었는지(수치 인용), (3) **다음에 시도할 개선안 2가지**를 쓰세요.
""")

run(r"""
def _analyze_4():
    pv = ps.predict_regressor(q4_model, gX[g_va], device=DEVICE)
    for name, t in [("고정 0.5,1.5,..", [0.5, 1.5, 2.5, 3.5]), ("최적화", q4_th)]:
        print(f"{name:14s} 임계값 QWK {qwk(g_y[g_va], to_grade(pv, t)):.3f}")
    gv = to_grade(pv, q4_th)
    print(pd.DataFrame(confusion_matrix(g_y[g_va], gv, labels=range(5)), index=[f"실제{k}" for k in range(5)], columns=[f"예측{k}" for k in range(5)]))
    print("등급별 재현율", np.round(recall_score(g_y[g_va], gv, labels=range(5), average=None), 2), "| 임계값", np.round(q4_th, 2))
attempt("4-6", _analyze_4)
""")

write(r"""
q4_analysis = ""  # 작성하세요
""", r"""
q4_analysis = ""  # 작성하세요
""", r"""
q4_analysis = (
    "__Q4_ANALYSIS__"
)
""")

# ======================================================================  Q5
md(r"""
---
## Q5 정보 추출 프롬프트와 구조화 출력 (프롬프트 엔지니어링 · 구현 + 기준)

가상의 진료 메모(`data/notes/notes.csv`: `id, split, note`)에서 정보를 뽑아 **소프트웨어가 바로 쓸 수 있는 JSON**으로 만듭니다. 메모는 모두 합성된 가상 자료입니다.
- `dev` 40개는 정답(`dev_labels.json`)이 있고, `test` 30개는 정답이 없습니다.
- **FAQ: 지정 모델은 이미지 모델이라 채팅이 안 됩니다.** 그래서 이 문항은 언어모델을 실행하지 않습니다. 대신 `ps.mock_llm(note_id)`가 **미리 만든 응답 문자열**을 돌려줍니다(여러분의 프롬프트와 무관). 실제 응답에서 흔한 문제(코드 블록, 앞뒤 설명문, 문자열 숫자, 단위 미변환, `unknown`, 여분의 키, 깨진 JSON)를 일부러 섞었습니다.
- **기준(가정): dev 필드 정확도 ≥ 0.95, 자료형 위반 레코드 0건.** 제출: `제출물/q5_submission.jsonl` (줄마다 `{"id": ..., "record": {...}}`)

**출력 스키마(필드 8개, 모르면 null)**

| 필드 | 자료형 | 규칙 |
|---|---|---|
| age | 정수 또는 null | 나이(세) |
| sex | `"M"`, `"F"` 또는 null | |
| chief_complaint | 문자열 enum | `fever, cough, headache, abdominal_pain, rash` 중 하나 |
| symptom_days | 정수 또는 null | 증상 기간(일). 일주일=7, 2주=14. "며칠"처럼 불확실하면 null |
| temperature_c | 실수 또는 null | 섭씨, 소수 첫째 자리. 화씨(F)는 섭씨로 변환. "미열"은 null |
| medications | 배열 | 원소 `{"name": 성분명 enum, "dose_mg": 실수|null, "times_per_day": 정수|null}`. g는 mg로. "필요시"는 null |
| allergy | true/false/null | "알레르기 없음"은 **false**(명시적 부정), 언급이 없으면 **null**(미확정) |
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

print(dev.note[0]); print("---- 모의 응답 ----"); print(ps.mock_llm(dev.id[0]))
""")

md(r"""
### 5-1 프롬프트 설계
`build_prompt(note)`: 언어모델에 보낼 **자연어 프롬프트 문자열**을 반환합니다. 다음을 모두 담으세요.
- 역할과 작업, 출력은 **JSON 객체 하나만**(설명문·코드 블록 금지)
- 8개 필드 이름·자료형·허용값(위 표)
- 정규화 규칙: 섭씨·소수 첫째 자리, 화씨 변환, 일 단위, mg 단위, 성분명, 성별 M/F
- **미확정 정보 처리**: 메모에 없거나 불확실하면 추측하지 말고 `null`. 명시적 부정("없음")은 `false`
- 예시 1개(few-shot), 메모 원문을 구분 기호로 감싸서 넣기

확인 셀의 `ps.check_prompt`는 빠진 요소만 점검하며 품질을 채점하지 않습니다.
""")

write(r"""
FEW_SHOT_NOTE = "여, 34세. 2주 전부터 마른기침. 체온 100.4F. 아목시실린 0.5g 하루 세 번 처방. 호전 없으면 재방문."
FEW_SHOT_JSON = {"age": 34, "sex": "F", "chief_complaint": "cough", "symptom_days": 14, "temperature_c": 38.0,
                 "medications": [{"name": "amoxicillin", "dose_mg": 500, "times_per_day": 3}],
                 "allergy": None, "follow_up_days": None}

def build_prompt(note):
    prompt = f'''당신은 진료 메모에서 정보를 추출해 ___으로 변환하는 도구입니다.
아래 [메모]를 읽고 ___ 하나만 출력하세요. 설명 문장, 코드 블록(```), 추가 키는 출력하지 마세요.

[출력 스키마] 키 8개를 모두 포함합니다.
- "age": 정수(세) 또는 null
- "sex": "M" 또는 "F" 또는 null (남/남성/남아 → "M", 여/여성/여아 → "F")
- "chief_complaint": "fever", "cough", "headache", "abdominal_pain", "rash" 중 하나
- "symptom_days": 정수(일) 또는 null (일주일=7, 2주=14)
- "temperature_c": ___(°C) 실수, 소수 첫째 자리 또는 null (화씨 F는 (F-32)×5/9로 변환)
- "medications": 배열. 원소는 {{"name": 성분명, "dose_mg": 숫자(___) 또는 null, "times_per_day": 정수 또는 null}}
  성분명은 "acetaminophen"(타이레놀), "ibuprofen"(애드빌·이부프로펜), "amoxicillin", "cetirizine" 중 하나. g는 mg로 변환(0.5g=500)
- "allergy": true, false 또는 null
- "follow_up_days": 재방문까지 일수(정수) 또는 null

[미확정 정보 규칙]
___

[예시]
메모: {FEW_SHOT_NOTE}
출력: {json.dumps(FEW_SHOT_JSON, ensure_ascii=False)}

[메모]
<<<
{___}
>>>
출력:'''
    return prompt
""", r"""
def build_prompt(note):
    '''위 요구사항을 모두 담은 프롬프트 문자열'''
    prompt = ""  # 작성하세요
    return prompt
""", r"""
# 예시 1개(few-shot): 규칙이 실제로 어떻게 적용되는지 보여 주어 형식 오류를 줄인다
FEW_SHOT_NOTE = "여, 34세. 2주 전부터 마른기침. 체온 100.4F. 아목시실린 0.5g 하루 세 번 처방. 호전 없으면 재방문."
FEW_SHOT_JSON = {"age": 34, "sex": "F", "chief_complaint": "cough", "symptom_days": 14, "temperature_c": 38.0,
                 "medications": [{"name": "amoxicillin", "dose_mg": 500, "times_per_day": 3}],
                 "allergy": None, "follow_up_days": None}

def build_prompt(note):
    # f-string 안의 JSON 중괄호는 {{ }}로 이스케이프
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
"알레르기 없음"처럼 명시적으로 부정한 정보는 false이고, 언급이 없으면 null입니다.
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
    assert "___" not in p, "빈칸이 남아 있습니다"
    missing = ps.check_prompt(p, dev.note[0], SCHEMA_KEYS)
    print(p[:600] + ("..." if len(p) > 600 else ""))
    print("\n빠진 요소:", missing or "없음", "| 길이", len(p))
    assert not missing
attempt("5-1", _check_5_1)
""")

md(r"""
### 5-2 JSON 스키마
`RECORD_SCHEMA`를 **JSON Schema** 형식의 dict로 작성하세요. 모든 키 필수(`required`), 여분 키 금지(`additionalProperties: false`), 위 표의 자료형·enum·null 허용을 반영합니다. `medications`의 원소도 객체 스키마로 정의합니다. (확인 셀은 dev 정답 40개가 모두 통과하는지, 잘못된 예 6개가 걸러지는지 봅니다.)
""")

write(r"""
RECORD_SCHEMA = {
    "type": "object",
    "required": ___,                                  # 8개 키 모두
    "additionalProperties": ___,                      # 여분 키 금지
    "properties": {
        "age": {"type": [___, "null"], "minimum": 0, "maximum": 120},
        "sex": {"enum": ___},                         # null도 허용
        "chief_complaint": {"enum": COMPLAINTS},
        "symptom_days": {"type": ["integer", "null"], "minimum": 0},
        "temperature_c": {"type": [___, "null"], "minimum": 30, "maximum": 45},
        "medications": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "dose_mg", "times_per_day"],
                "additionalProperties": False,
                "properties": {
                    "name": {"enum": ___},
                    "dose_mg": {"type": ["number", "null"], "exclusiveMinimum": 0},
                    "times_per_day": {"type": ["integer", "null"], "minimum": 1, "maximum": 6},
                },
            },
        },
        "allergy": {"type": [___, "null"]},
        "follow_up_days": {"type": ["integer", "null"], "minimum": 0},
    },
}
""", r"""
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
`extract_json(raw)`: 응답 문자열에서 JSON 객체(dict)를 꺼냅니다. 실패하면 `None`. 다음 순서로 시도하세요.
1. 첫 `{`부터 마지막 `}`까지 잘라내기(코드 블록·앞뒤 설명문 제거)
2. `json.loads`
3. 실패하면 `}`나 `]` 앞의 끝 쉼표를 정규식으로 지우고 다시 `json.loads`
4. 그래도 실패하면 `ast.literal_eval`(파이썬 dict 표기 대비). 결과가 dict가 아니면 None
""")

write(r"""
def extract_json(raw):
    if not isinstance(raw, str):
        return None
    start, end = raw.find("{"), raw.___("}")              # 마지막 }
    if start == -1 or end <= start:
        return None
    body = raw[start:___]
    candidates = [body, re.sub(r",\s*([}\]])", ___, body)]   # 끝 쉼표 제거본
    for text in candidates:
        try:
            obj = json.___(text)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    try:
        obj = ast.___(body)
        return obj if isinstance(obj, dict) else None
    except (ValueError, SyntaxError):
        return None
""", r"""
def extract_json(raw):
    '''위 1~4 순서로 dict를 꺼낸다. 실패하면 None'''
    # 작성하세요
""", r"""
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
- 키: 스키마의 8개만 남기고(스키마 순서), 없는 키는 null
- 문자열은 앞뒤 공백 제거·소문자로 비교. `UNKNOWN`에 속하면 null
- `age`, `symptom_days`, `follow_up_days`, `times_per_day`: 문자열에서 숫자를 뽑아 정수. `"1주"`→7, `"2주"`→14, `"일주일"`→7
- `temperature_c`: 실수로 바꾸고 **45보다 크면 화씨로 보고 섭씨로 변환**, 소수 첫째 자리 반올림
- `dose_mg`: `"0.5g"`→500, `"500mg"`/`"500 mg"`→500
- `sex`, `chief_complaint`, 약 이름, `allergy`: 제공한 사전(`SEX_MAP` 등)으로 변환. 사전에 없으면 null (약은 이름이 null이면 그 약을 제외)
- **불리언과 정수 구분**: `True`는 파이썬에서 `int`이기도 하므로 정수 필드에 불리언이 오면 null
""")

write(r"""
def _is_unknown(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in ___)

def _to_number(v):
    if isinstance(v, ___) or _is_unknown(v):          # True==1 함정: bool은 숫자로 보지 않음
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(___, str(v))                        # 문자열 속 첫 숫자(소수 포함)
    return float(m.group()) if m else None

def _to_int(v):
    x = _to_number(v)
    return int(round(x)) if x is not None else None

def _to_days(v):
    if isinstance(v, str):
        s = v.strip()
        if "일주일" in s:
            return ___
        if "주" in s and _to_number(s) is not None:
            return int(_to_number(s) * ___)
    return _to_int(v)

def _to_temp_c(v):
    x = _to_number(v)
    if x is None:
        return None
    if x > ___:                                       # 화씨로 판단
        x = ___
    return round(x, ___)

def _to_dose_mg(v):
    x = _to_number(v)
    if x is None:
        return None
    if isinstance(v, str) and re.search(r"\d\s*g\b", v.strip().lower()) and "mg" not in v.lower():
        x *= ___
    return x

def _map(v, table):
    if not isinstance(v, str) or _is_unknown(v):
        return None
    return table.get(v.strip().lower(), table.get(v.strip()))

def _to_bool(v):
    if isinstance(v, bool):
        return v
    return _map(v, ___)

def normalize_record(obj):
    obj = obj if isinstance(obj, dict) else {}
    rec = {}
    rec["age"] = _to_int(obj.get("age"))
    rec["sex"] = obj.get("sex") if obj.get("sex") in ("M", "F") else _map(obj.get("sex"), ___)
    rec["chief_complaint"] = _map(obj.get("chief_complaint"), COMPLAINT_MAP)
    rec["symptom_days"] = ___(obj.get("symptom_days"))
    rec["temperature_c"] = _to_temp_c(obj.get("temperature_c"))
    meds = []
    for m in obj.get("medications") or []:
        if not isinstance(m, dict):
            continue
        name = _map(m.get("name"), DRUG_MAP)
        if name is None:
            ___                                       # 성분을 알 수 없는 약은 제외
        meds.append({"name": name, "dose_mg": _to_dose_mg(m.get("dose_mg")),
                     "times_per_day": _to_int(m.get("times_per_day"))})
    rec["medications"] = meds
    rec["allergy"] = _to_bool(obj.get("allergy"))
    rec["follow_up_days"] = _to_days(obj.get("follow_up_days"))
    return {k: rec[k] for k in ___}                   # 스키마 키 순서, 여분 키 없음
""", r"""
def normalize_record(obj):
    '''위 규칙으로 스키마를 지키는 레코드(dict)를 반환. 도우미 함수를 나눠 써도 됩니다.'''
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
    if set(rec) != set(SCHEMA_KEYS):
        errors.append(f"키 불일치: {sorted(set(rec) ^ set(SCHEMA_KEYS))}")
        return errors
    is_int = lambda v: isinstance(v, int) and not isinstance(v, ___)
    is_num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    for k in ["age", "symptom_days", "follow_up_days"]:
        if rec[k] is not None and not ___(rec[k]):
            errors.append(f"{k}: 정수/null이어야 함 ({rec[k]!r})")
    if rec["temperature_c"] is not None and not is_num(rec["temperature_c"]):
        errors.append(f"temperature_c: 숫자/null이어야 함 ({rec['temperature_c']!r})")
    if rec["sex"] not in ("M", "F", ___):
        errors.append(f"sex: 허용값 아님 ({rec['sex']!r})")
    if rec["chief_complaint"] not in ___:
        errors.append(f"chief_complaint: 허용값 아님 ({rec['chief_complaint']!r})")
    if rec["allergy"] is not None and not isinstance(rec["allergy"], ___):
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
""", r"""
def type_errors(rec):
    '''자료형 위반 설명 문자열 리스트 (없으면 [])'''
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
    print("기준(필드 정확도 ≥ 0.95, 위반 0건):", bool(acc.mean() >= 0.95 and n_err == 0))
    raw_acc = np.mean([[(_extract(ps.mock_llm(n)) or {}).get(k) == dev_gold[n][k] for k in SCHEMA_KEYS] for n in dev.id])
    print(f"(참고) 정규화 없이 응답 그대로 쓸 때 필드 정확도 {raw_acc:.3f}")
    out = [{"id": nid, "record": normalize_record(_extract(ps.mock_llm(nid)))} for nid in test_notes.id]
    ps.save_jsonl(out, "q5_submission.jsonl")
    print("q5 형식 점검:", ps.check_submission("q5") or "통과")
_ = attempt("5-5", _run_5_5)
""")

md(r"""
---
## 제출 전 점검 (제출가이드 + 코드 리뷰 대비)
""")

run(r"""
for k in ["q3", "q4", "q5"]:
    print(k, "→", ps.check_submission(k) or "형식 통과")
for name in ["q1_visual_note", "q2_aug_note", "q3_analysis", "q4_analysis"]:
    print(f"{name}: {'작성함' if str(globals().get(name, '')).strip() else '비어 있음'}")
""")

md(r"""
- 형식 오류가 하나라도 있으면 그 문항은 0점입니다(FAQ). 위 셀이 모두 `형식 통과`인지 확인하세요.
- 코드 리뷰 관점: 함수마다 한 가지 일, 의미 있는 이름, 핵심 결정에 짧은 주석, 크기·개수 하드코딩 없음(`in_features`, `len(...)`), seed 고정, test 정답·test 점수 미사용, 원본 데이터 불변(`copy`, `clone`), 장치 일치(`.to(DEVICE)`).
- 커널 재시작 → 모두 실행(Run All) 후 출력이 남은 상태로 저장하세요.
- 풀이가 끝나면 `python -I 정답_채점용/score_local.py`로 test 기준 통과 여부를 확인하고 `정답_채점용/채점기준.md`와 비교하세요.
""")


# ======================================================================
def build(kind):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for i, (cell_kind, a, b, c) in enumerate(CELLS):
        if cell_kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(a))
        elif cell_kind == "run":
            nb.cells.append(nbf.v4.new_code_cell(a))
        else:
            src = {"A": "# [작성하세요] ___ 를 채우세요\n" + a,
                   "B": "# [작성하세요]\n" + b,
                   "answer": "# [정답 예시]\n" + c}[kind]
            nb.cells.append(nbf.v4.new_code_cell(src))
    head = {"A": "\n\n> **버전 A — 핵심 빈칸.** `___` 자리만 채우면 됩니다(대부분 한 단어·한 줄). 서술 답안은 직접 씁니다.",
            "B": "\n\n> **버전 B — 직접 구현.** 함수 이름·인자·설명만 주어집니다. 실전에 가장 가까운 판입니다.",
            "answer": "\n\n> **정답·해설용입니다.** `[정답 예시]` 셀은 줄마다 이유를 주석으로 단 풀이 하나이며, 같은 요구사항을 만족하는 다른 코드도 정답입니다."}[kind]
    nb.cells[0].source += head
    return nb


def fill_answers(nb, answers):
    for cell in nb.cells:
        for key, text in answers.items():
            if key in cell.source:
                cell.source = cell.source.replace(key, text)
    return nb


if __name__ == "__main__":
    import json, sys
    answers_path = Path(__file__).with_name("analysis_answers.json")
    answers = json.loads(answers_path.read_text(encoding="utf-8")) if answers_path.exists() else {}
    nbf.write(build("A"), ROOT / "문제_A_빈칸.ipynb")
    nbf.write(build("B"), ROOT / "문제_B_직접구현.ipynb")
    nbf.write(fill_answers(build("answer"), answers), ROOT / "정답_채점용" / "정답_해설.ipynb")
    print("cells:", len(CELLS), "| 작성 셀:", sum(k == "write" for k, *_ in CELLS))
