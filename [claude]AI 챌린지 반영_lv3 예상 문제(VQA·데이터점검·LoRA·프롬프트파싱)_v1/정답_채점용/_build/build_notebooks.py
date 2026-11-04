"""AI 챌린지 반영 VQA 모의문제 노트북 생성기.  실행: python -I build_notebooks.py

환경 2종(local / colab) × 문제 버전 3종(v1 핵심 빈칸, v2 긴 빈칸, v3 직접 구현) + 정답 2종.
코드 셀은 C(answer, v1, v2, v3, env=None)로 정의한다. v1/v2/v3가 None이면 정답과 같은 코드(실행만).
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
CELLS = []   # (kind, env, payload)


def M(text, env=None):
    CELLS.append(("md", env, text.strip("\n")))


def C(answer, v1=None, v2=None, v3=None, env=None):
    CELLS.append(("code", env, {"answer": answer.strip("\n"),
                                "v1": None if v1 is None else v1.strip("\n"),
                                "v2": None if v2 is None else v2.strip("\n"),
                                "v3": None if v3 is None else v3.strip("\n")}))


VERSION_NOTE = {
    "v1": "**버전 1 — 핵심 빈칸.** `___` 자리만 채우면 됩니다(대부분 한 줄).",
    "v2": "**버전 2 — 긴 빈칸.** 함수 안의 `# 작성하세요` 부분을 몇 줄로 채웁니다. 주석의 단계 순서를 따르세요.",
    "v3": "**버전 3 — 직접 구현.** 함수 이름·인자·설명만 주어집니다. 설명을 만족하도록 직접 구현합니다.",
    "answer": "**정답·해설.** 줄마다 '무엇을, 왜' 하는지 주석을 달았습니다. 같은 동작을 하는 다른 코드도 정답입니다.",
}

# ======================================================================
# 0. 머리말
# ======================================================================
M(r"""
# [claude] AI 챌린지 반영 Lv3 모의문제 — 이미지 객관식 VQA (로컬 CPU판)

SSAFY AI 챌린지 baseline(`(260908)_baseline_colab.ipynb`)과 **같은 흐름**으로 만든 창작 연습문제입니다. 실제 시험 문제가 아닙니다.

- 이 판은 GPU 없이 돌아가도록 **작은 연습용 모델(`mini_vlm.MiniVLM`)** 을 씁니다. **실제 Qwen2.5-VL이 아닙니다.** 함수·변수 이름과 순서는 baseline과 같게 맞췄습니다. 실제 모델 코드는 `Colab_실제판` 폴더에 있습니다.
- 데이터는 도형 그림에 대한 합성 객관식 문제입니다(`../data`). 원래 챌린지 데이터가 아닙니다.
- 전체 실행: 약 5~8분(CPU).

{VERSION}

| 단계 | baseline에서의 위치 | Lv3 공지 항목 |
|---|---|---|
| 1 데이터 점검 | (없음 → 추가) | 데이터셋 구성·품질·분포 점검, 육안 검사 |
| 2 프롬프트 | 프롬프트 템플릿 | 프롬프트 설계 |
| 3 증강 | (없음 → 추가) | 데이터 변환/증강, 학습 데이터 재구성 |
| 4 Dataset·Collator | Custom Dataset, Collator | 손실 함수의 이해(정답 토큰만 손실) |
| 5 모델·LoRA | 모델, Processor | 전이학습, 학습 범위 제어 |
| 6 학습 | fine-tuning | 딥러닝 모델 학습, 하이퍼파라미터 |
| 7 추론·파싱 | inference | 구조화 출력 파싱 |
| 8 성능 개선 | (없음 → 추가) | 성능 분석, 개선안 도출 → 개선된 CSV 제출 |
""", env="local")

M(r"""
# [claude] AI 챌린지 반영 Lv3 모의문제 — 이미지 객관식 VQA (Colab 실제판)

SSAFY AI 챌린지 baseline(`(260908)_baseline_colab.ipynb`)과 **같은 모델·라이브러리**(Qwen2.5-VL-3B, 4bit 양자화, LoRA)로 만든 창작 연습문제입니다. 실제 시험 문제가 아닙니다.

- **Colab GPU(T4 등)** 에서 실행합니다: 런타임 → 런타임 유형 변경 → GPU.
- 데이터는 합성 도형 그림 객관식 문제 `vqa_mock_data.zip`입니다. 구글 드라이브 `MyDrive/`에 올려 두세요. 원래 챌린지 데이터가 아닙니다.
- **이 판의 코드는 작성자가 실행해 보지 못했습니다**(작성 PC에 GPU 없음). 같은 코드를 로컬 CPU판에서 작은 모델로 검증했고, 실제 모델 부분은 baseline 코드를 따랐습니다. 4-3의 확인 셀로 정답 토큰 마스크가 맞는지 꼭 확인하세요.
- 시간(대략): 모델 다운로드 10~20분, 기준 학습 10~20분, 추론 10분 안팎.

{VERSION}

| 단계 | baseline에서의 위치 | Lv3 공지 항목 |
|---|---|---|
| 1 데이터 점검 | (없음 → 추가) | 데이터셋 구성·품질·분포 점검, 육안 검사 |
| 2 프롬프트 | 프롬프트 템플릿 | 프롬프트 설계 |
| 3 증강 | (없음 → 추가) | 데이터 변환/증강, 학습 데이터 재구성 |
| 4 Dataset·Collator | Custom Dataset, Collator | 손실 함수의 이해(정답 토큰만 손실) |
| 5 모델·LoRA | 모델, Processor | 전이학습, 학습 범위 제어(양자화 + LoRA) |
| 6 학습 | fine-tuning | 딥러닝 모델 학습, 하이퍼파라미터 |
| 7 추론·파싱 | inference | 구조화 출력 파싱 |
| 8 성능 개선 | (없음 → 추가) | 성능 분석, 개선안 도출 → 개선된 CSV 제출 |
""", env="colab")

# ======================================================================
# 환경·데이터·설정
# ======================================================================
M(r"""
# 환경 준비
이 판은 numpy, pandas, torch, pillow, matplotlib만 씁니다. `mini_vlm.py`가 transformers·peft의 이름을 흉내 냅니다.
""", env="local")
C(r"""
import torch
print("Torch version:", torch.__version__)
""", env="local")

M(r"""
# 환경 준비
baseline과 같습니다. 아래 셀 실행 후 **런타임 → 세션 다시 시작**.
""", env="colab")
C(r"""
!pip -q install git+https://github.com/huggingface/transformers accelerate
!pip -q install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
!pip -q install "peft>=0.13.2" "bitsandbytes==0.46.1" datasets pillow pandas --upgrade
""", env="colab")
C(r"""
import torch
print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
""", env="colab")

M(r"""
# 데이터 준비
`vqa_mock_data.zip`을 구글 드라이브 `MyDrive/`에 올린 뒤 실행합니다. `/content/`에 `train.csv, test.csv, sample_submission.csv, train/, test/`가 풀립니다.
""", env="colab")
C(r"""
from google.colab import drive
drive.mount('/content/drive')
""", env="colab")
C(r"""
!unzip -q -o "/content/drive/MyDrive/vqa_mock_data.zip" -d /content/
DATA_DIR = "/content"
""", env="colab")

M(r"""
# 라이브러리, 데이터, 설정
`BASE_CFG`는 baseline과 같은 설정입니다(200개만 학습, 1 epoch, lr 1e-4, LoRA r=8). 8단계에서 이 값을 바꿔 성능을 올립니다.
""")
C(r"""
import os, re, math, random, gc
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Any
import matplotlib.pyplot as plt
import sys
for _p in [".", "../로컬_CPU판"]:                 # mini_vlm.py 위치(문제 폴더 또는 정답 폴더에서 실행)
    if os.path.exists(os.path.join(_p, "mini_vlm.py")):
        sys.path.insert(0, _p)
from mini_vlm import (load_pretrained, MiniProcessor, LoraConfig, get_peft_model,
                      prepare_model_for_kbit_training, get_linear_schedule_with_warmup)
try:
    from tqdm.auto import tqdm
except ImportError:                      # 로컬에 tqdm이 없으면 간단한 대체 사용
    from mini_vlm import simple_tqdm as tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
torch.set_num_threads(4)

DATA_DIR = "../data"
OUT_DIR = "./제출물"                     # 제출 CSV 저장 위치
MAX_NEW_TOKENS = 4
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

BASE_CFG = dict(tag="baseline", n_train=200, epochs=1, lr=1e-4, grad_accum=4,
                lora_r=8, lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                label_mask=False, option_shuffle=False)
os.makedirs(OUT_DIR, exist_ok=True)
""", env="local")
C(r"""
import os, re, math, random, gc
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Any
import matplotlib.pyplot as plt
from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
                          BitsAndBytesConfig, get_linear_schedule_with_warmup)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm.auto import tqdm

Image.MAX_IMAGE_PIXELS = None
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_SIZE = 384
OUT_DIR = "/content"
MAX_NEW_TOKENS = 4
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

BASE_CFG = dict(tag="baseline", n_train=200, epochs=1, lr=1e-4, grad_accum=4,
                lora_r=8, lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                label_mask=False, option_shuffle=False)
""", env="colab")

# ======================================================================
# 1. 데이터 점검
# ======================================================================
M(r"""
# 1. 데이터 점검 (공지: 데이터셋 구성·품질·분포 점검, 육안 검사)
baseline은 데이터를 바로 학습에 썼습니다. 여기서는 먼저 점검합니다. **train에는 품질 문제가 섞여 있습니다.**
- 1-1 `normalize_answer(x)`: `' B'`, `'(c)'`, `'d.'`처럼 흔들린 정답 표기를 `'a'~'d'`로. 알 수 없으면 `None`
- 1-2 `find_bad_images(df, data_dir)`: 사진 파일이 **없는** 행과 **단색으로 깨진**(픽셀 표준편차 < 1) 행의 id 목록 두 개를 반환
""")
C(r"""
VALID = ["a", "b", "c", "d"]

def normalize_answer(x):
    s = str(x).strip().lower()      # 앞뒤 공백 제거, 대문자 → 소문자
    s = s.strip("().")              # 양 끝의 괄호·마침표 제거: '(c)' → 'c', 'd.' → 'd'
    return s if s in VALID else None   # 그래도 a~d가 아니면 '알 수 없음'

def find_bad_images(df, data_dir):
    missing, broken = [], []
    for row in df.itertuples():
        path = os.path.join(data_dir, row.path)
        if not os.path.exists(path):              # 파일 자체가 없음
            missing.append(row.id)
            continue
        pixels = np.asarray(Image.open(path).convert("RGB"))
        if pixels.std() < 1.0:                    # 모든 픽셀이 거의 같은 값 → 검은 화면 등 정보 없음
            broken.append(row.id)
    return missing, broken
""", v1=r"""
VALID = ["a", "b", "c", "d"]

def normalize_answer(x):
    s = str(x).___().lower()        # 앞뒤 공백 제거, 소문자
    s = s.strip("().")              # 괄호·마침표 제거
    return s if s in ___ else None

def find_bad_images(df, data_dir):
    missing, broken = [], []
    for row in df.itertuples():
        path = os.path.join(data_dir, row.path)
        if not ___:                               # 파일이 없으면
            missing.append(row.id)
            continue
        pixels = np.asarray(Image.open(path).convert("RGB"))
        if ___ < 1.0:                             # 픽셀 표준편차가 1 미만이면 깨진 사진
            broken.append(row.id)
    return missing, broken
""", v2=r"""
VALID = ["a", "b", "c", "d"]

def normalize_answer(x):
    # 작성하세요: (1) 문자열로 바꾸고 공백 제거·소문자 (2) 양 끝 '(', ')', '.' 제거 (3) a~d가 아니면 None
    pass

def find_bad_images(df, data_dir):
    missing, broken = [], []
    for row in df.itertuples():
        path = os.path.join(data_dir, row.path)
        # 작성하세요: 파일이 없으면 missing에 id 추가 후 다음 행으로,
        #            있으면 RGB 배열로 읽어 표준편차 < 1.0이면 broken에 id 추가
        pass
    return missing, broken
""", v3=r"""
VALID = ["a", "b", "c", "d"]

def normalize_answer(x):
    '''흔들린 정답 표기(' B', '(c)', 'd.')를 'a'~'d'로 정리한다. 알 수 없으면 None.'''
    # 작성하세요

def find_bad_images(df, data_dir):
    '''(missing, broken): 사진 파일이 없는 행 id 목록, 픽셀 표준편차 < 1.0인 행 id 목록.'''
    # 작성하세요
""")

C(r"""
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print("train", train_df.shape, "test", test_df.shape)
print("원래 정답 표기:", sorted(train_df["answer"].astype(str).unique()))

assert normalize_answer(" B") == "b" and normalize_answer("(c)") == "c" and normalize_answer("d.") == "d"
assert normalize_answer("e") is None
train_df["answer"] = train_df["answer"].map(normalize_answer)
missing, broken = find_bad_images(train_df, DATA_DIR)
print(f"사진 없음 {len(missing)}개, 깨진 사진 {len(broken)}개, 정답 불명 {train_df['answer'].isna().sum()}개")

clean_df = train_df[~train_df["id"].isin(missing + broken) & train_df["answer"].notna()].reset_index(drop=True)
print("정제 후", len(clean_df), "개")
print("정답 글자 분포(train):", clean_df["answer"].value_counts(normalize=True).round(3).to_dict())
""")

M(r"""
### 1-3 육안 검사와 분포 점검
아래 셀은 무작위 8장과 질문·보기·정답을 보여 줍니다. **그림을 보고 정답이 맞는지 직접 확인하세요.** 위에서 본 정답 글자 분포도 생각해 보세요. 정답이 고르게 퍼져 있지 않다면 모델은 어떻게 배울까요?
""")
C(r"""
sample = clean_df.sample(8, random_state=0)
fig, axes = plt.subplots(1, 8, figsize=(16, 2.4))
for ax, row in zip(axes, sample.itertuples()):
    ax.imshow(Image.open(os.path.join(DATA_DIR, row.path)))
    ax.set_title(f"{row.id[-4:]}  ans={row.answer}", fontsize=9)
    ax.axis("off")
plt.show()
for row in sample.itertuples():
    print(f"{row.id[-4:]} | {row.question} | a){row.a} b){row.b} c){row.c} d){row.d} | 정답 {row.answer}")
""")

C(r"""
# 서술: 육안 검사·분포 점검에서 발견한 점과 학습에 주는 의미 (2~3문장)
data_note = ("train 정답의 약 49%가 a라서, 그대로 학습하면 모델이 내용과 상관없이 a를 고르는 쪽으로 기울 수 있다. "
             "test는 정답이 고르게 퍼져 있을 수 있으므로 보기 순서를 섞는 증강(3단계)으로 이 쏠림을 없앤다. "
             "육안 검사에서 그림과 정답이 일치함을 확인했고, 사진이 없거나 검은 화면인 행은 제거했다.")
""", v1=r"""
# 서술: 육안 검사·분포 점검에서 발견한 점과 학습에 주는 의미 (2~3문장)
data_note = ""
""", v2=r"""
# 서술: 육안 검사·분포 점검에서 발견한 점과 학습에 주는 의미 (2~3문장)
data_note = ""
""", v3=r"""
# 서술: 육안 검사·분포 점검에서 발견한 점과 학습에 주는 의미 (2~3문장)
data_note = ""
""")

M(r"""
### 1-4 검증 데이터 고정
여러 설정을 **공정하게** 비교하려면 검증 데이터가 매번 같아야 합니다. baseline은 학습 200개의 뒤 10%(20개)를 검증으로 썼기 때문에, 학습 데이터 수를 바꾸면 검증 데이터도 바뀌었습니다. 여기서는 정제된 데이터에서 **100개를 먼저 떼어 고정**하고, 나머지(`pool_df`)에서 학습 데이터를 뽑습니다.
""")
C(r"""
valid_df = clean_df.sample(n=100, random_state=SEED).reset_index(drop=True)
pool_df = clean_df[~clean_df["id"].isin(valid_df["id"])].reset_index(drop=True)
print("검증", len(valid_df), "| 학습 후보", len(pool_df))
""")

# ======================================================================
# 2. 프롬프트
# ======================================================================
M(r"""
# 2. 프롬프트 템플릿 (공지: 프롬프트 설계)
`build_mc_prompt(question, a, b, c, d)`: 질문 다음 줄부터 `(a) 보기` 형식으로 네 줄, 빈 줄, 마지막에 "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."를 붙인 문자열. (baseline과 같은 형식입니다. 이 판의 연습용 모델이 이 형식으로 사전학습되어 있습니다.)
""")
C(r"""
SYSTEM_INSTRUCT = (
    "You are a helpful visual question answering assistant. "
    "Answer using exactly one letter among a, b, c, or d. No explanation."
)

def build_mc_prompt(question, a, b, c, d):
    return (
        f"{question}\n"                                  # 질문
        f"(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n"        # 보기 네 줄 + 빈 줄
        "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."   # 출력 형식 지시
    )
""", v1=r"""
SYSTEM_INSTRUCT = (
    "You are a helpful visual question answering assistant. "
    "Answer using exactly one letter among a, b, c, or d. No explanation."
)

def build_mc_prompt(question, a, b, c, d):
    options_text = ___      # f-string으로 "(a) 보기\n(b) 보기\n(c) 보기\n(d) 보기\n\n" 형식 만들기
    return f"{question}\n" + options_text + "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
""", v2=r"""
SYSTEM_INSTRUCT = (
    "You are a helpful visual question answering assistant. "
    "Answer using exactly one letter among a, b, c, or d. No explanation."
)

def build_mc_prompt(question, a, b, c, d):
    # 작성하세요: 질문 / (a)~(d) 네 줄 / 빈 줄 / 출력 형식 지시문 순서의 문자열을 반환
    pass
""", v3=r"""
SYSTEM_INSTRUCT = (
    "You are a helpful visual question answering assistant. "
    "Answer using exactly one letter among a, b, c, or d. No explanation."
)

def build_mc_prompt(question, a, b, c, d):
    '''위 설명의 형식으로 사용자 프롬프트 문자열을 만든다.'''
    # 작성하세요
""")
C(r"""
p = build_mc_prompt("Q?", "x", "y", "z", "w")
assert p == "Q?\n(a) x\n(b) y\n(c) z\n(d) w\n\n정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요.", p
print(p)
""")

# ======================================================================
# 3. 증강
# ======================================================================
M(r"""
# 3. 데이터 증강 (공지: 데이터 변환/증강, 학습 데이터 재구성)
`shuffle_options(row, rng)`: 한 행의 보기 a~d 순서를 무작위로 섞고, **정답 글자를 정답 내용이 옮겨 간 위치로** 바꾼 새 행(`row.copy()`)을 반환합니다. 같은 문제라도 정답 글자가 매번 달라지므로 1-3에서 본 정답 글자 쏠림을 줄입니다.

서술 `aug_note`: 이 데이터에서 **이미지 좌우 반전**을 증강으로 쓰면 안 되는 질문은 무엇이고, 왜인가요?
""")
C(r"""
def shuffle_options(row, rng):
    options = [row["a"], row["b"], row["c"], row["d"]]
    correct_text = options[VALID.index(row["answer"])]   # 정답 '글자'가 아니라 '내용'을 기억
    order = rng.permutation(4)                            # 예: [2, 0, 3, 1] → 새 a는 옛 c
    new_row = row.copy()                                  # 원래 행은 바꾸지 않음
    new_options = [options[k] for k in order]
    for letter, text in zip(VALID, new_options):
        new_row[letter] = text
    new_row["answer"] = VALID[new_options.index(correct_text)]   # 정답 내용이 간 자리의 글자
    return new_row

aug_note = ("'가장 왼쪽/오른쪽에 있는 도형' 질문은 좌우 반전하면 정답이 바뀐다. "
            "반전은 레이블을 바꾸지 않는 질문(색, 개수, 색으로 찾는 모양)에만 쓸 수 있고, 보기 섞기는 모든 질문에 안전하다.")
""", v1=r"""
def shuffle_options(row, rng):
    options = [row["a"], row["b"], row["c"], row["d"]]
    correct_text = options[VALID.index(row["answer"])]
    order = rng.permutation(4)
    new_row = row.copy()
    new_options = [options[k] for k in order]
    for letter, text in zip(VALID, new_options):
        new_row[letter] = text
    new_row["answer"] = ___                               # 정답 내용이 옮겨 간 자리의 글자
    return new_row

aug_note = ""
""", v2=r"""
def shuffle_options(row, rng):
    options = [row["a"], row["b"], row["c"], row["d"]]
    # 작성하세요: (1) 정답 내용(correct_text) 기억 (2) rng.permutation(4)로 새 순서
    #            (3) row.copy()에 섞인 보기를 a~d로 넣기 (4) 정답 글자를 새 위치로 바꾸기
    pass

aug_note = ""
""", v3=r"""
def shuffle_options(row, rng):
    '''보기 a~d의 순서를 섞고 정답 글자를 새 위치로 바꾼 새 행을 반환한다(원래 행은 그대로).'''
    # 작성하세요

aug_note = ""
""")
C(r"""
_rng = np.random.default_rng(0)
_row = clean_df.iloc[0]
for _ in range(5):
    _new = shuffle_options(_row, _rng)
    assert _new[_new["answer"]] == _row[_row["answer"]], "정답 내용이 바뀌었습니다"
assert clean_df.iloc[0]["a"] == _row["a"], "원래 행이 바뀌었습니다"
print(_row[["a", "b", "c", "d", "answer"]].tolist(), "→", _new[["a", "b", "c", "d", "answer"]].tolist())
print("서술 작성:", bool(aug_note.strip()))
""")

# ======================================================================
# 4. Dataset · Collator
# ======================================================================
M(r"""
# 4. Custom Dataset, Collator (공지: 손실 함수의 이해)
### 4-1 Dataset
baseline의 `VQAMCDataset`에 `option_shuffle` 인자를 더했습니다. 학습용이고 `option_shuffle=True`면 3단계 함수로 보기를 섞은 뒤 대화 메시지를 만듭니다. 학습용이면 마지막에 assistant의 정답 글자를 붙입니다.
""")
C(r"""
class VQAMCDataset(Dataset):
    def __init__(self, df, processor, train=True, option_shuffle=False, seed=SEED):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.train = train
        self.option_shuffle = option_shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        if self.train and self.option_shuffle:
            row = shuffle_options(row, self.rng)       # 꺼낼 때마다 새로 섞음 → epoch마다 다른 순서
        img = Image.open(os.path.join(DATA_DIR, row["path"])).convert("RGB")
        user_text = build_mc_prompt(row["question"], row["a"], row["b"], row["c"], row["d"])
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": user_text}]},
        ]
        if self.train:                                  # 학습용: 모델이 따라 말해야 할 정답 글자
            messages.append({"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]})
        return {"messages": messages, "image": img}
""", v1=r"""
class VQAMCDataset(Dataset):
    def __init__(self, df, processor, train=True, option_shuffle=False, seed=SEED):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.train = train
        self.option_shuffle = option_shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        if self.train and self.option_shuffle:
            row = ___                                   # 보기 섞기
        img = Image.open(os.path.join(DATA_DIR, row["path"])).convert("RGB")
        user_text = build_mc_prompt(row["question"], row["a"], row["b"], row["c"], row["d"])
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": user_text}]},
        ]
        if self.train:
            messages.append(___)                        # assistant의 정답 글자 메시지
        return {"messages": messages, "image": img}
""", v2=r"""
class VQAMCDataset(Dataset):
    def __init__(self, df, processor, train=True, option_shuffle=False, seed=SEED):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.train = train
        self.option_shuffle = option_shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        # 작성하세요:
        # (1) 학습용이고 option_shuffle이면 보기 섞기
        # (2) 사진 열기(RGB), 사용자 프롬프트 만들기
        # (3) system → user(image, text) 메시지 리스트
        # (4) 학습용이면 assistant 정답 메시지 추가
        # (5) {"messages": ..., "image": ...} 반환
        pass
""", v3=r"""
class VQAMCDataset(Dataset):
    '''baseline의 VQAMCDataset + option_shuffle(학습용일 때만 보기 섞기).
    __getitem__은 {"messages": [system, user(image, text), (학습용이면) assistant 정답], "image": PIL 이미지}를 반환.'''
    def __init__(self, df, processor, train=True, option_shuffle=False, seed=SEED):
        # 작성하세요
        pass
    # 작성하세요: __len__, __getitem__
""")

M(r"""
### 4-2 Collator와 정답 토큰 마스크
baseline은 `labels = input_ids.clone()`이라서 **시스템 지시문·질문·사진 토큰까지 모두** 따라 말하도록 손실을 계산했습니다. 우리가 원하는 것은 **assistant의 정답 글자**뿐입니다.

`build_labels(input_ids, attention_mask, im_start_id, prefix_len, label_mask)`:
1. `labels = input_ids.clone()`, 패딩 위치(`attention_mask == 0`)는 `-100`(손실 제외)
2. `label_mask=True`면, 행마다 **마지막 `<|im_start|>`** 위치를 찾고, 그 뒤 `assistant\n`(토큰 `prefix_len`개)까지 포함한 **앞부분 전체를 `-100`**
3. `labels` 반환
""")
C(r"""
def build_labels(input_ids, attention_mask, im_start_id, prefix_len, label_mask):
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100                  # 패딩은 손실에서 제외
    if label_mask:
        for i in range(len(labels)):
            starts = (input_ids[i] == im_start_id).nonzero().flatten()   # <|im_start|> 위치들
            answer_start = starts[-1].item() + 1 + prefix_len           # 마지막(=assistant) 뒤 'assistant\n' 다음
            labels[i, :answer_start] = -100              # 지시문·질문·사진 토큰은 손실 제외 → 정답만 학습
    return labels

@dataclass
class DataCollator:
    processor: Any
    train: bool = True
    label_mask: bool = False

    def __post_init__(self):
        tok = self.processor.tokenizer
        self.im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
        self.prefix_len = len(tok.encode("assistant\n", add_special_tokens=False))

    def __call__(self, batch):
        texts = [self.processor.apply_chat_template(s["messages"], tokenize=False, add_generation_prompt=False)
                 for s in batch]
        images = [s["image"] for s in batch]
        enc = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        if self.train:
            enc["labels"] = build_labels(enc["input_ids"], enc["attention_mask"],
                                         self.im_start_id, self.prefix_len, self.label_mask)
        return enc
""", v1=r"""
def build_labels(input_ids, attention_mask, im_start_id, prefix_len, label_mask):
    labels = input_ids.clone()
    labels[___] = -100                                  # 패딩 위치는 손실 제외
    if label_mask:
        for i in range(len(labels)):
            starts = (input_ids[i] == im_start_id).nonzero().flatten()
            answer_start = ___                          # 마지막 <|im_start|> + 1 + prefix_len
            labels[i, :answer_start] = -100
    return labels

@dataclass
class DataCollator:
    processor: Any
    train: bool = True
    label_mask: bool = False

    def __post_init__(self):
        tok = self.processor.tokenizer
        self.im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
        self.prefix_len = len(tok.encode("assistant\n", add_special_tokens=False))

    def __call__(self, batch):
        texts = [self.processor.apply_chat_template(s["messages"], tokenize=False, add_generation_prompt=False)
                 for s in batch]
        images = [s["image"] for s in batch]
        enc = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        if self.train:
            enc["labels"] = build_labels(enc["input_ids"], enc["attention_mask"],
                                         self.im_start_id, self.prefix_len, self.label_mask)
        return enc
""", v2=r"""
def build_labels(input_ids, attention_mask, im_start_id, prefix_len, label_mask):
    # 작성하세요: 위 1~3단계
    pass

@dataclass
class DataCollator:
    processor: Any
    train: bool = True
    label_mask: bool = False

    def __post_init__(self):
        tok = self.processor.tokenizer
        self.im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
        self.prefix_len = len(tok.encode("assistant\n", add_special_tokens=False))

    def __call__(self, batch):
        # 작성하세요:
        # (1) 각 샘플의 messages를 apply_chat_template(tokenize=False, add_generation_prompt=False)로 문자열화
        # (2) processor(text=..., images=..., padding=True, return_tensors="pt")
        # (3) 학습용이면 enc["labels"] = build_labels(...)
        pass
""", v3=r"""
def build_labels(input_ids, attention_mask, im_start_id, prefix_len, label_mask):
    '''위 1~3단계대로 labels 텐서를 만든다.'''
    # 작성하세요

@dataclass
class DataCollator:
    '''batch(list of Dataset 출력) → processor 출력 dict. 학습용이면 "labels" 추가.
    __post_init__에서 im_start_id(= "<|im_start|>" 토큰 id), prefix_len(= "assistant\n" 토큰 수)을 구해 둔다.'''
    processor: Any
    train: bool = True
    label_mask: bool = False
    # 작성하세요: __post_init__, __call__
""")

M(r"""
### 4-3 Processor 준비와 마스크 확인
아래 셀은 학습 샘플 하나를 Collator에 넣고, **손실에 들어가는 토큰만** 다시 글자로 풀어 보여 줍니다. `label_mask=True`일 때 정답 글자(+끝 토큰)만 남아야 합니다.
""")
C(r"""
_, processor = load_pretrained()        # 지정 모델의 processor (모델은 5단계에서 다시 만듦)
""", env="local")
C(r"""
processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=IMAGE_SIZE*IMAGE_SIZE,
                                          max_pixels=IMAGE_SIZE*IMAGE_SIZE, trust_remote_code=True)
""", env="colab")
C(r"""
_ds = VQAMCDataset(pool_df.head(2), processor, train=True)
for flag in [False, True]:
    _enc = DataCollator(processor, train=True, label_mask=flag)([_ds[0]])
    kept = _enc["labels"][0][_enc["labels"][0] != -100]
    print(f"label_mask={flag}: 손실 토큰 {len(kept)}개 →", repr(processor.tokenizer.decode(kept[-12:], skip_special_tokens=False)))
print("정답:", pool_df.iloc[0]["answer"])
""")

# ======================================================================
# 5. 모델 · LoRA
# ======================================================================
M(r"""
# 5. 모델, LoRA (공지: 전이학습, 학습 범위 제어)
`build_model(cfg)`: 지정 사전학습 모델을 불러오고 LoRA를 붙여 반환합니다. **사전학습 가중치는 고정**하고, `target_modules` 층에 붙인 작은 행렬(rank `r`)만 학습합니다. `cfg["lora_r"]`, `cfg["lora_alpha"]`, `cfg["target_modules"]`를 사용하세요.
""")
C(r"""
def build_model(cfg):
    base_model, _ = load_pretrained()                    # 사전학습된 연습용 모델(실제 Qwen 아님)
    base_model = prepare_model_for_kbit_training(base_model)   # (CPU판에서는 아무 일도 하지 않음)
    lora_config = LoraConfig(
        r=cfg["lora_r"],                                 # LoRA 행렬의 rank: 클수록 학습 파라미터↑
        lora_alpha=cfg["lora_alpha"],                    # LoRA 출력 배율(alpha / r)
        lora_dropout=0.05,
        bias="none",
        target_modules=cfg["target_modules"],            # LoRA를 붙일 Linear 층 이름
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)      # 원래 가중치는 고정, LoRA만 requires_grad=True
    model.print_trainable_parameters()
    return model.to(device)
""", v1=r"""
def build_model(cfg):
    base_model, _ = load_pretrained()
    base_model = prepare_model_for_kbit_training(base_model)
    lora_config = LoraConfig(
        r=___,
        lora_alpha=___,
        lora_dropout=0.05,
        bias="none",
        target_modules=___,
        task_type="CAUSAL_LM",
    )
    model = ___(base_model, lora_config)                 # LoRA를 붙인 PEFT 모델
    model.print_trainable_parameters()
    return model.to(device)
""", v2=r"""
def build_model(cfg):
    base_model, _ = load_pretrained()
    base_model = prepare_model_for_kbit_training(base_model)
    # 작성하세요: cfg 값으로 LoraConfig 만들기 → get_peft_model → print_trainable_parameters → device로 이동해 반환
    pass
""", v3=r"""
def build_model(cfg):
    '''사전학습 모델(load_pretrained) + LoRA(cfg의 lora_r, lora_alpha, target_modules)를 붙인 모델을 반환.'''
    # 작성하세요
""", env="local")
C(r"""
def build_model(cfg):
    bnb_config = BitsAndBytesConfig(                     # 4bit 양자화: GPU 메모리를 크게 줄임
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    base_model = prepare_model_for_kbit_training(base_model)   # 양자화 모델 학습 준비
    base_model.gradient_checkpointing_enable()                 # 메모리 절약(속도 약간↓)
    lora_config = LoraConfig(
        r=cfg["lora_r"],                                 # LoRA 행렬의 rank
        lora_alpha=cfg["lora_alpha"],                    # LoRA 출력 배율(alpha / r)
        lora_dropout=0.05,
        bias="none",
        target_modules=cfg["target_modules"],            # LoRA를 붙일 Linear 층 이름
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)      # 원래 가중치는 고정, LoRA만 학습
    model.print_trainable_parameters()
    return model                                         # device_map="auto"가 이미 GPU에 올림
""", v1=r"""
def build_model(cfg):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    base_model = prepare_model_for_kbit_training(base_model)
    base_model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=___,
        lora_alpha=___,
        lora_dropout=0.05,
        bias="none",
        target_modules=___,
        task_type="CAUSAL_LM",
    )
    model = ___(base_model, lora_config)
    model.print_trainable_parameters()
    return model
""", v2=r"""
def build_model(cfg):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    base_model = prepare_model_for_kbit_training(base_model)
    base_model.gradient_checkpointing_enable()
    # 작성하세요: cfg 값으로 LoraConfig 만들기 → get_peft_model → print_trainable_parameters → 반환
    pass
""", v3=r"""
def build_model(cfg):
    '''4bit 양자화(BitsAndBytesConfig nf4)로 MODEL_ID를 불러오고, kbit 학습 준비·gradient checkpointing 후
    LoRA(cfg의 lora_r, lora_alpha, target_modules)를 붙인 모델을 반환.'''
    # 작성하세요
""", env="colab")

# ======================================================================
# 6. 학습
# ======================================================================
M(r"""
# 6. fine-tuning (공지: 딥러닝 모델 학습, 하이퍼파라미터)
`train_model(cfg, train_df, valid_df)`: baseline의 학습 셀을 함수로 묶은 것입니다. 배치 크기는 1이고, **`GRAD_ACCUM`개 배치의 기울기를 모은 뒤** 한 번 갱신합니다(작은 배치로 큰 배치 효과).
- 배치마다: 순전파 → `loss = outputs.loss / GRAD_ACCUM` → `scaler.scale(loss).backward()`
- `step`이 `GRAD_ACCUM`의 배수이거나 **마지막 배치**면: `scaler.step(optimizer)` → `scaler.update()` → `optimizer.zero_grad()` → `scheduler.step()`
- epoch 끝: 평가 모드·기울기 없이 검증 손실
""")
C(r"""
def train_model(cfg, train_df, valid_df):
    random.seed(SEED); torch.manual_seed(SEED)
    model = build_model(cfg)
    collate = DataCollator(processor, train=True, label_mask=cfg["label_mask"])
    train_ds = VQAMCDataset(train_df, processor, train=True, option_shuffle=cfg["option_shuffle"])
    valid_ds = VQAMCDataset(valid_df, processor, train=True)            # 검증은 섞지 않음
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

    GRAD_ACCUM = cfg["grad_accum"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
    num_training_steps = cfg["epochs"] * math.ceil(len(train_loader) / GRAD_ACCUM)   # 실제 '갱신' 횟수
    scheduler = get_linear_schedule_with_warmup(optimizer, int(num_training_steps * 0.03), num_training_steps)
    use_amp = device == "cuda"                          # GPU에서만 혼합 정밀도(CPU판은 끔)
    amp_device = "cuda" if use_amp else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)

    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", unit="batch")
        for step, batch in enumerate(progress_bar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss / GRAD_ACCUM        # 모아서 더할 것이므로 나눠 둠
            scaler.scale(loss).backward()               # 기울기 '누적' (아직 갱신 안 함)
            running += loss.item()
            if step % GRAD_ACCUM == 0 or step == len(train_loader):   # 마지막 자투리 배치도 갱신
                scaler.step(optimizer)                  # 모인 기울기로 한 번 갱신
                scaler.update()
                optimizer.zero_grad(set_to_none=True)   # 다음 누적을 위해 기울기 비우기
                scheduler.step()
                progress_bar.set_postfix({"loss": f"{running:.3f}"})
                running = 0.0

        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad(), torch.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=use_amp):
            for vb in valid_loader:
                vb = {k: v.to(device) for k, v in vb.items()}
                val_loss += model(**vb).loss.item()
                val_steps += 1
        print(f"[{cfg['tag']} | Epoch {epoch+1}] valid loss {val_loss / val_steps:.4f}")
    return model
""", v1=r"""
def train_model(cfg, train_df, valid_df):
    random.seed(SEED); torch.manual_seed(SEED)
    model = build_model(cfg)
    collate = DataCollator(processor, train=True, label_mask=cfg["label_mask"])
    train_ds = VQAMCDataset(train_df, processor, train=True, option_shuffle=cfg["option_shuffle"])
    valid_ds = VQAMCDataset(valid_df, processor, train=True)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

    GRAD_ACCUM = cfg["grad_accum"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
    num_training_steps = cfg["epochs"] * math.ceil(len(train_loader) / GRAD_ACCUM)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(num_training_steps * 0.03), num_training_steps)
    use_amp = device == "cuda"
    amp_device = "cuda" if use_amp else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)

    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", unit="batch")
        for step, batch in enumerate(progress_bar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(**batch)
                loss = ___                              # 누적할 손실
            scaler.scale(loss).___()                    # 기울기 계산(누적)
            running += loss.item()
            if step % GRAD_ACCUM == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                ___                                     # 기울기 비우기
                scheduler.step()
                progress_bar.set_postfix({"loss": f"{running:.3f}"})
                running = 0.0

        model.___()                                     # 평가 모드
        val_loss, val_steps = 0.0, 0
        with torch.___(), torch.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=use_amp):
            for vb in valid_loader:
                vb = {k: v.to(device) for k, v in vb.items()}
                val_loss += model(**vb).loss.item()
                val_steps += 1
        print(f"[{cfg['tag']} | Epoch {epoch+1}] valid loss {val_loss / val_steps:.4f}")
    return model
""", v2=r"""
def train_model(cfg, train_df, valid_df):
    random.seed(SEED); torch.manual_seed(SEED)
    model = build_model(cfg)
    collate = DataCollator(processor, train=True, label_mask=cfg["label_mask"])
    train_ds = VQAMCDataset(train_df, processor, train=True, option_shuffle=cfg["option_shuffle"])
    valid_ds = VQAMCDataset(valid_df, processor, train=True)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

    GRAD_ACCUM = cfg["grad_accum"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
    num_training_steps = cfg["epochs"] * math.ceil(len(train_loader) / GRAD_ACCUM)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(num_training_steps * 0.03), num_training_steps)
    use_amp = device == "cuda"
    amp_device = "cuda" if use_amp else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)

    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", unit="batch")
        for step, batch in enumerate(progress_bar, start=1):
            # 작성하세요:
            # (1) batch의 텐서를 device로  (2) autocast 안에서 순전파, 손실 / GRAD_ACCUM
            # (3) scaler로 backward       (4) GRAD_ACCUM 배수이거나 마지막 배치면
            #     scaler.step → scaler.update → zero_grad → scheduler.step, 진행 표시
            pass

        # 작성하세요: 평가 모드·기울기 없이 valid_loader 평균 손실을 구해 출력
    return model
""", v3=r"""
def train_model(cfg, train_df, valid_df):
    '''build_model(cfg)로 모델을 만들고, cfg(epochs, lr, grad_accum, label_mask, option_shuffle)대로
    batch_size=1 + 기울기 누적(GRAD_ACCUM)으로 학습한 모델을 반환.
    AdamW(학습 파라미터만), get_linear_schedule_with_warmup(warmup 3%), GPU면 bf16 autocast + GradScaler.
    마지막 자투리 배치에서도 갱신하고, epoch마다 검증 손실을 출력.'''
    # 작성하세요
""")

# ======================================================================
# 7. 추론 · 파싱
# ======================================================================
M(r"""
# 7. inference (공지: 구조화 출력 파싱)
- `extract_choice(text)`: 모델 응답에서 선지를 뽑습니다. 소문자로 바꾼 뒤 (1) `{"answer": "b"}` 같은 JSON 형식이 있으면 그 글자, (2) 없으면 **마지막 줄부터 거꾸로** 앞뒤가 영문자가 아닌 단독 `a~d`를 찾고(`"(b)"`, `"정답: c"`, `"d) 파랑"` 모두 가능), (3) 못 찾으면 `"a"`.
- `predict(model, df)`: 행마다 system + user 메시지 → `apply_chat_template(add_generation_prompt=True)` → `generate(do_sample=False)`. baseline은 프롬프트까지 포함한 전체 출력을 디코드했지만, 여기서는 **입력 길이 뒤의 새 토큰만** 디코드합니다. `(예측 글자 리스트, 원문 응답 리스트)` 반환.
""")
C(r"""
def extract_choice(text: str) -> str:
    text = text.strip().lower()
    m = re.search(r'"answer"\s*:\s*"?([abcd])', text)   # (1) JSON 형식 {"answer": "b"}
    if m:
        return m.group(1)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):                        # (2) 마지막 줄부터 거꾸로
        m = re.search(r"(?<![a-z])([abcd])(?![a-z])", line)   # 앞뒤가 영문자가 아닌 단독 a~d
        if m:
            return m.group(1)
    return "a"                                          # (3) 못 찾으면 기본값

def predict(model, df, max_new_tokens=MAX_NEW_TOKENS):
    model.eval()                                        # 드롭아웃 끄기
    preds, raws = [], []
    for i in tqdm(range(len(df)), desc="Inference", unit="sample"):
        row = df.iloc[i]
        img = Image.open(os.path.join(DATA_DIR, row["path"])).convert("RGB")
        user_text = build_mc_prompt(row["question"], row["a"], row["b"], row["c"], row["d"])
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": user_text}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,   # 탐욕 디코딩: 재현 가능
                                     eos_token_id=processor.tokenizer.eos_token_id)
        new_ids = out_ids[:, inputs["input_ids"].shape[1]:]   # 프롬프트 뒤에 '새로 생성된' 토큰만
        output_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        raws.append(output_text)
        preds.append(extract_choice(output_text))
    return preds, raws

def accuracy(preds, answers):
    return float(np.mean(np.array(preds) == np.array(answers)))
""", v1=r"""
def extract_choice(text: str) -> str:
    text = text.strip().lower()
    m = re.search(r'"answer"\s*:\s*"?([abcd])', text)
    if m:
        return m.group(1)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in ___:                                    # 마지막 줄부터 거꾸로
        m = re.search(r"(?<![a-z])([abcd])(?![a-z])", line)
        if m:
            return ___
    return "a"

def predict(model, df, max_new_tokens=MAX_NEW_TOKENS):
    model.eval()
    preds, raws = [], []
    for i in tqdm(range(len(df)), desc="Inference", unit="sample"):
        row = df.iloc[i]
        img = Image.open(os.path.join(DATA_DIR, row["path"])).convert("RGB")
        user_text = build_mc_prompt(row["question"], row["a"], row["b"], row["c"], row["d"])
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": user_text}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=___)   # 답할 차례 표시
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=___,
                                     eos_token_id=processor.tokenizer.eos_token_id)
        new_ids = out_ids[:, ___:]                      # 입력 길이 뒤의 새 토큰만
        output_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        raws.append(output_text)
        preds.append(extract_choice(output_text))
    return preds, raws

def accuracy(preds, answers):
    return float(np.mean(np.array(preds) == np.array(answers)))
""", v2=r"""
def extract_choice(text: str) -> str:
    # 작성하세요: (1) 소문자 (2) JSON "answer" 형식 (3) 마지막 줄부터 단독 a~d (4) 기본값 "a"
    pass

def predict(model, df, max_new_tokens=MAX_NEW_TOKENS):
    model.eval()
    preds, raws = [], []
    for i in tqdm(range(len(df)), desc="Inference", unit="sample"):
        row = df.iloc[i]
        img = Image.open(os.path.join(DATA_DIR, row["path"])).convert("RGB")
        # 작성하세요:
        # (1) system + user(image, text) 메시지   (2) apply_chat_template(add_generation_prompt=True)
        # (3) processor(...).to(device)          (4) no_grad + generate(do_sample=False, eos_token_id=...)
        # (5) 새 토큰만 디코드 → raws, extract_choice → preds
        pass
    return preds, raws

def accuracy(preds, answers):
    return float(np.mean(np.array(preds) == np.array(answers)))
""", v3=r"""
def extract_choice(text: str) -> str:
    '''위 설명 (1)~(3)의 규칙으로 응답에서 'a'~'d' 하나를 뽑는다.'''
    # 작성하세요

def predict(model, df, max_new_tokens=MAX_NEW_TOKENS):
    '''df의 각 행을 추론해 (예측 글자 리스트, 새로 생성된 응답 문자열 리스트)를 반환.'''
    # 작성하세요

def accuracy(preds, answers):
    return float(np.mean(np.array(preds) == np.array(answers)))
""")
C(r"""
tests = {"b": "b", " (c) ": "c", "정답: D": "d", "d) 파랑": "d", '{"answer": "b"}': "b",
         "answer\na": "a", "모르겠습니다": "a", "cab": "a"}
for t, want in tests.items():
    assert extract_choice(t) == want, f"{t!r} → {extract_choice(t)!r} (기대 {want!r})"
print("파서 확인 통과")
""")

# ======================================================================
# baseline 실행
# ======================================================================
M(r"""
# baseline 실행과 제출
`BASE_CFG`(baseline과 같은 설정)로 학습하고, 고정 검증 데이터 정확도를 잰 뒤 test 예측을 `submission_baseline.csv`로 저장합니다.
""")
C(r"""
results = {}
train_base = pool_df.sample(n=BASE_CFG["n_train"], random_state=SEED).reset_index(drop=True)   # baseline: 200개만
model = train_model(BASE_CFG, train_base, valid_df)
val_preds, val_raws = predict(model, valid_df)
results["baseline"] = accuracy(val_preds, valid_df["answer"])
print("baseline 검증 정확도:", round(results["baseline"], 3), "| 응답 예시:", val_raws[:3])
print("예측 글자 분포:", pd.Series(val_preds).value_counts().to_dict())

test_preds, _ = predict(model, test_df)
submission = pd.DataFrame({"id": test_df["id"], "answer": test_preds})
submission.to_csv(os.path.join(OUT_DIR, "submission_baseline.csv"), index=False)
print("저장:", os.path.join(OUT_DIR, "submission_baseline.csv"))
""")

# ======================================================================
# 8. 성능 개선
# ======================================================================
M(r"""
# 8. 성능 개선 (공지: 성능 분석, 모델·학습 설정 개선안 도출)
baseline 결과를 보고 **바꿀 곳**을 고릅니다. 아래 표는 효과가 클 것으로 예상되는 순서입니다(`[claude]Lv3_점수개선_매뉴얼.md` 참고). **한 번에 하나씩** 바꾸고, 고정 검증 정확도로만 채택하세요. test 정답은 쓰지 않습니다.

| 순서 | 바꿀 곳 (`cfg` 키) | 왜 |
|---|---|---|
| 1 | `n_train`: 200 → `len(pool_df)` | baseline은 학습 데이터의 일부만 씀 |
| 2 | `label_mask`: False → True | 지시문·질문까지 따라 말하는 데 쓰던 학습을 **정답 글자**에 집중(4-2) |
| 3 | `option_shuffle`: False → True | 정답 글자 쏠림(a ≈ 49%) 제거(1-3, 3) |
| 4 | `epochs`, `lr` | 갱신 횟수·크기. 데이터를 늘리면 갱신 횟수도 늘어남 |
| 5 | `lora_r`, `target_modules` | 학습 범위(파라미터 수)와 표현력 |
| 6 | 프롬프트·파서 | 응답 형식이 흔들리면 파서가 놓침(7) |

`IMPROVED_CFG`를 정하고 `improve_note`에 무엇을 왜 바꿨는지 쓰세요. 아래 실행 셀이 학습·검증·제출(`submission_improved.csv`)을 합니다.
""")
C(r"""
IMPROVED_CFG = dict(BASE_CFG, tag="improved",
                    n_train=len(pool_df),     # 1) 정제된 학습 후보 전부 사용
                    label_mask=True,          # 2) 정답 글자에만 손실
                    option_shuffle=True,      # 3) 보기 섞기로 정답 글자 쏠림 제거
                    epochs=2, lr=5e-4)        # 4) 갱신 횟수·학습률 증가(작은 모델 기준, 검증으로 확인)
improve_note = ("baseline은 200개만, 지시문까지 따라 말하도록 학습했고 정답 글자가 a에 쏠려 있었다. "
                "데이터 전체 사용, 정답 토큰만 손실, 보기 섞기, 갱신 횟수·학습률 증가를 적용했다.")
""", v1=r"""
IMPROVED_CFG = dict(BASE_CFG, tag="improved",
                    n_train=___,
                    label_mask=___,
                    option_shuffle=___,
                    epochs=___, lr=___)
improve_note = ""
""", v2=r"""
IMPROVED_CFG = dict(BASE_CFG, tag="improved")   # 작성하세요: 바꿀 키를 추가
improve_note = ""
""", v3=r"""
IMPROVED_CFG = dict(BASE_CFG, tag="improved")   # 작성하세요: 바꿀 키를 추가
improve_note = ""
""", env="local")
C(r"""
IMPROVED_CFG = dict(BASE_CFG, tag="improved",
                    n_train=len(pool_df),     # 1) 정제된 학습 후보 전부 사용 (학습 시간 약 4배)
                    label_mask=True,          # 2) 정답 글자에만 손실
                    option_shuffle=True)      # 3) 보기 섞기로 정답 글자 쏠림 제거
# 실제 모델은 이미 강하므로 lr·epochs는 baseline 그대로 두고, 위 세 가지의 효과부터 확인한다.
improve_note = ("baseline은 200개만, 지시문까지 따라 말하도록 학습했고 정답 글자가 a에 쏠려 있었다. "
                "데이터 전체 사용, 정답 토큰만 손실, 보기 섞기를 적용했다.")
""", v1=r"""
IMPROVED_CFG = dict(BASE_CFG, tag="improved",
                    n_train=___,
                    label_mask=___,
                    option_shuffle=___)
improve_note = ""
""", v2=r"""
IMPROVED_CFG = dict(BASE_CFG, tag="improved")   # 작성하세요: 바꿀 키를 추가
improve_note = ""
""", v3=r"""
IMPROVED_CFG = dict(BASE_CFG, tag="improved")   # 작성하세요: 바꿀 키를 추가
improve_note = ""
""", env="colab")
C(r"""
del model; gc.collect()                               # 이전 모델 메모리 해제
if device == "cuda":
    torch.cuda.empty_cache()
train_imp = pool_df.sample(n=IMPROVED_CFG["n_train"], random_state=SEED).reset_index(drop=True)
model = train_model(IMPROVED_CFG, train_imp, valid_df)
val_preds, val_raws = predict(model, valid_df)
results["improved"] = accuracy(val_preds, valid_df["answer"])
print(pd.Series(results).round(3).to_string())
print("예측 글자 분포:", pd.Series(val_preds).value_counts().to_dict(), "| 서술 작성:", bool(improve_note.strip()))

test_preds, _ = predict(model, test_df)
pd.DataFrame({"id": test_df["id"], "answer": test_preds}).to_csv(os.path.join(OUT_DIR, "submission_improved.csv"), index=False)
print("저장:", os.path.join(OUT_DIR, "submission_improved.csv"))
""")

M(r"""
# 9. 로컬 채점 (Kaggle 제출 대신)
실제 시험에서는 CSV를 Kaggle에 제출해 점수를 받는 것으로 예상됩니다(미확정). 이 연습에서는 `정답_채점용/score_local.py`가 test 정답과 비교합니다. **채점 결과를 보고 설정을 다시 고치지 마세요**(시험에서는 test 정답이 없습니다).
""", env="local")
C(r"""
import subprocess, sys
for name in ["submission_baseline.csv", "submission_improved.csv"]:
    out = subprocess.run([sys.executable, "-I", "-X", "utf8", "../정답_채점용/score_local.py", os.path.join(OUT_DIR, name)],
                         capture_output=True, text=True, encoding="utf-8")
    print(out.stdout.strip() or out.stderr.strip())
""", env="local")
M(r"""
# 9. 제출 파일 받기
`/content/submission_baseline.csv`, `/content/submission_improved.csv`를 내려받습니다(왼쪽 파일 탭). 로컬 PC에서 아래처럼 채점할 수 있습니다(모의문제 폴더에서):
```bash
python 정답_채점용/score_local.py submission_improved.csv
```
""", env="colab")


# ======================================================================
def build(env, version):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    first = True
    for kind, cell_env, payload in CELLS:
        if cell_env not in (None, env):
            continue
        if kind == "md":
            text = payload.replace("{VERSION}", VERSION_NOTE[version]) if first else payload
            first = False
            nb.cells.append(nbf.v4.new_markdown_cell(text))
        else:
            code = payload["answer"] if version == "answer" or payload[version] is None else payload[version]
            tag = "# [정답 예시]\n" if version == "answer" and payload["v1"] is not None else (
                "# [작성하세요]\n" if payload[version if version != "answer" else "v1"] is not None else "# [실행만 하세요]\n")
            nb.cells.append(nbf.v4.new_code_cell(("" if code.startswith("!") else tag) + code))
    return nb


if __name__ == "__main__":
    names = {"v1": "버전1_핵심빈칸", "v2": "버전2_긴빈칸", "v3": "버전3_직접구현"}
    for env, folder in [("local", "로컬_CPU판"), ("colab", "Colab_실제판")]:
        for v, nm in names.items():
            nbf.write(build(env, v), ROOT / folder / f"{nm}_문제.ipynb")
        nbf.write(build(env, "answer"), ROOT / "정답_채점용" / f"{folder}_정답.ipynb")
    print("cells:", len(CELLS))
