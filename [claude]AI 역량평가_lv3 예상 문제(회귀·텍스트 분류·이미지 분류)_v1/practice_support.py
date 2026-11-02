"""[claude] Lv3 모의 v1 실행 도우미 (창작 연습용).

실제 SSAFY 공식 도우미(exam_data.py, model_support.py 등)가 아닙니다.
데이터 경로, 오류를 잡고 다음 문항으로 넘어가는 attempt, 제출 파일 형식 확인,
Q3에서 쓰는 사전학습 CNN 백본 정의만 담고 있습니다.
"""
from pathlib import Path
import json
import random
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
SUBMIT = ROOT / "제출물"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(2)
    except ImportError:
        pass


def attempt(name, fn, *args, **kwargs):
    """fn을 실행하고 실패하면 오류만 출력한 뒤 None을 돌려줍니다."""
    try:
        out = fn(*args, **kwargs)
        print(f"[{name}] 실행 완료")
        return out
    except Exception as exc:  # 연습용: 다음 문항 진행을 위해 모든 오류를 잡음
        print(f"[{name}] 실패: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=1)
        return None


def csv(q, name):
    return pd.read_csv(DATA / q / name)


# ---------------- Q2: 텍스트 ----------------
def q2_embeddings():
    """모의 인코더로 미리 계산한 임베딩 캐시. 실제 LLM 출력이 아닙니다."""
    with np.load(DATA / "q2" / "mock_embeddings.npz", allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def q2_label_info():
    return json.loads((DATA / "q2" / "labels.json").read_text(encoding="utf-8"))


# ---------------- Q3: 이미지 ----------------
def q3_images(split):
    """split: 'train' 또는 'test'. (ids, images[N,H,W,3] uint8) 반환."""
    with np.load(DATA / "q3" / f"{split}_images.npz", allow_pickle=False) as z:
        return z["ids"].astype(str).copy(), z["images"].copy()


def _torch():
    import torch
    from torch import nn
    return torch, nn


def make_backbone():
    """사전학습 백본 구조. 입력 (B,3,H,W) → 출력 (B,64,H//8,W//8).

    out_channels 속성으로 마지막 채널 수(64)를 알려 줍니다.
    공간 크기는 입력 크기에 따라 달라집니다.
    """
    torch, nn = _torch()

    class Backbone(nn.Module):
        out_channels = 64

        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
            self.block2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
            self.block3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
            self.block4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())

        def forward(self, x):
            return self.block4(self.block3(self.block2(self.block1(x))))

    return Backbone()


def load_pretrained_backbone():
    torch, _ = _torch()
    backbone = make_backbone()
    state = torch.load(ASSETS / "q3_backbone.pt", map_location="cpu", weights_only=True)
    backbone.load_state_dict(state)
    return backbone


# ---------------- 독립 진행용 준비 입력 ----------------
# 앞 소문항을 못 풀었을 때만 쓰는 대체 입력입니다. 해당 소문항 점수와 무관합니다.
def q1_ready_features(df):
    out = df.copy()
    t = pd.to_datetime(out["order_time"], format="%Y-%m-%d %H:%M")
    out["hour"] = t.dt.hour
    out["dayofweek"] = t.dt.dayofweek
    out["is_peak"] = out["hour"].isin([11, 12, 13, 17, 18, 19, 20]).astype(int)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    return out


def q2_ready_embeddings(train_ids, test_ids):
    cache = q2_embeddings()
    tr = pd.Series(range(len(cache["train_ids"])), index=cache["train_ids"]).loc[list(train_ids)].to_numpy()
    te = pd.Series(range(len(cache["test_ids"])), index=cache["test_ids"]).loc[list(test_ids)].to_numpy()
    return cache["train_emb"][tr], cache["test_emb"][te]


def q3_ready_tensors(train_images, train_idx, test_images):
    torch, _ = _torch()
    def chw(a):
        return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float() / 255.0
    x, xt = chw(train_images), chw(test_images)
    mean = x[train_idx].mean(dim=(0, 2, 3)).view(1, -1, 1, 1)
    std = x[train_idx].std(dim=(0, 2, 3)).view(1, -1, 1, 1)
    return (x - mean) / std, (xt - mean) / std


# ---------------- 제출 ----------------
def save_submission(frame, name, sample, folder=None):
    """sample_submission과 열·id 집합이 같은지 확인한 뒤 제출물/(또는 folder)에 저장합니다."""
    if list(frame.columns) != list(sample.columns):
        raise ValueError(f"열이 다릅니다: {list(frame.columns)} != {list(sample.columns)}")
    if len(frame) != len(sample) or frame["id"].duplicated().any():
        raise ValueError("행 수가 다르거나 id가 중복되었습니다.")
    if not frame["id"].astype(str).reset_index(drop=True).equals(sample["id"].astype(str).reset_index(drop=True)):
        raise ValueError("id 순서가 sample_submission과 다릅니다.")
    if frame.isna().any().any():
        raise ValueError("결측값이 있습니다.")
    folder = SUBMIT if folder is None else Path(folder)
    folder.mkdir(exist_ok=True)
    path = folder / name
    frame.to_csv(path, index=False)
    print("저장:", path.relative_to(ROOT), frame.shape)
    return path
