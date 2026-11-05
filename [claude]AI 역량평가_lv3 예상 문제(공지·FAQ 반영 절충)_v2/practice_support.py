"""[claude] Lv3 모의 v2 제공 도우미 (창작 연습용, 공식 도우미 아님).

시험의 '제공 코드'에 해당하는 부분입니다.
- 데이터 읽기, 육안 검사용 격자, attempt(한 소문항의 실패가 다음을 막지 않게)
- 지정 모델: 사전학습된 TinyResNet(torchvision ResNet과 같은 속성 이름)
- Q4 제공 학습기(train_regressor / predict_regressor)
- Q5 모의 응답(mock_llm): 지정 모델은 이미지 모델이라 채팅이 안 되므로, 미리 만든 응답 문자열을 씁니다
- 제출 형식 점검(check_submission): Kaggle 업로드 때의 형식 검사 역할
- ready_*: 앞 소문항을 못 풀었을 때만 쓰는 진행용 대체물(해당 소문항 평가와 무관)
"""
from pathlib import Path
import json
import random
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
SUBMIT = ROOT / "제출물"
CLASSES = ["healthy", "rust", "scab", "multiple"]

# 모델 카드(assets/모델카드.md)와 같은 값
PRETRAIN_MEAN = [0.5, 0.5, 0.5]
PRETRAIN_STD = [0.25, 0.25, 0.25]


def get_transforms_module():
    """torchvision.transforms.v2가 있으면 그것을, 없으면 같은 이름의 연습용 대체를 돌려줍니다."""
    try:
        from torchvision.transforms import v2 as T
        return T
    except ImportError:
        from tv_compat import transforms_v2 as T
        return T


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def attempt(name, fn, *args, **kwargs):
    """fn을 실행하고, 실패하면 오류만 보여 준 뒤 None을 돌려줍니다."""
    try:
        out = fn(*args, **kwargs)
        print(f"[{name}] 실행 완료")
        return out
    except Exception as exc:  # 연습용: 다음 문항으로 넘어갈 수 있게 모든 오류를 잡음
        print(f"[{name}] 실패: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=1)
        return None


def csv(folder, name):
    return pd.read_csv(DATA / folder / name)


def load_images(folder, split):
    """(ids, images[N,48,48,3] uint8 RGB)"""
    with np.load(DATA / folder / f"{split}_images.npz", allow_pickle=False) as z:
        return z["ids"].astype(str).copy(), z["images"].copy()


def show_images(images, titles=None, ncols=8, size=1.4):
    """육안 검사용 격자. images: (N,H,W,3) uint8"""
    import matplotlib.pyplot as plt
    n = len(images)
    if n == 0:
        print("표시할 이미지가 없습니다.")
        return
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size * 1.15))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i])
            if titles is not None:
                ax.set_title(str(titles[i]), fontsize=7)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- 지정 모델
class _BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + idt)


class TinyResNet(nn.Module):
    """torchvision ResNet과 같은 이름: conv1, bn1, relu, layer1~layer4, avgpool, fc"""

    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = _BasicBlock(16, 16, 1)
        self.layer2 = _BasicBlock(16, 32, 2)
        self.layer3 = _BasicBlock(32, 64, 2)
        self.layer4 = _BasicBlock(64, 96, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def load_pretrained_model():
    """지정 모델: 사전학습 가중치를 불러온 TinyResNet(CPU). fc는 사전학습 과제용 6출력입니다.

    시험의 `torchvision.models.resnet18()` + `load_state_dict(torch.load(로컬 가중치))`에 해당합니다.
    """
    model = TinyResNet(6)
    state = torch.load(ASSETS / "pretrained_tinyresnet.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


# ---------------------------------------------------------------- 진행용 대체물
def ready_clean_leaf():
    """Q1 정답과 같은 기준으로 정제한 학습 데이터 (ids, images, labels)."""
    with np.load(DATA / "leaf" / "_ready_clean.npz", allow_pickle=False) as z:
        return z["ids"].astype(str).copy(), z["images"].copy(), z["labels"].astype(str).copy()


def ready_transforms(train):
    T = get_transforms_module()
    steps = [T.ToImage()]
    if train:
        steps += [T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5)]
    steps += [T.ToDtype(torch.float32, scale=True)]
    if train:
        steps += [T.ColorJitter(brightness=(0.4, 1.1))]
    steps += [T.Normalize(PRETRAIN_MEAN, PRETRAIN_STD)]
    return T.Compose(steps)


class ReadyLeafDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.transform = images, transform
        self.y = None if labels is None else [CLASSES.index(c) for c in labels]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        y = -1 if self.y is None else self.y[i]
        return self.transform(self.images[i]), torch.tensor(y, dtype=torch.long)


def ready_train_loader(dataset, labels, batch_size=64, balanced=False, seed=42):
    g = torch.Generator().manual_seed(seed)
    if not balanced:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)
    counts = pd.Series(labels).value_counts()
    w = torch.tensor([1.0 / counts[c] for c in labels], dtype=torch.double)
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True, generator=g)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


# ---------------------------------------------------------------- Q4 제공 학습기
def to_float_tensor(images):
    """(N,H,W,3) uint8 → (N,3,H,W) float [0,1]"""
    return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 255.0


def _normalize_batch(x):
    m = torch.tensor(PRETRAIN_MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(PRETRAIN_STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def _augment_batch(xb, generator):
    xb = xb.clone()
    n = len(xb)
    m = torch.rand(n, generator=generator) < 0.5
    xb[m] = xb[m].flip(3)
    m = torch.rand(n, generator=generator) < 0.5
    xb[m] = xb[m].flip(2)
    scale = torch.empty(n, 1, 1, 1).uniform_(0.6, 1.1, generator=generator)
    return (xb * scale.to(xb.device)).clamp(0, 1)


def train_regressor(model, x, y, train_idx, scope=("fc",), lr=1e-3, epochs=5, augment=True,
                    batch_size=64, weight_decay=0.0, seed=42, device="cpu"):
    """Q4 제공 학습기. x: [0,1] float (N,3,H,W), y: float 등급.

    scope: 학습할 파라미터 이름 접두어. 범위 밖 파라미터는 고정하고 범위 밖 BN은 eval()로 둡니다.
    augment=True면 배치마다 좌우·상하 반전과 밝기 배율 U(0.6, 1.1)을 적용합니다.
    augment에 함수 fn(xb, generator) -> xb 를 주면 그 함수로 증강합니다(xb: CPU의 [0,1] 배치, 정규화 전).
    손실은 MSE입니다.
    """
    scope = tuple(scope)
    model.to(device)
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(scope)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    idx = torch.as_tensor(np.asarray(train_idx))
    for _ in range(epochs):
        model.train()
        for name, mod in model.named_modules():
            if isinstance(mod, nn.BatchNorm2d) and not name.startswith(scope):
                mod.eval()
        order = idx[torch.randperm(len(idx), generator=g)]
        for s in range(0, len(order), batch_size):
            b = order[s:s + batch_size]
            xb = x[b]
            if augment:
                fn = augment if callable(augment) else _augment_batch
                xb = fn(xb, g)
            xb, yb = _normalize_batch(xb.to(device)), y[b].to(device)
            opt.zero_grad()
            loss_fn(model(xb).squeeze(1), yb).backward()
            opt.step()
    return model


def predict_regressor(model, x, batch_size=256, device="cpu"):
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            out.append(model(_normalize_batch(x[i:i + batch_size].to(device))).squeeze(1).cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- Q5 모의 응답
_MOCK = None


def mock_llm(note_id):
    """**모의 응답**(미리 만든 문자열). 실제 언어모델 호출이 아니며 프롬프트와 무관합니다.

    FAQ: 지정 모델은 이미지 모델이라 채팅이 안 됩니다. 그래서 이 세트는 '프롬프트 작성 +
    응답 파싱·정규화·검증 코드'를 연습하고, 응답은 실제 LLM에서 흔한 형식 오류를 섞어 미리 만들었습니다.
    """
    global _MOCK
    if _MOCK is None:
        _MOCK = json.loads((DATA / "notes" / "mock_llm_outputs.json").read_text(encoding="utf-8"))
    return _MOCK[note_id]


def check_prompt(prompt, note, schema_keys):
    """프롬프트에 필수 요소가 들어 있는지만 봅니다(품질 채점이 아님). 빠진 항목 리스트를 반환."""
    missing = []
    if note not in prompt:
        missing.append("메모 원문 포함")
    for k in schema_keys:
        if k not in prompt:
            missing.append(f"필드 '{k}'")
    for word, label in [("JSON", "JSON 출력 지시"), ("null", "미확정 정보 → null 규칙")]:
        if word not in prompt:
            missing.append(label)
    if not any(w in prompt for w in ["섭씨", "°C", "℃"]):
        missing.append("체온 단위 규칙")
    if "mg" not in prompt:
        missing.append("용량 단위 규칙")
    return missing


# ---------------------------------------------------------------- 제출 형식 점검
from submission_check import GUIDE, check_submission  # noqa: E402,F401  (torch 없이도 쓰도록 분리)


def save_jsonl(records, name, folder=None):
    """[{'id':..., 'record':{...}}, ...]를 JSON Lines로 저장."""
    folder = SUBMIT if folder is None else Path(folder)
    folder.mkdir(exist_ok=True)
    path = folder / name
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("저장:", path.relative_to(ROOT), len(records), "줄")
    return path


class Timer:
    """with ps.Timer("이름"): ... → 걸린 시간 출력"""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        print(f"[{self.name}] {time.time() - self.t0:.1f}초")
