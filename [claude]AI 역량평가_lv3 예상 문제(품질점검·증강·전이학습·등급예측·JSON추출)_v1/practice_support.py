"""[claude] Lv3 공지 후 모의 v1 실행 도우미 (창작 연습용).

실제 SSAFY 공식 도우미가 아닙니다. 데이터 경로, attempt, 이미지 보기,
'지정 모델' 역할의 TinyResNet(torchvision ResNet과 같은 속성 이름),
모의 LLM 응답 조회, 제출 파일 확인을 제공합니다.
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
CLASSES = ["healthy", "rust", "scab", "multiple"]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(4)
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


def csv(folder, name):
    return pd.read_csv(DATA / folder / name)


def load_images(folder, split):
    """(ids, images[N,48,48,3] uint8 RGB) 반환."""
    with np.load(DATA / folder / f"{split}_images.npz", allow_pickle=False) as z:
        return z["ids"].astype(str).copy(), z["images"].copy()


def show_images(images, titles=None, ncols=8, size=1.4):
    """육안 검사용 격자 출력. images: (N,H,W,3) uint8."""
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


# ---------------- '지정 모델': TinyResNet ----------------
def _nn():
    import torch
    from torch import nn
    return torch, nn


def make_tinyresnet(num_classes=6):
    """torchvision ResNet과 같은 이름의 작은 ResNet.

    conv1 → bn1 → relu → layer1(16) → layer2(32, /2) → layer3(64, /2) → layer4(96, /2)
    → avgpool(AdaptiveAvgPool2d(1)) → fc(Linear(96, num_classes)).
    """
    torch, nn = _nn()

    class BasicBlock(nn.Module):
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
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = BasicBlock(16, 16, 1)
            self.layer2 = BasicBlock(16, 32, 2)
            self.layer3 = BasicBlock(32, 64, 2)
            self.layer4 = BasicBlock(64, 96, 2)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(96, num_classes)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            return self.fc(torch.flatten(self.avgpool(x), 1))

    return TinyResNet()


def load_pretrained_model():
    """사전학습 가중치를 불러온 TinyResNet. fc는 사전학습 과제(6클래스)용입니다."""
    torch, _ = _nn()
    model = make_tinyresnet(6)
    model.load_state_dict(torch.load(ASSETS / "pretrained_tinyresnet.pt", map_location="cpu", weights_only=True))
    return model


PRETRAIN_MEAN = [0.5, 0.5, 0.5]   # 사전학습 때 쓴 정규화 값(모델 카드)
PRETRAIN_STD = [0.25, 0.25, 0.25]


# ---------------- 독립 진행용 준비 입력 ----------------
# 앞 소문항을 못 풀었을 때만 쓰는 대체 입력입니다. 해당 소문항 점수와 무관합니다.
def ready_clean_leaf():
    """정제된 학습 데이터: (ids, images RGB, labels str). Q1 정답과 같은 기준."""
    with np.load(DATA / "leaf" / "_ready_clean.npz", allow_pickle=False) as z:
        return z["ids"].astype(str).copy(), z["images"].copy(), z["labels"].astype(str).copy()


def ready_to_tensor(images):
    torch, _ = _nn()
    return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 255.0


def ready_normalize(x):
    torch, _ = _nn()
    return (x - torch.tensor(PRETRAIN_MEAN).view(1, 3, 1, 1)) / torch.tensor(PRETRAIN_STD).view(1, 3, 1, 1)


def ready_augment(xb):
    torch, _ = _nn()
    xb = xb.clone()
    m = torch.rand(len(xb)) < 0.5
    xb[m] = xb[m].flip(3)
    m = torch.rand(len(xb)) < 0.5
    xb[m] = xb[m].flip(2)
    return (xb * torch.empty(len(xb), 1, 1, 1).uniform_(0.4, 1.1)).clamp(0, 1)


def ready_balanced_order(labels, generator):
    torch, _ = _nn()
    labels = torch.as_tensor(labels)
    w = 1.0 / torch.bincount(labels).float()[labels]
    return torch.multinomial(w, len(labels), replacement=True, generator=generator)


# ---------------- Q4: 회귀 학습기(제공) ----------------
def train_regressor(model, x, y, train_idx, scope, lr, epochs, batch_size=64, seed=42):
    """Q4용 제공 학습기. x: [0,1] float (N,3,H,W), y: float 등급. scope 밖 파라미터·BN은 고정.

    증강(좌우·상하 반전, 밝기)과 사전학습 정규화를 내부에서 적용합니다.
    """
    torch, nn = _nn()
    for name, p in model.named_parameters():
        p.requires_grad = any(name.startswith(s) for s in scope)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    idx = torch.as_tensor(np.asarray(train_idx))
    for _ in range(epochs):
        model.train()
        for name, mod in model.named_modules():
            if isinstance(mod, nn.BatchNorm2d) and not any(name.startswith(s) for s in scope):
                mod.eval()
        order = idx[torch.randperm(len(idx), generator=g)]
        for s in range(0, len(order), batch_size):
            b = order[s:s + batch_size]
            opt.zero_grad()
            loss_fn(model(ready_normalize(ready_augment(x[b]))).squeeze(1), y[b]).backward()
            opt.step()
    return model


def predict_regressor(model, x, batch_size=256):
    torch, _ = _nn()
    model.eval()
    with torch.no_grad():
        return torch.cat([model(ready_normalize(x[i:i + batch_size])).squeeze(1)
                          for i in range(0, len(x), batch_size)]).numpy()


# ---------------- Q5: 모의 LLM ----------------
def mock_llm(note_id):
    """**모의 LLM 응답**(미리 만든 문자열). 실제 LLM 호출이 아니며 프롬프트 내용과 무관합니다."""
    global _MOCK
    try:
        _MOCK
    except NameError:
        _MOCK = json.loads((DATA / "notes" / "mock_llm_outputs.json").read_text(encoding="utf-8"))
    return _MOCK[note_id]


def check_prompt(prompt, note, schema_keys):
    """프롬프트 형식 점검(내용 품질 채점이 아님). 빠진 항목 리스트를 반환."""
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


# ---------------- 제출 ----------------
def save_submission(frame, name, sample, folder=None):
    """sample_submission과 열·id 순서가 같은지 확인한 뒤 저장합니다."""
    if list(frame.columns) != list(sample.columns):
        raise ValueError(f"열이 다릅니다: {list(frame.columns)} != {list(sample.columns)}")
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
