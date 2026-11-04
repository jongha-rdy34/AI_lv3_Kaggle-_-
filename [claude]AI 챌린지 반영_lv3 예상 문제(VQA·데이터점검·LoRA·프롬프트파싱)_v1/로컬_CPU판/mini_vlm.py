"""[claude] 로컬 CPU 연습용 '작은 비전-언어 모델' 도우미.

**실제 Qwen2.5-VL이 아닙니다.** baseline 노트북과 같은 순서·이름으로 연습할 수 있도록
transformers / peft의 일부 이름을 흉내 낸 작은 구현입니다.

- MiniProcessor : AutoProcessor 흉내 (apply_chat_template, __call__, batch_decode, tokenizer)
- MiniVLM       : Qwen2_5_VLForConditionalGeneration 흉내 (forward(**batch).loss, generate)
                  층 이름은 q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- LoraConfig, get_peft_model : peft의 같은 이름 함수 흉내 (Linear 층에 LoRA 덧붙이기)
- get_linear_schedule_with_warmup : transformers의 같은 이름 스케줄러 흉내
양자화(BitsAndBytes)·혼합 정밀도(GradScaler)는 GPU 전용이라 이 판에서는 쓰지 않습니다.
"""
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SPECIALS = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>", "<image>"]
N_IMAGE_TOKENS = 4          # 사진 한 장 = <image> 토큰 4개 (세로 띠 4개)
IMAGE_SIZE = 64             # 이 판의 입력 크기 (Colab판 IMAGE_SIZE=384와 다름)


# ======================================================================
# Processor (토크나이저 + 이미지 전처리)
# ======================================================================
class _BatchEncoding(dict):
    """dict처럼 쓰면서 .to(device)를 지원 (transformers BatchEncoding 흉내)."""

    def to(self, device):
        return _BatchEncoding({k: v.to(device) for k, v in self.items()})


_PUNCT = r"(),.?:!\"'{}\[\]<>|"
_TOKEN_RE = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + r")|(\s)|([" + _PUNCT + r"])|([^\s" + _PUNCT + r"]+)")


def split_tokens(text):
    """특수 토큰 / 공백 한 칸 / 문장부호 / 단어 단위로 자른다."""
    return [m.group(0) for m in _TOKEN_RE.finditer(text)]


class MiniTokenizer:
    """단어 단위 토크나이저. 사전에 없는 단어는 글자 단위로 쪼갠다(글자도 없으면 <unk>)."""

    def __init__(self, vocab_tokens):
        self.itos = SPECIALS + sorted(set(vocab_tokens) - set(SPECIALS))
        self._index()

    def _index(self):
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad_token_id = self.stoi["<pad>"]
        self.eos_token_id = self.stoi["<|im_end|>"]
        self.image_token_id = self.stoi["<image>"]

    def encode(self, text, add_special_tokens=False):
        ids = []
        for tok in split_tokens(text):
            if tok in self.stoi:
                ids.append(self.stoi[tok])
            else:
                ids.extend(self.stoi.get(ch, self.stoi["<unk>"]) for ch in tok)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for i in ids:
            tok = self.itos[int(i)]
            if skip_special_tokens and tok in SPECIALS:
                continue
            out.append(tok)
        return "".join(out)

    def convert_tokens_to_ids(self, token):
        return self.stoi.get(token, self.stoi["<unk>"])

    def __len__(self):
        return len(self.itos)


class MiniProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    # Qwen 계열과 같은 ChatML 형식
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = ""
        for m in messages:
            text += f"<|im_start|>{m['role']}\n"
            for c in m["content"]:
                text += "<image>" * N_IMAGE_TOKENS if c["type"] == "image" else c["text"]
            text += "<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    @staticmethod
    def image_to_tensor(img):
        img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (x - 0.5) / 0.25

    def __call__(self, text, images=None, padding=True, return_tensors="pt"):
        ids = [self.tokenizer.encode(t) for t in text]
        L = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), L), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(ids), L), dtype=torch.long)
        for i, x in enumerate(ids):                     # 오른쪽 패딩
            input_ids[i, :len(x)] = torch.tensor(x)
            attention_mask[i, :len(x)] = 1
        enc = {"input_ids": input_ids, "attention_mask": attention_mask}
        if images is not None:
            enc["pixel_values"] = torch.stack([self.image_to_tensor(im) for im in images])
        return _BatchEncoding(enc)

    def batch_decode(self, ids, skip_special_tokens=True):
        return [self.tokenizer.decode(row, skip_special_tokens) for row in ids]

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "vocab.json").write_text(json.dumps(self.tokenizer.itos, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path=None):
        path = Path(path) if path else HERE / "assets"
        itos = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        tok = MiniTokenizer([])
        tok.itos = itos
        tok._index()
        return cls(tok)


# ======================================================================
# 작은 비전-언어 모델
# ======================================================================
# 실제 VLM은 거대한 트랜스포머가 사진·질문·보기를 한꺼번에 읽습니다. CPU에서 짧게 학습할 수 있도록
# 이 연습용 모델은 구조를 단순화했습니다.
#  (1) 정답 자리: 질문 단어 평균 + 보기 단어 평균 + 보기 글자(a~d)가 사진 토큰을 cross-attention → 보기마다 점수 → a~d 로짓
#  (2) 나머지 자리: 같은 MLP 층을 지나는 '다음 토큰 예측'(작은 언어모델)
# 두 경로가 같은 층(q_proj ... down_proj)을 공유하므로, 정답 이외 토큰까지 손실에 넣으면(label mask 없음)
# LoRA가 그 토큰들을 맞히는 데에도 힘을 나눠 쓰게 됩니다. 실제 모델에서도 생기는 현상입니다.
class Attention(nn.Module):
    """보기(질의)가 사진 토큰(키·값)을 보는 cross-attention."""

    def __init__(self, d, n_heads):
        super().__init__()
        self.h = n_heads
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x, kv):
        B, K, D = x.shape
        N = kv.shape[1]
        q = self.q_proj(x).view(B, K, self.h, D // self.h).transpose(1, 2)
        k = self.k_proj(kv).view(B, N, self.h, D // self.h).transpose(1, 2)
        v = self.v_proj(kv).view(B, N, self.h, D // self.h).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.o_proj(out.transpose(1, 2).reshape(B, K, D))


class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate_proj, self.up_proj = nn.Linear(d, hidden), nn.Linear(d, hidden)
        self.down_proj = nn.Linear(hidden, d)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn, self.mlp = Attention(d, n_heads), MLP(d, d * 2)

    def forward(self, h, img):           # 정답 점수 경로
        h = h + self.self_attn(self.input_layernorm(h), img)
        return h + self.mlp(self.post_attention_layernorm(h))

    def token_forward(self, x):          # 다음 토큰 예측 경로(MLP 공유)
        return x + self.mlp(self.post_attention_layernorm(x))


class Visual(nn.Module):
    """사진 (B,3,64,64) → 시각 토큰 (B,4,d)  (왼쪽부터 세로 띠 4개)."""

    def __init__(self, d):
        super().__init__()
        c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(), nn.MaxPool2d(2))
        self.conv = nn.Sequential(c(3, 16), c(16, 32), c(32, 64), c(64, 96))   # 64 → 4
        self.pool = nn.AdaptiveMaxPool2d((1, 4))
        self.merger = nn.Linear(96, d)
        self.pos = nn.Parameter(torch.zeros(1, N_IMAGE_TOKENS, d))

    def forward(self, x):
        f = self.pool(self.conv(x)).flatten(2).transpose(1, 2)  # (B,4,96)
        return self.merger(f) + self.pos


class MiniVLM(nn.Module):
    def __init__(self, vocab_size, d=128, n_heads=4, n_layers=2, max_len=512):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size, d=d, n_heads=n_heads, n_layers=n_layers, max_len=max_len)
        self.visual = Visual(d)
        self.embed_tokens = nn.Embedding(vocab_size, d)
        self.embed_positions = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size)
        self.score_head = nn.Linear(d, 1)
        self.ids = None              # set_special_ids에서 설정

    def set_special_ids(self, tok):
        g = tok.convert_tokens_to_ids
        self.ids = SimpleNamespace(image=g("<image>"), im_start=g("<|im_start|>"), eos=g("<|im_end|>"),
                                   letters=[g(c) for c in "abcd"], open=g("("), close=g(")"), newline=g("\n"))

    def _parse(self, row):
        """한 줄의 토큰에서 질문 위치, 보기 4개 위치, 정답을 말할 위치를 찾는다."""
        ids = self.ids
        starts = [i for i, t in enumerate(row) if t == ids.im_start]
        last_start = starts[-1] if starts else len(row) - 1
        img = [i for i, t in enumerate(row) if t == ids.image]
        u0 = img[-1] + 1 if img else 0
        nl = next((i for i in range(u0, last_start) if row[i] == ids.newline), last_start)
        question = list(range(u0, nl))
        options = [[] for _ in range(4)]
        for p in range(u0, max(u0, last_start - 2)):
            if row[p] == ids.open and row[p + 1] in ids.letters and row[p + 2] == ids.close:
                k = ids.letters.index(row[p + 1])
                e = next((i for i in range(p + 3, last_start) if row[i] == ids.newline), last_start)
                options[k] = list(range(p + 3, e))
        answer_pos = last_start + 2 if starts else len(row) - 1      # <|im_start|> assistant \n 다음
        return question, options, answer_pos

    def forward(self, input_ids, attention_mask=None, pixel_values=None, labels=None, **kwargs):
        B, T = input_ids.shape
        emb = self.embed_tokens(input_ids)
        # (2) 다음 토큰 예측 경로
        x = emb + self.embed_positions(torch.arange(T, device=emb.device))[None]
        for layer in self.layers:
            x = layer.token_forward(x)
        logits = self.lm_head(self.norm(x))
        # (1) 정답 점수 경로
        if pixel_values is not None and self.ids is not None:
            img = self.visual(pixel_values)
            mean = lambda b, idx: emb[b, idx].mean(0) if idx else emb.new_zeros(emb.shape[-1])
            parsed = [self._parse(input_ids[b].tolist()) for b in range(B)]
            letter_emb = self.embed_tokens(torch.tensor(self.ids.letters, device=emb.device))   # (4,d) 보기 글자 a~d
            h = torch.stack([torch.stack([mean(b, q) + mean(b, o) for o in opts]) for b, (q, opts, _) in enumerate(parsed)])
            h = h + letter_emb[None]                                       # 보기 위치(글자) 정보 → 글자 쏠림도 배울 수 있음
            for layer in self.layers:
                h = layer(h, img)
            scores = self.score_head(self.norm(h)).squeeze(-1)            # (B,4)
            logits = logits.clone()
            letters = torch.tensor(self.ids.letters, device=logits.device)
            for b, (_, _, pos) in enumerate(parsed):
                if pos < T:
                    row = torch.full_like(logits[b, pos], -20.0)
                    row[letters] = scores[b]
                    logits[b, pos] = row
                if pos + 1 < T:                                       # 글자 다음에는 끝 토큰
                    logits[b, pos + 1, self.ids.eos] = logits[b, pos + 1].max().detach() + 10.0
        loss = None
        if labels is not None:
            # 다음 토큰 예측: t 위치의 출력으로 t+1 토큰을 맞힌다. -100은 손실에서 제외
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, pixel_values=None, max_new_tokens=8, do_sample=False,
                 eos_token_id=None, temperature=1.0, **kwargs):
        """탐욕(또는 샘플링) 디코딩. 반환값은 '프롬프트 + 생성 토큰' (HF generate와 같음)."""
        ids = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(ids, None, pixel_values).logits[:, -1]
            if do_sample:
                nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
            if eos_token_id is not None and (nxt == eos_token_id).all():
                break
        return ids

    # transformers 흉내
    def gradient_checkpointing_enable(self):
        pass

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "model.pt")
        (Path(path) / "config.json").write_text(json.dumps(vars(self.config)), encoding="utf-8")


def load_pretrained(path=None):
    """사전학습된 MiniVLM과 MiniProcessor를 함께 불러온다."""
    path = Path(path) if path else HERE / "assets"
    processor = MiniProcessor.from_pretrained(path)
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model = MiniVLM(cfg["vocab_size"], cfg["d"], cfg["n_heads"], cfg["n_layers"], cfg["max_len"])
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.set_special_ids(processor.tokenizer)
    return model, processor


# ======================================================================
# LoRA (peft 흉내)
# ======================================================================
@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    task_type: str = "CAUSAL_LM"


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) · B(A(dropout(x))).  W는 고정, A·B만 학습."""

    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base_layer = base
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)              # 처음에는 원래 모델과 똑같이 동작
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class PeftModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = model

    def forward(self, *a, **k):
        return self.base_model(*a, **k)

    def generate(self, *a, **k):
        return self.base_model.generate(*a, **k)

    def print_trainable_parameters(self):
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {n_train:,} || all params: {n_all:,} || trainable%: {100 * n_train / n_all:.4f}")

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save({k: v for k, v in self.state_dict().items() if "lora_" in k}, Path(path) / "adapter.pt")


def get_peft_model(model, config):
    """모든 파라미터를 고정하고, 이름이 target_modules로 끝나는 Linear에 LoRA를 붙인다."""
    for p in model.parameters():
        p.requires_grad = False
    targets = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and child_name in config.target_modules:
                targets.append((module, child_name, child))
    if not targets:
        raise ValueError(f"target_modules {config.target_modules}에 맞는 Linear 층이 없습니다.")
    for parent, child_name, child in targets:
        setattr(parent, child_name, LoRALinear(child, config.r, config.lora_alpha, config.lora_dropout))
    return PeftModel(model)


def prepare_model_for_kbit_training(model):
    """GPU 양자화 학습 준비 함수의 자리. CPU판에서는 아무것도 하지 않는다."""
    return model


# ======================================================================
# tqdm이 없을 때 쓰는 간단한 진행 표시
# ======================================================================
class simple_tqdm:
    def __init__(self, iterable, desc="", unit="", **kwargs):
        self.iterable, self.desc, self.postfix = iterable, desc, ""

    def __iter__(self):
        n = len(self.iterable) if hasattr(self.iterable, "__len__") else None
        for i, x in enumerate(self.iterable, 1):
            yield x
            if n and (i == n or i % max(1, n // 4) == 0):
                print(f"{self.desc} {i}/{n} {self.postfix}", flush=True)

    def __len__(self):
        return len(self.iterable)

    def set_postfix(self, d=None, **kwargs):
        d = dict(d or {}, **kwargs)
        self.postfix = " ".join(f"{k}={v}" for k, v in d.items())


# ======================================================================
# 스케줄러 (transformers 흉내)
# ======================================================================
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def f(step):
        if step < num_warmup_steps:
            return (step + 1) / max(1, num_warmup_steps)
        return max(0.0, (num_training_steps - step) / max(1, num_training_steps - num_warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
