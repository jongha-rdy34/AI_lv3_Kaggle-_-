# 공통 참고자료 (공지 반영 모의 v1)

시험 중 제공된다고 가정한 요약입니다. 공식 자료가 아닙니다.

## 데이터 점검

```python
images.shape, images.dtype                       # (N,H,W,C), uint8
images.reshape(len(images), -1).std(axis=1)     # 사진별 표준편차 → 0에 가까우면 단색·깨짐
images[..., c].mean(axis=(1, 2))                # 사진별 채널 c 평균 (0=R, 1=G, 2=B)
hashlib.md5(x.tobytes()).hexdigest()            # 바이트가 같은 사진 찾기
img[..., ::-1]                                  # 채널 순서 뒤집기 (BGR ↔ RGB)
s.str.strip().str.lower(); s.value_counts()
set(a) - set(b)                                 # 한쪽에만 있는 id
np.isin(ids, other_ids)
```

## 변환·증강·재구성 (PyTorch)

```python
torch.from_numpy(a).permute(0, 3, 1, 2).float() / 255   # (N,H,W,C) uint8 → (N,C,H,W) [0,1]
(x - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
x.flip(3)  # 좌우     x.flip(2)  # 상하      torch.rot90(x, k, dims=[2, 3])
torch.rand(B) < 0.5                                       # 샘플별 무작위 선택
torch.empty(B, 1, 1, 1).uniform_(lo, hi)                  # 샘플별 배율
x.clamp(0, 1); x.clone()
train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
torch.bincount(y); torch.multinomial(w, n, replacement=True, generator=g)
```

레이블을 바꾸지 않는 변환만 증강에 씁니다. 색이 레이블의 근거라면 색조 변환은 금지입니다. 증강은 [0,1] 범위에서 한 뒤 정규화합니다.

## 전이학습·학습 범위

```python
model.fc.in_features; model.fc = nn.Linear(in_f, n)          # torchvision ResNet과 같은 방식
nn.Sequential(nn.Dropout(p), nn.Linear(in_f, 1))              # 회귀 헤드
for name, p in model.named_parameters(): p.requires_grad = name.startswith(("layer4", "fc"))
for name, m in model.named_modules():
    if isinstance(m, nn.BatchNorm2d) and not name.startswith(scope): m.eval()   # model.train() 다음에
torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
```

`requires_grad=False`로는 BatchNorm의 `running_mean/var` 갱신이 멈추지 않습니다. 고정할 BN은 `eval()`로 둡니다.

## 손실 함수

```python
log_p = F.log_softmax(logits, dim=1)                 # softmax 후 log보다 수치적으로 안정
log_p_true = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
# 가중 CE: -(w[y] * log_p_true).sum() / w[y].sum()    (PyTorch CrossEntropyLoss(weight=w)와 같은 정의)
# focal : (-(1 - p_t) ** gamma * log_p_true).mean(),  p_t = log_p_true.exp()
nn.CrossEntropyLoss(weight=w, label_smoothing=0.0); nn.MSELoss(); nn.SmoothL1Loss()
```

## 지표

- macro-F1: `f1_score(y, p, average="macro")`. 클래스별 재현율: `recall_score(y, p, average=None)`
- QWK: O = 관측 행렬, W[i,j] = (i−j)²/(N−1)², E = outer(실제 분포, 예측 분포)/n, κ = 1 − ΣWO/ΣWE
- `np.digitize(pred, sorted_thresholds)`: 연속값 → 등급. `confusion_matrix(y, p, labels=range(k))`

## 하이퍼파라미터 탐색

```python
for values in itertools.product(*grid.values()): params = dict(zip(grid, values))
time.time()                                           # 시간 예산 관리
```

## 구조화 출력

```python
json.loads(s); json.dumps(obj, ensure_ascii=False)
s.find("{"), s.rfind("}")
re.sub(r",\s*([}\]])", r"\1", s)                      # 끝 쉼표 제거
ast.literal_eval(s)                                   # 파이썬 dict 표기
re.search(r"-?\d+(\.\d+)?", s)
isinstance(True, int)  # → True!  정수 검사는 isinstance(v, int) and not isinstance(v, bool)
```

JSON Schema 예: `{"type": ["integer", "null"]}`, `{"enum": ["M", "F", None]}`, `"required": [...]`, `"additionalProperties": False`, 배열은 `"items": {...}`. `jsonschema.Draft202012Validator(schema).iter_errors(obj)` (라이브러리가 있을 때).

## 실제 로컬 LLM을 쓸 때 (실행 검증 안 함)

이 PC에는 `transformers`와 LLM 가중치가 없어 **아래 코드는 실행해 보지 않았습니다.** 시험에서 지정 모델이 주어졌을 때의 일반적인 흐름입니다.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
llm = AutoModelForCausalLM.from_pretrained(LOCAL_DIR).eval()
def call_llm(prompt, max_new_tokens=300):
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")  # 채팅 모델일 때
    with torch.no_grad():
        out = llm.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)           # 재현성: 탐욕 디코딩
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
```

생성 결과는 항상 `extract_json → normalize_record → 자료형 검사`를 거칩니다. 실패하면 null 템플릿 같은 대체 규칙을 둡니다.
