# 손실 함수 · 옵티마이저 · 학습 루프

## 손실 함수

### `nn.CrossEntropyLoss(weight=None, reduction="mean", label_smoothing=0.0)` / `F.cross_entropy(input, target, ...)`
- `input`: **logits** `(B, C)` (softmax를 미리 하지 않음). `target`: 정수 클래스 `(B,)` `torch.long`.
- 표본 i의 손실: `l_i = -w[y_i] · log softmax(x_i)[y_i]` (`weight`가 없으면 w=1)
- `reduction="mean"`이고 `weight`가 있으면 **`Σ l_i / Σ w[y_i]`**(가중치 합으로 나눔). 표본 수로 나누는 것이 아닙니다.
- `label_smoothing=ε`: 정답 분포를 `(1-ε)·one-hot + ε/C`로 부드럽게.

### `F.log_softmax(input, dim)`
- `log(softmax(x))`를 수치적으로 안정하게 계산. `softmax` 후 `log`를 따로 하면 큰 logit에서 `-inf`/NaN이 날 수 있습니다.

### Focal loss (라이브러리 함수 아님, 정의)
- `p_t` = 정답 클래스 확률, `FL = -(1 - p_t)^γ · log p_t`. 잘 맞히는 쉬운 표본(p_t≈1)의 손실을 줄여 어려운 표본에 집중합니다. `γ=0`이면 CE와 같습니다.

### `nn.MSELoss(reduction="mean")`, `nn.SmoothL1Loss()`, `nn.BCEWithLogitsLoss()`
- 회귀: 예측 `(B,)`와 정답 `(B,)`의 모양을 맞춥니다(`(B,1)` vs `(B,)`이면 브로드캐스팅 경고·오답).

## 옵티마이저

### `torch.optim.Adam(params, lr=1e-3, weight_decay=0)` / `torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)`
- `params`: 학습할 파라미터 이터러블. 고정한 파라미터를 빼고 넘기려면 `[p for p in model.parameters() if p.requires_grad]`.
- 층마다 다른 학습률: `[{"params": model.fc.parameters(), "lr": 1e-3}, {"params": model.layer4.parameters(), "lr": 1e-4}]`
- `optimizer.zero_grad()`: 기울기 초기화 · `optimizer.step()`: 갱신

### 학습률 스케줄러 (선택)
- `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)`: epoch마다 `scheduler.step()`
- `torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1)`

## 학습 루프의 기본 형태

```python
model.to(device)
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()            # 1. 이전 기울기 지우기
        loss = loss_fn(model(xb), yb)    # 2. 순전파 + 손실
        loss.backward()                  # 3. 역전파(기울기 계산)
        optimizer.step()                 # 4. 갱신
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(xb.to(device)).cpu() for xb, _ in valid_loader])
```

- `loss.item()`: 스칼라 텐서 → 파이썬 float. 평균 손실은 `loss.item() * len(xb)`를 더해 전체 표본 수로 나누면 마지막 작은 배치까지 정확합니다.
- 과적합 진단: train 손실은 내려가는데 검증 지표가 나빠지면 과적합, 둘 다 나쁘면 과소적합(학습률·epoch·학습 범위를 늘려 봄).
