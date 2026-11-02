# [claude] Lv3 점수 개선 매뉴얼 — "주어진 모델의 어디를 바꿀까"

일반적인 Kaggle 관행에 `[claude]…_v1` 데이터 실측(2026-11-02)을 더한 개인 학습용 매뉴얼입니다. **실제 Level 3의 제출 횟수·리더보드·지표·허용 패키지는 미공개**이므로, 시험 당일 공지가 이 매뉴얼과 다르면 공지를 따르세요. v1 수치는 합성 데이터 결과이므로 **효과 크기가 아니라 순서와 방향**만 참고하세요.

---

## 0. 외울 한 줄: **지 · 기 · 검 · 진 · 레 · 후 · 합**

| | 단계 | 한 문장 |
|---|---|---|
| **지** | 지표 | 지표 이름, 방향(↑/↓), 출력 형태(값·레이블·확률)를 먼저 확인한다 |
| **기** | 기준선 | 주어진 코드를 그대로 돌려 **제출 파일부터 하나 확보**한다 |
| **검** | 검증 | split·seed를 고정하고, 모든 실험을 같은 검증으로 비교한다 |
| **진** | 진단 | 과소적합인지, 과대적합인지, 분포 변화인지, 버그인지 가린다 |
| **레** | 레버 | 진단에 맞는 레버를 **한 번에 하나씩** 바꾸고 기록한다 |
| **후** | 후처리 | 지표에 맞게 출력을 다듬는다(clip, 확률, 임계값) |
| **합** | 합치기 | 시간이 남으면 앙상블·TTA로 마무리한다 |

---

## 1. 문항마다 시작 15분 체크리스트

- [ ] 지표와 방향, 그리고 **제출할 값의 종류**(실수 / 레이블 / 클래스별 확률)
- [ ] `sample_submission`의 열 이름, id 순서, 자료형
- [ ] 주어진 모델 확인: `print(model)`, 파라미터 수, **입력 전처리**(정규화 통계, 해상도, 토크나이저, `max_length`), 체크포인트 로드 방식
- [ ] 쓸 수 있는 라이브러리를 `try: import lightgbm`처럼 확인합니다. 네트워크가 막혀 있어 `pip install`은 안 된다고 가정합니다
- [ ] 데이터: 크기, 클래스 분포, 결측, **train/test 차이**(test에만 있는 범주, 처음 보는 단어 비율, 수치 범위)
- [ ] **1 epoch(또는 fit 1회)에 걸리는 시간**을 재고, 남은 시간에 실험을 몇 번 할 수 있는지 계산
- [ ] 기준선 제출 파일을 저장합니다. 이것이 안전망입니다

---

## 2. 진단표 — 어디가 막혔나

| 증상 | 진단 | 바꿀 곳 |
|---|---|---|
| train 점수도 낮다, loss가 아직 내려가는 중이다 | **과소적합** | 학습량(epoch↑, lr 조정) → 학습 범위(동결 해제) → 특징·모델 용량 |
| train ≫ validation | **과대적합** | 증강, 정규화(weight_decay, Ridge `alpha`↑, `min_samples_leaf`↑), best 체크포인트, 학습 파라미터 줄이기 |
| validation은 좋은데 test(리더보드)가 나쁘다 | **분포 변화·누수** | 검증 설계 재검토, 분포 점검, 일반화가 잘 되는 표현(사전학습 임베딩), 누수 제거 |
| fold마다 점수가 크게 흔들린다 | **잡음** | fold 표준편차보다 작은 개선은 채택하지 않음, seed 평균 |
| loss가 안 줄거나 NaN이 나온다 | **버그** | lr 과대, `zero_grad` 누락, 레이블 dtype(`long`), softmax를 두 번 적용, `train/eval` 모드 |

> **v1 실측:** Q3 기본 답안은 head 8 epoch + block4 8 epoch였고, train loss 0.53으로 아직 내려가는 중이었습니다(과소적합). 그래서 증강보다 **학습 범위와 epoch를 늘리는 것**이 맞았습니다. 결과는 아래 5절 표에 있습니다.

---

## 3. 레버 우선순위 (효과 대비 시간)

```
① 버그·제출 형식  →  ② 학습량·학습 범위  →  ③ 입력 표현·특징  →  ④ 하이퍼파라미터
→  ⑤ 정규화·증강(과대적합일 때만)  →  ⑥ 후처리  →  ⑦ 앙상블·TTA
```

원칙 세 가지:
1. **한 번에 하나만** 바꿉니다. 두 개를 같이 바꾸면 어느 쪽이 효과였는지 알 수 없습니다.
2. 모든 실험을 **기록**합니다(10절 표).
3. **검증 점수로만 채택**합니다. 이름난 기법도 v1에서는 여러 번 점수를 떨어뜨렸습니다.

---

## 4. 표 데이터 (회귀·분류, scikit-learn)

### v1 Q1 실측 (RMSE ↓)

| 변경 | CV | test | 판정 |
|---|---:|---:|---|
| Ridge 기준 | 7.567 | 6.420 | 기준 |
| + 결측 표시(`add_indicator=True`) | 7.572 | 6.413 | 효과 없음. 결측이 무작위였기 때문 |
| **+ 상호작용 특징(피크×악천후, 거리²)** | **7.449** | **6.328** | **채택** |
| + log 타깃 | 7.641 | 6.631 | 악화. 지표가 RMSE이기 때문 |
| HGB 기본 → 튜닝 | 8.020 → 8.015 | 6.96 → 6.90 | 튜닝으로 모델 차이를 뒤집지 못함 |
| Ridge 0.7 + HGB 0.3 혼합 | 7.485 | 6.381 | CV가 단독보다 나빠 채택 안 함 |

교훈: **표 데이터의 1순위 레버는 특징**입니다. 모델 튜닝보다 도메인 특징 하나가 더 큽니다.

### 4-1 특징 (가장 큰 레버)
- 날짜·시간: 시, 요일, 월, 주말, 피크 시간
- 도메인 상호작용: 곱·비율·차이(예: 거리/속도, 피크×날씨). 선형 모델은 이것 없이 비선형 관계를 못 잡습니다
- 결측 표시: 결측 자체에 의미가 있을 때 `SimpleImputer(..., add_indicator=True)`
- 그룹 통계·타깃 인코딩: **fold 안에서만** 계산합니다(누수 주의)
- test에만 있는 범주: `OneHotEncoder(handle_unknown="ignore")`

### 4-2 모델별 손잡이

| 모델 | 바꿀 매개변수 | 시작값 → 탐색 |
|---|---|---|
| Ridge / LogisticRegression | `alpha` / `C` | 0.1, 1, 10, 100 (10배 단위) |
| HistGradientBoosting | `learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization` | 0.05~0.1 / 200~1000 + `early_stopping=True` / 15~63 / 20~100 / 0~1 |
| RandomForest | `n_estimators`, `min_samples_leaf`, `max_features` | 200~500 / 1~5 / `"sqrt"`, 0.5 |
| 분류 공통 | `class_weight="balanced"` | 불균형 데이터 + macro-F1일 때 |

### 4-3 타깃 변환
`TransformedTargetRegressor(regressor=..., func=np.log1p, inverse_func=np.expm1)`는 **지표가 RMSLE이거나 타깃 꼬리가 매우 길 때**만 씁니다. 지표가 RMSE이면 오히려 손해일 수 있습니다(v1에서 악화).

### 4-4 후처리
값 범위로 자르기(`np.clip`). RMSE 지표에서는 정수로 반올림하지 않습니다.

---

## 5. 이미지 (사전학습 CNN, PyTorch)

### v1 Q3 실측 (accuracy ↑)

| 단계 | best val | test |
|---|---:|---:|
| head 8 + block4 8 epoch (기본 답안) | 0.796 | 0.821 |
| 전체 동결 해제, 16 epoch, lr 1e-3 | 0.921 | 0.914 |
| 전체 해제, **30 epoch** | 0.962 | 0.964 |
| + 증강(좌우 반전, 90° 회전) | 0.983 | 0.981 |
| **+ TTA(반전 평균)** | **0.992** | **0.988** |
| *악화:* 백본 lr 3e-4 / head lr 3e-3 차등, 16 epoch | 0.863 | 0.880 |
| *악화:* cosine 스케줄, 30 epoch | 0.925 | 0.944 |
| *악화:* 16 epoch에 증강 추가 | 0.812 | 0.814 |

교훈: **과소적합을 먼저 풀고(학습 범위·epoch), 그다음에 증강과 TTA**를 씁니다. 학습이 부족한 상태에서 증강을 넣거나 lr을 줄이면 오히려 손해입니다.

### 바꿀 곳 (순서대로)
1. **학습 범위**: head → 마지막 블록 → 전체. `requires_grad`를 바꾼 뒤에는 **optimizer를 다시 만듭니다.**
2. **lr·epoch**: lr은 3배 단위로 탐색(3e-4, 1e-3, 3e-3). epoch은 `1 epoch 시간 × 남은 실험 수`로 정합니다. **best val 체크포인트**를 저장합니다.
3. **증강**: 레이블을 바꾸지 않는 변환만 씁니다. 숫자 6/9에는 회전을 쓰지 않고, 좌우에 의미가 있으면 반전을 쓰지 않습니다. 과대적합이 보일 때 넣습니다.
4. **입력 전처리**: 사전학습 때와 같은 정규화 통계·해상도를 씁니다. 주어진 mean/std가 있으면 그것을 씁니다.
5. **BatchNorm 백본을 고정했다면** `model.train()` 뒤에 `model.backbone.eval()`을 호출합니다. 그러지 않으면 BN 통계가 바뀝니다.
6. **손실**: `nn.CrossEntropyLoss(label_smoothing=0.1)`, 클래스 불균형이면 `weight=`. log loss 지표에서 과신을 줄일 때 유용합니다.
7. **TTA**: 원본과 반전 이미지의 softmax 확률을 평균합니다.
8. 시간이 남으면 seed를 바꾼 모델 2~3개의 확률을 평균합니다.

```python
# 학습 범위 바꾼 뒤 optimizer 재생성 + best 체크포인트
for p in model.backbone.parameters(): p.requires_grad = True
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
best_acc, best_state = 0.0, None
for ep in range(EPOCHS):
    train_one_epoch(model, train_loader, optimizer, loss_fn)
    acc = evaluate(model, val_loader)
    if acc > best_acc:
        best_acc, best_state = acc, copy.deepcopy(model.state_dict())
model.load_state_dict(best_state)

# 증강(배치 단위, 레이블 불변일 때만)
flip = torch.rand(len(xb)) < 0.5
xb = xb.clone(); xb[flip] = torch.flip(xb[flip], dims=[3])

# TTA
with torch.no_grad():
    prob = (model(x).softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)) / 2
```

---

## 6. 텍스트 · LLM

### v1 Q2 실측 (macro-F1 ↑, test에 처음 보는 표현 22%)

| 변경 | test |
|---|---:|
| TF-IDF char_wb(2,4) + LR | 0.732 |
| TF-IDF 설정 변경(char_wb(1,3), sublinear, C=10) | 0.745 (TF-IDF의 한계) |
| TF-IDF 단어(1,2) + LinearSVC | 0.689 |
| **모의 임베딩 + 코사인 kNN / 정규화 후 LR** | **0.983 / 0.984** |
| TF-IDF와 임베딩의 확률 평균 | 0.973 (약한 모델을 섞어 손해) |

교훈: **어떤 벡터를 쓰느냐(표현)가 분류기 튜닝보다 큰 레버**입니다. 분포 변화가 있으면 특히 그렇습니다. (v1 임베딩은 가짜 모의 인코더이므로 실제 LLM이 같은 점수를 낸다는 뜻이 아닙니다.)

### 바꿀 곳 (순서대로)
1. **TF-IDF 기준선**: `analyzer`(한국어는 조사 때문에 `char_wb`가 대체로 유리), `ngram_range`, `sublinear_tf=True`, `min_df`, 분류기 `C`
2. **지정 임베딩 모델**: 풀링(attention mask를 쓴 mean vs CLS), **L2 정규화**, `max_length`, 배치 크기. 그 위에 LR / kNN / 프로토타입을 올립니다. train과 test는 같은 모델·같은 설정으로 임베딩합니다
3. **생성 LLM 프롬프트**: 레이블 목록을 명시하고, 출력 형식을 "레이블 하나"로 고정합니다. train에서 클래스별 1~2개를 few-shot 예시로 넣고, `do_sample=False`, `max_new_tokens`는 작게 둡니다. **파싱 실패 시 대체 규칙**(기준선 예측)을 둡니다
4. **미세조정**(허용되고 시간이 있을 때): Lv2 Q5처럼 마지막 블록과 head만 풀고, lr은 작게(1e-5~5e-5), 짧게 학습합니다
5. **결합**: 확률 평균은 두 모델의 성능이 비슷할 때만 씁니다
6. **긴 문장**: `truncation=True`. 중요한 정보가 앞에 있는지 뒤에 있는지 확인합니다

---

## 7. 지표별 출력 맞추기 (외우기)

| 지표 | 제출할 것 | 개선 포인트 |
|---|---|---|
| RMSE ↓ | 평균에 가까운 예측 | 이상치 영향이 큼, 범위 clip, log 타깃은 신중히 |
| MAE ↓ | 중앙값에 가까운 예측 | HGB `loss="absolute_error"` |
| RMSLE ↓ | 음수가 아닌 값 | log1p 타깃으로 학습 → expm1 |
| accuracy ↑ | 레이블(argmax) | 다수 클래스가 점수를 좌우함 |
| macro-F1 ↑ | 레이블 | 소수 클래스가 중요 → `class_weight="balanced"`, 클래스별 오류 확인 |
| F1(이진) ↑ | 레이블 | **임계값을 validation에서 탐색**(0.5 고정 금지) |
| log loss ↓ | 클래스별 **확률** | 0/1을 그대로 내지 않음(`np.clip(p, 1e-6, 1-1e-6)` 후 재정규화), 과신 줄이기(앙상블 평균, label smoothing) |
| AUC ↑ | 점수·확률 | 순위만 중요하므로 임계값과 무관 |

```python
# 이진 F1 임계값 탐색 (validation 확률 p_val, 정답 y_val)
ths = np.linspace(0.05, 0.95, 19)
best_t = max(ths, key=lambda t: f1_score(y_val, (p_val >= t).astype(int)))
```

---

## 8. 검증 설계

- 회귀는 `KFold(shuffle=True)`, 분류는 `StratifiedKFold`. 같은 사용자·상품이 반복되면 `GroupKFold`, 미래를 예측하는 문제면 시간 순서로 나눕니다
- **모든 실험에서 split과 seed를 같게** 둡니다
- 채택 규칙(경험칙): 평균 개선폭이 fold 표준편차보다 크거나, 5 fold 중 4 fold 이상에서 좋아졌을 때만 채택합니다
- 앙상블 가중치는 **OOF(out-of-fold) 예측**으로 정합니다(`cross_val_predict`)
- 리더보드가 있다면(가정): 공개 점수는 test 일부만 반영할 수 있습니다. CV와 함께 보고, 리더보드 점수만 보고 미세하게 계속 고치는 일은 피합니다. 제출 횟수 제한을 먼저 확인합니다

---

## 9. 4시간 운영 (3문항 가정)

| 시간 | 할 일 |
|---|---|
| 0:00–0:30 | 세 문항을 모두 읽고 **기준선 제출 파일 3개 확보** |
| 0:30–3:20 | 문항별로 약 55분씩 개선. 실험 1회는 10분 이내. **20분 동안 개선이 없으면 다음 문항으로** |
| 3:20–3:30 | 새 실험 중단. 문항별 최고 설정 확정 |
| 3:30–4:00 | 최종 예측, 형식 점검, 커널 재시작 후 전체 재실행, 출력 저장 |

- 시간이 오래 걸리는 실험은 먼저 **짧게(epoch 적게, 데이터 일부)** 돌려 경향을 봅니다
- 제출 파일은 버전 이름으로 저장합니다(`q3_v3_full30_aug.csv`). 최종본만 정해진 이름으로 복사합니다

---

## 10. 실험 기록 표 (시험 중에 이것만 적기)

| # | 문항 | 바꾼 것 (하나) | CV / val | 이전 대비 | 채택 | 시간 |
|---|---|---|---|---|---|---|
| 1 | Q3 | 기준선 | 0.796 | — | ✓ | 1분 |
| 2 | Q3 | 전체 해제 30ep | 0.962 | +0.166 | ✓ | 1분 |
| 3 | Q3 | + 증강 | 0.983 | +0.021 | ✓ | 1분 |

```python
LOG = []
def log(q, change, score, keep):
    LOG.append(dict(q=q, change=change, score=round(score, 4), keep=keep))
    print(pd.DataFrame(LOG).tail(5).to_string(index=False))
```

---

## 11. 하지 말 것

- test 레이블을 쓰거나, test 점수로 모델을 고르기
- test 데이터로 무언가를 fit하기(TF-IDF 어휘 학습 같은 비지도 사용도 **규칙을 확인하기 전에는** 금지)
- 여러 레버를 한꺼번에 바꾸기
- 과소적합 상태에서 증강·정규화부터 넣기 (v1 Q3: 0.821 → 0.814)
- 약한 모델을 강한 모델에 섞기 (v1 Q2: 0.984 → 0.973)
- 마지막 30분에 큰 변경하기, 제출 형식 확인 없이 덮어쓰기
- 위치(행 번호)로 예측을 붙이기. 항상 **id 기준**으로 맞춥니다

---

### 직접 적용해 보기
`[claude]AI 역량평가_lv3 예상 문제(회귀·텍스트 분류·이미지 분류)_v1/Lv3_v1_점수개선_실습.ipynb`에서 이 매뉴얼의 순서대로 빈칸 11개를 채워 봅니다. 정답판 결과(test): Q1 6.420 → 6.328, Q2 0.983 → 0.984, Q3 0.821 → 0.986.

### 출처와 한계
- 일반 기법: 공개 Kaggle 관행과 scikit-learn·PyTorch 공식 API
- 실측: `[claude]AI 역량평가_lv3 예상 문제(회귀·텍스트 분류·이미지 분류)_v1`의 합성 데이터, CPU, seed 42 한 번 실행. 다른 seed나 실제 데이터에서는 수치가 달라질 수 있습니다
- 실제 LLM 경로(6절 2~4)는 이 PC에 `transformers`와 가중치가 없어 실행 검증하지 않았습니다
