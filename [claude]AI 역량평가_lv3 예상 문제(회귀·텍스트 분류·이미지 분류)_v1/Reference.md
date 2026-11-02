# 공통 참고자료 (모의 v1)

시험 중 제공된다고 가정한 요약입니다. Lv2의 공식 `SSAFY_Common_Reference.pdf`를 흉내 낸 창작물이며 공식 자료가 아닙니다.

## 데이터 (pandas / scikit-learn)

```python
t = pd.to_datetime(df["order_time"], format="%Y-%m-%d %H:%M")
t.dt.hour, t.dt.dayofweek            # 월=0 … 일=6
s.isin([1, 2]).astype(int)           # 조건 → 0/1 정수
df.copy()                            # 원본을 바꾸지 않으려면 복사

SimpleImputer(strategy="median" | "most_frequent")
StandardScaler()
OneHotEncoder(handle_unknown="ignore")   # 학습 때 없던 범주 → 모두 0
Pipeline([("이름", 변환기), ...]);  ColumnTransformer([("이름", 변환기, 열목록), ...])
KFold(n_splits=5, shuffle=True, random_state=42).split(X)   # (학습 위치, 검증 위치)
X.iloc[위치배열]                     # 위치로 행 선택 / X.loc[라벨] 은 인덱스 라벨로 선택
min(d, key=lambda k: d[k][0])        # 사전에서 값이 가장 작은 키
np.clip(a, 0, None)
pd.Series(값, index=id).reindex(새_id순서)  / sub["id"].map(id→값 Series)
```

RMSE = sqrt(mean((y − ŷ)²)).  macro-F1 = 클래스별 F1의 단순 평균: `f1_score(y, p, average="macro")`.

## 텍스트

```python
TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))   # 어절 경계 안의 글자 2~4-gram
LogisticRegression(max_iter=2000)
train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
```

코사인 유사도: cos(a, b) = a·b / (‖a‖ ‖b‖). 행렬 형태: 각 행을 L2 노름으로 나눈 뒤 `A_unit @ B_unit.T` → `(n, m)`.

```python
np.linalg.norm(A, axis=1, keepdims=True)   # (n, 1)
np.atleast_2d(v)                            # (d,) → (1, d)
np.maximum(x, eps)                          # 0 나누기 방지
np.argsort(-scores)[:k]                     # 큰 순서 위치. 길이 < k이면 있는 만큼만 나옴
```

## PyTorch

```python
torch.from_numpy(a).permute(0, 3, 1, 2).float() / 255   # (N,H,W,C) → (N,C,H,W)
x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3))              # 채널별 통계 (C,)
mean.view(1, -1, 1, 1)                                   # 브로드캐스팅용 (1,C,1,1)
nn.AdaptiveAvgPool2d(1)          # (B,C,H,W) → (B,C,1,1), H·W와 무관
torch.flatten(t, 1)              # (B,C,1,1) → (B,C)
p.requires_grad = False          # 고정. 옵티마이저에는 requires_grad=True인 파라미터만 넘김
getattr(module, "block4")        # 이름으로 하위 모듈 꺼내기
nn.CrossEntropyLoss()            # 입력은 softmax 전 logits
model.train() / model.eval();  with torch.no_grad(): ...
optimizer.zero_grad(); loss.backward(); optimizer.step()
logits.argmax(dim=1)
```

## 실제 로컬 언어모델로 문장 임베딩을 만들 때 (실행 검증 안 함)

이 세트에는 실제 모델 가중치가 없고 이 PC에는 `transformers`도 설치되어 있지 않아 **아래 코드는 실행해 보지 않았습니다.** 실제 시험에서 지정 모델과 라이브러리가 주어졌을 때의 일반적인 흐름입니다.

```python
from transformers import AutoTokenizer, AutoModel   # 지정 라이브러리가 있을 때만
tok = AutoTokenizer.from_pretrained(LOCAL_DIR)        # 로컬 폴더 경로. 네트워크 다운로드 없음
model = AutoModel.from_pretrained(LOCAL_DIR).eval()

def embed(texts, batch_size=32, max_length=128):
    outs = []
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            enc = tok(texts[s:s + batch_size], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
            h = model(**enc).last_hidden_state              # (B, L, H)
            m = enc["attention_mask"].unsqueeze(-1).float()  # 패딩 위치 제외
            outs.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).numpy())
    return np.concatenate(outs)                              # (N, H), 정규화 안 됨
```

- train·test를 **같은 모델·같은 설정**으로 임베딩해야 합니다. 그 뒤 과정은 Q2의 코사인·kNN·프로토타입과 같습니다.
- Lv2에서 쓴 SmolLM2-135M 같은 생성형(causal) LM의 평균 풀링은 문장 임베딩 전용 모델(MiniLM 등)보다 품질이 낮을 수 있습니다. 어느 쪽이 나은지는 validation으로 확인합니다.
- 프롬프트로 레이블을 생성하게 하는 방식은 출력에서 레이블 문자열을 안정적으로 뽑아야 하고, 모르는 출력은 기본 레이블이나 다른 모델의 예측으로 대신하는 규칙이 필요합니다.
