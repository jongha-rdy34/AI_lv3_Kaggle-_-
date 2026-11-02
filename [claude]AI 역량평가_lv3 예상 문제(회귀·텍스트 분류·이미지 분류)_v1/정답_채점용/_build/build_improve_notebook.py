"""점수 개선 실습 노트북 생성기. 실행: python -I build_improve_notebook.py

build_notebooks.py와 같은 방식: 코드 셀마다 (문제용, 정답용) 두 버전.
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
CELLS = []


def md(text):
    CELLS.append(("md", text.strip("\n"), None))


def run(code):
    CELLS.append(("code", "# [실행만 하세요]\n" + code.strip("\n"), None))


def write(problem, answer):
    CELLS.append(("code", "# [작성하세요]\n" + problem.strip("\n"), "# [정답 예시]\n" + answer.strip("\n")))


# ======================================================================
md("""
# [claude] Lv3 v1 — 점수 개선 실습

`[claude]Lv3_점수개선_매뉴얼.md`의 **지 · 기 · 검 · 진 · 레 · 후 · 합**을 v1 데이터에 그대로 적용해 보는 연습입니다. 창작 연습물이며 실제 시험 양식이 아닙니다.

> **v1 본 문제(`Lv3_모의_v1_문제.ipynb`)를 먼저 푸세요.** 이 노트북은 v1 정답 함수를 기본 코드로 제공합니다.

규칙
- **모든 채택 결정은 validation(CV) 점수로만** 합니다. test 점수는 맨 마지막 셀에서 한 번만 확인하며, 그 결과를 보고 되돌아가 설정을 바꾸지 않습니다.
- 레버는 **한 번에 하나**씩 바꾸고 `log(...)`로 기록합니다.
- 개선한 제출 파일은 `제출물_개선/`에 저장합니다. v1 풀이의 `제출물/`은 건드리지 않습니다.
- 전체 실행 시간은 CPU에서 약 3~4분입니다(Q3 학습이 대부분).
""")

run("""
from pathlib import Path
import sys, copy, time, subprocess
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "practice_support.py").exists()), None)
if ROOT is None:
    raise FileNotFoundError("모의문제 폴더(또는 그 하위 폴더)에서 실행하세요.")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from practice_support import (seed_everything, attempt, csv, save_submission, q1_ready_features,
                              q2_embeddings, q2_ready_embeddings, q2_label_info, q3_images,
                              q3_ready_tensors, load_pretrained_backbone)
from sklearn.model_selection import KFold, train_test_split, cross_val_predict

seed_everything(42)
IMPROVED = ROOT / "제출물_개선"
LOG = []
MIN_GAIN = 0.01   # 이보다 작은 개선은 잡음으로 보고 채택하지 않음(매뉴얼 8절)

def log(q, change, score, keep, note=""):
    LOG.append(dict(q=q, change=change, score=round(float(score), 4), keep=keep, note=note))
    print(pd.DataFrame(LOG).tail(6).to_string(index=False))

print("준비 완료")
""")

# ---------------------------------------------------------------- Q1
md("""
---
## 개선 1 — Q1 회귀 (RMSE ↓)

매뉴얼 4절: **표 데이터의 1순위 레버는 특징** → 그다음 정규화 강도 → 앙상블은 OOF로 확인.
아래 셀은 v1 정답의 전처리와 5-fold OOF RMSE 도우미를 제공합니다.
""")

run("""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

q1_train, q1_test, q1_sample = csv("q1", "train.csv"), csv("q1", "test.csv"), csv("q1", "sample_submission.csv")
q1_X, q1_XT, q1_y = q1_ready_features(q1_train), q1_ready_features(q1_test), q1_train["delivery_min"]
NUM_COLS = ["distance_km", "n_items", "prep_min", "courier_rating", "hour", "is_peak", "is_weekend"]
CAT_COLS = ["region", "weather"]
KF = KFold(n_splits=5, shuffle=True, random_state=42)

def build_preprocessor(num_cols, cat_cols=CAT_COLS):
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

def cv_oof(model, X, y, num_cols):
    \"\"\"fold마다 전처리+모델을 새로 학습한 OOF 예측과 그 RMSE.\"\"\"
    pipe = Pipeline([("prep", build_preprocessor(num_cols)), ("model", model)])
    oof = cross_val_predict(pipe, X, y, cv=KF)
    return rmse(y, oof), oof

q1_base_rmse, q1_base_oof = cv_oof(Ridge(alpha=1.0), q1_X, q1_y, NUM_COLS)
log("Q1", "기준선 Ridge(alpha=1)", q1_base_rmse, True)
""")

md("""
### 1-1 도메인 특징 추가
`add_domain_features(df)`: 복사본에 두 열을 추가해 반환합니다.
- `peak_bad`: 피크 시간(`is_peak == 1`)이면서 날씨가 `비` 또는 `눈`이면 1, 아니면 0 (날씨 결측은 0)
- `dist2`: `distance_km`의 제곱 (결측은 NaN으로 두면 전처리에서 대치됩니다)

이유: 선형 모델은 "피크 시간에 비까지 오면 더 늦다", "거리가 멀수록 더 가파르게 늘어난다" 같은 관계를 특징 없이는 잡지 못합니다.
""")

write("""
def add_domain_features(df):
    out = df.copy()
    out["peak_bad"] = None  # 작성하세요
    out["dist2"] = None     # 작성하세요
    return out
""", """
def add_domain_features(df):
    out = df.copy()
    out["peak_bad"] = (out["is_peak"].eq(1) & out["weather"].isin(["비", "눈"])).astype(int)
    out["dist2"] = out["distance_km"] ** 2
    return out
""")

run("""
def _run_1_1():
    X2 = add_domain_features(q1_X)
    assert set(X2["peak_bad"].unique()) <= {0, 1} and "peak_bad" not in q1_X.columns
    score, oof = cv_oof(Ridge(alpha=1.0), X2, q1_y, NUM_COLS + ["peak_bad", "dist2"])
    keep = score < q1_base_rmse - MIN_GAIN
    log("Q1", "+ peak_bad, dist2", score, keep)
    return keep
q1_use_domain = bool(attempt("1-1", _run_1_1))
Q1_X = add_domain_features(q1_X) if q1_use_domain else q1_X
Q1_XT = add_domain_features(q1_XT) if q1_use_domain else q1_XT
Q1_NUM = NUM_COLS + (["peak_bad", "dist2"] if q1_use_domain else [])
print("도메인 특징 사용:", q1_use_domain)
""")

md("""
### 1-2 정규화 강도 탐색
`search_alpha(alphas, X, y, num_cols)`: 각 `alpha`로 `cv_oof(Ridge(alpha=a), ...)`를 계산해 `{alpha: RMSE}`를 반환합니다. 10배 단위 탐색입니다.
""")

write("""
def search_alpha(alphas, X, y, num_cols):
    scores = {}
    # 작성하세요
    return scores
""", """
def search_alpha(alphas, X, y, num_cols):
    scores = {}
    for a in alphas:
        scores[a] = cv_oof(Ridge(alpha=a), X, y, num_cols)[0]
    return scores
""")

run("""
q1_alpha_scores = attempt("1-2", search_alpha, [0.1, 1.0, 10.0, 100.0], Q1_X, q1_y, Q1_NUM) or {}
print({a: round(s, 4) for a, s in q1_alpha_scores.items()})
Q1_ALPHA = min(q1_alpha_scores, key=q1_alpha_scores.get) if q1_alpha_scores else 1.0
q1_ridge_rmse, q1_ridge_oof = cv_oof(Ridge(alpha=Q1_ALPHA), Q1_X, q1_y, Q1_NUM)
log("Q1", f"Ridge alpha={Q1_ALPHA}", q1_ridge_rmse, True, "탐색 최솟값")
""")

md("""
### 1-3 앙상블은 OOF로 확인
`best_blend_weight(oof_a, oof_b, y, weights)`: 각 `w`에 대해 `w*oof_a + (1-w)*oof_b`의 RMSE를 구해 **가장 작은 `(w, rmse)`** 를 반환합니다. `weights`에 1.0이 들어 있으므로 "섞지 않기"도 후보입니다. 개선폭이 `MIN_GAIN`(0.01)보다 작으면 실행 셀이 채택하지 않습니다.
""")

write("""
def best_blend_weight(oof_a, oof_b, y, weights):
    # 작성하세요
    return None
""", """
def best_blend_weight(oof_a, oof_b, y, weights):
    results = [(w, rmse(y, w * oof_a + (1 - w) * oof_b)) for w in weights]
    return min(results, key=lambda t: t[1])
""")

run("""
HGB = lambda: HistGradientBoostingRegressor(learning_rate=0.05, max_iter=400, max_leaf_nodes=15,
                                            min_samples_leaf=40, l2_regularization=1.0, random_state=42)
q1_hgb_rmse, q1_hgb_oof = cv_oof(HGB(), Q1_X, q1_y, Q1_NUM)
log("Q1", "HGB(튜닝) 단독", q1_hgb_rmse, False, "비교용")
blend = attempt("1-3", best_blend_weight, q1_ridge_oof, q1_hgb_oof, q1_y, [1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
Q1_W = 1.0
if blend:
    gain = q1_ridge_rmse - blend[1]
    keep = blend[0] < 1.0 and gain > MIN_GAIN
    log("Q1", f"Ridge*{blend[0]} + HGB*{1 - blend[0]:.1f}", blend[1], keep, f"개선 {gain:.4f}")
    Q1_W = blend[0] if keep else 1.0
""")

run("""
# 최종: 위에서 채택된 설정으로 전체 train 학습 → test 예측
ridge = Pipeline([("prep", build_preprocessor(Q1_NUM)), ("model", Ridge(alpha=Q1_ALPHA))]).fit(Q1_X, q1_y)
pred = Q1_W * ridge.predict(Q1_XT)
if Q1_W < 1.0:
    hgb = Pipeline([("prep", build_preprocessor(Q1_NUM)), ("model", HGB())]).fit(Q1_X, q1_y)
    pred = pred + (1 - Q1_W) * hgb.predict(Q1_XT)
sub = q1_sample[["id"]].copy()
sub["delivery_min"] = sub["id"].map(pd.Series(np.clip(pred, 0, None), index=q1_test["id"])).to_numpy()
_ = save_submission(sub, "q1_submission.csv", q1_sample, folder=IMPROVED)
print(f"Q1 최종: 도메인특징={q1_use_domain}, alpha={Q1_ALPHA}, ridge 가중치={Q1_W}")
""")

# ---------------------------------------------------------------- Q2
md("""
---
## 개선 2 — Q2 텍스트 분류 (macro-F1 ↑)

매뉴얼 6절: **표현(어떤 벡터를 쓰나)이 분류기 튜닝보다 큰 레버**, 특히 분포 변화가 있을 때. 진단부터 합니다.
`mock_embeddings`는 실제 LLM이 아닌 **가짜 모의 인코더**입니다.
""")

run("""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

q2_train, q2_test, q2_sample = csv("q2", "train.csv"), csv("q2", "test.csv"), csv("q2", "sample_submission.csv")
q2_E, q2_ET = q2_ready_embeddings(q2_train["id"], q2_test["id"])
q2_y = q2_train["label"].to_numpy()
tr2, va2 = train_test_split(np.arange(len(q2_train)), test_size=0.2, random_state=42, stratify=q2_y)
macro_f1 = lambda a, b: float(f1_score(a, b, average="macro"))

def tfidf_model(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=False, C=1.0):
    return Pipeline([("tfidf", TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, sublinear_tf=sublinear_tf)),
                     ("clf", LogisticRegression(max_iter=2000, C=C))])

base = tfidf_model().fit(q2_train["text"].iloc[tr2], q2_y[tr2])
q2_base_f1 = macro_f1(q2_y[va2], base.predict(q2_train["text"].iloc[va2]))
log("Q2", "기준선 TF-IDF char_wb(2,4)", q2_base_f1, True)
""")

md("""
### 2-1 진단: 분포 변화 점검 (레이블 불필요)
`oov_ratio(ref_texts, texts)`: `ref_texts`에 공백 기준으로 한 번도 나오지 않은 어절이 `texts` 전체 어절 중 몇 %인지(0~1) 반환합니다.
""")

write("""
def oov_ratio(ref_texts, texts):
    seen = None   # 작성하세요: ref_texts의 어절 집합
    words = None  # 작성하세요: texts의 모든 어절 리스트
    return None   # 작성하세요
""", """
def oov_ratio(ref_texts, texts):
    seen = {w for t in ref_texts for w in t.split()}
    words = [w for t in texts for w in t.split()]
    return float(np.mean([w not in seen for w in words])) if words else 0.0
""")

run("""
def _run_2_1():
    assert abs(oov_ratio(["a b"], ["a c", "d"]) - 2 / 3) < 1e-9
    v = oov_ratio(q2_train["text"].iloc[tr2], q2_train["text"].iloc[va2])
    t = oov_ratio(q2_train["text"], q2_test["text"])
    print(f"처음 보는 어절 비율 | validation {v:.3f} | test {t:.3f}")
    return v, t
q2_oov = attempt("2-1", _run_2_1)
""")

md("""
**진단 메모:** validation은 train과 분포가 같아 test 어려움을 재현하지 못합니다. 그래서 validation 점수가 비슷하면 **처음 보는 표현에 강한 쪽**을 고르는 것이 합리적입니다.

### 2-2 TF-IDF 레버 탐색
`tfidf_grid(configs)`: 각 설정 dict를 `tfidf_model(**cfg)`에 넣어 `tr2`로 학습하고 `va2` macro-F1을 계산해 `[(cfg, f1), ...]`를 반환합니다.
""")

write("""
def tfidf_grid(configs):
    results = []
    # 작성하세요
    return results
""", """
def tfidf_grid(configs):
    results = []
    for cfg in configs:
        m = tfidf_model(**cfg).fit(q2_train["text"].iloc[tr2], q2_y[tr2])
        results.append((cfg, macro_f1(q2_y[va2], m.predict(q2_train["text"].iloc[va2]))))
    return results
""")

run("""
CONFIGS = [dict(analyzer="char_wb", ngram_range=(1, 3), sublinear_tf=True, C=10.0),
           dict(analyzer="word", ngram_range=(1, 2)),
           dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)]
for cfg, f in attempt("2-2", tfidf_grid, CONFIGS) or []:
    log("Q2", "TF-IDF " + ",".join(f"{k}={v}" for k, v in cfg.items()), f, False, "val만으로는 차이 작음")
""")

md("""
### 2-3 표현 바꾸기: 정규화 임베딩 + 로지스틱 회귀
- `l2_normalize(E)`: 행마다 L2 노름으로 나눕니다(영벡터는 그대로).
- `emb_logreg(E, y, C=1.0)`: 정규화한 임베딩으로 `LogisticRegression(max_iter=3000, C=C)`를 학습해 반환합니다.
""")

write("""
def l2_normalize(E):
    return None  # 작성하세요

def emb_logreg(E, y, C=1.0):
    return None  # 작성하세요
""", """
def l2_normalize(E):
    E = np.asarray(E, dtype=np.float64)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)

def emb_logreg(E, y, C=1.0):
    return LogisticRegression(max_iter=3000, C=C).fit(l2_normalize(E), y)
""")

run("""
def _run_2_3():
    m = emb_logreg(q2_E[tr2], q2_y[tr2])
    f = macro_f1(q2_y[va2], m.predict(l2_normalize(q2_E[va2])))
    log("Q2", "모의 임베딩(정규화) + LR", f, True, "분포 변화에 강한 표현")
attempt("2-3", _run_2_3)
""")

md("""
### 2-4 최종 선택
`Q2_FINAL`을 `"tfidf"`, `"emb_lr"`, `"blend"`(두 모델 확률 평균) 중 하나로 정하고 이유를 쓰세요. 매뉴얼 11절: **약한 모델을 강한 모델에 섞으면 손해**일 수 있습니다.
""")

write("""
Q2_FINAL = ""         # 작성하세요
q2_final_reason = ""  # 작성하세요
""", """
Q2_FINAL = "emb_lr"
q2_final_reason = ("validation에서는 차이가 거의 없지만 test의 처음 보는 어절 비율이 훨씬 높다. "
                   "글자 조각에 기대는 TF-IDF를 섞으면 test에서 오히려 흔들릴 수 있어 임베딩 모델만 쓴다.")
""")

run("""
def _run_2_4():
    emb = emb_logreg(q2_E, q2_y)
    if Q2_FINAL == "emb_lr":
        pred = emb.predict(l2_normalize(q2_ET))
    elif Q2_FINAL == "tfidf":
        pred = tfidf_model().fit(q2_train["text"], q2_y).predict(q2_test["text"])
    elif Q2_FINAL == "blend":
        tf = tfidf_model().fit(q2_train["text"], q2_y)
        assert list(tf.classes_) == list(emb.classes_)
        prob = 0.5 * tf.predict_proba(q2_test["text"]) + 0.5 * emb.predict_proba(l2_normalize(q2_ET))
        pred = emb.classes_[prob.argmax(1)]
    else:
        raise ValueError("Q2_FINAL을 정하세요.")
    sub = q2_sample[["id"]].copy()
    sub["label"] = pd.Series(pred, index=q2_test["id"]).reindex(sub["id"]).to_numpy()
    print("선택:", Q2_FINAL, "| 이유 작성:", bool(q2_final_reason.strip()))
    return save_submission(sub, "q2_submission.csv", q2_sample, folder=IMPROVED)
_ = attempt("2-4", _run_2_4)
""")

# ---------------------------------------------------------------- Q3
md("""
---
## 개선 3 — Q3 이미지 분류 (accuracy ↑)

매뉴얼 5절: **과소적합을 먼저 풀고(학습 범위·epoch), 그다음 증강·TTA.** 진단 → 레버 → 재진단 순서로 진행합니다.
아래 셀은 v1 정답의 모델·파라미터 선택·1 epoch 학습·평가 함수를 제공합니다.
""")

run("""
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

CLASSES = ["circle", "square", "triangle", "cross"]
ALL_BLOCKS = ("block1", "block2", "block3", "block4")
q3_ids, q3_img = q3_images("train")
q3_test_ids, q3_test_img = q3_images("test")
q3_y = torch.tensor([CLASSES.index(c) for c in csv("q3", "train_labels.csv").set_index("id").loc[q3_ids, "label"]])
q3_sample = csv("q3", "sample_submission.csv")
tr3, va3 = train_test_split(np.arange(len(q3_y)), test_size=0.2, random_state=42, stratify=q3_y.numpy())
q3_x, q3_xt = q3_ready_tensors(q3_img, tr3, q3_test_img)

class ShapeClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone, self.pool = backbone, nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(backbone.out_channels, num_classes)
    def forward(self, x):
        return self.head(torch.flatten(self.pool(self.backbone(x)), 1))

def set_trainable(model, train_blocks=()):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for name in train_blocks:
        for p in getattr(model.backbone, name).parameters():
            p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]

def train_one_epoch(model, loader, optimizer, loss_fn, augment=None):
    model.train()
    for xb, yb in loader:
        if augment is not None:
            xb = augment(xb)
        optimizer.zero_grad()
        loss_fn(model(xb), yb).backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    correct = count = 0
    with torch.no_grad():
        for xb, yb in loader:
            correct += (model(xb).argmax(1) == yb).sum().item()
            count += len(yb)
    return correct / count

def loaders():
    g = torch.Generator().manual_seed(42)
    train = DataLoader(TensorDataset(q3_x[tr3], q3_y[tr3]), batch_size=64, shuffle=True, generator=g)
    train_eval = DataLoader(TensorDataset(q3_x[tr3], q3_y[tr3]), batch_size=256)
    val = DataLoader(TensorDataset(q3_x[va3], q3_y[va3]), batch_size=256)
    return train, train_eval, val

def new_model():
    seed_everything(42)
    return ShapeClassifier(load_pretrained_backbone(), len(CLASSES))
print(q3_x.shape, q3_xt.shape)
""")

md("""
### 3-1 best 체크포인트를 남기는 학습 함수
`fit(model, epochs, lr, train_blocks, augment=None)`:
1. `set_trainable(model, train_blocks)`로 학습 범위를 정하고 **그 파라미터로 optimizer(Adam)를 새로 만듭니다.**
2. 매 epoch `train_one_epoch` → `evaluate(val)`. 가장 좋은 val 정확도의 `state_dict`를 `copy.deepcopy`로 저장합니다.
3. 끝나면 best 가중치를 다시 불러오고 `{"val": best_val, "train": train_eval 정확도}`를 반환합니다.
""")

write("""
def fit(model, epochs, lr, train_blocks, augment=None):
    train_loader, train_eval_loader, val_loader = loaders()
    loss_fn = nn.CrossEntropyLoss()
    best_val, best_state = -1.0, None
    # 작성하세요
    return {"val": best_val, "train": None}
""", """
def fit(model, epochs, lr, train_blocks, augment=None):
    train_loader, train_eval_loader, val_loader = loaders()
    loss_fn = nn.CrossEntropyLoss()
    best_val, best_state = -1.0, None
    optimizer = torch.optim.Adam(set_trainable(model, train_blocks), lr=lr)
    for ep in range(epochs):
        train_one_epoch(model, train_loader, optimizer, loss_fn, augment)
        acc = evaluate(model, val_loader)
        if acc > best_val:
            best_val, best_state = acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {"val": best_val, "train": evaluate(model, train_eval_loader)}
""")

md("""
### 3-2 진단 함수
`diagnose(train_acc, val_acc, low=0.9, gap=0.05)`:
- `train_acc < low` → `"과소적합"` (학습 범위·epoch·lr부터)
- 아니고 `train_acc - val_acc > gap` → `"과대적합"` (증강·정규화)
- 그 외 → `"양호"` (후처리·TTA·앙상블)
""")

write("""
def diagnose(train_acc, val_acc, low=0.9, gap=0.05):
    return None  # 작성하세요
""", """
def diagnose(train_acc, val_acc, low=0.9, gap=0.05):
    if train_acc < low:
        return "과소적합"
    if train_acc - val_acc > gap:
        return "과대적합"
    return "양호"
""")

run("""
def _run_3_base():
    assert diagnose(0.7, 0.68) == "과소적합" and diagnose(0.99, 0.9) == "과대적합" and diagnose(0.97, 0.95) == "양호"
    m = new_model()
    fit(m, 8, 5e-3, ())                 # v1 기본: head만 8 epoch
    r = fit(m, 8, 1e-3, ("block4",))    #         + block4 8 epoch
    log("Q3", "기준선 head8 + block4 8", r["val"], True, f"train {r['train']:.3f} → {diagnose(r['train'], r['val'])}")
    return r
q3_base = attempt("3 기준선", _run_3_base)
""")

md("""
### 3-3 레버 ①: 학습 범위·학습량 (과소적합 대응)
기준선 진단이 과소적합이면 **전체 블록을 풀고 30 epoch**를 학습합니다(약 25초).
""")

run("""
def _run_3_3():
    m = new_model()
    r = fit(m, 30, 1e-3, ALL_BLOCKS)
    log("Q3", "전체 해제 30ep lr1e-3", r["val"], r["val"] > q3_base["val"], f"train {r['train']:.3f} → {diagnose(r['train'], r['val'])}")
    return m, r
q3_full = attempt("3-3", _run_3_3)
""")

md("""
### 3-4 레버 ②: 증강
`augment_batch(xb)`: 배치 `(B,3,H,W)`의 **샘플마다 50% 확률로 좌우 반전**하고, **배치 전체를 0~3회 무작위 90° 회전**합니다(`torch.rot90`, `dims=[2, 3]`). 원/사각형/삼각형/십자는 반전·90° 회전에도 레이블이 바뀌지 않습니다. 입력 텐서를 직접 수정하지 마세요.
""")

write("""
def augment_batch(xb):
    xb = xb.clone()
    # 작성하세요
    return xb
""", """
def augment_batch(xb):
    xb = xb.clone()
    flip = torch.rand(len(xb)) < 0.5
    xb[flip] = torch.flip(xb[flip], dims=[3])
    k = int(torch.randint(0, 4, (1,)).item())
    return torch.rot90(xb, k, dims=[2, 3])
""")

run("""
def _run_3_4():
    z = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).view(2, 3, 4, 4)
    out = augment_batch(z)
    assert out.shape == z.shape and torch.equal(z, torch.arange(96, dtype=torch.float32).view(2, 3, 4, 4)), "입력이 바뀌었습니다"
    m = new_model()
    r = fit(m, 30, 1e-3, ALL_BLOCKS, augment=augment_batch)
    prev = q3_full[1]["val"] if q3_full else 0.0
    log("Q3", "+ 증강(반전·90°회전)", r["val"], r["val"] > prev, f"train {r['train']:.3f} → {diagnose(r['train'], r['val'])}")
    return m, r
q3_aug = attempt("3-4", _run_3_4)
""")

md("""
### 3-5 레버 ③: TTA
`predict_proba(model, x, batch_size=256, tta=False)`: 평가 모드·기울기 없이 배치별 softmax 확률 `(N, C)`를 반환합니다. `tta=True`면 원본과 좌우 반전 이미지의 확률을 평균합니다. 마지막 짧은 배치도 포함해야 합니다.
""")

write("""
def predict_proba(model, x, batch_size=256, tta=False):
    # 작성하세요
    return None
""", """
def predict_proba(model, x, batch_size=256, tta=False):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            xb = x[s:s + batch_size]
            p = model(xb).softmax(1)
            if tta:
                p = (p + model(torch.flip(xb, dims=[3])).softmax(1)) / 2
            out.append(p)
    return torch.cat(out)
""")

run("""
def _run_3_5():
    candidates = {}
    for name, pack in [("전체 해제", q3_full), ("+증강", q3_aug)]:
        if pack is None:
            continue
        m = pack[0]
        for tta in [False, True]:
            p = predict_proba(m, q3_x[va3], batch_size=100, tta=tta)
            assert p.shape == (len(va3), len(CLASSES))
            candidates[(name, tta)] = (m, float((p.argmax(1) == q3_y[va3]).float().mean()))
    for (name, tta), (_, acc) in candidates.items():
        print(f"{name:6s} TTA={tta!s:5s} val {acc:.3f}")
    best_key = max(candidates, key=lambda k: candidates[k][1])   # validation으로만 선택
    log("Q3", f"최종 후보 {best_key[0]} TTA={best_key[1]}", candidates[best_key][1], True, "val 최고")
    model, tta = candidates[best_key][0], best_key[1]
    pred = predict_proba(model, q3_xt, tta=tta).argmax(1).numpy()
    sub = q3_sample[["id"]].copy()
    sub["label"] = pd.Series(np.array(CLASSES)[pred], index=q3_test_ids).reindex(sub["id"]).to_numpy()
    return save_submission(sub, "q3_submission.csv", q3_sample, folder=IMPROVED)
_ = attempt("3-5", _run_3_5)
""")

md("""
---
## 마무리 — 실험 기록과 test 확인 (한 번만)
아래 셀은 실험 기록 전체를 보여 주고, 로컬 채점기로 **v1 기본 답안**과 **개선본**의 test 점수를 비교합니다. 이 점수를 보고 앞으로 돌아가 설정을 바꾸지 마세요(시험에서는 test 정답이 없습니다).
""")

run("""
print(pd.DataFrame(LOG).to_string(index=False))
scorer = ROOT / "정답_채점용" / "score_local.py"
for label, folder in [("v1 기본 답안", ROOT / "정답_채점용" / "정답_제출예시"), ("개선본", IMPROVED)]:
    print(f"\\n== {label} ==")
    print(subprocess.run([sys.executable, "-I", "-X", "utf8", str(scorer), str(folder)],
                         capture_output=True, text=True, encoding="utf-8").stdout.strip())
""")


# ======================================================================
def build(answer):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, src, ans in CELLS:
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(src))
        else:
            nb.cells.append(nbf.v4.new_code_cell(ans if (answer and ans) else src))
    if answer:
        nb.cells[0].source += "\n\n> **정답·해설용입니다.** `[정답 예시]`와 같은 동작을 하는 다른 코드도 정답입니다."
    return nb


if __name__ == "__main__":
    nbf.write(build(False), ROOT / "Lv3_v1_점수개선_실습.ipynb")
    nbf.write(build(True), ROOT / "정답_채점용" / "Lv3_v1_점수개선_실습_정답.ipynb")
    print("cells:", len(CELLS))
