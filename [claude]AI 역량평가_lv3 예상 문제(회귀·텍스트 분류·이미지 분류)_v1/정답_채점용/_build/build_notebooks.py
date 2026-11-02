"""문제/정답 노트북 생성기. 실행: python -I build_notebooks.py

각 코드 셀은 (문제용, 정답용) 두 버전을 가진다. 문제용은 '작성하세요' 빈칸.
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
CELLS = []


def md(text):
    CELLS.append(("md", text.strip("\n"), None))


def run(code):
    """실행만 하세요 셀(문제·정답 동일)."""
    CELLS.append(("code", "# [실행만 하세요]\n" + code.strip("\n"), None))


def write(problem, answer):
    """작성하세요 셀."""
    CELLS.append(("code", "# [작성하세요]\n" + problem.strip("\n"), "# [정답 예시]\n" + answer.strip("\n")))


# ======================================================================
md("""
# [claude] Lv3 모의평가 v1 — 회귀 · 텍스트 분류 · 이미지 분류

**창작 모의문제입니다. 실제 Level 3 문제·배점·지표·제출 형식이 아닙니다.** 규칙은 `README.md`를 먼저 읽으세요.

- 독립된 3문항, 권장 4시간(Q1 70분 / Q2 70분 / Q3 70분 / 검토 30분). 앞 문항을 못 풀어도 다음 문항을 진행할 수 있습니다.
- `[작성하세요]` 셀의 `None`·빈 문자열을 채웁니다. **함수 이름·매개변수·반환 형식은 바꾸지 마세요.** 중간 변수·반복문은 자유입니다.
- `[실행만 하세요]` 셀은 수정하지 않고 실행합니다. 확인 셀은 형태·간단한 성질만 검사하며 점수를 알려주는 채점기가 아닙니다.
- 데이터 준비 소문항(1-1, 2-1, 3-1)이나 1-3·1-4를 못 풀면 확인 셀이 **진행용 준비 입력**으로 대신 이어 갑니다(해당 소문항 점수와 무관).
- 소문항 배점(괄호)은 **과정 채점 가정**입니다. 각 문항의 예측 CSV(`제출물/`)는 Kaggle식 지표로 따로 확인하는 연습용입니다.
- **test 정답을 학습·모델 선택에 쓰지 않습니다.** `정답_채점용/` 폴더는 풀이가 끝날 때까지 열지 마세요.
""")

run("""
from pathlib import Path
import sys
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "practice_support.py").exists()), None)
if ROOT is None:
    raise FileNotFoundError("모의문제 폴더(또는 그 하위 폴더)에서 노트북을 실행하세요.")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from practice_support import seed_everything, attempt, csv, save_submission
from practice_support import q1_ready_features, q2_ready_embeddings, q3_ready_tensors  # 독립 진행용

seed_everything(42)
print("준비 완료:", ROOT.name)
""")

# ---------------------------------------------------------------- Q1
md("""
---
## Q1 데이터 — 배달 소요시간 회귀 (35점)

`data/q1/train.csv`(2,400행)와 `test.csv`(800행)는 음식 배달 주문 기록입니다(합성 자료). 목표 `delivery_min`은 주문부터 도착까지 걸린 **분(실수)** 입니다.

| 열 | 설명 |
|---|---|
| id | 주문 ID |
| order_time | `YYYY-MM-DD HH:MM` 문자열 |
| region, weather | 범주형. weather에 결측 있음. **test에는 train에 없는 region이 있습니다.** |
| distance_km, n_items, prep_min, courier_rating | 수치형. 일부 결측 |

평가지표(가정): **RMSE, 낮을수록 좋음.** 제출: `제출물/q1_submission.csv` — `id,delivery_min`.
**주의: `sample_submission.csv`의 행 순서는 `test.csv`와 다릅니다.**
""")

run("""
q1_train = csv("q1", "train.csv")
q1_test = csv("q1", "test.csv")
q1_sample = csv("q1", "sample_submission.csv")
print(q1_train.shape, q1_test.shape, q1_sample.shape)
print(q1_train.isna().sum().to_dict())
q1_train.head()
""")

md("""
### 1-1 (5점) 시간 특징 만들기
`add_time_features(df)`를 완성하세요. 입력 `df`는 **수정하지 않고** 복사본에 다음 열을 추가해 반환합니다.
- `hour`: 주문 시(0~23), `dayofweek`: 요일(월=0 … 일=6)
- `is_peak`: `hour`가 `PEAK_HOURS`에 있으면 1, 아니면 0 (정수)
- `is_weekend`: 토·일이면 1, 아니면 0 (정수)
""")

write("""
PEAK_HOURS = [11, 12, 13, 17, 18, 19, 20]

def add_time_features(df):
    out = df.copy()
    t = None                 # 작성하세요: order_time 문자열 → datetime
    out["hour"] = None       # 작성하세요
    out["dayofweek"] = None  # 작성하세요
    out["is_peak"] = None    # 작성하세요
    out["is_weekend"] = None # 작성하세요
    return out
""", """
PEAK_HOURS = [11, 12, 13, 17, 18, 19, 20]

def add_time_features(df):
    out = df.copy()
    t = pd.to_datetime(out["order_time"], format="%Y-%m-%d %H:%M")
    out["hour"] = t.dt.hour
    out["dayofweek"] = t.dt.dayofweek
    out["is_peak"] = out["hour"].isin(PEAK_HOURS).astype(int)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    return out
""")

run("""
def _check_1_1():
    a, b = add_time_features(q1_train), add_time_features(q1_test)
    assert "hour" not in q1_train.columns, "입력 df가 수정되었습니다."
    for col in ["hour", "dayofweek", "is_peak", "is_weekend"]:
        assert a[col].notna().all() and b[col].notna().all(), f"{col} 미작성"
    assert set(a["is_peak"].unique()) <= {0, 1} and set(a["is_weekend"].unique()) <= {0, 1}
    print(a[["order_time", "hour", "dayofweek", "is_peak", "is_weekend"]].head(3))
    print("peak 비율:", round(float(a["is_peak"].mean()), 3))
    return a, b
q1_feats = attempt("1-1", _check_1_1)
if q1_feats is None:
    print("→ 이후 진행용 준비 특징을 사용합니다(1-1 점수와 무관).")
    q1_feats = (q1_ready_features(q1_train), q1_ready_features(q1_test))
q1_X, q1_XT = q1_feats
""")

md("""
### 1-2 (8점) 전처리기
`build_preprocessor(num_cols, cat_cols)`를 완성하세요.
- 수치형: **중앙값 대치 → 표준화**
- 범주형: **최빈값 대치 → 원-핫 인코딩**, 학습 때 없던 범주는 오류 없이 무시(모두 0)
""")

write("""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUM_COLS = ["distance_km", "n_items", "prep_min", "courier_rating", "hour", "is_peak", "is_weekend"]
CAT_COLS = ["region", "weather"]

def build_preprocessor(num_cols, cat_cols):
    num_pipe = None  # 작성하세요
    cat_pipe = None  # 작성하세요
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
""", """
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUM_COLS = ["distance_km", "n_items", "prep_min", "courier_rating", "hour", "is_peak", "is_weekend"]
CAT_COLS = ["region", "weather"]

def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
""")

run("""
def _check_1_2():
    prep = build_preprocessor(NUM_COLS, CAT_COLS).fit(q1_X)   # train에만 fit
    a, b = prep.transform(q1_X), prep.transform(q1_XT)         # test의 새 region도 통과해야 함
    a = a.toarray() if hasattr(a, "toarray") else a
    b = b.toarray() if hasattr(b, "toarray") else b
    assert not np.isnan(a).any() and not np.isnan(b).any(), "변환 결과에 NaN"
    print("train", a.shape, "test", b.shape, "| 수치 열 평균≈0:", np.allclose(a[:, :len(NUM_COLS)].mean(0), 0, atol=1e-6))
attempt("1-2", _check_1_2)
""")

md("""
### 1-3 (10점) 교차검증 RMSE
- `rmse(y_true, y_pred)`: RMSE를 실수로 반환
- `cv_rmse(make_model, X, y, n_splits=5, seed=42)`: 각 fold마다 **새 Pipeline(전처리+모델)** 을 학습 fold에만 fit하고 검증 fold RMSE를 모읍니다. `(평균, fold별 리스트)`를 반환합니다. 행 선택은 `iloc`을 쓰세요.
- 서술(2점) `q1_reason`: 전처리를 fold마다 학습 fold에서만 fit하는 이유를 1~2문장으로.
""")

write("""
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

def rmse(y_true, y_pred):
    return None  # 작성하세요

def cv_rmse(make_model, X, y, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr_idx, va_idx in kf.split(X):
        pipe = Pipeline([("prep", build_preprocessor(NUM_COLS, CAT_COLS)), ("model", make_model())])
        # 작성하세요: tr_idx로 학습, va_idx로 예측 후 RMSE를 scores에 추가
    return float(np.mean(scores)), scores

CANDIDATES = {
    "ridge": lambda: Ridge(alpha=1.0),
    "hgb": lambda: HistGradientBoostingRegressor(random_state=42),
}

q1_reason = ""  # 작성하세요
""", """
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def cv_rmse(make_model, X, y, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr_idx, va_idx in kf.split(X):
        pipe = Pipeline([("prep", build_preprocessor(NUM_COLS, CAT_COLS)), ("model", make_model())])
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        scores.append(rmse(y.iloc[va_idx], pipe.predict(X.iloc[va_idx])))
    return float(np.mean(scores)), scores

CANDIDATES = {
    "ridge": lambda: Ridge(alpha=1.0),
    "hgb": lambda: HistGradientBoostingRegressor(random_state=42),
}

q1_reason = ("검증 fold의 중앙값·평균·표준편차·범주 정보가 학습에 섞이면 검증 점수가 낙관적으로 나오는 데이터 누수가 생긴다. "
             "Pipeline을 fold마다 학습 fold에만 fit하면 실제 test를 처음 보는 상황과 같아진다.")
""")

run("""
q1_y = q1_train["delivery_min"]
cv_results = {}
for name, make_model in CANDIDATES.items():
    res = attempt(f"1-3 {name}", cv_rmse, make_model, q1_X, q1_y)
    if res is not None and np.isfinite(res[0]) and len(res[1]) == 5:
        cv_results[name] = res
        print(f"  {name}: 평균 RMSE {res[0]:.3f} | folds {np.round(res[1], 2)}")
    else:
        print(f"  {name}: fold 점수 5개가 만들어지지 않았습니다.")
print("기준(평균값 예측) RMSE:", round(float(q1_y.std(ddof=0)), 3))
print("서술 작성 여부:", bool(q1_reason.strip()))
""")

md("""
### 1-4 (7점) 최종 모델 학습과 test 예측
`fit_and_predict(cv_results, X_train, y_train, X_test)`: CV 평균 RMSE가 **가장 작은** 후보를 고르고(test 사용 금지), 전체 train으로 다시 학습해 test를 예측합니다. 소요시간은 음수가 될 수 없으므로 0 미만은 0으로 자릅니다. `(best_name, pred)` 반환.
""")

write("""
def fit_and_predict(cv_results, X_train, y_train, X_test):
    best_name = None  # 작성하세요: cv_results = {이름: (평균, folds)}
    pipe = Pipeline([("prep", build_preprocessor(NUM_COLS, CAT_COLS)), ("model", CANDIDATES[best_name]())])
    pred = None       # 작성하세요: 학습 → 예측 → 0 미만 자르기
    return best_name, pred
""", """
def fit_and_predict(cv_results, X_train, y_train, X_test):
    best_name = min(cv_results, key=lambda name: cv_results[name][0])
    pipe = Pipeline([("prep", build_preprocessor(NUM_COLS, CAT_COLS)), ("model", CANDIDATES[best_name]())])
    pipe.fit(X_train, y_train)
    pred = np.clip(pipe.predict(X_test), 0, None)
    return best_name, pred
""")

run("""
if len(cv_results) < len(CANDIDATES):
    print("→ 1-3 결과 대신 진행용 고정값을 사용합니다(1-3 점수와 무관).")
    cv_results = {"ridge": (7.45, []), "hgb": (7.92, [])}
q1_out = attempt("1-4", fit_and_predict, cv_results, q1_X, q1_y, q1_XT)
if q1_out is not None and q1_out[1] is not None:
    q1_best, q1_pred = q1_out
    print("선택:", q1_best, "| 예측", q1_pred.shape, "최소", round(float(q1_pred.min()), 2))
else:
    print("→ 1-5 진행용 임시 예측(학습 목표 중앙값)을 사용합니다.")
    q1_pred = np.full(len(q1_test), float(q1_y.median()))
""")

md("""
### 1-5 (5점) 제출 파일
`make_q1_submission(sample, test_ids, pred)`: `pred[i]`는 `test_ids[i]`의 예측입니다. **sample의 id 순서**에 맞춘 `id,delivery_min` DataFrame을 반환하세요. 위치(행 번호)로 붙이면 안 됩니다.
""")

write("""
def make_q1_submission(sample, test_ids, pred):
    sub = sample[["id"]].copy()
    sub["delivery_min"] = None  # 작성하세요
    return sub
""", """
def make_q1_submission(sample, test_ids, pred):
    sub = sample[["id"]].copy()
    pred_by_id = pd.Series(np.asarray(pred, dtype=float), index=pd.Index(test_ids))
    sub["delivery_min"] = sub["id"].map(pred_by_id).to_numpy()
    return sub
""")

run("""
def _check_1_5():
    # 숨은 확인: 순서를 섞은 작은 입력에서도 id로 맞춰지는가
    toy = make_q1_submission(pd.DataFrame({"id": ["a", "b", "c"]}), ["c", "a", "b"], [3.0, 1.0, 2.0])
    assert toy["delivery_min"].tolist() == [1.0, 2.0, 3.0], "id가 아닌 위치로 붙였습니다."
    sub = make_q1_submission(q1_sample, q1_test["id"].to_numpy(), q1_pred)
    return save_submission(sub, "q1_submission.csv", q1_sample)
_ = attempt("1-5", _check_1_5)
""")

# ---------------------------------------------------------------- Q2
md("""
---
## Q2 LLM 활용 — 고객 문의 텍스트 분류 (35점)

`data/q2/train.csv`(1,200행: id,text,label)와 `test.csv`(600행: id,text). 레이블 5개: `delivery, refund, payment, account, defect` (`labels.json`에 설명).
평가지표(가정): **macro-F1, 높을수록 좋음.** 제출: `제출물/q2_submission.csv` — `id,label`.

`mock_embeddings.npz`에는 문장 임베딩 캐시(`train_ids, train_emb, test_ids, test_emb`, 64차원)가 있습니다.
> **주의: 이 임베딩은 실제 LLM·문장임베딩 모델의 출력이 아닙니다.** 동의어 사전을 쓰는 *가짜 모의 인코더*로 만든 연습용 벡터이며, 사전학습 임베딩이 '처음 보는 표현도 의미가 비슷하면 가깝게 놓는다'는 성질만 흉내 냅니다. **벡터는 정규화되어 있지 않고**, 캐시의 행 순서는 CSV와 다릅니다. 실제 로컬 모델로 임베딩을 직접 계산하는 방법은 `Reference.md`를 보세요.

운영 안내(모의 가정): **test 문의에는 train에서 보기 어려운 표현(동의어·구어체)이 다수 포함됩니다.**
""")

run("""
from practice_support import q2_embeddings, q2_label_info
from sklearn.model_selection import train_test_split

q2_train = csv("q2", "train.csv")
q2_test = csv("q2", "test.csv")
q2_sample = csv("q2", "sample_submission.csv")
q2_cache = q2_embeddings()
LABELS = list(q2_label_info())
print(q2_train.shape, q2_test.shape, {k: v.shape for k, v in q2_cache.items()})
print(q2_train["label"].value_counts().to_dict())
tr_idx2, va_idx2 = train_test_split(np.arange(len(q2_train)), test_size=0.2, random_state=42,
                                    stratify=q2_train["label"])
q2_train.head(3)
""")

md("""
### 2-1 (5점) 임베딩을 id로 정렬
`align_embeddings(target_ids, cache_ids, cache_emb)`: `target_ids` 순서대로 `cache_emb`의 행을 골라 `(len(target_ids), D)` 배열을 반환합니다. 캐시에 없는 id가 있으면 `KeyError`를 발생시키세요.
""")

write("""
def align_embeddings(target_ids, cache_ids, cache_emb):
    pos = None  # 작성하세요: id → 캐시 행 위치
    return None # 작성하세요
""", """
def align_embeddings(target_ids, cache_ids, cache_emb):
    pos = {cid: i for i, cid in enumerate(cache_ids)}
    missing = [t for t in target_ids if t not in pos]
    if missing:
        raise KeyError(f"캐시에 없는 id {len(missing)}개: {missing[:3]}")
    return cache_emb[[pos[t] for t in target_ids]]
""")

run("""
def _check_2_1():
    toy = align_embeddings(["b", "a"], np.array(["a", "b", "c"]), np.array([[1.0], [2.0], [3.0]]))
    assert toy.tolist() == [[2.0], [1.0]]
    try:
        align_embeddings(["z"], np.array(["a"]), np.array([[1.0]]))
        raise AssertionError("없는 id에서 KeyError가 나야 합니다.")
    except KeyError:
        pass
    tr = align_embeddings(q2_train["id"].tolist(), q2_cache["train_ids"], q2_cache["train_emb"])
    te = align_embeddings(q2_test["id"].tolist(), q2_cache["test_ids"], q2_cache["test_emb"])
    norms = np.linalg.norm(tr, axis=1)
    print("train", tr.shape, "test", te.shape, "| 노름 범위", norms.min().round(2), "~", norms.max().round(2))
    return tr, te
q2_E = attempt("2-1", _check_2_1)
if q2_E is None:
    print("→ 이후 진행용 정렬 임베딩을 사용합니다(2-1 점수와 무관).")
    q2_E = q2_ready_embeddings(q2_train["id"], q2_test["id"])
q2_emb_train, q2_emb_test = q2_E
""")

md("""
### 2-2 (6점) TF-IDF 기준선 (LLM 아님)
- `build_tfidf_model()`: `TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))` + `LogisticRegression(max_iter=2000)` Pipeline
- `macro_f1(y_true, y_pred)`: macro 평균 F1
""")

write("""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def build_tfidf_model():
    return None  # 작성하세요

def macro_f1(y_true, y_pred):
    return None  # 작성하세요
""", """
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def build_tfidf_model():
    return Pipeline([("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
                     ("clf", LogisticRegression(max_iter=2000))])

def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro"))
""")

run("""
q2_val_scores = {}
def _run_2_2():
    m = build_tfidf_model().fit(q2_train["text"].iloc[tr_idx2], q2_train["label"].iloc[tr_idx2])
    return macro_f1(q2_train["label"].iloc[va_idx2], m.predict(q2_train["text"].iloc[va_idx2]))
s = attempt("2-2", _run_2_2)
if s is not None:
    q2_val_scores["tfidf"] = s
    print("TF-IDF validation macro-F1:", round(s, 4))
""")

md("""
### 2-3 (8점) 코사인 유사도 행렬
`cosine_similarity_matrix(A, B, eps=1e-12)`: `A`는 `(n,d)` 또는 `(d,)`, `B`는 `(m,d)` 또는 `(d,)`. **각 행을 L2 노름으로 나눈 뒤** 곱해 `(n,m)` 배열을 반환합니다(1차원 입력은 행 1개로 취급). 노름이 0인 행은 0 나누기 없이 유사도 0이 되게 하세요. n·m·d가 바뀌어도 동작해야 합니다(반복 횟수 하드코딩 금지).
""")

write("""
def cosine_similarity_matrix(A, B, eps=1e-12):
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    return None  # 작성하세요
""", """
def cosine_similarity_matrix(A, B, eps=1e-12):
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    A_unit = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), eps)
    B_unit = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), eps)
    return A_unit @ B_unit.T
""")

run("""
def _check_2_3():
    rng = np.random.default_rng(0)
    for n, m, d in [(1, 3, 4), (5, 9, 8), (2, 1, 3)]:
        A, B = rng.normal(size=(n, d)), rng.normal(size=(m, d))
        S = cosine_similarity_matrix(A, B)
        ref = np.array([[a @ b / np.linalg.norm(a) / np.linalg.norm(b) for b in B] for a in A])
        assert S.shape == (n, m) and np.allclose(S, ref), f"({n},{m},{d}) 불일치"
    v = rng.normal(size=6)
    assert cosine_similarity_matrix(v, 3.7 * v).shape == (1, 1)
    assert np.isclose(cosine_similarity_matrix(v, 3.7 * v)[0, 0], 1.0), "크기 배율에 따라 값이 변합니다(노름으로 나누었나요?)"
    Z = cosine_similarity_matrix(np.zeros(6), rng.normal(size=(2, 6)))
    assert np.all(np.isfinite(Z)) and np.allclose(Z, 0), "영벡터 처리"
    print("모양·값·배율 불변·영벡터 확인 통과")
attempt("2-3", _check_2_3)
""")

md("""
### 2-4 (8점) top-k와 kNN 분류
- `top_k_indices(scores, k)`: 1차원 점수에서 **큰 순서대로** 최대 k개 위치(정수 배열). `len(scores) < k`이면 전부 반환.
- `knn_predict(sim, train_labels, k=5)`: `sim`은 `(n_query, n_train)` 유사도. 각 행의 top-k 이웃 레이블별로 **유사도를 합산**해 가장 큰 레이블을 예측합니다.
""")

write("""
def top_k_indices(scores, k):
    scores = np.asarray(scores)
    return None  # 작성하세요

def knn_predict(sim, train_labels, k=5):
    train_labels = np.asarray(train_labels)
    preds = []
    for row in sim:
        idx = top_k_indices(row, k)
        votes = {}
        # 작성하세요: 레이블별 유사도 합 → 최댓값 레이블을 preds에 추가
    return np.array(preds)
""", """
def top_k_indices(scores, k):
    scores = np.asarray(scores)
    k = min(k, len(scores))
    return np.argsort(-scores, kind="stable")[:k]

def knn_predict(sim, train_labels, k=5):
    train_labels = np.asarray(train_labels)
    preds = []
    for row in sim:
        idx = top_k_indices(row, k)
        votes = {}
        for j in idx:
            votes[train_labels[j]] = votes.get(train_labels[j], 0.0) + float(row[j])
        preds.append(max(votes, key=votes.get))
    return np.array(preds)
""")

run("""
def _check_2_4():
    assert top_k_indices([0.1, 0.9, 0.5], 2).tolist() == [1, 2]
    assert top_k_indices([0.3, 0.2], 5).tolist() == [0, 1], "개수가 k보다 적을 때"
    assert len(top_k_indices(np.arange(9), 3)) == 3
    toy = knn_predict(np.array([[0.9, 0.8, 0.1], [0.1, 0.2, 0.95]]), ["x", "x", "y"], k=2)
    assert toy.tolist() == ["x", "y"]
    y_all = q2_train["label"].to_numpy()
    S = cosine_similarity_matrix(q2_emb_train[va_idx2], q2_emb_train[tr_idx2])
    for k in [1, 5, 15]:
        q2_val_scores[f"knn{k}"] = macro_f1(y_all[va_idx2], knn_predict(S, y_all[tr_idx2], k))
        print(f"kNN k={k:2d} validation macro-F1:", round(q2_val_scores[f"knn{k}"], 4))
attempt("2-4", _check_2_4)
""")

md("""
### 2-5 (5점) 클래스 프로토타입 분류
`class_prototypes(emb, labels, label_names)`: 각 임베딩을 L2 정규화한 뒤 클래스별 평균을 구해 `(C, D)` 배열(행 순서 = `label_names`)을 반환합니다. 예측은 프로토타입과의 코사인 유사도가 가장 큰 클래스입니다.
""")

write("""
def class_prototypes(emb, labels, label_names):
    labels = np.asarray(labels)
    unit = None  # 작성하세요
    return None  # 작성하세요

def prototype_predict(query_emb, prototypes, label_names):
    return None  # 작성하세요: cosine_similarity_matrix 활용
""", """
def class_prototypes(emb, labels, label_names):
    labels = np.asarray(labels)
    unit = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return np.stack([unit[labels == name].mean(axis=0) for name in label_names])

def prototype_predict(query_emb, prototypes, label_names):
    sim = cosine_similarity_matrix(query_emb, prototypes)
    return np.asarray(label_names)[sim.argmax(axis=1)]
""")

run("""
def _run_2_5():
    y_all = q2_train["label"].to_numpy()
    P = class_prototypes(q2_emb_train[tr_idx2], y_all[tr_idx2], LABELS)
    assert P.shape == (len(LABELS), q2_emb_train.shape[1])
    q2_val_scores["prototype"] = macro_f1(y_all[va_idx2], prototype_predict(q2_emb_train[va_idx2], P, LABELS))
    print("prototype validation macro-F1:", round(q2_val_scores["prototype"], 4))
attempt("2-5", _run_2_5)

# 분포 변화 점검(레이블 불필요): 처음 보는 어절 비율
seen = set(" ".join(q2_train["text"].iloc[tr_idx2]).split())
oov = lambda texts: np.mean([w not in seen for w in " ".join(texts).split()])
print("처음 보는 어절 비율 | validation:", round(oov(q2_train["text"].iloc[va_idx2]), 3), "| test:", round(oov(q2_test["text"]), 3))
print("validation 점수:", {k: round(v, 4) for k, v in q2_val_scores.items()})
""")

md("""
### 2-6 (3점) 최종 방법 선택과 제출
validation 점수와 위 분포 점검을 근거로 `Q2_METHOD`를 `"tfidf"`, `"knn"`, `"prototype"` 중 하나로 정하고, `q2_choice_reason`에 이유를 1~2문장 쓰세요. 아래 셀이 **전체 train**으로 다시 학습해 test를 예측·저장합니다.
""")

write("""
Q2_METHOD = ""         # 작성하세요
Q2_K = 5               # knn 선택 시 사용할 k (선택)
q2_choice_reason = ""  # 작성하세요
""", """
Q2_METHOD = "knn"
Q2_K = 5
q2_choice_reason = ("validation에서는 세 방법이 비슷하지만 test에는 train에 없던 어절 비율이 크게 높다. "
                    "글자 조각을 외우는 TF-IDF보다 의미가 비슷한 표현을 가깝게 두는 임베딩 기반 kNN이 test에 더 잘 일반화될 것으로 판단했다.")
""")

run("""
def _run_2_6():
    y_all = q2_train["label"].to_numpy()
    if Q2_METHOD == "tfidf":
        pred = build_tfidf_model().fit(q2_train["text"], y_all).predict(q2_test["text"])
    elif Q2_METHOD == "knn":
        pred = knn_predict(cosine_similarity_matrix(q2_emb_test, q2_emb_train), y_all, Q2_K)
    elif Q2_METHOD == "prototype":
        pred = prototype_predict(q2_emb_test, class_prototypes(q2_emb_train, y_all, LABELS), LABELS)
    else:
        raise ValueError("Q2_METHOD를 정하세요.")
    sub = q2_sample[["id"]].copy()
    sub["label"] = pd.Series(pred, index=q2_test["id"]).reindex(sub["id"]).to_numpy()
    print("예측 분포:", sub["label"].value_counts().to_dict(), "| 이유 작성:", bool(q2_choice_reason.strip()))
    return save_submission(sub, "q2_submission.csv", q2_sample)
_ = attempt("2-6", _run_2_6)
""")

# ---------------------------------------------------------------- Q3
md("""
---
## Q3 AI 모델 — 도형 이미지 분류 + 사전학습 백본 (30점)

`data/q3/train_images.npz`(1,200장)·`test_images.npz`(800장): `ids`, `images` — `(N, 32, 32, 3)` **HWC, uint8**. 학습 레이블은 `train_labels.csv`(id,label, 행 순서가 이미지와 다름). 클래스 4개: `circle, square, triangle, cross` (합성 자료).
`assets/q3_backbone.pt`는 **다른 도형 6종으로 미리 학습한** 작은 CNN 백본입니다(지정 모델 가정). `load_pretrained_backbone()`으로 불러오며 출력은 `(B, 64, H//8, W//8)`, `backbone.out_channels == 64`.

평가지표(가정): **accuracy.** 제출: `제출물/q3_submission.csv` — `id,label`. CPU만 사용합니다.
""")

run("""
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from practice_support import q3_images, load_pretrained_backbone

CLASSES = ["circle", "square", "triangle", "cross"]
q3_ids, q3_img = q3_images("train")
q3_test_ids, q3_test_img = q3_images("test")
q3_lab = csv("q3", "train_labels.csv").set_index("id").loc[q3_ids, "label"]
q3_y = torch.tensor([CLASSES.index(c) for c in q3_lab], dtype=torch.long)
q3_sample = csv("q3", "sample_submission.csv")
tr_idx3, va_idx3 = train_test_split(np.arange(len(q3_y)), test_size=0.2, random_state=42, stratify=q3_y.numpy())
print(q3_img.shape, q3_img.dtype, q3_test_img.shape, np.bincount(q3_y.numpy()))
""")

md("""
### 3-1 (6점) 텐서 변환과 채널 정규화
- `to_chw_float(images)`: `(N,H,W,3)` uint8 → `torch.float32` `(N,3,H,W)`, 값 범위 [0,1]
- `channel_stats(x)`: `(N,3,H,W)`에서 채널별 평균·표준편차 `(3,)` 두 개 반환
- `normalize(x, mean, std)`: 채널별 `(x - mean) / std` (브로드캐스팅)

통계는 **학습 분할에서만** 계산해 검증·test에 같이 적용합니다(아래 실행 셀).
""")

write("""
def to_chw_float(images):
    return None  # 작성하세요

def channel_stats(x):
    mean = None  # 작성하세요
    std = None   # 작성하세요
    return mean, std

def normalize(x, mean, std):
    return None  # 작성하세요
""", """
def to_chw_float(images):
    return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 255.0

def channel_stats(x):
    mean = x.mean(dim=(0, 2, 3))
    std = x.std(dim=(0, 2, 3))
    return mean, std

def normalize(x, mean, std):
    return (x - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
""")

run("""
def _run_3_1():
    x_all = to_chw_float(q3_img)
    assert x_all.shape == (len(q3_img), 3, 32, 32) and x_all.dtype == torch.float32 and float(x_all.max()) <= 1.0
    mean, std = channel_stats(x_all[tr_idx3])
    assert mean.shape == (3,) and std.shape == (3,)
    x_all = normalize(x_all, mean, std)
    x_test = normalize(to_chw_float(q3_test_img), mean, std)
    print("train 분할 정규화 후 채널 평균:", x_all[tr_idx3].mean(dim=(0, 2, 3)).numpy().round(3))
    return x_all, x_test
q3_X = attempt("3-1", _run_3_1)
if q3_X is None:
    print("→ 이후 진행용 준비 텐서를 사용합니다(3-1 점수와 무관).")
    q3_X = q3_ready_tensors(q3_img, tr_idx3, q3_test_img)
q3_x, q3_xt = q3_X
""")

md("""
### 3-2 (7점) 분류 모델
`ShapeClassifier(backbone, num_classes)`: `self.backbone` → `nn.AdaptiveAvgPool2d(1)` → 평탄화 → `nn.Linear(backbone.out_channels, num_classes)` (`self.head`). `forward`는 softmax 전 **logits** `(B, num_classes)`를 반환합니다. **입력 크기가 32×32가 아니어도 동작해야 합니다**(공간 크기 하드코딩 금지).
""")

write("""
class ShapeClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.pool = None  # 작성하세요
        self.head = None  # 작성하세요

    def forward(self, x):
        return None       # 작성하세요
""", """
class ShapeClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(backbone.out_channels, num_classes)

    def forward(self, x):
        return self.head(torch.flatten(self.pool(self.backbone(x)), 1))
""")

run("""
def _check_3_2():
    m = ShapeClassifier(load_pretrained_backbone(), len(CLASSES))
    for size in [32, 48, 24]:
        out = m(torch.zeros(2, 3, size, size))
        assert out.shape == (2, len(CLASSES)), f"{size}x{size} 입력 출력 {tuple(out.shape)}"
    print("32·48·24 크기 입력 모두 (2, 4) 출력")
attempt("3-2", _check_3_2)
""")

md("""
### 3-3 (5점) 학습할 파라미터 선택
`set_trainable(model, train_blocks=())`: 백본의 모든 파라미터를 고정(`requires_grad=False`)하되, 이름이 `train_blocks`에 있는 백본 블록(예: `"block4"`)은 학습을 허용합니다. `head`는 항상 학습합니다. **학습 가능한 파라미터 리스트**를 반환합니다.
""")

write("""
def set_trainable(model, train_blocks=()):
    # 작성하세요
    return None
""", """
def set_trainable(model, train_blocks=()):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for name in train_blocks:
        for p in getattr(model.backbone, name).parameters():
            p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]
""")

run("""
def _check_3_3():
    m = ShapeClassifier(load_pretrained_backbone(), len(CLASSES))
    n_head = sum(p.numel() for p in set_trainable(m))
    n_last = sum(p.numel() for p in set_trainable(m, ["block4"]))
    print("head만:", n_head, "| head+block4:", n_last)
    assert n_head == 64 * 4 + 4 and n_last == n_head + 64 * 64 * 9 + 64
attempt("3-3", _check_3_3)
""")

md("""
### 3-4 (8점) 학습 1 epoch와 평가
- `train_one_epoch(model, loader, optimizer, loss_fn)`: 학습 모드에서 배치마다 기울기 초기화 → 순전파 → 손실 → 역전파 → 갱신. **샘플 수 가중 평균 손실**을 반환.
- `evaluate(model, loader)`: 평가 모드·기울기 계산 없이 정확도(0~1) 반환.

아래 실행 셀은 ① head만 8 epoch → ② `block4`까지 풀어 8 epoch 순으로 학습합니다(약 10초).
""")

write("""
def train_one_epoch(model, loader, optimizer, loss_fn):
    total, count = 0.0, 0
    # 작성하세요
    return total / count

def evaluate(model, loader):
    correct, count = 0, 0
    # 작성하세요
    return correct / count
""", """
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total, count = 0.0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(yb)
        count += len(yb)
    return total / count

def evaluate(model, loader):
    model.eval()
    correct, count = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
            count += len(yb)
    return correct / count
""")

run("""
def _run_3_4():
    seed_everything(42)
    model = ShapeClassifier(load_pretrained_backbone(), len(CLASSES))
    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(TensorDataset(q3_x[tr_idx3], q3_y[tr_idx3]), batch_size=64, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(q3_x[va_idx3], q3_y[va_idx3]), batch_size=256)
    loss_fn = nn.CrossEntropyLoss()
    for stage, blocks, lr in [("① head만", (), 5e-3), ("② +block4", ("block4",), 1e-3)]:
        optimizer = torch.optim.Adam(set_trainable(model, blocks), lr=lr)
        for ep in range(8):
            loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        print(f"{stage}: 마지막 train loss {loss:.3f} | validation acc {evaluate(model, val_loader):.3f}")
    return model
q3_model = attempt("3-4", _run_3_4)
""")

md("""
### 3-5 (4점) 배치 추론과 제출
`predict_labels(model, x, batch_size=128)`: 평가 모드·기울기 없이 `batch_size`씩 나눠 예측한 클래스 번호를 `np.ndarray`(길이 N)로 반환합니다. **N이 batch_size로 나누어떨어지지 않아도** 모든 샘플을 예측해야 합니다.
""")

write("""
def predict_labels(model, x, batch_size=128):
    preds = []
    # 작성하세요
    return None
""", """
def predict_labels(model, x, batch_size=128):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            preds.append(model(x[start:start + batch_size]).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)
""")

run("""
def _run_3_5():
    for n in [1, 130]:
        assert len(predict_labels(q3_model, q3_xt[:n], batch_size=64)) == n, f"N={n}"
    pred = predict_labels(q3_model, q3_xt)
    sub = q3_sample[["id"]].copy()
    sub["label"] = pd.Series(np.array(CLASSES)[pred], index=q3_test_ids).reindex(sub["id"]).to_numpy()
    print("예측 분포:", sub["label"].value_counts().to_dict())
    return save_submission(sub, "q3_submission.csv", q3_sample)
_ = attempt("3-5", _run_3_5)
""")

md("""
---
## 제출 전 점검
- `제출물/`의 세 CSV: 열 이름·행 수·id 순서·결측값 (`save_submission`이 확인)
- 서술 2개(`q1_reason`, `q2_choice_reason`)를 채웠는지
- 커널 재시작 후 처음부터 끝까지 실행되는지, 출력이 저장되었는지
- 풀이를 마친 뒤 `정답_채점용/score_local.py`로 지표를 확인하고 `Answer.ipynb`와 비교하세요.
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
        nb.cells[0].source += "\n\n> **이 파일은 정답·해설용입니다.** `[정답 예시]` 셀은 하나의 풀이이며, 같은 동작을 하는 다른 코드도 정답입니다."
    return nb


if __name__ == "__main__":
    nbf.write(build(False), ROOT / "Lv3_모의_v1_문제.ipynb")
    nbf.write(build(True), ROOT / "정답_채점용" / "Lv3_모의_v1_정답.ipynb")
    print("cells:", len(CELLS))
