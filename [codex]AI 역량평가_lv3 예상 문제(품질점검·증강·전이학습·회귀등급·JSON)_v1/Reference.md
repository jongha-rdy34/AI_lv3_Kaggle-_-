# 공지 반영 모의평가 — 공통 참고자료

실제 시험 참고자료와 동일하다는 뜻은 아닙니다.

## 1. 품질과 분할
`df.duplicated('id')`, `df['label'].value_counts()`, `np.isfinite(x).all()`, `image.max()==0`, `train_test_split(...,stratify=labels,random_state=42)`. 이미지를 육안으로 볼 때 `plt.imshow(image)`를 사용합니다. 중복·누락·범위 오류를 먼저 제거하고 train/valid ID가 겹치지 않게 합니다.

## 2. Dataset과 샘플링
`torch.from_numpy(image.copy()).permute(2,0,1).float()/255.0`; train에서만 `np.flip(image,axis=1).copy()`를 적용합니다. 레이블을 바꿀 수 있는 증강은 사용하지 않습니다. `weights = train.label.map(lambda label: 1 / counts[label])`; `WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)`는 validation/test에 사용하지 않습니다. 클래스 역빈도 샘플링과 손실 가중치를 함께 쓰면 중복 보정이 될 수 있으니 근거를 가지고 비교합니다.

## 3. 전이학습
`backbone = load_backbone()`; `for p in backbone.parameters(): p.requires_grad=False`; `backbone.block4.parameters()`를 다시 True로 설정할 수 있습니다. 새로운 head는 `nn.AdaptiveAvgPool2d(1) → nn.Flatten() → nn.Linear(64,4)`; CrossEntropyLoss는 softmax 전 logits와 long 레이블을 사용합니다. 고정/해제 후 `Adam([p for p in model.parameters() if p.requires_grad],lr=...)`를 생성합니다. optimizer를 만든 이후 새로운 파라미터를 해제하면 optimizer에 반영해야 합니다.

## 4. 학습과 검증
`optimizer.zero_grad() → forward → loss → backward() → optimizer.step()`. 검증은 `model.eval()` 및 `with torch.no_grad():`. `f1_score(...,average='macro')`, `sqrt(mean_squared_error(y,pred))`, `cohen_kappa_score(y_grade,p_grade,weights='quadratic',labels=[0,1,2,3,4])`. F1/QWK는 클수록, RMSE는 작을수록 좋습니다. 최고 점수가 바뀔 때 `copy.deepcopy(model.state_dict())`로 복사하고 끝에 `model.load_state_dict(best_state)`합니다.

Q4 손실은 score/100에 대한 MSE, 평가·제출은 0~100 단위로 환산한 RMSE입니다. 고정 경계는 `np.digitize(scores, thresholds)`로 등급 0~4를 계산합니다. 임계값은 제공된 학습 split에서만 만들었으며 validation/test 점수를 만들 때 재계산하지 않습니다.

## 5. JSON / prompt
`json.loads(raw_text)`는 JSON 문자열을 Python 객체로 읽습니다. JSON null/true/false는 Python의 None/True/False에 대응합니다. `json.dumps(obj,ensure_ascii=False)`로 한글 JSON을 출력합니다. 구조 검사 통과와 정보가 사실에 근거한다는 것은 다른 문제입니다.

예시 데이터는 합성된 일반 제품 문의입니다. 모델명·증상·배터리 시간(분)·보증/수리 여부를 추출하고, 모르면 null, 명시적 부정은 false로 구별합니다. issues는 battery/fan/storage/display 중 목록이며 정해진 종류가 없으면 빈 배열입니다.
