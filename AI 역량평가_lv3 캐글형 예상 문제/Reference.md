# 연습용 Reference

실제 시험의 라이브러리와 지표는 별도 공지 전까지 미확정입니다.

- split: train_test_split(...,test_size=.2,random_state=42,stratify=y). 회귀는 stratify 생략. test는 validation이 아닙니다.
- 회귀: 결측 중앙값 SimpleImputer, 범주 OneHotEncoder(handle_unknown='ignore'), ColumnTransformer와 Pipeline으로 누출 방지. sqrt(mean_squared_error(y,p)).
- 다중분류: `nn.CrossEntropyLoss()` 입력은 logits `(B,10)`, 정답은 long `(B,)`. 평가·제출 확률은 `torch.softmax(logits,dim=1)`; train/eval과 no_grad 구분. log_loss(y,prob,labels=list(range(10))).
- 텍스트 기본선: TfidfVectorizer(analyzer='char',ngram_range=(2,4)) + LogisticRegression. TF-IDF는 학습 분할에만 fit.
- 제출: sample[['id']].merge(predictions,on='id',how='left',validate='one_to_one'), 열 이름·순서·행 수·중복·결측·예측 범위 확인. CSV index=False.

## 실제 로컬 LLM으로 Q3을 풀어보려면

이 모의평가의 기본 배포물에는 실제 LLM이나 transformers가 없습니다. 기존 Lv2 공식 환경 ZIP에서 assets/language_model 전체와 weights를 별도 준비하고 호환 transformers가 사전 설치된 경우, AutoTokenizer / AutoModelForCausalLM.from_pretrained(경로, local_files_only=True)로 CPU에서 불러올 수 있습니다. 네트워크 연결이 필요하지 않도록 사전에 준비하세요. 제공된 SmolLM2-135M은 base 모델이라 한국어 지시 수행이 약할 수 있습니다. 토큰 길이와 실행 시간을 제한하고, 예측 라벨이 셋 중 하나인지 반드시 검증하세요.

프롬프트 예: `문의의 주제를 fan, battery, upgrade 중 하나로만 답하세요. 문의: {text} 답:`. 생성된 새 토큰만 decode하고 문자열을 검증하세요. 파싱 불가능한 출력은 고정 기본값을 쓰거나 별도 재시도하는 등 방침을 명시하고 validation에서 평가하세요. 이 예시는 실제 모델의 성능을 보장하지 않으며 로컬 실모델 실행은 이번 검증 범위가 아닙니다.

다음은 선택 실습의 예시 코드입니다. `model_dir`은 실제 압축 해제 위치로 바꾸세요. 이 코드는 실제 모델이 없는 현재 환경에서 실행 검증하지 않았습니다. 검증셋에서 먼저 분류 정확도와 라벨 추출 실패율을 측정한 뒤 사용하세요.

```python
import re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = r'압축해제한_폴더/assets/language_model'
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True).to('cpu').eval()
labels = {'fan', 'battery', 'upgrade'}

def predict_with_llm(text):
    prompt = f'Choose exactly one label: fan, battery, upgrade. Question: {text} Label:'
    encoded = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.no_grad():
        output = model.generate(**encoded, do_sample=False, max_new_tokens=8, pad_token_id=pad)
    generated = tokenizer.decode(output[0, encoded['input_ids'].shape[1]:], skip_special_tokens=True)
    matches = re.findall(r'\b(fan|battery|upgrade)\b', generated.lower())
    # 연습용 대체 규칙. validation에서 실패율을 측정하고 개선하세요.
    return matches[0] if matches else 'fan'
```

## 로컬 결과 확인

세 제출 CSV를 만든 뒤 모의 응시를 마치고 `정답_채점용/로컬채점.py`를 실행하세요. 정답이 같은 컴퓨터에 있어 경쟁 환경의 보안과 독립 평가를 재현하지는 못합니다. 결과는 모의 점수이며 실제 시험과 무관합니다.
