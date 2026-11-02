# Lv3 연습 참고자료

이 문서는 창작 연습용이며 Level 3 공식 참고자료가 아닙니다.

## 데이터
- groupby('product_id', as_index=False)[['views','clicks']].sum()
- merge(...,on='product_id',how='left',validate='one_to_one',sort=False)
- SimpleImputer(strategy='median') 및 StandardScaler의 fit은 train만 사용하고 valid/test에는 transform. LinearRegression.fit/predict 후 np.clip(예측,0,1). RMSE=sqrt(mean((실제-예측)**2)).

## PyTorch
TensorDataset / DataLoader; nn.Linear, nn.ReLU, nn.Sigmoid; zero_grad → model(x) → MSE → backward → step. 검증은 model.eval()과 torch.no_grad(). copy.deepcopy(model.state_dict())로 최선 가중치를 복사하고 model.load_state_dict()로 복원합니다.

## 검색과 LLM
코사인 = (doc_vectors @ query_vector) / (np.linalg.norm(doc_vectors,axis=1)*np.linalg.norm(query_vector)). np.argsort(-scores)로 상위 위치를 구합니다. topic은 평가에만 사용합니다.

모델 입력: tokenizer(prompt,return_tensors='pt',truncation=True,max_length=128); 모든 텐서를 model.device로 이동합니다. eval/no_grad에서 model.generate(**encoded,max_new_tokens=32,do_sample=False,pad_token_id=...)를 호출합니다. 결과에서 input_ids 길이만큼을 제외하고 새 토큰만 decode합니다.

MockTokenizer / MockCausalLM은 인터페이스 점검용 가짜 객체입니다. 실제 언어모델의 답변 품질은 검증하지 않습니다.

### 실제 로컬 LLM 선택 실행
기존 Lv2 환경 ZIP의 assets/language_model 전체(269MB 가중치 포함)를 압축 해제하고 호환 transformers가 설치된 경우만 실행하세요. 경로는 직접 지정합니다. 네트워크를 사용하지 않는 설정입니다. base 모델은 지시를 잘 따르지 않을 수 있습니다.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model_dir = r'압축해제한_경로/assets/language_model'
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True).to('cpu').eval()
print(generate_answer(prompt, tokenizer, model))
```

모델과 transformers가 없어도 기본 검색·인터페이스 연습은 가능합니다.
