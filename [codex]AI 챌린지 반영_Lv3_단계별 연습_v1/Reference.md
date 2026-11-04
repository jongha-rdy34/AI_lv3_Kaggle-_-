# 입문 참고표

| 코드 | 뜻 |
|---|---|
| `None` / `pass` / `__V...__` + TODO | 문제의 빈칸. 힌트를 보고 채웁니다. |
| `df['label'].isin(CLASSES)` | 허용 클래스에 속하는지 확인한 참/거짓 배열 |
| `train_test_split(..., stratify=labels)` | 클래스 비율을 고려해 학습·검증 분리 |
| `Image.open(ROOT/path).convert('RGB')` | 파일 경로에서 RGB 사진 읽기 |
| `.permute(2,0,1).float()/255.0` | HWC 정수 사진을 CHW 0~1 실수로 변환 |
| `CrossEntropyLoss` | logits와 long 클래스 정답의 분류 손실 |
| `MSELoss` | 같은 크기의 실수 예측·정답에 대한 회귀 손실 |
| `loss.backward()` | 가중치의 기울기를 계산 |
| `optimizer.step()` | 계산한 기울기로 가중치를 갱신 |
| `model.eval()` + `torch.no_grad()` | 검증·추론 모드 및 기울기 기록 생략 |
| `copy.deepcopy(model.state_dict())` | 그 시점의 가중치를 별도 복사 |
| `model.load_state_dict(best_state)` | 가장 좋았던 가중치로 복원 |
| `torch.save(state, path)` | 가중치 파일 저장 |
| `to_csv(path,index=False)` | 불필요한 인덱스 열 없이 CSV 저장 |

RMSE는 작을수록, macro-F1/QWK는 클수록 좋습니다. 같은 분할·지표로 비교합니다. `model.train()`은 학습 모드이며 optimizer.step()을 대신하지 않습니다.

JSON의 null/true/false는 Python 객체에서는 None/True/False입니다. `prompt_template`은 자연어 문자열이므로 JSON 표기는 소문자로 씁니다. 미확정(null)과 명시적 부정(false)을 구별합니다.

V1/V2 빈칸별 완성 코드와 설명은 정답_채점용의 해설 문서를 참고하세요.
