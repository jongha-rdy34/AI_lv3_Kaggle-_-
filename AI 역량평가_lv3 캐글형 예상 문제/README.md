# 문제와 운영 안내

이 자료는 **Kaggle형 예측 경쟁을 가정한 창작 모의문제**입니다. 실제 시험 제출물이 CSV인지, 각 분야가 예측형인지, 평가 지표·배점·리더보드·제출 횟수·허용 도구는 미공개입니다. 사용자가 전달한 것은 Kaggle 기반이라는 사실과 네트워크 차단 환경입니다. 두 조건의 실제 연결 방식도 미확정입니다.

4시간 연습: Q1 데이터·회귀 70분, Q2 영상 모델 80분, Q3 텍스트·LLM 활용 70분, 검토 20분. 세 문제는 독립적입니다. 각 문제에서 train을 다시 train/validation으로 나눠 validation 성능을 확인하고, test는 예측만 합니다. test 정답은 학습 또는 모델 선택에 쓰지 마세요.

|문항|연습 목표|지표|제출 파일과 열|
|---|---|---|---|
|1|상품의 연속형 전환율 예측|RMSE 낮을수록 좋음|제출물/q1_submission.csv: id,prediction|
|2|8×8 숫자 이미지의 10개 범주 분류|다중분류 log loss 낮을수록 좋음|제출물/q2_submission.csv: id,p0,...,p9|
|3|고객 질문 유형 3개 분류|accuracy 높을수록 좋음|제출물/q3_submission.csv: id,label|

각 sample_submission은 열과 행 순서의 예시입니다. Q1 예측은 0~1 실수. Q2 확률은 각 행 합이 1이고 0~1; 열 순서는 p0~p9. Q3 label은 fan/battery/upgrade 중 하나입니다. 세 CSV의 id는 각 sample_submission과 정확히 일치해야 합니다. test 정답은 정답_채점용에 둔 로컬 검증용 파일이며 실제 비공개 리더보드가 아닙니다. 모의 응시 중에는 해당 폴더를 열지 마세요.

Q3은 LLM 사용 가능성을 염두에 둔 텍스트 분류 문제입니다. 제공 자료에는 LLM 가중치가 없으므로 답안은 TF-IDF + 선형 분류의 CPU 베이스라인을 제공합니다. 이는 LLM이 아닙니다. 실제 사전학습 LLM을 이용한 선택 연습에 필요한 조건과 프롬프트 예시는 Reference.md에 있습니다.

Q2 이미지는 scikit-learn의 내장 digits(8×8)를 오프라인으로 사용합니다. Q1과 Q3 데이터는 생성된 연습 자료입니다. 출처: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

Q3 합성 문장은 소수의 반복되는 문장 틀을 사용하므로 TF-IDF 기준선이 쉽게 높은 점수를 낼 수 있습니다. 이 점수로 실제 Level 3나 LLM 성능을 예측할 수 없습니다.
