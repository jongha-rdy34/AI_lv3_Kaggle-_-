# 자산 출처

- `pretrained_tinyresnet.pt`: v1 세트(`[claude]AI 역량평가_lv3 예상 문제(품질점검·증강·전이학습·등급예측·JSON추출)_v1`)와 같은 파일. Claude가 합성 무늬 6종 6,000장(48×48)으로 CPU에서 학습한 `TinyResNet(6)`의 state_dict입니다. 생성 코드는 v1의 `정답_채점용/_build/make_data.py`의 `pretrain()`. 모델 카드: `모델카드.md`.
- `data/leaf`, `data/grade`, `data/notes`: v1 세트의 합성 데이터를 그대로 복사했습니다(잎 사진·등급 사진·가상 진료 메모). 메모의 인물·진료 내용은 가상이며 실제 의료 정보가 아닙니다.
- `data/notes/mock_llm_outputs.json`: 정답 레코드에 규칙으로 형식 오류를 섞은 **모의 응답**입니다. 실제 LLM 출력이 아닙니다.
