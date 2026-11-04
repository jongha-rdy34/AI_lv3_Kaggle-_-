# 자산 출처

- `pretrained_tinyresnet.pt`: `practice_support.make_tinyresnet(6)`의 state_dict. Claude가 합성 무늬 6종(작은 점, 줄무늬, 체커, 고리, 큰 얼룩, 무늬 없는 타원) 6,000장(학습 5,000, 확인 1,000)을 48×48로 만들어 CPU에서 8 epoch 학습했습니다. 마지막 확인 정확도는 0.944입니다(중간 epoch 최고 0.986). 정규화는 mean 0.5, std 0.25입니다(`PRETRAIN_MEAN/STD`). 잎·등급 데이터는 사전학습에 쓰지 않았습니다. 생성 코드: `정답_채점용/_build/make_data.py`의 `pretrain()`.
- `data/notes/mock_llm_outputs.json`: 정답 레코드에 규칙으로 형식 오류를 섞은 **모의 응답**입니다. 실제 LLM 출력이 아닙니다.
- 모든 사진·메모는 합성 자료입니다. 메모의 인물·진료 내용은 가상이며 실제 의료 정보가 아닙니다.
