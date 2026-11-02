# 자산 출처

- `q3_backbone.pt`: `practice_support.make_backbone()` 구조(Conv 4블록, 64채널, 파라미터 약 6만 개)의 state_dict. Claude가 합성 도형 6종(ring, ellipse, rect, diamond, hollow_square, xmark) 5,000장으로 CPU에서 20 epoch 학습했습니다. Q3 목표 클래스(circle, square, triangle, cross)는 사전학습에 쓰지 않았습니다. 생성 코드: `정답_채점용/_build/make_data.py`의 `make_q3()`.
- `data/q2/mock_embeddings.npz`: 같은 생성기의 `MockEncoder`로 만든 **가짜** 임베딩입니다. 실제 LLM·문장 임베딩 모델이 아닙니다.
- 모든 CSV와 이미지는 합성 연습 자료이며 실제 개인·주문·상품 정보가 아닙니다.
