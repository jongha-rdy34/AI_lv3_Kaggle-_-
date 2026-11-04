# [claude] AI 챌린지 반영 Lv3 모의문제 — 이미지 객관식 VQA (버전 1·2·3)

SSAFY AI 챌린지에서 제공된 `(260908)_baseline_colab.ipynb`의 흐름(사전학습 VLM 로드 → 양자화·LoRA → 프롬프트 → Dataset·Collator → 학습 → 추론·파싱 → 제출 CSV)을 그대로 따르면서, Level 3 상세 공지의 항목(데이터 점검·증강·학습 범위·손실·성능 개선·구조화 출력 파싱)을 빈칸으로 넣은 **창작 연습문제**입니다. 실제 시험 문제가 아닙니다.

> 교육 프로님에게서 "AI 챌린지와 비슷한 방식(baseline 코드의 핵심 빈칸 채우기 → 성능을 개선한 CSV로 Kaggle 점수)"이라는 이야기를 들었다는 사용자의 정보를 반영한 **예상**입니다. 실제 형식은 11/6 공개되는 Kaggle 페이지와 당일 안내를 따르세요.

## 1. 어떤 것부터 풀까

| 순서 | 파일 | 하는 일 |
|---|---|---|
| 1 | `로컬_CPU판/버전1_핵심빈칸_문제.ipynb` | `___` 자리(대부분 한 줄)만 채우기 |
| 2 | `로컬_CPU판/버전2_긴빈칸_문제.ipynb` | 함수 안을 몇 줄로 채우기(주석에 단계가 있음) |
| 3 | `로컬_CPU판/버전3_직접구현_문제.ipynb` | 함수 이름·설명만 보고 직접 구현 |
| 4 | `Colab_실제판/버전?_…_문제.ipynb` | 같은 문제를 실제 Qwen2.5-VL로 (Colab GPU) |

세 버전은 **같은 과제·같은 데이터**이고 빈칸 크기만 다릅니다. 버전 1을 풀고 정답과 비교한 뒤 버전 2, 3으로 넘어가면 같은 코드를 점점 더 많이 직접 쓰게 됩니다. 정답(줄마다 주석)은 `정답_채점용/로컬_CPU판_정답.ipynb`, `정답_채점용/Colab_실제판_정답.ipynb`입니다.

## 2. 문제 구성 (baseline 대비)

| 단계 | baseline | 이 문제에서 추가·변경 | 공지 항목 |
|---|---|---|---|
| 1 데이터 점검 | 없음 | 정답 표기 정리, 사진 없음·깨진 사진 찾기, 정답 글자 분포, 육안 검사, 검증 100개 고정 | 품질·분포 점검, 육안 검사 |
| 2 프롬프트 | `build_mc_prompt` | 같음 | 프롬프트 설계 |
| 3 증강 | 없음 | 보기 순서 섞기(`shuffle_options`) + 좌우 반전을 쓰면 안 되는 질문 서술 | 변환/증강, 학습 데이터 재구성 |
| 4 Dataset·Collator | `labels = input_ids.clone()` | **정답 토큰만 손실**(`build_labels`, label mask) | 손실 함수의 이해 |
| 5 모델·LoRA | `LoraConfig` | `cfg`로 r·alpha·target_modules 지정 | 전이학습, 학습 범위 제어 |
| 6 학습 | 학습 셀 | 함수로 묶음, 마지막 자투리 배치도 갱신 | 학습, 하이퍼파라미터 |
| 7 추론·파싱 | 전체 출력 디코드, 마지막 줄만 확인 | **새 토큰만 디코드**, `(b)`·`정답: c`·`{"answer": "d"}`도 읽는 파서 | 구조화 출력 파싱 |
| 8 성능 개선 | 없음 | `IMPROVED_CFG` 정하기 → `submission_improved.csv` | 성능 분석, 개선안 도출 |
| 9 채점 | Kaggle | `정답_채점용/score_local.py`(정답률, 질문 유형별) | — |

## 3. 두 가지 판의 차이 (실제 LLM과 연습용 모델 구분)

| | 로컬 CPU판 | Colab 실제판 |
|---|---|---|
| 모델 | `mini_vlm.MiniVLM`: Claude가 합성 데이터로 사전학습한 **연습용 작은 모델**(약 70만 파라미터). 실제 Qwen2.5-VL이 아닙니다 | `Qwen/Qwen2.5-VL-3B-Instruct` (baseline과 같음) |
| 라이브러리 | `mini_vlm.py`가 `LoraConfig`, `get_peft_model`, `AutoProcessor`처럼 쓰는 함수를 흉내 냄 | transformers, peft, bitsandbytes |
| 양자화·혼합 정밀도 | 없음(GPU 전용) | 4bit nf4 + bf16 autocast (baseline과 같음) |
| 실행 검증 | **작성자가 처음부터 끝까지 실행해 확인** | **작성자가 실행하지 못함**(작성 PC에 GPU 없음). 로컬판과 같은 코드 + baseline의 모델 코드 |
| 시간 | 전체 약 5분 | 다운로드 10~20분 + 학습·추론 수십 분 |

연습용 모델은 CPU에서 짧게 학습되도록 구조를 단순화했습니다(질문·보기를 사진 토큰에 cross-attention으로 비교해 보기마다 점수를 냄). 그래서 **결과 수치로 실제 Qwen의 성능을 짐작할 수 없습니다.** 코드의 흐름과 "어디를 바꾸면 무엇이 달라지는지"를 연습하는 용도입니다.

## 4. 데이터

`data/`(로컬판이 사용)와 `Colab_실제판/vqa_mock_data.zip`(같은 내용)에는 합성 도형 그림 객관식 문제가 들어 있습니다. 원래 챌린지 데이터가 아닙니다.
- 열: `id, path, question, a, b, c, d, answer` (baseline과 같음). 64×64 PNG
- train 1,000개, test 400개. 질문 유형: 색, 개수, 색으로 찾는 모양, 가장 왼쪽·오른쪽 도형
- train에 일부러 넣은 문제: 사진 없음 8, 검은 화면 8, 정답 표기 흔들림 약 6%, **정답 글자 a 쏠림(약 49%)**. test는 정답 글자가 고르게 퍼져 있습니다

## 5. 설치와 실행

- 로컬: `로컬_CPU판/requirements.txt` (`pip install -r requirements.txt`). 이 PC에서 Python 3.12.2, torch 2.5.1+cpu, numpy 2.5.2, pandas 3.0.5, pillow 12.3.0, matplotlib 3.11.1로 확인했습니다.
- Colab: 노트북 첫 셀이 baseline처럼 설치합니다(`Colab_실제판/requirements_colab.txt` 참고). `vqa_mock_data.zip`을 구글 드라이브 `MyDrive/`에 올리세요.
- 채점(이 폴더에서):
  ```bash
  python 정답_채점용/score_local.py 로컬_CPU판/제출물/submission_improved.csv
  ```
