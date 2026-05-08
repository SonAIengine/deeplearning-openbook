# 05. Recurrent Neural Networks (RNN) — 오픈북 정리

> 핵심 두 질문 (★교수님 단골):
> 1. **"왜 RNN이 arbitrary length sequence를 처리 가능한가?"**
> 2. **"왜 RNN에서 vanishing gradient가 더 심한가?"**

---

## 0. 동기 — "왜 RNN이 필요한가?"

FNN/CNN의 한계:
- 입력 크기 **고정**.
- 입력 간 **순서 정보** 모름.
- 같은 패턴이라도 위치가 다르면 따로 학습 (CNN은 공간엔 sharing이지만 시간엔 안 함).

시퀀스 데이터(텍스트, 음성, 시계열)는:
- 길이가 가변.
- **이전 정보가 다음에 영향** (메모리 필요).

→ "**시간을 따라 펼친 가중치 공유 망**" = RNN.

---

## 1. RNN 구조

### 1.1 기본 식
$$h_t = \sigma(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

- $h_t$: 현재 hidden state (과거의 요약).
- 같은 $W_{xh}, W_{hh}, W_{hy}$를 **모든 시점에서 공유** (★).

### 1.2 펼치기 (unrolling)
시간 축으로 펼치면 깊이가 T인 FNN과 동등 — 단, **모든 층이 같은 가중치를 공유**.

```
x_1 → [RNN] → h_1 → [RNN] → h_2 → ... → h_T
       ↑              ↑              ↑
      W              W              W      (공유)
```

---

## 2. "왜 arbitrary length를 처리?" (★★ 시험 단골)

답:
1. **Recurrent 구조** = 같은 셀을 시간에 따라 **반복 적용**.
2. **가중치가 시간에 무관하게 공유** → 시퀀스 길이에 따라 파라미터 수 변하지 않음.
3. hidden state $h_t$가 **과거의 모든 정보를 압축한 요약**.
→ 길이 5짜리든 5000짜리든 같은 망으로 처리 가능.

생각해보라:
- FNN은 입력 크기에 따라 첫 층 가중치 수가 결정 → 길이 가변 처리 불가.
- CNN은 conv를 길이 따라 슬라이딩하면 가변 처리는 되지만 **위치별 의존성 모델링 약함** (long-range는 깊어야).
- RNN은 가중치를 시간에 sharing → 본질적 가변.

---

## 3. RNN의 task 형태 (★)

| 형태 | 입력 | 출력 | 예시 |
|---|---|---|---|
| **One-to-one** | 1 | 1 | 일반 분류 (사실 RNN 아님) |
| **One-to-many** | 1 | 시퀀스 | 이미지 캡셔닝 |
| **Many-to-one** | 시퀀스 | 1 | 감성 분류 |
| **Many-to-many (정렬)** | T → T | 같은 시점 출력 | 품사 태깅, 비디오 프레임 분류 |
| **Many-to-many (seq2seq)** | T_in → T_out | 길이 다름 | 번역, 요약 |

---

## 4. BPTT (Backpropagation Through Time)

펼친 망에서 일반 backprop을 적용. 단, 가중치가 공유되므로 **gradient를 모든 시점에서 합산**.

$$\frac{\partial L}{\partial W_{hh}} = \sum_t \frac{\partial L_t}{\partial W_{hh}}$$

### 4.1 Truncated BPTT
긴 시퀀스에서 메모리 폭발 → 일정 step만 backprop.
- 장점: 계산·메모리.
- 단점: long-term gradient 소실 (절단된 만큼).

---

## 5. Vanishing/Exploding Gradient — "왜 RNN에서 더 심한가?" (★★)

### 5.1 원인
hidden state의 미분:
$$\frac{\partial h_t}{\partial h_{t-k}} = \prod_{i=t-k+1}^{t} W_{hh}^T \cdot \text{diag}(\sigma'(\cdot))$$

→ **같은 가중치 행렬을 k번 곱한다**.
- $W_{hh}$의 최대 고유값 < 1 → **vanishing**.
- > 1 → **exploding**.

### 5.2 왜 RNN이 깊은 FNN보다 더 심한가?
- FNN은 층마다 **다른** 가중치 행렬. 모든 층이 동시에 < 1 또는 > 1일 확률은 낮음.
- RNN은 **같은** 행렬의 거듭제곱 → 한쪽으로 폭주 또는 소멸이 거의 보장.

### 5.3 처방
| 문제 | 처방 |
|---|---|
| Exploding | **Gradient clipping** (norm > τ면 자름) — 흔하게 쓰임 |
| Vanishing | **LSTM/GRU의 gating 구조** — 다음 챕터의 주제 |
| | Identity init, orthogonal init |
| | Skip connection (residual RNN) |
| | LayerNorm in RNN |

생각해보라: 왜 ReLU 쓰면 RNN에서 더 위험? → 양수에서 미분이 1이라 vanishing은 줄지만 exploding 위험↑. 더해서 dead ReLU. → tanh가 RNN에서 표준인 이유.

---

## 6. Bidirectional RNN

$$h_t^\rightarrow = \text{RNN}_\rightarrow(x_t, h_{t-1}^\rightarrow)$$
$$h_t^\leftarrow = \text{RNN}_\leftarrow(x_t, h_{t+1}^\leftarrow)$$
$$h_t = [h_t^\rightarrow ; h_t^\leftarrow]$$

**왜?** → 어떤 task는 미래 정보도 필요. 예: 품사 태깅에서 "bank"의 의미는 뒤 단어가 결정.

**언제 안 쓰나?** → 실시간 / 자기회귀 생성 (미래 모름).

---

## 7. Deep / Stacked RNN

여러 RNN 층을 쌓음. 위층의 입력은 아래층의 hidden.
- 표현력↑.
- 하지만 너무 깊으면 학습 어려움. 보통 2~4층.

---

## 8. Sequence-to-Sequence (Encoder-Decoder)

번역, 요약, QA 등에서 입력 길이 ≠ 출력 길이.

```
Encoder: x_1 ... x_T → context vector c
Decoder: c → y_1 ... y_T'
```

### 8.1 한계 — Attention의 동기
- Encoder가 모든 입력을 **하나의 context vector**로 압축 → 정보 병목.
- 입력이 길면 앞부분 정보가 사라짐.

### 8.2 Attention의 등장
"매 출력 시점마다 입력의 **관련 부분에 가중치**를 두자."
$$c_t = \sum_i \alpha_{t,i} h_i^{enc}$$

→ Bahdanau/Luong attention → Transformer로 발전.

생각해보라: Attention은 RNN의 **vanishing 문제를 우회**한다. context를 멀리서 직접 가져오기 때문. 이게 왜 LSTM도 부족한 long-range에서 큰 도움이 되었는가의 답.

---

## 9. RNN vs CNN vs Transformer (시퀀스 처리)

| | 장점 | 단점 |
|---|---|---|
| **RNN/LSTM** | 가변 길이 자연스러움, 메모리 작음 | **순차적 → 병렬화 어려움**, long-range 약함 |
| **1D CNN** | 병렬 가능, 국소 패턴 강함 | RF 한계 (깊이 필요) |
| **Transformer** | 완전 병렬, long-range 직접, 강력 | O(n²) 메모리·계산, 위치 정보 별도 필요 |

**왜 NLP가 Transformer로 갔나?** → GPU 병렬 활용 + long-range + 큰 데이터에서 scaling. RNN은 시퀀스 길이만큼 step이 직렬.

---

## 10. Language Model (RNN의 대표 응용)

### 10.1 정의
$P(w_t | w_{<t})$를 추정.

### 10.2 학습
- 입력: $w_1, ..., w_{T-1}$
- 정답: $w_2, ..., w_T$
- Loss: per-step CE.

### 10.3 추론 (생성)
1. $w_1$에서 hidden 계산.
2. softmax로 $w_2$ 샘플링 / argmax.
3. $w_2$를 입력으로 다시 forward.
4. 반복.

생각해보라: greedy decode vs beam search vs sampling — 왜 다 다른 결과? → 분포의 mode를 어떻게 탐색하느냐의 차이. greedy는 근시안적, beam은 trade-off, sampling은 다양성.

---

## 11. Teacher Forcing

학습 시 decoder 입력으로 **정답 이전 토큰**을 사용 (모델 출력 X).
- **장점**: 학습 안정·빠름.
- **단점**: train-test mismatch (**exposure bias**) — test 때는 자기 출력을 입력으로 쓰므로 오류 누적.
- **처방**: scheduled sampling, RL fine-tuning.

---

## 12. RNN 학습 실패 패턴

| 증상 | 원인 | 처방 |
|---|---|---|
| Loss NaN, gradient 폭발 | Exploding | gradient clipping |
| 학습 멈춤, 긴 의존성 못 잡음 | Vanishing | LSTM/GRU, attention |
| 너무 느림 | 시퀀스 길이 | mini-batch + truncated BPTT |
| Overfit | 작은 데이터 | dropout (수직 방향만), weight decay |

**왜 RNN의 dropout은 수직(층 방향)만?** → 시간 방향 dropout은 hidden state 흐름을 끊어 학습 어렵게 함. **변종**: Variational dropout (시간 방향 같은 mask 사용).

---

## 13. 응용 영역

- 언어 모델, 번역, 요약
- 음성 인식 (RNN-T, listen-attend-spell)
- 시계열 예측 (주식, 수요)
- 비디오 분류 (CNN+RNN)
- 음악 생성

---

## 14. 한 줄 요약

- RNN의 본질 = **시간 축으로 가중치 공유 → 가변 길이 처리**.
- BPTT = 펼친 후 일반 backprop. 가중치 공유라 합산.
- **Vanishing이 더 심한 이유**: 같은 행렬의 거듭제곱.
- 처방: clipping(exploding), LSTM/GRU(vanishing), attention(long-range).
- Encoder-Decoder의 병목 → Attention → Transformer로 발전.
- RNN의 진짜 약점은 **순차성** → 병렬 안 됨, 그래서 NLP가 Transformer로.
