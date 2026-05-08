# 05. Recurrent Neural Networks (RNN) — 심층 정리

> **이 문서의 핵심 두 질문 (★ 교수님 단골)**:
> 1. **"왜 RNN이 arbitrary length sequence를 처리 가능한가?"**
> 2. **"왜 RNN에서 vanishing gradient가 더 심한가?"**
>
> 이 두 질문에 한 단락씩 깔끔히 답할 수 있으면 시험·면접의 절반이 끝난다. 그 답을 *수식 + 직관 + 한계*까지 풀어내는 게 이 챕터의 목적.

---

## 0. 큰 그림 — 시퀀스 데이터에는 왜 RNN이 필요한가

### 0.1 시퀀스 데이터의 특성

시퀀스 데이터는 다음과 같은 본질적 성질을 갖는다:

**(1) 가변 길이**: 문장은 5단어일 수도, 500단어일 수도 있다. 음성은 1초일 수도, 10분일 수도 있다. 시계열은 100 step일 수도, 100,000 step일 수도 있다.

**(2) 시간적 순서**: 토큰의 *순서*가 의미를 결정한다. "I love you"와 "You love I"는 다른 문장. 순서를 무시하면 정보 손실.

**(3) 시간적 의존성**: 현재 시점이 *과거 시점에 의존*. "그는 어제 학교에 갔다"의 "갔다"를 이해하려면 "어제"라는 과거 토큰이 필요.

**(4) 장거리 의존성 (long-range dependency)**: "파리에서 태어난 그녀는 ... 30년 후 ... 프랑스어를 능숙하게 구사한다"에서 "프랑스어"의 의미는 50단어 전 "파리"에 의존. 멀리 떨어진 토큰 사이의 관계.

이 네 성질을 모두 잘 다룰 모델 architecture가 필요하다.

### 0.2 FNN과 CNN의 한계

**FNN**:
- 입력 크기 *고정* → 가변 길이 처리 불가.
- 위치 정보 모름 → 순서 무시.
- 같은 패턴이 다른 위치에 있으면 별도 학습.

**CNN (1D)**:
- 가변 길이는 sliding window로 처리 가능.
- 시간 축 weight sharing이 있음 → 위치 무관.
- *국소* receptive field → long-range 약함. 깊이로 보충 가능하지만 한계.

→ 시퀀스에 맞는 architecture: **시간 축에 weight sharing + memory를 가진 구조** = RNN.

### 0.3 RNN의 핵심 아이디어

"같은 cell을 시간에 따라 반복 적용. 이전 시점의 정보를 hidden state로 다음 시점에 전달."

```
x_1 → [cell] → h_1 → [cell] → h_2 → [cell] → h_3 → ...
       ↑ W            ↑ W            ↑ W
       ↑ 같은 가중치를 모든 시점에 sharing
```

이 구조에서:
- **가변 길이**: 시퀀스 길이에 따라 cell을 더 반복하면 됨. 가중치 수 변하지 않음.
- **순서**: 시간 순서대로 처리. 과거가 hidden state에 누적.
- **시간 의존성**: hidden state $h_t$가 과거의 요약.
- **시간 sharing**: 같은 cell. 한 시점 학습이 모든 시점에 적용.

이게 RNN의 모든 것. 단순하지만 powerful한 아이디어.

### 0.4 본 챕터의 흐름

1. RNN의 구조와 forward (§1)
2. 왜 arbitrary length 처리 가능 (§2 — 첫 핵심 질문)
3. RNN이 풀 수 있는 task의 다양한 형태 (§3)
4. BPTT (Backpropagation Through Time) (§4)
5. Vanishing/Exploding gradient — RNN에서 더 심한 이유 (§5 — 두 번째 핵심 질문)
6. RNN의 발전 — bidirectional, deep, seq2seq, attention (§6~9)
7. Language Model과 자기회귀 생성 (§10)
8. Teacher forcing과 exposure bias (§11)
9. RNN vs CNN vs Transformer for sequences (§12)
10. 면접 Q&A (§13)
11. 생각해보라 (§14)

---

## 1. RNN 구조 — 시간 축 sharing의 구체

### 1.1 기본 RNN 식

각 시점 $t$에서:

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

여기서:
- $x_t$: 시점 $t$의 입력. 차원 $d_x$.
- $h_t$: 시점 $t$의 hidden state. 차원 $d_h$ (hyperparameter).
- $y_t$: 시점 $t$의 출력. 차원 $d_y$ (task에 따라).
- $W_{xh}, W_{hh}, W_{hy}$: 가중치 행렬. **모든 시점에서 sharing**.
- $b_h, b_y$: bias. 마찬가지로 sharing.
- $h_0$: 초기 hidden state. 보통 0 vector.

### 1.2 직관 — Markov 체인의 신경망 버전

각 시점에서 hidden state $h_t$는 두 입력의 함수:
- 현재 입력 $x_t$
- 이전 hidden state $h_{t-1}$

$h_{t-1}$이 *과거의 모든 정보의 요약*이라 가정. 이 요약 + 현재 입력으로 새 요약 생성. 이걸 시간 따라 반복.

수학적으로 이건 **Markov 가정의 학습된 버전**. 1차 Markov는 $P(x_t | x_{<t}) = P(x_t | x_{t-1})$ 가정 — 현재가 직전 시점만 의존. RNN은 이걸 일반화 — $h_{t-1}$이 직전 시점의 *압축*이지만 그 안에 더 먼 과거 정보도 들어있음 (학습으로).

이론상 RNN은 *임의 길이의 의존성*을 hidden state에 저장 가능. 실용적으론 vanishing gradient 때문에 어려움 (§5).

### 1.3 펼치기 (Unrolling)

RNN을 시간 축으로 펼치면 깊이 $T$인 깊은 신경망과 동등 — 단 *모든 layer가 같은 가중치를 공유*.

```
x_1 → [W] → h_1 → [W] → h_2 → [W] → ... → [W] → h_T
                                                    → y_T

(여기서 [W]는 같은 가중치 행렬 W_xh, W_hh)
```

이 펼침이 **BPTT의 토대**. 펼친 망에 일반 backprop을 적용 + 가중치 공유로 인한 gradient 합산.

### 1.4 다양한 RNN cell들

위 식이 가장 단순한 "vanilla RNN" 또는 "Elman RNN". 다양한 변종:

**Jordan RNN**: hidden 대신 출력을 다음 시점에 feedback.

**LSTM (Long Short-Term Memory)**: gating 메커니즘 도입 — 다음 챕터의 주제.

**GRU (Gated Recurrent Unit)**: LSTM의 단순화 — 다음 챕터.

본 챕터에서는 vanilla RNN 위주로 다룬다. LSTM/GRU는 별도.

### 1.5 Hidden state의 차원 선택

$d_h$는 hyperparameter. 일반 가이드:
- 작은 task (간단한 분류): 64~128
- 중간 task (감성 분석, 짧은 번역): 256~512
- 큰 task (긴 문서 요약): 512~1024

너무 작으면 정보 압축 한계, 너무 크면 overfit + 학습 느림.

---

## 2. "왜 arbitrary length 처리 가능?" — 첫 핵심 질문

### 2.1 답의 구조

이 질문에 답은 다음 세 부분으로:

**(1) 시간 축 weight sharing** — 같은 가중치를 모든 시점에서 사용.

**(2) Hidden state의 압축 역할** — 과거의 가변 길이 정보를 *고정 크기*의 hidden state로 요약.

**(3) Architecture와 길이의 분리** — 모델 구조는 길이에 독립, 길이는 forward pass에서 결정.

### 2.2 시간 축 sharing — 왜 그것이 핵심인가

만약 시간 축에서도 다른 가중치를 쓴다면? 즉:

$$h_t = \tanh(W^{(t)}_{xh} x_t + W^{(t)}_{hh} h_{t-1} + b^{(t)}_h)$$

이러면 시점 $t$마다 다른 가중치 행렬. 시퀀스 길이가 $T$면 가중치 수가 $T$에 비례. 길이가 가변이면 모델 자체가 가변 — 학습된 모델을 다른 길이 시퀀스에 적용 불가.

시간 축 sharing은 이 문제를 정확히 해결한다. 하나의 cell만 학습하면 *어떤 길이의 시퀀스든 같은 cell을 반복*해 처리. **모델 자체는 길이에 독립**.

### 2.3 Hidden state의 정보 압축 역할

가변 길이 입력 $x_1, ..., x_T$를 받아 어떻게 처리? 답: hidden state가 과거를 *압축*.

각 시점에서 $h_t = f(x_t, h_{t-1})$. 이건 재귀 정의 — $h_T$는 모든 입력 $x_1, ..., x_T$의 함수. 단 차원은 고정 ($d_h$).

가변 길이의 정보를 고정 크기에 압축 → 정보 손실 위험. 단 학습된 cell이 *task-relevant 정보를 우선 보존*하도록.

이 압축이 RNN의 강점이자 약점:
- 강점: 어떤 길이든 처리.
- 약점: 매우 긴 시퀀스에선 정보 병목 (LSTM/Attention의 동기).

### 2.4 Architecture와 길이의 분리

학습 시점에 시퀀스 길이가 가변일 수도 있다. 학습 데이터의 한 batch에:
- 시퀀스 1: 길이 10
- 시퀀스 2: 길이 50
- 시퀀스 3: 길이 100

모델은 같은 cell을 반복해 적용 — 짧으면 10번, 길면 100번. 메모리·연산은 길이에 비례하지만 *모델 가중치*는 변하지 않음.

PyTorch에서 이걸 효율적으로 처리하는 두 가지:
- **Padding**: 짧은 시퀀스를 0으로 채워 batch 내 길이 통일.
- **Packing**: `pack_padded_sequence`로 padding 부분 무시.

### 2.5 결론 — 답안 골격

> "시간 축 weight sharing 덕분입니다. 같은 cell을 시간에 따라 반복 적용하므로 시퀀스 길이가 5이든 5000이든 같은 cell을 그만큼 반복하면 됩니다. Hidden state $h_t$가 과거 정보의 고정 크기 요약이라 가변 길이 정보를 고정 차원에 압축. 모델 가중치는 길이에 독립이고, forward pass의 반복 횟수만 길이에 의존합니다."

이게 표준 답안. 시험·면접에 그대로.

### 2.6 비교 — FNN과 CNN의 가변 길이 처리

**FNN**: 입력 차원 고정. 길이 다른 시퀀스 처리 불가. *Sliding window*로 우회 가능하지만 long-range 정보 잃음.

**CNN (1D)**: Conv는 가변 길이에 자연스러움 — 같은 커널을 시퀀스 전체에 슬라이딩. 단 receptive field 한계 있음 (깊이 필요).

**RNN**: 시간 축 sharing + hidden state. 자연스러움.

**Transformer**: 자유로움. 위치 인코딩으로 처리. 단 메모리가 $O(T^2)$ — 매우 긴 시퀀스에 부담.

각 architecture가 길이를 다르게 다루지만, RNN의 *재귀* 구조가 시퀀스의 본질에 가장 가까움.

---

## 3. RNN의 task 형태 — 다양한 입출력 매핑

### 3.1 다섯 가지 패턴

RNN은 입력·출력의 시간 구조에 따라 다양한 task에 적용:

| 패턴 | 입력 | 출력 | 예시 |
|---|---|---|---|
| **One-to-one** | 1 | 1 | 일반 분류 (사실 RNN 아님) |
| **One-to-many** | 1 | T 시퀀스 | 이미지 캡셔닝 (이미지 → 문장) |
| **Many-to-one** | T 시퀀스 | 1 | 감성 분류, 다음 단어 예측 |
| **Many-to-many (정렬)** | T → T (같은 시점) | 시점별 출력 | 품사 태깅, 비디오 프레임 분류 |
| **Many-to-many (seq2seq)** | T_in → T_out (다른 길이) | 다른 길이 시퀀스 | 번역, 요약, QA |

### 3.2 One-to-Many — 이미지 캡셔닝

이미지 → 문장 생성:

```
Image (CNN으로 feature 추출)
  ↓
h_0 = CNN_features
  ↓
[RNN cell] → h_1 → "The"
  ↓
[RNN cell] → h_2 → "cat"
  ↓
[RNN cell] → h_3 → "sits"
  ...
[RNN cell] → h_T → <END>
```

특징:
- 첫 hidden state를 이미지 feature로.
- 매 시점 출력이 다음 시점 입력 (자기회귀).
- <END> 토큰까지 생성.

### 3.3 Many-to-One — 감성 분석

문장 → 감성 (긍정/부정):

```
"This movie is great"
   ↓     ↓    ↓    ↓
   x_1   x_2  x_3  x_4
    ↓     ↓    ↓    ↓
   h_1 → h_2 → h_3 → h_4
                       ↓
                     FC → softmax
                       ↓
                    긍정/부정
```

마지막 hidden state $h_T$를 분류기에 전달. 또는 모든 hidden state를 평균/max pool.

### 3.4 Many-to-Many (정렬) — 품사 태깅

각 시점에 라벨:

```
"I  love  you"
 ↓   ↓    ↓
 x_1 x_2  x_3
  ↓   ↓    ↓
 h_1 h_2  h_3
  ↓   ↓    ↓
 PR  V    PR  (Pronoun, Verb, Pronoun)
```

각 hidden state에서 출력. 시점별 cross-entropy loss.

### 3.5 Many-to-Many (seq2seq) — 번역

입력 길이 ≠ 출력 길이. 두 단계:

**Encoder**: 입력 시퀀스를 고정 크기 vector로 압축.
**Decoder**: 그 vector에서 출력 시퀀스 생성.

```
Encoder:
"I love you"
 ↓  ↓   ↓
 h_1 h_2 h_3 → context vector c

Decoder:
c → h'_1 → "Je"
    ↓
    h'_2 → "t'"
    ↓
    h'_3 → "aime"
    ↓
    h'_4 → <END>
```

Encoder의 마지막 hidden state $h_T$가 decoder의 첫 hidden state. 또는 별도 context vector.

문제: 모든 입력 정보를 단일 context vector에 압축 → 긴 문장에서 정보 병목 → **Attention**의 등장 (§9).

### 3.6 입출력 패턴별 구현 차이

| 패턴 | 핵심 디자인 |
|---|---|
| Many-to-one | 마지막 $h_T$만 사용 |
| Many-to-many (정렬) | 모든 $h_t$에서 출력 |
| Seq2seq | Encoder + Decoder 분리, context vector |
| One-to-many | 첫 hidden을 input feature로, 자기회귀 |

이 패턴들이 RNN의 거의 모든 응용을 커버.

---

## 4. BPTT — Backpropagation Through Time

### 4.1 펼친 망에 backprop

RNN을 시간 축으로 펼치면 깊이 $T$의 깊은 망. 일반 backprop을 그대로 적용. 단 가중치 공유 처리.

각 시점 $t$의 loss를 $L_t$, 전체 loss $L = \sum_t L_t$.

### 4.2 가중치 sharing의 gradient 처리

같은 가중치 $W$가 시점 1, 2, ..., T에서 사용. 각 시점에서의 gradient 기여:

$$\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W} = \sum_{t=1}^{T} \sum_{s \le t} \frac{\partial L_t}{\partial h_s} \cdot \frac{\partial h_s}{\partial W}$$

여기서 $\frac{\partial h_s}{\partial W}$는 시점 $s$에서 $W$의 직접적 영향, $\frac{\partial L_t}{\partial h_s}$는 시점 $s$의 hidden state가 시점 $t$의 loss에 미치는 영향 (chain rule로).

다시 말해 시점 $s$의 가중치 $W$는:
- 시점 $s$ 이후 모든 시점 ($s, s+1, ..., T$)에 영향.
- 각 시점에서의 영향력을 모두 합산.

### 4.3 Hidden state의 backward

$\frac{\partial L_t}{\partial h_s}$를 계산하려면 chain rule을 시간 거꾸로 적용:

$$\frac{\partial L_t}{\partial h_s} = \prod_{k=s+1}^{t} \frac{\partial h_k}{\partial h_{k-1}} \cdot \frac{\partial L_t}{\partial h_t}$$

$\frac{\partial h_k}{\partial h_{k-1}}$는 한 step의 Jacobian. $h_k = \tanh(W x_k + U h_{k-1})$이면:

$$\frac{\partial h_k}{\partial h_{k-1}} = \text{diag}(\tanh'(z_k)) \cdot U$$

여기서 $z_k = W x_k + U h_{k-1}$, $U = W_{hh}$.

이게 시간 축에서 같은 행렬 $U$가 거듭제곱되는 본질.

### 4.4 Truncated BPTT

매우 긴 시퀀스 (수천 step) 학습은 메모리·연산 부담 큼. **Truncated BPTT**: 일정 step만 backward.

```
1. Forward 전체: x_1 ... x_T
2. Backward 일부: 마지막 K step만
```

장점:
- 메모리 일정.
- 학습 속도.

단점:
- $K$ step 너머 long-range gradient 없음.

흔히 $K = 32, 64$. Language modeling에서 자주 사용.

### 4.5 BPTT의 시간 복잡도

Forward: $O(T \cdot d^2)$ (행렬 곱이 dominant).
Backward: $O(T \cdot d^2)$.
메모리: $O(T \cdot d)$ (모든 hidden state 저장).

매우 긴 시퀀스에선 메모리 문제. Gradient checkpointing으로 메모리 줄일 수 있음 (계산 다시 함).

---

## 5. "왜 RNN에서 vanishing이 더 심한가?" — 두 번째 핵심 질문

### 5.1 답의 핵심

답: **같은 가중치 행렬이 시간 축에 거듭제곱**되기 때문.

FNN의 vanishing은 매 layer의 *다른* 행렬의 곱. 운 좋게 어떤 layer는 < 1, 다른 layer는 > 1로 균형 가능. RNN은 *같은* 행렬의 거듭제곱 — 한쪽 방향으로 폭주 또는 소멸이 거의 보장된다.

### 5.2 수식적 분석

Hidden state의 미분:

$$\frac{\partial h_t}{\partial h_{s}} = \prod_{k=s+1}^{t} \frac{\partial h_k}{\partial h_{k-1}} = \prod_{k=s+1}^{t} U^T \cdot \text{diag}(\tanh'(z_k))$$

(편의상 행렬 곱 순서 단순화)

이 product의 norm:

$$\left\| \frac{\partial h_t}{\partial h_s} \right\| \le \prod_{k=s+1}^{t} \|U^T\| \cdot \|\tanh'\|$$

$\|U^T\|$가 norm. $\|\tanh'\| \le 1$ (sigmoid는 0.25).

만약 $\|U^T\| \cdot \max|\tanh'| < 1$이면 product가 $(< 1)^{(t-s)}$로 지수 감쇠 → **vanishing**.

만약 $\|U^T\| \cdot \max|\tanh'| > 1$이면 폭발 → **exploding**.

### 5.3 고유값으로 보는 분석

$U$의 고유값 분해: $U = Q \Lambda Q^{-1}$. 거듭제곱:

$$U^t = Q \Lambda^t Q^{-1}$$

$\Lambda$의 대각 원소(고유값)의 거듭제곱이 결과를 좌우.
- 모든 고유값 절댓값 < 1 → 0으로 수렴.
- 어떤 고유값 > 1 → 발산.

활성화 미분이 들어가면 좀 더 복잡하지만 본질은 같음. 같은 행렬의 거듭제곱은 *고유값에 따라 폭주 또는 소멸*.

### 5.4 FNN과 비교 — 왜 RNN이 더 심한가

**FNN의 backward**:
$$\delta^{(l)} = \prod_{k=l+1}^{L} W^{(k)} \cdot \sigma'$$

매 layer의 $W^{(k)}$가 *다른* 행렬. $\|W^{(1)}\| \approx 0.9, \|W^{(2)}\| \approx 1.1, ...$ 식으로 우연히 균형 가능. 또 학습이 진행되면 적절히 조정됨.

**RNN의 backward**:
$$\frac{\partial h_t}{\partial h_s} = (U^T)^{t-s} \cdot \prod \text{diag}(\tanh')$$

같은 $U$의 거듭제곱. $\|U^T\| < 1$이면 *모든 시점*에서 vanishing, > 1이면 *모든 시점*에서 exploding. 균형이 매우 어려움.

이래서 RNN의 vanishing은 *구조적*. FNN의 그것보다 훨씬 심한 문제.

### 5.5 처방

| 문제 | 처방 |
|---|---|
| Exploding | **Gradient clipping** (norm > 임계면 자름) |
| Vanishing | **LSTM/GRU의 gating** |
| | Identity init, orthogonal init |
| | Skip connection (residual RNN) |
| | LayerNorm |
| | Attention (장거리 정보 직접) |

가장 중요한 건 **LSTM/GRU**. 이건 다음 챕터의 주제.

### 5.6 Gradient Clipping — Exploding의 표준 처방

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

작동: gradient의 norm이 임계값을 넘으면 비례 축소.

```
g_clipped = g if ||g|| < c else c * g / ||g||
```

매우 단순하지만 효과적. RNN/Transformer 학습에 거의 필수.

### 5.7 활성화 선택 — 왜 RNN에 tanh?

RNN의 표준 활성화는 *tanh*. 왜 ReLU 안 쓰나?

ReLU의 양수 영역 미분이 1 → vanishing은 줄지만 *exploding 위험 ↑*. 같은 행렬을 거듭제곱하면서 미분도 1이면 폭발 거의 확정.

Tanh는 미분이 ≤ 1, 평균은 더 작음 (0~0.5 정도). Vanishing은 약간 위험하지만 exploding은 없음. RNN의 시간 축 거듭제곱 구조에서 tanh가 안전한 선택.

LSTM/GRU에서도 hidden은 tanh. 단 gate는 sigmoid (0~1 비율 의미).

### 5.8 결론 — 답안 골격

> "RNN은 같은 가중치 행렬 $W_{hh}$가 시간 축에 거듭제곱되기 때문입니다. FNN은 매 layer마다 다른 $W$의 곱이라 우연히 균형 가능하지만, RNN은 같은 행렬의 거듭제곱이라 고유값이 < 1이면 즉시 vanishing, > 1이면 즉시 exploding. 한쪽 방향으로 폭주 또는 소멸이 거의 보장됩니다. 처방은 exploding엔 gradient clipping, vanishing엔 LSTM/GRU의 gating 또는 attention."

---

## 6. Bidirectional RNN — 양방향 정보

### 6.1 동기

기본 RNN은 *forward 방향*. 시점 $t$의 hidden state가 *과거* 정보만 가짐. 어떤 task는 *미래* 정보도 필요.

예: 품사 태깅에서 "bank"의 의미. "I went to the *bank*"는 이전 단어로 충분 (강기슭 또는 은행). "I went to the *bank* of the river"는 *뒤* 단어 "river"가 결정.

이걸 풀려면 양방향 모두 필요.

### 6.2 BiRNN 구조

두 RNN을 동시에 — forward와 backward:

$$h_t^\rightarrow = \text{RNN}_\rightarrow(x_t, h_{t-1}^\rightarrow)$$
$$h_t^\leftarrow = \text{RNN}_\leftarrow(x_t, h_{t+1}^\leftarrow)$$
$$h_t = [h_t^\rightarrow ; h_t^\leftarrow]$$

(concat)

각 시점의 출력은 양방향 hidden의 결합. 정보가 양쪽에서.

### 6.3 BiLSTM

LSTM 위에 적용. NLP의 표준 (BERT 이전):
- 감성 분석
- NER (Named Entity Recognition)
- 품사 태깅

### 6.4 한계

자기회귀 생성 (language model, translation generation)에선 사용 불가 — 미래를 모르기 때문. Encoder에는 BiRNN, decoder에는 unidirectional이 표준 패턴.

---

## 7. Deep RNN — 여러 layer 쌓기

### 7.1 Stacked RNN

여러 RNN을 위로 쌓음. 위 layer의 입력은 아래 layer의 hidden state:

```
Layer 3:  h_1^(3) → h_2^(3) → ...
            ↑          ↑
Layer 2:  h_1^(2) → h_2^(2) → ...
            ↑          ↑
Layer 1:  h_1^(1) → h_2^(1) → ...
            ↑          ↑
Inputs:   x_1       x_2     ...
```

위 layer는 더 추상적 표현.

### 7.2 깊이 선택

| Task | 일반적 깊이 |
|---|---|
| 단순 분류 | 1 |
| 일반 NLP | 2~3 |
| 큰 모델 | 4~8 |

너무 깊으면 학습 어려움 — vanishing 가중. ResNet 식 skip connection이나 LayerNorm 보조.

### 7.3 RNN에 ResNet 식 skip 적용

```
h_t^(l+1) = h_t^(l) + LSTM(x_t, h_{t-1}^(l+1))
```

또는 Highway-style gating. 깊은 RNN 학습 안정화.

---

## 8. Sequence-to-Sequence — Encoder-Decoder

### 8.1 동기

번역, 요약 같은 task: 입력 길이 ≠ 출력 길이. 단순 RNN으로는 안 됨.

해결: 두 RNN을 분리.

### 8.2 구조

```
Encoder (BiLSTM 자주):
"I love you"
 ↓  ↓   ↓
 h_1 h_2 h_3 → context vector c (마지막 hidden 또는 별도 압축)

Decoder (unidirectional):
c → h'_1 → "Je"
        ↓
    h'_2 → "t'"
        ↓
    h'_3 → "aime"
        ↓
    h'_4 → <END>
```

Decoder는 자기회귀 — 이전 시점 출력이 다음 시점 입력.

### 8.3 학습 시 — Teacher Forcing

학습 중엔 decoder 입력으로 *정답 이전 토큰*을 줌:

```
Decoder 입력:    <START> Je      t'      aime
Decoder 출력:    Je      t'      aime    <END>
정답 (label):    Je      t'      aime    <END>
```

각 시점의 출력과 정답을 cross-entropy로 loss.

### 8.4 추론 시

자기회귀 — 모델 출력을 다음 시점 입력으로:

```
1. <START> → "Je"
2. <START> Je → "t'"
3. <START> Je t' → "aime"
4. <START> Je t' aime → <END>
```

### 8.5 한계 — 정보 병목

Encoder가 모든 입력 정보를 *단일 vector $c$*에 압축 → 긴 시퀀스에선 정보 손실. "100단어 문장의 모든 의미를 512차원 vector에"는 어려움.

특히 입력 끝 부분의 정보가 약함 (여러 시점 거치며 잊힘 — vanishing).

이게 **Attention의 동기**.

---

## 9. Attention — Long-range의 답

### 9.1 핵심 아이디어

Decoder의 매 시점에서 *encoder의 모든 시점에 가중치를 두고 정보 가져오기*. 단일 context vector 대신 *동적으로* 결합.

### 9.2 Bahdanau Attention (2014)

각 decoder step $t$에서:

1. **Score**: encoder의 각 hidden $h_i^{enc}$에 대한 점수
$$e_{t,i} = \text{score}(h'_t, h_i^{enc})$$

2. **Weight**: softmax 정규화
$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}$$

3. **Context**: 가중합
$$c_t = \sum_i \alpha_{t,i} h_i^{enc}$$

4. **Output**: $c_t$와 $h'_t$로 다음 hidden과 출력 계산.

Score 함수: MLP, dot product 등. Bahdanau은 MLP, Luong은 dot product.

### 9.3 효과

- **Long-range 직접**: encoder의 어느 시점이든 한 번에 접근. Vanishing 우회.
- **해석 가능**: $\alpha_{t,i}$가 "decoder $t$가 encoder $i$를 얼마나 보는가" — 번역에서 단어 정렬 시각화.
- **정보 병목 해소**: 단일 context 대신 동적 결합.

### 9.4 Self-Attention과 Transformer

Bahdanau attention은 *encoder-decoder 사이*의 attention. Self-attention은 *같은 시퀀스 내*의 attention — 모든 토큰이 모든 토큰에 attention.

Transformer (2017)는 RNN을 완전 제거하고 self-attention만으로 시퀀스 처리. 병렬 + long-range + scaling. NLP의 새 표준.

이건 본 챕터 범위 밖이지만 RNN→Attention→Transformer의 흐름은 시퀀스 모델링의 큰 그림.

---

## 10. Language Model — RNN의 핵심 응용

### 10.1 정의

$P(w_t | w_{<t})$ — 이전 단어들로 다음 단어 분포 추정.

$P(\text{the}, \text{cat}, \text{sat}) = P(\text{the}) \cdot P(\text{cat}|\text{the}) \cdot P(\text{sat}|\text{the}, \text{cat})$

자기회귀 분해.

### 10.2 RNN으로 구현

```
Input:  <s>     The     cat     sat
        ↓       ↓       ↓       ↓
        h_1 →   h_2 →   h_3 →   h_4
        ↓       ↓       ↓       ↓
        FC + softmax over vocab
        ↓       ↓       ↓       ↓
Output: P(.|.)  P(.|<s>The)  P(.|<s>The cat)  P(.|<s>The cat sat)
Target: The     cat          sat              </s>
```

각 시점에서 다음 단어 분포 예측. Per-step CE loss.

### 10.3 학습 — Perplexity

평가 metric: **Perplexity** = $\exp(\text{average CE})$. 낮을수록 좋음.

직관: 평균 CE가 $\log V$ (V = vocab size)이면 random — perplexity = V. 학습된 모델은 perplexity ≪ V.

GPT-2 (2019)가 WikiText에서 perplexity 약 18 — 사람 수준에 근접.

### 10.4 생성 (Sampling)

학습된 LM에서 새 텍스트 생성:

1. 시작 토큰 $w_1$ 입력.
2. $P(w_2 | w_1)$ 계산.
3. 그 분포에서 $w_2$ sampling 또는 argmax.
4. $w_2$를 입력으로 다시 forward.
5. 반복 until <END>.

### 10.5 Decoding 전략

| 전략 | 방식 |
|---|---|
| **Greedy** | 매 시점 argmax | 단순. 자주 반복적·재미없음 |
| **Beam search** | k개 후보 유지, 끝에서 best | 더 좋은 확률, 단 단조로움 |
| **Sampling (temperature)** | 분포에서 sampling | 다양성 ↑. T로 조절 |
| **Top-k sampling** | 상위 k개에서만 | 다양성 + 안정 |
| **Top-p (nucleus)** | 누적 확률 p까지에서 | 동적 |

생성 task의 trade-off: **품질 vs 다양성**. Greedy/beam은 품질 ↑ 다양성 ↓, sampling은 반대.

### 10.6 RNN-based LM의 한계

- Long-range 약함 — 100단어 전 정보 거의 못 씀.
- 병렬화 불가 — 학습 매우 느림.
- Hidden state 크기에 의한 정보 병목.

Transformer-based LM (GPT 등)이 이 한계를 모두 해결 → NLP 패러다임 전환.

---

## 11. Teacher Forcing & Exposure Bias

### 11.1 Teacher Forcing

학습 중 decoder 입력으로 *정답 이전 토큰*을 사용:

```
Train:
 Decoder input:  <s>      The     cat     sat
                  ↓        ↓       ↓       ↓
                  h_1 →   h_2 →  h_3 →   h_4
                  ↓        ↓       ↓       ↓
 Predict:        The      cat     sat     </s>
 Target:         The      cat     sat     </s>
```

매 시점에서 *정답*을 입력으로. 모델이 자기 예측을 입력으로 쓰지 않음.

### 11.2 왜 Teacher Forcing?

장점:
- **빠른 학습**: 모든 시점의 입력이 알려져 있으니 *parallel forward 가능* (학습 시).
- **안정**: 모델이 초기에 잘못 예측해도 다음 시점 입력은 정답이라 cascade 안 함.
- **Cross-entropy 잘 정의**: 매 시점 정답과 비교.

단점:
- **Train-test mismatch**: Test 시엔 자기 예측을 입력으로 → train과 분포 다름.

### 11.3 Exposure Bias

위의 mismatch가 exposure bias. Train에서는 항상 정답을 보다가, test에서는 자기 (잠재적으로 잘못된) 출력을 입력으로 봄. 한 번 잘못 예측하면 누적되어 출력이 점점 이상해짐.

### 11.4 처방

**Scheduled Sampling**: 학습 중 점진적으로 모델 출력을 입력으로 사용. 처음엔 100% 정답, 점차 모델 출력 비율 증가.

**RL Fine-tuning**: 학습된 모델을 reward (예: BLEU score)로 fine-tune. 자기 출력을 입력으로 쓰며 학습.

**Beam search**: 추론 시 여러 후보 유지 → 한 번 잘못 가도 다른 후보가 보완.

이 처방들이 어느 정도 도움이 되나, 근본 해결은 어려움. Transformer + masked language modeling 같은 새 패러다임이 더 효과적.

---

## 12. RNN vs CNN vs Transformer — 시퀀스 비교

### 12.1 핵심 비교표

| | RNN/LSTM | 1D CNN | Transformer |
|---|---|---|---|
| 가변 길이 | 자연스러움 | 자연스러움 | 자연스러움 (단 PE) |
| 시간 sharing | ✓ 시간 축 | ✓ 시간 축 (커널 sliding) | ✓ 토큰 무관 |
| Long-range | 약함 | 깊이 필요 | 매우 강함 |
| 병렬 | ✗ 순차 | ✓ | ✓ 매우 잘 |
| 메모리 | $O(T \cdot d)$ | $O(T \cdot d)$ | $O(T^2 \cdot d)$ |
| 데이터 효율 | 좋음 | 좋음 | 많이 필요 |
| 위치 정보 | 자동 (sequential) | 자동 (kernel position) | PE 필요 |

### 12.2 언제 무엇을?

```
시퀀스 task?
├── 짧고 데이터 적음 (1k~10k)
│   → RNN/LSTM 또는 1D CNN
├── 중간 길이, 데이터 충분 (10k~100k)
│   → LSTM 또는 작은 Transformer
├── 긴 시퀀스, long-range 중요 (10k+)
│   → Transformer
└── 매우 큰 데이터 (100k+)
    → Transformer
```

### 12.3 왜 NLP가 RNN→Transformer로 갔나

(1) **병렬화**: RNN은 시점 t를 계산하려면 t-1이 끝나야 → GPU 활용 못함. Transformer는 모든 시점 동시 → GPU 폭발.

(2) **Long-range**: RNN은 거리에 따라 정보 소실. Transformer는 모든 토큰을 한 layer에서 직접 연결.

(3) **Scaling**: Transformer가 모델·데이터 키울수록 잘 따라감. RNN은 한계점 빨리 옴.

(4) **Foundation models**: GPT, BERT 등 매우 큰 사전학습 모델이 Transformer로 가능. RNN으론 비현실.

### 12.4 RNN의 자리는 사라졌나?

아니다. 여전히 자리:
- **작은 데이터**: Transformer 학습 불충분, RNN/LSTM이 더 안정.
- **메모리 한계**: $O(T^2)$가 부담스러운 환경.
- **Edge devices**: 모바일·IoT에서 LSTM이 더 효율.
- **일부 시계열**: 강한 trend·seasonality가 있는 단순 시계열엔 LSTM이 충분.

NLP의 큰 모델 영역은 Transformer 독점이지만, "모든 시퀀스 task = Transformer"는 과장.

---

## 13. 면접 단골 Q&A

### Q1. RNN이 arbitrary length를 처리하는 원리?
"시간 축 weight sharing 덕분입니다. 같은 cell을 시간에 따라 반복 적용. 시퀀스 길이가 5이든 5000이든 같은 cell을 그만큼 반복. Hidden state $h_t$가 과거 정보의 고정 크기 요약 — 가변 길이 정보를 고정 차원에 압축. 모델 가중치는 길이에 독립이고 forward pass의 반복 횟수만 길이에 의존."

### Q2. RNN에서 vanishing gradient가 더 심한 이유?
"같은 가중치 행렬 $W_{hh}$가 시간 축에 거듭제곱되기 때문. FNN은 매 layer마다 다른 $W$의 곱이라 우연히 균형 가능. RNN은 같은 행렬의 거듭제곱 — 고유값 < 1이면 즉시 vanishing, > 1이면 즉시 exploding. 한쪽 방향 폭주 또는 소멸이 거의 보장. 처방: gradient clipping (exploding), LSTM/GRU의 gating (vanishing), attention (장거리 직접)."

### Q3. BPTT vs 일반 backprop 차이?
"본질적으로 같습니다. RNN을 시간 축으로 펼치면 깊이 T의 깊은 망 + 가중치 공유. 그 망에 일반 backprop 적용 후 같은 가중치의 gradient를 합산. 차이는 (1) 가중치 sharing의 합산, (2) 매우 깊은 망이 되어 vanishing/exploding 위험 高, (3) 메모리·연산 부담 → truncated BPTT로 보완."

### Q4. Truncated BPTT가 무엇이고 trade-off는?
"매우 긴 시퀀스의 BPTT는 메모리 부담 — 모든 hidden state 저장. Truncated은 forward는 전체, backward는 마지막 K step만. 메모리 일정. Trade-off: K step 너머 long-range gradient 없음 → 매우 긴 의존성 학습 어려움. K=32~64가 흔함. Language modeling에서 자주 사용."

### Q5. Bidirectional RNN의 동기?
"기본 RNN은 forward 방향 — 시점 t의 hidden state가 *과거* 정보만. 어떤 task는 *미래* 정보도 필요. 예: '강가의 bank'에서 'bank' 의미는 뒤 단어 'river'가 결정. BiRNN은 forward와 backward 두 RNN을 동시. 양방향 hidden을 concat. NLP의 분류·태깅에서 표준. 단 자기회귀 생성에는 사용 불가 (미래 모름)."

### Q6. Seq2Seq의 정보 병목?
"Encoder가 모든 입력 정보를 단일 context vector로 압축. 짧은 문장은 OK, 긴 문장은 정보 손실. 100단어 문장의 의미를 512차원 vector에 압축은 어려움. 또 입력 끝 부분 정보가 약함 (vanishing). 이게 attention의 동기 — 단일 context 대신 decoder 매 시점에서 encoder 모든 시점에 동적 가중. Transformer로 발전."

### Q7. Attention이 RNN의 vanishing을 해결하는 방식?
"단일 context vector 대신 decoder 각 시점에서 encoder의 모든 시점에 직접 접근. Soft selection — 가중합. Long-range 정보를 거리 무관하게 전달. RNN의 chain 구조를 우회. 이게 attention이 깊은 vanishing 문제를 풀고, 결국 Transformer로 발전한 흐름."

### Q8. Teacher Forcing이 무엇이고 단점?
"학습 중 decoder 입력으로 *정답 이전 토큰* 사용. 장점: 빠른 학습, 안정, parallel forward 가능. 단점: train-test mismatch — exposure bias. Train에선 항상 정답 보다가 test에선 자기 출력 보면 분포 다름. 한 번 잘못 예측 시 cascade. 처방: scheduled sampling, RL fine-tuning, beam search."

### Q9. RNN에 ReLU 안 쓰는 이유?
"ReLU의 양수 미분 1 → vanishing은 줄지만 *exploding 위험 ↑*. RNN은 같은 행렬을 거듭제곱하면서 미분도 1이면 폭발 거의 확정. Tanh는 미분 ≤ 1, 평균 더 작음. Vanishing 약간 위험하지만 exploding은 거의 없음. RNN의 시간 축 거듭제곱 구조에 tanh가 안전. LSTM의 hidden도 tanh, gate는 sigmoid (0~1 비율)."

### Q10. RNN vs Transformer 어느 쪽?
"task에 따라. Transformer가 NLP의 표준이 된 이유: (1) 병렬화 — RNN은 시점 t가 t-1 의존, 순차. (2) Long-range — 한 layer에서 전역 연결. (3) Scaling — 큰 모델·데이터에 잘 따라감. RNN의 자리: 작은 데이터, 메모리 제약 (O(T²) 부담), edge device, 단순 시계열. NLP 큰 모델 영역은 Transformer 독점."

### Q11. Encoder-Decoder의 역할 분리?
"Encoder: 입력 시퀀스를 표현으로 압축. Decoder: 그 표현에서 출력 시퀀스 생성. 이 분리가 (1) 입출력 길이 다름 처리, (2) 양방향 입력 + 단방향 출력 (BiLSTM encoder + unidirectional decoder), (3) attention으로 동적 결합 가능. 번역·요약·QA 등 다양한 seq2seq의 표준 패턴."

### Q12. Beam search가 greedy보다 좋은 이유?
"Greedy는 매 시점 argmax → 근시안적. 한 시점 잘못 선택이 누적되면 전체 문장 망함. Beam search는 k개 후보 유지 — 한 시점에서 차선택이 다음 시점에서 더 좋은 확률 기여. 결국 더 높은 sequence likelihood. 단점: 단조로움 — 항상 *가장 그럴듯한* 출력 → 다양성 ↓. 창의적 생성엔 sampling 더 적합."

### Q13. RNN의 hidden state size 결정?
"Task 복잡도에 비례. (1) 단순 분류 (감성 등) — 64~128. (2) 일반 NLP (번역, 요약) — 256~512. (3) 큰 task (긴 문서) — 512~1024. 너무 작으면 정보 압축 한계, 너무 크면 overfit + 학습 느림. Validation에서 결정. 일반적으로 128부터 시작해 키워가며 시도."

### Q14. RNN의 dropout 적용 방식?
"수직 (layer 방향) 만 적용이 표준. 시간 방향 dropout은 hidden state 흐름 끊김 → 학습 어려움. 변종 'Variational Dropout'은 시퀀스 내내 같은 mask 사용 — 시간적 일관성. 또는 weight tying(input과 output embedding 공유)으로 정규화 효과. RNN-specific regularization은 까다로움."

---

## 14. 생각해보라 — 단락 답안

**Q. 왜 RNN이 시퀀스의 inductive bias로 자연스러운가?**

시퀀스 데이터의 두 핵심 성질:
- **시간 순서 의미**: 토큰 순서가 의미 결정.
- **시간 무관성 (translation in time)**: "the cat" 패턴이 어디 등장해도 같은 패턴.

RNN의 두 디자인:
- **순차 처리**: 시간 순서 보존.
- **시간 축 weight sharing**: 같은 cell을 모든 시점에 → 시간 무관성.

이 두 성질이 정확히 시퀀스의 본질과 매칭. CNN의 공간 sharing이 자연 이미지에 자연스러운 것처럼, RNN의 시간 sharing이 시퀀스에 자연스러움. *모델 구조와 데이터 구조의 정렬*이 효율적 학습의 비밀.

**Q. Hidden state 크기가 정보 병목이라는 의미?**

매 시점 hidden state $h_t$는 *고정 차원*. 과거의 모든 정보를 이 고정 차원에 압축. 시퀀스가 길수록 더 많은 정보를 같은 크기에 압축해야 함 → **정보 손실 누적**.

수치적 예: 100차원 hidden state. 100단어 문장에서 1000단어 문장으로 가도 hidden은 여전히 100차원. 정보 밀도 10배. 모든 정보가 보존될 수 없음.

이게 LSTM에도 남는 문제 — gating이 정보 선택을 도와주지만 hidden 크기 자체의 제약은 그대로. Transformer는 이걸 우회 — hidden을 단일 vector 아니라 *모든 토큰의 representation 집합*으로 유지. 시퀀스 길이만큼의 정보 capacity.

**Q. 왜 RNN이 NLP의 SOTA에서 밀려났는가?**

세 축에서:
(1) **Scaling**: GPT-3 (175B params)는 RNN으로 거의 불가능. RNN의 sequential dependency가 GPU 활용을 막음. Transformer는 매우 큰 모델 학습 가능.

(2) **Long-range**: 1000+ 토큰 문서 이해. RNN/LSTM은 100~200 토큰이 한계. Transformer는 한 layer에서 전역.

(3) **Foundation model 패러다임**: 대량 데이터로 사전학습 → fine-tune. Transformer가 sample efficient + scaling. BERT/GPT의 폭발적 성공.

RNN은 이론적으로 더 나은 inductive bias (시간 순서)를 가지지만, 실용적으론 Transformer의 위 세 장점이 압도. *이론과 실용의 또 다른 갭*.

**Q. Truncated BPTT가 long-range를 못 잡는 이유?**

K=32 truncated이면 backward는 마지막 32 step만. Loss의 gradient가 32 step 너머의 가중치 update에 영향 못 줌. 그 가중치는 *학습 신호가 없어* 무엇이 좋은지 모름.

Forward는 전체이므로 hidden state는 100 step 전 정보를 *어떤 식으로* 들고는 있지만, 학습이 안 되니 그 보존이 random. Long-range가 *우연히* 잘 작동할 수도 있고 안 할 수도 있음.

처방: K를 키우면 메모리·연산 比례 증가. Trade-off.

근본 해결은 attention/transformer — 매 시점이 모든 시점에 직접 연결, 거리 무관 학습 가능.

**Q. Attention과 RNN의 관계?**

처음 attention(Bahdanau 2014)은 RNN encoder-decoder의 *보조*로 등장 — context vector의 정보 병목 해결. 단일 vector 대신 동적 결합.

이후 self-attention 발전 — 같은 시퀀스 내에서 토큰 간 attention. 그리고 Transformer (2017) — RNN을 *완전 제거*하고 attention만으로.

진화의 의미:
- 1단계 (RNN + attention): RNN으로 시간 정보, attention으로 long-range.
- 2단계 (Transformer): attention만으로 모든 것. 단 위치 인코딩으로 시간 정보 보충.

현재의 큰 모델은 모두 Transformer 기반. RNN은 niche로.

**Q. RNN의 미래는?**

큰 모델 영역에서는 거의 사라짐. 단 niche에서 살아남음:
- **메모리 제약 환경**: $O(T^2)$ Transformer 부담스러운 환경.
- **Streaming 데이터**: RNN의 incremental processing이 자연스러움. 음성 인식의 RNN-T.
- **단순 시계열**: GBM과 함께 baseline.
- **이론적 연구**: 시퀀스 모델링의 본질적 inductive bias 이해.

또 최근 연구에서 RNN의 부활 시도 (RWKV, Mamba 등) — Transformer의 $O(T^2)$ 한계를 극복하려는. State Space Model (S4, Mamba)이 본질적으로 RNN과 같은 정신.

긴 시퀀스 (수만 토큰)에서는 RNN 식 모델이 다시 효율적일 수 있음. 미래는 hybrid.

**Q. 왜 LSTM이 RNN보다 long-term을 잘 잡는가?**

다음 챕터의 주제지만 핵심만: LSTM의 cell state $c_t$는 **덧셈 update** ($c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$). 이 덧셈 구조에서 $\partial c_t/\partial c_{t-1} = f_t$ (scalar) — 행렬 곱이 아닌 scalar 곱. ResNet skip connection과 본질 같음.

Forget gate가 1 근처를 학습하면 정보가 시간 축에 *그대로* 흐름. Vanishing 거의 없음. 이게 LSTM이 100+ step long-term을 잡을 수 있는 이유.

여전히 한계가 있지만 (수천 step 어려움) RNN보다 압도적 우위. 자세한 건 다음 챕터.

---

## 15. 한 줄 요약 (시험 직전)

- RNN의 본질 = **시간 축 weight sharing + hidden state**. 가변 길이 처리 자연.
- **Arbitrary length 처리 원리** = 같은 cell의 반복. 모델 가중치는 길이 독립.
- **Vanishing이 더 심한 이유** = 같은 행렬의 거듭제곱. FNN의 다른 행렬 곱과 본질 차이.
- **BPTT** = 펼친 망에 backprop + 가중치 sharing 합산. Truncated으로 메모리 절약.
- **Gradient clipping**으로 exploding, **LSTM/GRU**로 vanishing.
- **Tanh**가 RNN 표준 — exploding 약함. ReLU는 위험.
- **BiRNN** = 양방향 정보. 자기회귀 생성엔 unsuited.
- **Seq2seq** = encoder + decoder. 단일 context vector의 정보 병목 → attention 등장.
- **Attention** = 매 시점 동적 결합. RNN의 vanishing 우회. Transformer로 발전.
- **Teacher forcing**의 exposure bias = train-test mismatch. Scheduled sampling 보완.
- **NLP의 RNN→Transformer**: 병렬 + long-range + scaling의 압도적 우위.
- RNN은 niche에서 살아남음 — 작은 데이터, 메모리 제약, streaming.
