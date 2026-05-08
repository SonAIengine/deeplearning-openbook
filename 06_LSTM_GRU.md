# 06. LSTM & GRU — 오픈북 정리

> 핵심 질문 (★교수님 단골):
> 1. **"왜 LSTM은 RNN의 vanishing gradient를 완화하는가?"**
> 2. **"각 gate는 왜 필요한가? 없으면 어떻게 되나?"**
> 3. **"LSTM vs GRU, 언제 무엇을?"**

---

## 0. 동기 — "RNN으로 안 되는 이유"

기본 RNN의 hidden state 업데이트:
$$h_t = \tanh(W x_t + U h_{t-1})$$
- 새 정보가 매 시점마다 hidden을 **덮어쓴다**.
- 과거 정보를 멀리 전달하려면 같은 행렬을 거듭제곱 → **vanishing**.

→ 필요한 것: **"기억을 선택적으로 유지·삭제·갱신"하는 메커니즘** = **gating**.

---

## 1. LSTM 구조 — "왜 cell state를 도입?"

### 1.1 핵심 분리
LSTM은 **cell state $c_t$** (장기 기억)와 **hidden state $h_t$** (출력) 분리.

cell state는 "고속도로"처럼 **gradient가 곱이 아닌 합으로 전달**되는 경로.

### 1.2 식 (★ 외우기보다 의미 이해)

$$\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)} \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(input gate)} \\
\tilde{c}_t &= \tanh(W_c [h_{t-1}, x_t] + b_c) \quad \text{(candidate)} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(cell update)} \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(output gate)} \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}$$

### 1.3 각 gate의 역할 — "없으면 어떻게 되나?" (★)

| Gate | 역할 | 없으면? |
|---|---|---|
| **Forget $f_t$** | "과거 cell state 중 뭘 버릴까?" | 모든 과거 누적 → 폭발/잡음 |
| **Input $i_t$** | "새 정보 중 뭘 cell에 쓸까?" | 매 step 모두 기록 → 노이즈 |
| **Output $o_t$** | "cell의 어느 부분을 노출할까?" | 항상 전부 노출 → task 무관 정보까지 |
| **Candidate $\tilde{c}_t$** | "쓸 새 정보 후보" | 갱신 못함 |

생각해보라:
- forget=1, input=0이면 **완벽 보존**.
- forget=0, input=1이면 **완전 덮어쓰기** (RNN과 유사).
- gate가 **상황에 따라 다르게 작동**하기 때문에 task-적응적.

---

## 2. "왜 LSTM이 vanishing을 완화?" (★★ 시험 단골)

### 2.1 cell state의 미분 경로
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$\frac{\partial c_t}{\partial c_{t-1}} = f_t \quad (\text{element-wise})$$

→ gradient가 곱해지는 게 **forget gate 값**(0~1).

### 2.2 RNN과의 차이
- **기본 RNN**: $\partial h_t/\partial h_{t-1} = W^T \cdot \text{diag}(\sigma'(\cdot))$ — **행렬의 거듭제곱**.
- **LSTM**: $\partial c_t/\partial c_{t-1} = f_t$ — **scalar(per-element) 곱**.

### 2.3 왜 이게 더 좋나?
1. forget gate를 **1에 가깝게 학습**하면 gradient가 거의 그대로 흐름 (additive 경로).
2. 행렬 곱이 아니라 element-wise → 차원별로 **다른 시간 스케일** 가능.
3. 식이 $c_t = f \cdot c_{t-1} + i \cdot \tilde{c}$ — 덧셈 구조라 **ResNet의 skip connection과 본질적으로 같음**.

### 2.4 한계
- 완전히 해결은 아님. 매우 긴 의존성(수천 step)에서는 여전히 약함.
- 그래서 attention/Transformer가 필요.

---

## 3. LSTM의 직관적 비유

- **cell state** = 컨베이어 벨트. 기본적으로 그대로 흘러감.
- 위에 **forget gate**가 부분 삭제, **input gate**가 부분 추가.
- **output gate**는 벨트의 일부만 외부에 보여줌.

이 비유로 보면 **각 gate의 필요성이 자연스럽다**.

---

## 4. Forget gate bias 초기화 (실용 트릭)

$b_f$를 1~2로 초기화 → 학습 초기에 forget=1에 가까움 → 정보 보존 → 안정적 학습.
이 단순한 변경이 큰 차이를 만든다는 보고 多.

---

## 5. GRU — "단순한 LSTM"

### 5.1 식
$$\begin{aligned}
z_t &= \sigma(W_z [h_{t-1}, x_t]) \quad \text{(update gate)} \\
r_t &= \sigma(W_r [h_{t-1}, x_t]) \quad \text{(reset gate)} \\
\tilde{h}_t &= \tanh(W [r_t \odot h_{t-1}, x_t]) \\
h_t &= (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{aligned}$$

### 5.2 LSTM과의 차이
- **cell state 없음**: hidden state 하나만.
- **gate 2개 (update, reset)**: forget+input을 하나로 합침.
- output gate 없음.

### 5.3 각 gate
- **Update $z_t$**: 과거 vs 새것의 비율. (1−z) 만큼 과거 유지, z만큼 새로 갱신.
- **Reset $r_t$**: 새 candidate 만들 때 과거를 얼마나 무시할지.

### 5.4 LSTM과 본질
- update gate가 **forget·input을 합친 형태** ($f = 1-z$, $i = z$).
- 파라미터·계산 약 25% 적음.

---

## 6. LSTM vs GRU — "언제 무엇을?" (★)

| | LSTM | GRU |
|---|---|---|
| Gate 수 | 3 (+ candidate) | 2 |
| State | cell + hidden | hidden만 |
| 파라미터 | 많음 | ~75% |
| 학습 속도 | 느림 | 빠름 |
| 표현력 | 약간 더 (이론적) | 약간 덜 |
| 실무 성능 | 비슷 (task-dependent) | 비슷 |

**결론**:
- 데이터·계산 적으면 → GRU.
- 매우 긴 의존성, 큰 데이터 → LSTM (약간 우위 보고 있음).
- **실제로는 거의 동급** — 두 개 다 시도해보는 게 정석.

---

## 7. Bidirectional LSTM (BiLSTM)

forward + backward LSTM을 합침.
- NLP에서 BERT 이전 표준 (감성 분석, NER, 품사 태깅).
- 자기회귀 생성에는 불가 (미래 모름).

---

## 8. Stacked LSTM

여러 LSTM을 쌓음. 보통 2~4층.
- 위층은 더 추상적 표현.
- 너무 깊으면 학습 어려움 → residual + LayerNorm 보조.

---

## 9. LSTM의 한계 — "왜 결국 Transformer로?"

1. **순차 처리**: 시점 t를 계산하려면 t−1이 끝나야 함 → **GPU 병렬화 불가**.
2. **여전한 long-range 약점**: 100~1000 step 넘어가면 정보 흐릿.
3. **고정 크기 hidden**: 정보 병목.

→ Self-attention은 **모든 시점을 동시에**, **거리 무관하게** 정보 전달.

생각해보라: LSTM의 forget gate가 1에 가까운 학습이 가능하면 정보가 보존되는데, **왜 여전히 long-range가 약한가?** → 가능하지만 학습 신호가 끝까지 잘 전달되어야 그렇게 학습됨. 그 학습 자체가 어려움. 또 hidden 크기 제약으로 정보를 압축해야 함.

---

## 10. 다른 변종

| 변종 | 특징 |
|---|---|
| **Peephole LSTM** | gate 입력에 cell state도 포함 |
| **ConvLSTM** | gate를 conv로 (영상 시퀀스) |
| **Coupled LSTM** | $i_t = 1 - f_t$ (gate 묶음) |
| **Layer Norm LSTM** | gate 내부에 LN |
| **Residual LSTM** | layer 간 skip |

---

## 11. 실용 체크리스트

- forget bias = 1~2로 init.
- gradient clipping (norm 1~5).
- LayerNorm 또는 BatchNorm은 LSTM gate 계산에 조심스럽게.
- Dropout은 **시간 방향 X, 층 방향만** (또는 variational).
- 시퀀스 packing/padding으로 가변 길이 효율 처리.

---

## 12. RNN → LSTM/GRU → Transformer 흐름 정리 (★ 빅 픽처)

| 시대 | 모델 | 해결 | 새 한계 |
|---|---|---|---|
| 1986~ | RNN | 시퀀스 처리 가능 | vanishing |
| 1997 | LSTM | gating으로 vanishing 완화 | 여전히 순차, long-range 약함 |
| 2014 | GRU | LSTM 단순화 | 동일 |
| 2014~ | Attention | long-range 직접 전달 | RNN과 결합 시 여전히 순차 |
| 2017 | Transformer | self-attention만, 완전 병렬 | O(n²) |

→ "**모든 단계는 직전 단계의 한계에 대한 답**." 이 흐름을 외우면 면접·시험 모두 강해진다.

---

## 13. 한 줄 요약

- LSTM의 핵심 = **cell state + 3 gate**. additive update로 vanishing 완화.
- gate들은 "기억을 무엇을 버리고/추가하고/노출할지"를 **데이터-의존적으로** 결정.
- $\partial c_t/\partial c_{t-1} = f_t$ → 행렬 곱 대신 **element-wise scalar 곱** → ResNet의 skip과 같은 원리.
- GRU는 LSTM의 단순화 버전, 성능은 거의 동급.
- LSTM/GRU도 **병렬화 불가 + long-range 한계** → Transformer 등장.
