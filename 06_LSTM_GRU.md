# 06. LSTM & GRU — 심층 정리

> **이 문서의 핵심 세 질문 (★ 교수님 단골)**:
> 1. **"왜 LSTM은 RNN의 vanishing gradient를 완화하는가?"**
> 2. **"각 gate는 정확히 어떤 역할을 하고, 없으면 어떻게 되는가?"**
> 3. **"LSTM과 GRU의 차이는? 언제 무엇을 선택하는가?"**
>
> 이 세 질문에 *수식·직관·한계*까지 답할 수 있게 만든다. 그리고 LSTM의 cell state가 ResNet의 skip connection과 *본질적으로 같은 정신*임을 보여 다른 챕터와 연결한다.

---

## 0. 큰 그림 — 왜 gating이 필요한가

### 0.1 RNN의 한계 다시

지난 챕터에서 본 RNN의 vanishing gradient 문제는 깊은 구조적 문제다:

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1})$$

매 시점에서 hidden state가 *완전히 새로 계산*된다. 이전 hidden $h_{t-1}$은 가중치를 거쳐 변환되고, $x_t$의 영향과 합쳐져 새 hidden $h_t$가 된다. 즉 **이전 정보가 매 시점마다 덮어쓰기** 된다.

이게 두 문제를 만든다:

(1) **Vanishing**: 정보가 시간 축으로 흐르려면 같은 $W_{hh}$를 거듭제곱. 고유값이 1 미만이면 빠르게 0으로 수렴 → 멀리 있는 정보는 사라짐.

(2) **선택의 부재**: 모든 새 정보가 같은 가중치로 통합. 중요한 정보든 noise이든 똑같이 hidden에 흡수.

### 0.2 Gating의 아이디어

이 두 문제의 답: **"어떤 정보를 보존할지, 어떤 새 정보를 추가할지를 학습 가능하게 하자."**

기본 RNN의 hidden update는 모든 시점에서 *같은 함수*. Gating은 매 시점마다 *상황에 맞게* 동작:
- 중요한 정보가 hidden에 있으면 → 거의 변경 없이 유지
- noise만 있으면 → 새 입력으로 갱신
- 새 입력이 정말 중요하면 → 강하게 흡수
- 새 입력이 무관하면 → 무시

이걸 수식으로 표현하는 방법이 **gate** — 0과 1 사이의 값으로 정보 흐름을 조절.

### 0.3 두 가지 gating 모델

- **LSTM** (1997, Hochreiter-Schmidhuber): 처음 등장한 gating RNN. **3개 gate + cell state**.
- **GRU** (2014, Cho et al.): LSTM의 단순화. **2개 gate, cell state 없음**.

둘 다 같은 정신 — gating으로 정보 선택. 차이는 정교함과 단순함의 trade-off.

### 0.4 본 챕터의 흐름

1. LSTM 구조와 forward (§1)
2. 각 gate의 의미 (§2)
3. 왜 vanishing 완화 — 수식적 분석 (§3 — 첫 핵심 질문)
4. 각 gate를 빼면 어떻게 되는가 (§4 — 두 번째 핵심 질문)
5. GRU 구조와 LSTM 비교 (§5)
6. LSTM vs GRU — 언제 무엇을 (§6 — 세 번째 핵심 질문)
7. LSTM의 변종들 (§7)
8. LSTM의 한계와 attention의 등장 (§8)
9. 면접 Q&A (§9)
10. 생각해보라 (§10)

---

## 1. LSTM 구조 — Cell State와 3 Gate

### 1.1 핵심 분리 — Cell vs Hidden

LSTM은 두 개의 state를 가진다:

- **Cell state $c_t$**: 장기 기억. 시간 축에 *덧셈으로 update*. ResNet의 skip과 본질 같음.
- **Hidden state $h_t$**: 단기 기억 + 외부 출력. Cell state의 일부를 노출.

이 분리가 LSTM의 가장 본질적 디자인. 왜 분리?
- Cell state는 *정보 보존* 전용 — vanishing 약하게.
- Hidden state는 *외부 출력* 전용 — task-relevant 정보만.

### 1.2 LSTM 식 (★ 외우기보다 의미 이해)

각 시점에서:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(input gate)}$$
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad \text{(candidate)}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(cell update)}$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(output gate)}$$
$$h_t = o_t \odot \tanh(c_t) \quad \text{(hidden output)}$$

기호 설명:
- $\sigma$: sigmoid. 출력 (0, 1) — 비율로 해석.
- $\tanh$: hyperbolic tangent. 출력 (-1, 1) — 정규화된 값.
- $\odot$: element-wise 곱.
- $[h_{t-1}, x_t]$: concatenation.
- $f_t, i_t, o_t$: forget, input, output gate. 각각 sigmoid라 (0, 1).
- $\tilde{c}_t$: candidate. 새로 cell에 추가될 정보 후보.
- $c_t$: cell state.
- $h_t$: hidden state (외부 출력).

### 1.3 단계별 직관 풀이

매 시점에 LSTM이 하는 일:

**Step 1**: "이전 cell의 어느 부분을 잊을까?" → forget gate $f_t$ 계산.
- $f_t \approx 1$이면 보존, $f_t \approx 0$이면 삭제.
- 결정 근거: 이전 hidden $h_{t-1}$과 현재 입력 $x_t$.

**Step 2**: "새 정보 후보를 만들자." → candidate $\tilde{c}_t$ 계산.
- $h_{t-1}$과 $x_t$로 만든 잠재적 cell update 값.
- $\tanh$라 (-1, 1) 범위 — 정규화.

**Step 3**: "새 정보 중 어느 부분을 추가할까?" → input gate $i_t$ 계산.
- $i_t \approx 1$이면 candidate 강하게 추가, $\approx 0$이면 무시.

**Step 4**: "Cell 업데이트": $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$.
- 이전 cell의 일부 보존 + 새 정보의 일부 추가.
- **덧셈 구조**가 핵심.

**Step 5**: "Cell의 어느 부분을 노출할까?" → output gate $o_t$ 계산.

**Step 6**: "Hidden 출력": $h_t = o_t \odot \tanh(c_t)$.
- $\tanh(c_t)$로 cell 정규화 후, $o_t$로 일부만 외부에 노출.

각 단계가 분리된 역할. 이게 단순 RNN의 *모든 정보를 한 번에 통합*과의 본질적 차이.

### 1.4 수식의 직관적 비유 — 컨베이어 벨트

이해를 돕는 비유 (Olah blog 식):

**Cell state = 컨베이어 벨트**. 시간 축으로 정보가 그대로 흘러간다. 매 시점에서 약간 수정될 뿐.

**Forget gate**: 벨트 위 어느 부분을 *지울지* 결정. $f_t \cdot c_{t-1}$.

**Input gate + candidate**: 벨트에 *어느 새 정보를 추가할지*. $i_t \cdot \tilde{c}_t$.

**Output gate**: 벨트의 *어느 부분을 외부에 보여줄지*. $o_t \cdot \tanh(c_t)$.

이 비유로 보면 cell state가 컨베이어 벨트로서 정보를 *길게 운반*하고, gate들이 *국소적으로* 수정만. 자연스럽게 long-term 정보 보존.

### 1.5 파라미터 수

LSTM cell의 파라미터:
- 4개의 가중치 행렬 ($W_f, W_i, W_c, W_o$). 각 크기 $d_h \times (d_h + d_x)$.
- 4개의 bias.

총: $4 \cdot d_h \cdot (d_h + d_x + 1)$.

기본 RNN($d_h \cdot (d_h + d_x + 1)$)의 약 **4배**. 표현력 ↑, 비용 ↑.

### 1.6 PyTorch 구현 (개념)

```python
class LSTMCell:
    def __init__(self, d_in, d_h):
        self.W_f = nn.Linear(d_in + d_h, d_h)
        self.W_i = nn.Linear(d_in + d_h, d_h)
        self.W_c = nn.Linear(d_in + d_h, d_h)
        self.W_o = nn.Linear(d_in + d_h, d_h)

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([h, x], dim=-1)
        f = torch.sigmoid(self.W_f(combined))
        i = torch.sigmoid(self.W_i(combined))
        c_tilde = torch.tanh(self.W_c(combined))
        c_new = f * c + i * c_tilde
        o = torch.sigmoid(self.W_o(combined))
        h_new = o * torch.tanh(c_new)
        return h_new, (h_new, c_new)
```

실제 PyTorch는 이 4개 행렬 곱을 하나로 합쳐 효율화 (병렬 GEMM).

---

## 2. 각 Gate의 의미

### 2.1 Forget Gate $f_t$

**역할**: "과거 cell state 중 무엇을 버릴까?"

$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$

각 차원 $f_{t,i} \in (0, 1)$:
- $f_{t,i} \approx 1$: 그 차원의 정보 보존.
- $f_{t,i} \approx 0$: 그 차원의 정보 삭제.

**언제 forget?**:
- 새 sentence 시작: 이전 sentence의 정보를 일부 잊음.
- 컨텍스트 전환: 주제가 바뀜.
- 오래된 정보가 더 이상 관련 없음.

**Forget bias 초기화**:
보통 bias를 0으로. 단 LSTM에서는 $b_f$를 1~2로 init하는 것이 권장. 학습 초기에 forget gate가 1에 가깝게 (정보 보존). 학습이 진행되며 *어느 정보를 잊을지* 학습.

이 작은 변경이 학습 안정성에 큰 차이.

### 2.2 Input Gate $i_t$ + Candidate $\tilde{c}_t$

**역할**: "새 정보 중 무엇을 cell에 쓸까?"

두 부분으로:
- $\tilde{c}_t$: 새로 추가될 *후보* 값. (-1, 1).
- $i_t$: 그 후보 중 *얼마나* 추가할지의 비율. (0, 1).

곱한 결과 $i_t \odot \tilde{c}_t$가 cell에 더해짐.

**왜 분리?**: 후보를 만드는 것과 그것을 받아들일지 결정하는 것을 분리. 표현력 ↑.

GRU에서는 이 둘을 합쳤다 (다음 섹션).

**언제 input?**:
- 중요한 새 단어 등장.
- 컨텍스트 전환 시점에서 새 컨텍스트 인식.
- 의문문/평서문 등 문장 종류 결정.

### 2.3 Output Gate $o_t$

**역할**: "Cell의 어느 부분을 외부에 노출할까?"

$h_t = o_t \odot \tanh(c_t)$.

Cell state는 *모든 정보 보존*. 단 hidden state로 *나가는 건 일부만*. 어떤 부분을 노출할지가 task에 의존.

**왜 분리?**:
- Cell은 long-term 보존.
- Hidden은 다음 시점 계산 + 외부 출력.
- 두 역할이 다름.

예: 분류 task에서 어떤 cell 차원은 "지금까지의 토큰 수" 정보일 수 있다. 이건 다음 시점 처리에는 유용하지만 *분류 출력*엔 무관. Output gate가 그 차원을 닫음.

### 2.4 세 gate의 조합 동작

특정 패턴들:

**완벽 보존** ($f = 1, i = 0$): $c_t = c_{t-1}$. 정보 그대로.

**완전 덮어쓰기** ($f = 0, i = 1$): $c_t = \tilde{c}_t$. 기본 RNN과 비슷.

**일부 update** ($f = 0.7, i = 0.3$): 70% 보존 + 30% 새 정보. 균형.

이 조합이 *상황에 따라 다르게* 학습됨. 그래서 LSTM이 다양한 시퀀스 task에 robust.

---

## 3. "왜 LSTM이 vanishing을 완화?" — 첫 핵심 질문

### 3.1 답의 핵심 — 덧셈 구조

기본 RNN: $h_t = \tanh(W h_{t-1} + ...)$. **곱셈** 구조 — 같은 $W$가 거듭제곱.

LSTM: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. **덧셈** 구조 — cell state가 시간 축에 덧셈으로 흐름.

이 덧셈 구조가 vanishing 완화의 본질.

### 3.2 수식적 분석

Cell state의 미분:

$$\frac{\partial c_t}{\partial c_{t-1}} = \frac{\partial}{\partial c_{t-1}} \left( f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \right) = f_t$$

(여기서 $f_t, i_t, \tilde{c}_t$ 자체가 $h_{t-1}$ 통해 $c_{t-1}$의 함수지만, 직접적 영향은 $f_t$가 dominant. 정확한 분석은 더 복잡하지만 핵심은 같다.)

이 미분이 **scalar (per-element)**. 행렬 곱이 아님.

긴 시간 축의 미분:
$$\frac{\partial c_t}{\partial c_s} = \prod_{k=s+1}^{t} f_k$$

Forget gate $f_k$가 1 근처면 product가 1 근처 → gradient 그대로 흐름.

### 3.3 RNN과의 비교 — 행렬 곱 vs scalar 곱

**기본 RNN**:
$$\frac{\partial h_t}{\partial h_{t-1}} = W^T \cdot \text{diag}(\tanh'(\cdot))$$

깊이 t-s에서:
$$\frac{\partial h_t}{\partial h_s} = \prod (\text{행렬 곱})$$

행렬 곱의 누적 → 고유값에 따라 폭주 또는 소멸.

**LSTM**:
$$\frac{\partial c_t}{\partial c_s} = \prod_{k=s+1}^{t} f_k \quad (\text{element-wise scalar 곱})$$

Element-wise 곱 → 차원마다 독립적 동역학. 한 차원이 vanishing이라도 다른 차원은 살아있을 수 있음.

또 forget gate가 *학습된 값*이라, 학습 신호가 그 gate를 *1에 가깝게* 만들면 자동으로 vanishing 완화.

### 3.4 왜 이게 ResNet과 본질 같은가

ResNet의 skip:
$$h_{l+1} = h_l + F(h_l)$$
$$\frac{\partial h_{l+1}}{\partial h_l} = I + \frac{\partial F}{\partial h_l}$$

Identity 항 $I$가 있어 곱셈 누적이 아닌 덧셈 경로.

LSTM의 cell:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

Forget=1, input=0이면 $c_t = c_{t-1}$ — 정확히 identity. 일반적으로 forget이 1 근처면 거의 identity skip.

**둘 다 곱셈 누적을 덧셈 경로로 우회**. 다른 자리(공간 vs 시간)지만 같은 정신.

### 3.5 차원별 다른 시간 스케일

LSTM의 또 다른 강점: cell state 각 차원이 *독립적인 forget gate*. 어떤 차원은 forget 매우 강함 (단기 정보), 어떤 차원은 forget 매우 약함 (장기 정보).

이게 학습으로 자동 결정. 시퀀스의 다양한 시간 스케일을 *동시에* 모델링.

기본 RNN은 모든 차원이 같은 행렬 $W$를 공유 → 시간 스케일이 통합. LSTM은 차원별 독립성으로 다양성.

### 3.6 한계 — 완전 해결은 아님

LSTM도 완벽하진 않다:

(1) **수천 step에선 여전히 약함**. Forget gate가 정확히 1을 학습하기 어려움. 작은 오차가 누적되면 100+ step에서 정보 약해짐.

(2) **Hidden 크기 제약**. $c_t$의 차원이 고정 → 정보 압축 한계.

(3) **Sequential 처리**. 병렬 불가. 학습 매우 느림.

이 한계들이 attention/Transformer 등장의 동기.

### 3.7 답안 골격

> "Cell state의 update가 덧셈 구조 ($c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$)이기 때문입니다. 미분이 $\partial c_t/\partial c_{t-1} = f_t$ — 행렬 곱이 아닌 element-wise scalar 곱. RNN의 행렬 거듭제곱 대신 forget gate만 곱해지므로, gate가 1 근처를 학습하면 gradient가 시간 축에 거의 그대로 흐릅니다. 이게 ResNet의 skip connection과 본질적으로 같은 정신 — 곱셈 누적을 덧셈 경로로 우회. 단 수천 step에서는 여전히 약점이라 attention이 필요합니다."

---

## 4. 각 Gate를 빼면 어떻게 되나 — 두 번째 핵심 질문

### 4.1 Forget Gate 없이 ($f = 1$ 고정)

식: $c_t = c_{t-1} + i_t \odot \tilde{c}_t$

**무엇이 망가지나**:
- 모든 과거 정보가 *영원히 누적*. 잊기 불가.
- Cell state가 **폭발하거나 noise로 가득** 참.
- 긴 시퀀스에서 정보 포화.
- 컨텍스트 전환 시 이전 컨텍스트 못 잊음.

예: 두 sentence를 처리할 때 첫 sentence의 정보가 둘째 sentence 처리에 계속 영향. 분류, 번역 모두 망함.

**가장 중요한 gate**: forget. 없으면 LSTM이 무용지물.

### 4.2 Input Gate 없이 ($i = 1$ 고정)

식: $c_t = f_t \odot c_{t-1} + \tilde{c}_t$

**무엇이 망가지나**:
- 모든 candidate가 항상 cell에 추가.
- 노이즈에 매우 약함.
- "선택적 기록"이 안 됨.
- 매 시점의 모든 토큰이 *같은 가중치*로 cell에 영향.

예: "the"나 "a" 같은 무관 토큰의 정보가 cell에 강하게 들어감 → 중요 정보 희석.

### 4.3 Output Gate 없이 ($o = 1$ 고정)

식: $h_t = \tanh(c_t)$

**무엇이 망가지나**:
- Cell의 모든 정보가 외부에 노출.
- 다음 시점의 gate 계산 + 외부 출력 모두 같은 정보.
- Task-무관한 정보까지 출력으로 흘러 학습 신호 흐려짐.

상대적으로 덜 치명적. GRU는 output gate가 *없는* 형태에 가까움 (다음 섹션). 실제로 큰 손실 아님.

### 4.4 Forget + Input 합치기 ($i = 1 - f$)

식: $c_t = f_t \odot c_{t-1} + (1 - f_t) \odot \tilde{c}_t$

이게 **Coupled LSTM**. 또는 GRU의 정신.

직관: "보존 비율과 새 정보 비율의 합이 1." 즉 cell의 한 차원에서 *옛것과 새것 중 택일* (실제론 부드러운 비율).

표현력 약간 손해. 단 파라미터 적음. 실용적으론 거의 동급.

이게 *GRU의 update gate*가 정확히 하는 일.

### 4.5 Candidate 없이?

$\tilde{c}_t$ 없이는 cell update가 정의 안 됨. 새 정보가 들어올 곳 없음. LSTM의 본질적 부품.

### 4.6 Gate를 모두 빼면?

식이 $c_t = c_{t-1} + \tanh(W [h_{t-1}, x_t])$로 간단해지면 — 이게 **residual RNN** 또는 비슷한 형태. 단순한 skip connection 추가 RNN. Vanishing은 약간 완화하지만 LSTM만큼 강력하진 않음.

### 4.7 핵심 정리

| Gate 제거 | 효과 |
|---|---|
| Forget | **치명적**. 정보 포화. |
| Input | 큰 손해. 선택적 기록 불가. |
| Output | 덜 치명적. GRU가 증명. |
| Forget + Input 합침 | 약간 손해. GRU 식. |

**Forget이 가장 중요**, output은 거의 없어도 됨.

이게 GRU의 디자인 철학 — output gate 제거 + forget·input 합침으로 더 단순하게.

---

## 5. GRU — 단순화된 LSTM

### 5.1 GRU의 동기

LSTM은 효과적이지만 복잡 — 3개 gate + cell state. 단순화 가능?

GRU (Gated Recurrent Unit, 2014, Cho et al.): **2개 gate, cell state 없음**.

### 5.2 GRU 식

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t]) \quad \text{(update gate)}$$
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t]) \quad \text{(reset gate)}$$
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t]) \quad \text{(candidate)}$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### 5.3 LSTM과 매핑

| LSTM | GRU |
|---|---|
| Forget $f$ | 1 - z (update gate의 보수) |
| Input $i$ | z |
| Output $o$ | (없음) |
| Reset (없음) | r (candidate 만들 때 과거 영향 조절) |
| Cell + Hidden 분리 | Hidden만 |

본질적 통찰:
- LSTM의 forget·input을 update gate $z$ 하나로 합침 ($f = 1-z, i = z$).
- Output gate 없음 — cell·hidden 분리도 없음.
- 새로운 reset gate $r$ — candidate 만들 때 과거 정보 영향 조절.

### 5.4 각 gate의 의미

**Update gate $z_t$**:
- $z_t \approx 1$: 거의 새 정보. $h_t \approx \tilde{h}_t$.
- $z_t \approx 0$: 거의 보존. $h_t \approx h_{t-1}$.
- 정확히 LSTM의 forget·input의 합쳐진 형태.

**Reset gate $r_t$**:
- 새 candidate 만들 때 *얼마나 과거를 무시할지*.
- $r_t \approx 0$: 새 candidate은 거의 $x_t$만 기반. 과거 무시.
- $r_t \approx 1$: 과거 hidden과 결합한 candidate.

직관: reset gate는 "**컨텍스트 reset**" — 새 시작점에서 과거 무시.

### 5.5 Vanishing 완화 — GRU의 분석

$h_t = (1 - z_t) h_{t-1} + z_t \tilde{h}_t$

미분:
$$\frac{\partial h_t}{\partial h_{t-1}} \approx 1 - z_t \quad (\text{plus terms via } \tilde{h}_t)$$

$z_t \approx 0$이면 $\partial h_t/\partial h_{t-1} \approx 1$ — gradient 그대로.

LSTM과 거의 같은 메커니즘. Element-wise gating으로 vanishing 완화.

### 5.6 LSTM 대비 차이

**더 단순**: 2개 gate vs 3개 + candidate. 파라미터 약 25% 적음.

**Cell state 없음**: hidden 하나만. 정보가 한 곳에. 단 LSTM의 *분리*가 주는 표현력은 손해.

**Reset gate**: LSTM에 없는 새 메커니즘. Candidate 만들 때 과거 영향 조절.

---

## 6. LSTM vs GRU — 언제 무엇을? — 세 번째 핵심 질문

### 6.1 비교표

| | LSTM | GRU |
|---|---|---|
| Gate 수 | 3 + candidate | 2 |
| State | Cell + Hidden | Hidden만 |
| 파라미터 | 많음 | LSTM의 ~75% |
| 표현력 | 약간 더 (이론) | 약간 덜 |
| 학습 속도 | 느림 | 빠름 |
| 메모리 | 많음 | 적음 |
| 실무 성능 | 비슷 (task별) | 비슷 |

### 6.2 실증적 결과

여러 논문과 실험 결과:

(1) **Chung et al. 2014**: GRU와 LSTM의 비교. Music modeling, speech signal modeling. 결과: 거의 동급.

(2) **Jozefowicz et al. 2015**: 다양한 RNN 구조의 architecture search. 결론: LSTM의 forget gate가 가장 중요. GRU와 LSTM은 비슷.

(3) **Greff et al. 2017**: LSTM 변종 ablation. 결과: 모든 gate가 중요하지만 단순화된 형태 (GRU)도 경쟁력.

요약: **Task와 데이터에 따라 다르지만 거의 동급**. 매우 큰 차이 안 남.

### 6.3 결정 가이드

```
선택 기준?
├── 데이터 적음, 빨라야 함 → GRU
├── 매우 긴 의존성, 큰 데이터 → LSTM (약간 우위 보고)
├── 메모리 제약 (모바일·edge) → GRU
├── 정확도가 가장 중요, 비용 OK → 둘 다 시도
└── 잘 모르겠음 → 둘 다 시도, val로 결정
```

### 6.4 실무에서

(1) 대부분 task에서 두 모델을 모두 시도하고 validation으로 결정.

(2) GRU가 단순해 학습이 약간 더 안정적. Hyperparameter 튜닝 부담 적음.

(3) LSTM이 더 큰 모델·데이터에서 약간 우위 (Jozefowicz 등 보고).

(4) 큰 NLP 모델은 어차피 Transformer로 갔으므로 LSTM/GRU 선택 자체가 큰 의미 없는 영역도 多.

### 6.5 답안 골격

> "거의 동급입니다. LSTM은 3개 gate + cell/hidden 분리로 표현력 약간 더 풍부, GRU는 2개 gate + hidden 단일로 단순. 파라미터 GRU가 25% 적음, 학습 약간 빠름. 실증적으로는 task에 따라 다르지만 큰 차이 없다는 보고가 多. 데이터 적고 빨라야 하면 GRU, 매우 긴 의존성 + 큰 데이터면 LSTM이 약간 우위. 실무에선 둘 다 시도해 validation으로 결정."

---

## 7. LSTM의 변종들

### 7.1 Peephole LSTM

기본 LSTM의 gate는 $h_{t-1}$과 $x_t$만 본다. Peephole은 cell state $c_{t-1}$도 gate 계산에 사용:

$$f_t = \sigma(W_f [h_{t-1}, x_t, c_{t-1}] + b_f)$$

직관: "현재 cell 상태도 gate 결정에 참여."

큰 효과는 아니지만 일부 task에서 약간 우위 보고.

### 7.2 Coupled LSTM

Forget·input gate를 묶음: $i = 1 - f$.

파라미터 약간 절약, 표현력 약간 손해. GRU의 정신과 비슷.

### 7.3 ConvLSTM

Gate 계산을 fully connected가 아닌 convolution으로:

$$f_t = \sigma(\text{conv}(W_f, [h_{t-1}, x_t]) + b_f)$$

영상 시퀀스 처리에 적합 (각 시점이 2D 이미지). 영상 예측, 강수 예측 등.

### 7.4 Layer Normalization in LSTM

Gate 계산 전 input과 hidden을 LayerNorm. 학습 안정화. 깊은 LSTM에서 효과.

```
combined = LN(W [h_{t-1}, x_t])
f_t = sigmoid(combined.split(...))
```

### 7.5 Highway LSTM

Skip connection 추가:
$$h_t = \alpha h_{t-1} + (1-\alpha) \text{LSTM}(x_t, h_{t-1})$$

$\alpha$는 학습된 gate. ResNet 식 정보 흐름.

### 7.6 어느 변종을 쓰나?

대부분 task에서 표준 LSTM/GRU로 충분. 변종은 specific 상황 (영상 시퀀스, 매우 깊은 RNN 등)에 한정.

---

## 8. LSTM의 한계와 Attention의 등장

### 8.1 Long-range의 한계

LSTM이 RNN보다 훨씬 잘 long-term 잡지만 한계 있음:

- 100~200 step 정도까지는 잘.
- 1000+ step은 어려움.
- 이론상 가능하지만 학습이 *그 영역*에 도달 안 함.

### 8.2 Sequential 처리의 한계

LSTM도 RNN처럼 *시점 t를 계산하려면 t-1이 끝나야*. 병렬 불가.

큰 모델·데이터 시대에 이게 치명적:
- GPU 활용 못함.
- 학습 속도가 시퀀스 길이에 비례.
- Foundation model (수십억 파라미터)에 비현실적.

### 8.3 Hidden 크기 정보 병목

Cell state도 고정 차원. 매우 긴 시퀀스 + 풍부한 정보 → 압축 한계.

### 8.4 Attention의 답

Attention이 이 셋 모두 해결:

(1) **Long-range**: 거리 무관 직접 연결.
(2) **병렬**: 모든 시점 동시 계산.
(3) **정보 capacity**: Hidden state 단일이 아닌 *모든 토큰의 representation* 유지.

LSTM + Attention (Bahdanau 식) → 점진적으로 LSTM 비중 줄어듦 → Transformer (2017) — LSTM 완전 제거.

NLP의 RNN→LSTM→Transformer 흐름은 *각 단계의 한계가 다음 단계의 동기*가 된 좋은 예시.

### 8.5 그래도 LSTM의 자리

큰 NLP는 Transformer로 갔지만 LSTM은:
- **메모리 제약**: $O(T^2)$ Transformer 부담스러운 환경.
- **Streaming**: incremental processing이 자연.
- **단순 시계열**: 강한 trend·seasonality.
- **이론 연구**: gating의 본질 이해.

또 최근 RNN의 재부활 시도 (RWKV, Mamba 같은 SSM). Transformer의 quadratic 한계를 극복하려는 노력.

---

## 9. 면접 단골 Q&A

### Q1. LSTM이 RNN의 vanishing을 어떻게 완화?
"Cell state의 덧셈 update가 핵심. $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. 미분이 $\partial c_t/\partial c_{t-1} = f_t$ — 행렬 곱이 아닌 element-wise scalar 곱. RNN의 행렬 거듭제곱 대신 forget gate만 곱해지므로, gate가 1 근처를 학습하면 gradient가 시간 축에 거의 그대로 흐름. ResNet skip connection과 본질 같은 정신 — 곱셈 누적을 덧셈 경로로 우회. 단 수천 step에선 여전히 약점이라 attention 필요."

### Q2. LSTM의 각 gate 역할?
"Forget: 과거 cell 중 무엇을 버릴지. Input + candidate: 새 정보 중 무엇을 추가할지. Output: cell의 어느 부분을 외부에 노출할지. Forget이 가장 중요 — 없으면 정보 포화. Input은 선택적 기록. Output은 task-relevant 정보만 노출. 세 gate가 분리되어 *상황별 다른 동작* 가능. 이게 단순 RNN의 *모든 정보를 같은 방식으로 통합*과 본질 차이."

### Q3. Forget gate bias를 1로 init하는 이유?
"학습 초기에 forget gate가 1 근처 (정보 보존). 이게 두 효과. (1) 학습 안정 — 초기엔 'cell state를 거의 그대로' 흐르게 해서 학습 신호가 잘 전달. (2) 학습 동역학 — 어떤 정보를 잊을지를 점진적으로 학습. Bias 0으로 init하면 forget gate가 0.5 근처, 매 시점 절반 정보 손실. 학습이 어렵고 long-term 못 잡음. 이 단순한 변경이 큰 차이."

### Q4. LSTM vs GRU 어느 쪽?
"거의 동급입니다. LSTM은 3 gate + cell/hidden 분리, GRU는 2 gate + hidden 단일. 파라미터 GRU가 ~25% 적음, 학습 약간 빠름. 실증적으로 task별 차이 — 데이터 적고 빨라야 하면 GRU, 매우 긴 의존성 + 큰 데이터면 LSTM. 실무에선 둘 다 시도해 validation으로 결정. NLP 큰 모델은 어차피 Transformer라 둘 사이 선택 자체가 작은 영역."

### Q5. Output gate 없으면?
"Cell의 모든 정보가 외부에 노출. 다음 시점 gate 계산 + 외부 출력이 같은 정보. Task-무관 정보까지 출력으로 흘러 학습 신호 흐려짐. 단 상대적으로 덜 치명적 — GRU가 output gate 없이도 잘 작동. Forget이 가장 중요, output은 거의 없어도 됨. 이게 GRU의 디자인 철학."

### Q6. LSTM의 cell state와 hidden state의 차이?
"Cell state $c_t$: 장기 기억. 시간 축에 덧셈으로 흐름 — vanishing 약함. 모든 정보 보존. Hidden state $h_t$: 단기 기억 + 외부 출력. Cell의 일부만 노출 (output gate). 이 분리가 LSTM의 핵심. Cell은 정보 보존 전용, hidden은 외부 출력 전용. 두 역할을 한 state가 동시에 하면 task-relevant 정보 노출과 정보 보존이 충돌."

### Q7. LSTM의 cell state가 ResNet skip connection과 같다는 의미?
"두 디자인 모두 *곱셈 누적 → 덧셈 경로*의 정신. ResNet: $h_{l+1} = h_l + F(h_l)$, $\partial h_{l+1}/\partial h_l = I + \partial F$. LSTM: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$, $\partial c_t/\partial c_{t-1} = f_t$. 둘 다 identity 항이 있어 gradient가 직접 흐름. ResNet은 공간 축에서, LSTM은 시간 축에서. 다른 자리지만 같은 정신. Vanishing gradient의 chain rule 곱셈 누적 문제를 해결하는 일반 원리."

### Q8. GRU의 reset gate가 LSTM에 없는 새 메커니즘?
"Reset gate $r_t$는 새 candidate $\tilde{h}_t$를 만들 때 *과거 hidden의 영향을 얼마나 줄일지* 결정. $\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t])$. $r_t \approx 0$이면 candidate가 거의 $x_t$만 기반 — '컨텍스트 reset'. LSTM에 직접 대응 없음 (forget gate는 cell update에서만 작동, candidate에는 영향 없음). 이게 GRU만의 디자인 — 약간 다른 정보 흐름. 실용적 의미는 task별."

### Q9. LSTM 학습이 RNN보다 안정적인 이유?
"Vanishing 약화로 gradient가 잘 흐름 + gate들이 학습 동역학 부드럽게. Forget bias=1 init이 학습 시작점에서 안정. 또 cell·hidden 분리로 외부 출력과 정보 보존이 분리 → 한쪽 영향이 다른쪽으로 cascade 안 함. RNN은 모든 신호가 한 hidden을 통해 → unstable. LSTM의 architectural redundancy가 학습 안정성에 기여."

### Q10. LSTM도 결국 한계가 있는 이유?
"세 한계. (1) Long-range — 100~200 step 정도까지. 1000+ step은 여전히 약함. Forget gate를 정확히 1로 학습 어려움, 작은 오차 누적. (2) Sequential 처리 — 시점 t가 t-1 의존, 병렬 불가. 큰 모델·데이터 시대에 치명적. (3) Hidden 크기 정보 병목. 이 셋이 attention/Transformer 등장의 동기. NLP 큰 모델 영역은 Transformer로 갔음."

### Q11. LSTM의 forget gate가 항상 1 근처면 좋은가?
"아니. forget gate는 *상황에 맞게* 동작해야. 새 sentence 시작에서 이전 sentence 정보 일부 잊기. 컨텍스트 전환 인식. 항상 1이면 모든 정보 누적 → 결국 cell state 폭발. 핵심은 forget이 *task-relevant 정보를 보존*, *무관 정보를 잊음*. 학습이 이 균형을 맞춤. 단 학습 *초기*에 1 근처가 좋음 — 그래야 학습 신호가 잘 흐르고 점진적으로 'forget 어디서?' 학습."

### Q12. LSTM에 dropout 적용?
"수직 (layer 방향)만 적용 표준. 시간 방향 dropout은 hidden state 흐름을 끊어 학습 어려움. 변종 'Variational Dropout'은 시퀀스 내내 같은 mask 사용 — 시간적 일관성. 또는 weight tying, embedding dropout 등. RNN-specific regularization은 까다로움. PyTorch의 nn.LSTM(dropout=0.5)는 layer 사이만 적용."

### Q13. ConvLSTM의 동기와 사용처?
"기본 LSTM은 1D 입력 가정. 영상 시퀀스 (video, satellite imagery)는 각 시점이 2D 이미지. Fully connected gate가 큰 파라미터 + 위치 정보 잃음. ConvLSTM은 gate 계산을 conv로 — spatial 구조 보존. 강수 예측, 영상 예측, 동작 인식에 사용. CNN의 공간 inductive bias + LSTM의 시간 처리 결합."

### Q14. LSTM이 NLP의 큰 모델에서 왜 사라졌는가?
"세 이유. (1) Sequential 처리 — 시점 t가 t-1 의존. 병렬 불가 → GPU 활용 못함. 큰 모델 학습 비현실. (2) Long-range — 100+ step 어려움. 긴 문서 이해 한계. (3) Scaling 한계. 모델·데이터 키워도 한계점 빨리 옴. Transformer는 이 셋 모두 해결 — 병렬 + 거리 무관 + 잘 따라가는 scaling. NLP foundation model (BERT, GPT) 시대에 LSTM 자리 없음."

---

## 10. 생각해보라 — 단락 답안

**Q. LSTM의 cell state가 *덧셈*인 게 왜 그렇게 강력한가?**

수학적 관점: chain rule의 곱셈 누적이 vanishing의 본질. $\delta_l = \delta_{l+1} \cdot J_{l+1}$의 형태. $J$의 norm이 < 1이면 exponential decay.

덧셈 구조: $c_t = c_{t-1} + \Delta_t$. 미분 $\partial c_t / \partial c_{t-1} = I$ (identity). 곱이 1 근처라 누적해도 안 사라짐.

이 차이가 *지수적 vs 선형* 차이. 선형 누적은 t step에서 $O(t)$, 지수는 $O(c^t)$. 100 step에서 $0.5^{100} \approx 10^{-30}$ vs 100. 비교 불가의 차이.

이게 ResNet, LSTM, Transformer의 residual connection — 모든 modern deep learning이 이 정신을 채택. *덧셈으로 정보 흐름*은 단순한 트릭이 아닌 *깊은 학습의 핵심 원리*.

**Q. Forget gate가 task-별로 *다른 패턴*을 학습한다는 의미?**

LSTM의 cell state는 다차원. 각 차원이 *독립적인 forget gate*. 학습으로 각 차원이 다른 시간 스케일을 갖게 된다.

예: 언어 모델링.
- 어떤 차원: 짧은 trigger 정보 (직전 단어 영향). Forget 강함.
- 어떤 차원: 문장 전체 컨텍스트 (주어, 시제). Forget 중간.
- 어떤 차원: 문서 전체 주제. Forget 매우 약함.

이 다양한 시간 스케일을 *동시에* 모델링이 LSTM의 본질적 강점. 기본 RNN은 모든 차원이 같은 행렬 공유 → 시간 스케일 통합. LSTM은 차원별 독립.

연구에서 학습된 LSTM의 차원들을 분석하면 실제로 이런 패턴이 보임 (Karpathy의 RNN visualization).

**Q. GRU가 LSTM보다 표현력 약간 덜한 이유?**

GRU의 update gate $z$는 forget·input을 묶음 ($f = 1-z, i = z$). LSTM은 둘 *독립*.

독립의 의미: LSTM은 "정보 일부 보존 + 새 정보 일부 추가"가 동시에 가능. 예: $f = 0.7, i = 0.5$ — 70% 보존하면서 50% 새 정보 추가 (cell이 폭발할 수 있지만 가능).

GRU는 둘 합 = 1 강제. "70% 보존 + 30% 새 정보" 만 가능. 합이 1이라는 제약.

이 차이가 큰 영향은 아닌 듯 — 실제 학습에서 LSTM도 $f + i \approx 1$ 근처를 자주 학습. 그래서 GRU가 단순화로도 거의 동급.

또 GRU에는 LSTM에 없는 reset gate $r$. 이게 candidate 만들 때 다른 효과. 그래서 *완전 단순화*는 아니고 *다른 균형*.

**Q. LSTM이 Transformer 시대에도 살아남는 이유?**

세 자리:

(1) **메모리 제약**: Transformer의 self-attention은 $O(T^2)$ 메모리. 긴 시퀀스 (수만 토큰)에서 부담. LSTM은 $O(T)$. Edge device, 모바일.

(2) **Streaming**: Real-time 음성 인식, 동영상 처리. 입력이 점진적으로 도착. LSTM은 자연스럽게 *incremental*. Transformer는 매번 모든 컨텍스트 재계산.

(3) **단순 시계열**: 강한 trend·seasonality가 있는 시계열. RNN/LSTM이 충분, Transformer는 overkill.

또 최근 RNN의 *부활* — RWKV, Mamba 같은 SSM (State Space Model). Transformer의 $O(T^2)$ 한계를 극복하려는 시도. 본질적으로 RNN과 비슷한 정신 (recurrent state).

미래는 hybrid — 짧은 시퀀스엔 Transformer, 긴 시퀀스엔 RNN-like. 이미 진행 중.

**Q. LSTM의 hidden과 cell의 분리가 *없다면* 어떻게 될까?**

GRU가 정확히 그 case. Hidden state 하나로 *모든 일* — 정보 보존 + 외부 출력 + 다음 시점 gate 계산.

문제는 *출력 ≠ 보존*. Task-relevant 정보 (분류용)와 다음 시점 처리에 필요한 정보 (모든 컨텍스트)가 다를 수 있음. 한 state가 둘 다 하려면 *항상 모든 정보 노출* — task에 noisy.

LSTM의 분리는 *수직* 분업. Cell은 정보 보존 전용, hidden은 외부 출력 전용 (output gate로 cell의 일부만). 깔끔.

GRU는 분리 없이도 작동 — 그렇다는 게 실증. 단 *완전 동급*은 아니고 약간 다른 trade-off. LSTM의 정교함 vs GRU의 단순함.

**Q. LSTM이 매우 긴 (10,000+ step) 시퀀스에선 왜 안 되는가?**

Forget gate가 *정확히 1*을 학습하기 매우 어려움. 학습된 forget gate는 보통 0.95~0.99. 작아 보이지만 누적되면:
- 100 step: $0.95^{100} \approx 0.006$
- 1000 step: $0.95^{1000} \approx 10^{-23}$

Sigmoid의 saturation 영역에서 미분이 작아 정확한 1 학습 어려움. 이론상 가능, 학습 동역학으론 어려움.

또 cell state 차원 고정. 10,000 step의 정보를 100차원에 압축 — 정보 밀도 100. 손실 누적.

처방:
- **Hierarchical LSTM**: 시간 스케일 다른 여러 layer.
- **Attention 추가**: long-range를 attention으로.
- **Transformer**: long-range 자체에 강함.

매우 긴 시퀀스의 본질적 어려움은 architecture 변경 외엔 답 없음.

**Q. LSTM의 *생물학적 타당성*?**

LSTM이 발표된 후 신경과학자들의 평가는 "*기능적으론 비슷*하지만 *메커니즘은 다름*."

기능적 유사성:
- 작업 기억 (working memory)의 modeling.
- 장기 보존 + 단기 처리의 분리.
- 게이팅 메커니즘 (자극에 따라 정보 흐름 조절).

생리학적 차이:
- 실제 뉴런은 spiking + analog. LSTM은 continuous.
- Backprop은 비생물학적. 뇌는 다른 방식 (Hebbian, predictive coding).
- LSTM의 정확한 식이 뇌에 없음.

LSTM은 *공학적 솔루션*이지 *뇌 모델*이 아님. 기능적 영감은 있지만 메커니즘은 다름. 이게 ML과 신경과학의 일반적 관계.

**Q. GRU의 reset gate가 *없다면*?**

Update gate만 있고 reset 없으면:
$\tilde{h}_t = \tanh(W [h_{t-1}, x_t])$ — 새 candidate가 항상 *전체* $h_{t-1}$ 영향.

문제: 컨텍스트 reset 어려움. 새 sentence 시작에서 이전 sentence의 hidden을 *완전 무시*하고 시작하기 어려움. Update gate $z=1$로 새 정보 흡수해도 candidate 자체가 과거 영향.

Reset gate가 candidate에 *부분적 무시*. $r=0$이면 candidate = $x_t$만 기반.

이게 LSTM에는 *직접 대응*이 없는 메커니즘. LSTM의 forget이 cell update에서만 작동, candidate에는 영향 없음. GRU의 reset이 *다른 자리*에서 비슷한 일 (정보 reset).

실용적으로 reset gate의 영향은 task별. 어떤 task는 큰 차이, 어떤 건 거의 없음.

---

## 11. 한 줄 요약 (시험 직전)

- LSTM의 핵심 = **cell state + 3 gate**. Cell은 덧셈 update로 vanishing 완화.
- **각 gate의 역할**: forget (잊기), input (기록), output (노출).
- **Forget gate가 가장 중요** — 없으면 정보 포화. Bias=1 init.
- **Vanishing 완화 원리**: $\partial c_t/\partial c_{t-1} = f_t$. 행렬 곱이 아닌 scalar 곱. 덧셈 경로.
- **ResNet skip과 본질 같음** — 곱셈 누적을 덧셈으로 우회. 다른 자리, 같은 정신.
- **GRU = LSTM 단순화**. 2 gate (update, reset), cell 없음. 파라미터 ~25% 적음.
- **LSTM vs GRU**: 거의 동급. Task별 차이. 둘 다 시도가 정석.
- **LSTM의 한계** = 수천 step long-range, 병렬 불가, hidden 크기 병목. → Attention/Transformer.
- **NLP 큰 모델은 Transformer**, LSTM은 메모리 제약·streaming·단순 시계열에 자리.
- **차원별 다른 시간 스케일** 학습 — LSTM의 표현력의 본질.
