# Optimizer hyperparameters — Momentum β, Adam (β₁ / β₂ / ε / LR)

> 본문 §3 Gradient Descent의 optimizer 보강. SGD+momentum과 Adam의 내부 파라미터들. 본문에서는 한 줄 요약만 두고 여기서 깊게.

---

## Momentum β

**위치 — 본문 §3.5 Momentum**

순수 SGD에 관성을 더해 골짜기 진동을 완화하는 기법:

$$v_t = \beta v_{t-1} + \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

이전 update의 일부를 누적. 같은 방향이 반복되면 가속, 진동하면 상쇄.

### 표준값

- $\beta = 0.9$ — 거의 universal default. 한 step의 update가 약 10번 step의 효과를 누적하는 셈 (effective horizon $\approx 1/(1-\beta)$).
- $\beta = 0.99$ — 더 강한 관성. 매우 noisy한 환경(작은 batch, 큰 LR)에서 안정화에 도움. 단 방향 바뀜에 반응이 느림.
- $\beta = 0$ — 순수 SGD. momentum 없음.

### 언제 키우고 줄이는가

- 학습 곡선이 진동·zigzag → β 키움 (안정화).
- 학습이 minimum을 지나쳐 발산하거나 너무 느리게 반응 → β 줄임.
- 일반적으로 0.9가 sweet spot이라 거의 안 건드린다. *튜닝 우선순위는 LR > batch size ≫ β*.

### 함정

- LR schedule과 같이 쓸 때 — warmup 끝난 후 LR이 크게 떨어지면 momentum이 누적된 큰 update를 가져와 발산 가능. β를 약간 줄이거나 update를 클리핑.

---

## Adam

**위치 — 본문 §3.6 Adaptive LR**

RMSProp + momentum의 조합. 신경망 시대의 표준 optimizer:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1-\beta_1^t), \quad \hat{v}_t = v_t / (1-\beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

네 개의 hyperparameter — β₁, β₂, ε, LR.

### 표준값

각각 빈 줄로 분리해 italic 매칭 회피:

- $\beta_1 = 0.9$ — 1차 moment(평균)의 EMA 계수. momentum의 β와 같은 역할이라 같은 값.

- $\beta_2 = 0.999$ — 2차 moment(분산)의 EMA 계수. 매우 크게 잡아 LR adaptation을 부드럽게.

- $\epsilon = 10^{-8}$ — division 안정화. *하지만 큰 모델에서는 1e-6 또는 1e-4까지 키운다* (Transformer가 대표) — 작은 분산 영역에서 effective LR이 폭발하는 걸 막기 위해.

- LR $\eta = 10^{-3}$ — 거의 universal 시작점. 큰 모델은 1e-4, fine-tuning은 1e-5 정도.

### 왜 그 값인가

$\beta_2$를 0.999로 크게 잡는 이유는 2차 moment가 천천히 적응해야 LR이 gradient의 *최근 분산*을 매끄럽게 따라가기 때문. 너무 작으면 LR이 noise에 끌려가 학습 불안정.

$\beta_2 = 0.999$의 effective horizon은 약 1000 step. 학습이 그보다 짧으면 2차 moment 추정이 부정확해 **warmup이 사실상 필수** (본문 §3.8).

### 함정

- LR을 그대로 두고 batch size를 키우면 학습 불안정. linear scaling rule이 Adam에서도 거의 통하지만 매우 큰 batch에선 깨진다.
- AdamW가 사실상 표준 — `Adam(weight_decay=...)` 대신 `AdamW(weight_decay=...)` 권장 (본문 §3.7).

### 왜 Adam이 좋은가

파라미터별 effective LR이 자동 조정되고, momentum도 들어 있고, hyperparameter 튜닝 부담이 적다. "잘 모르겠으면 Adam"이 거의 default 조언이 됐다.

### 왜 Adam이 항상 최선이 아닌가

ImageNet 같은 task에서 SGD+momentum이 약간 더 일반화 잘하는 보고가 많다. Adam이 sharp minima로 가는 경향이라는 분석. 또 Adam의 실제 weight decay 처리가 부정확해서 AdamW가 등장했다.

---

## 본문 연결

- §3.5 Momentum → Momentum β
- §3.6 Adaptive LR → Adam (β₁ / β₂ / ε / LR)
- §3.7 AdamW (weight decay 정확한 처리)도 함께 참고
