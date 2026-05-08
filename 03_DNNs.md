# 03. Deep Neural Networks (DNN) — 오픈북 정리

> 핵심: **"왜 깊게 가면 학습이 안 되는가? 그걸 어떻게 풀었는가?"**
> DNN 챕터는 사실상 *학습 가능하게 만드는 기술 모음*.

---

## 0. "왜 deep?"의 답을 다시

- **표현력**: 같은 함수, 적은 파라미터.
- **계층적 추상화**: 이미지에서 edge → texture → part → object.
- **실증적 우수성**: 충분한 데이터·연산이 있을 때 shallow를 압도.

그럼에도 불구하고 깊이는 **공짜가 아니다**. 아래 문제들이 발생.

---

## 1. Vanishing / Exploding Gradient — "왜 발생?"

backprop의 gradient는 **층마다 곱**으로 누적:
$$\delta^{(l)} \propto \prod_{k=l+1}^{L} W^{(k)} \sigma'(z^{(k)})$$

- 곱하는 값이 < 1이면 → 0으로 수렴(**vanishing**).
- > 1이면 → 발산(**exploding**).

**왜 sigmoid/tanh에서 심한가?** → σ'의 최대가 0.25. 깊이 L에서 0.25^L → 0.

### 1.1 처방

| 문제 | 처방 | 메커니즘 |
|---|---|---|
| Vanishing | **ReLU 계열** | 양수 영역에서 σ'=1 |
| | **He init** | 분산을 깊이에 맞게 |
| | **Batch Norm** | 활성값 분포 안정 |
| | **Skip connection (ResNet)** | gradient 우회로 |
| Exploding | **Gradient clipping** | norm > τ면 잘라냄 |
| | **적절한 init** | 분산 폭발 방지 |

생각해보라: **왜 ResNet이 vanishing을 풀었나?** → $h_{l+1} = h_l + F(h_l)$. backward에서 $\partial h_{l+1}/\partial h_l = I + \partial F/\partial h_l$. **identity 항** 덕에 gradient가 직접 흐른다.

---

## 2. Weight Initialization — "왜 init이 중요?" (★)

처음 가중치가 너무 크면 → 활성값 폭발 → exploding.
너무 작으면 → 활성값 0 수렴 → vanishing.
모두 0이면 → 모든 뉴런이 동일하게 학습 → **symmetry breaking 실패**.

| Init | 공식 | 가정 |
|---|---|---|
| **Xavier (Glorot)** | Var(W) = 2/(n_in + n_out) | tanh, sigmoid (대칭) |
| **He (Kaiming)** | Var(W) = 2/n_in | ReLU (음수 절반 죽음 보정) |
| LeCun | Var(W) = 1/n_in | SELU |

**왜 He는 2/n_in?** → ReLU가 음수 절반을 0으로 만드므로 분산이 절반으로 줄어드는 걸 보상.

---

## 3. Batch Normalization — "왜 효과가 있나?" (★★ 단골)

각 mini-batch에서 층의 입력을 **정규화 + scale/shift**:
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

### 3.1 효과 (왜 좋은가?)
1. **Internal Covariate Shift 완화** (원논문 주장) — 층별 입력 분포가 안정.
2. **Loss landscape를 평활화** (후속 연구) — 더 큰 LR 사용 가능.
3. **암묵적 regularization** — mini-batch 통계의 noise.
4. **초기화에 덜 민감**.

### 3.2 train vs eval (★실수 포인트)
- **Train**: 현재 batch의 μ, σ 사용.
- **Eval**: 학습 중 누적한 **이동평균**의 μ, σ 사용.
→ `model.eval()` 호출 안 하면 추론 결과가 batch에 따라 들쭉날쭉.

### 3.3 한계 / 변종
- 작은 batch에서는 통계가 부정확 → **GroupNorm / LayerNorm** (Transformer는 LN).
- RNN에서는 시퀀스 길이가 가변이라 BN 불편 → **LayerNorm**.

생각해보라: 왜 LayerNorm이 NLP에서 표준? → 시퀀스마다 길이가 다르고, batch 통계가 token 분포에 의존하면 불안정. LN은 sample 내부에서 정규화 → batch 무관.

---

## 4. Dropout — "왜 ensemble처럼 작동?"

학습 중 각 뉴런을 확률 p로 0으로 마스킹.
$$h' = h \odot m, \quad m_i \sim \text{Bernoulli}(1-p)$$

### 4.1 왜 효과?
- 매 step마다 다른 sub-network 학습 → **암묵적 ensemble**.
- 특정 뉴런에 의존 못하게 → **co-adaptation 방지**.

### 4.2 train vs eval (★)
- Train: mask 적용.
- Eval: mask 없음. 대신 활성값을 (1−p)배 (또는 train 시 1/(1−p)배 — inverted dropout).

생각해보라: dropout과 BN을 같은 위치에 같이 쓰면? → 분산 변동이 충돌해 성능 저하 자주 보고됨. 보통 BN을 먼저, dropout은 FC 뒤. 혹은 둘 중 하나만.

---

## 5. Regularization 종합 비교

| 방법 | 어디에 작용 | 효과 |
|---|---|---|
| L2 (weight decay) | 가중치 크기 | smooth 솔루션 |
| L1 | 가중치 sparsity | feature selection |
| Dropout | 활성값 | ensemble |
| BN | 활성값 통계 | 부수적 정규화 |
| Data augmentation | 입력 | 분포 확장 |
| Early stopping | 학습 시간 | 시간적 정규화 |
| Label smoothing | 라벨 | 과신 방지 |
| Mixup / CutMix | 입력+라벨 | smoothing |

**왜 여러 가지를 같이?** → 작용하는 축이 달라서 보완적. 단 trade-off 존재(예: 강한 augmentation + dropout이면 underfit 가능).

---

## 6. Optimizer — "왜 SGD에서 Adam까지 발전?"

### 6.1 SGD
$$\theta \leftarrow \theta - \eta g$$
단순. noise 덕에 일반화 좋다는 보고 있음. **단점**: 골짜기 진동, 평탄 영역에서 느림.

### 6.2 Momentum
$$v \leftarrow \beta v + g, \quad \theta \leftarrow \theta - \eta v$$
이전 gradient를 누적 → **관성**. 골짜기에서 진동을 줄이고 가속.

### 6.3 Nesterov Accelerated Gradient
$$v \leftarrow \beta v + \nabla L(\theta - \eta \beta v), \quad \theta \leftarrow \theta - \eta v$$
"미리 가본" 위치에서 gradient. 더 빠른 수렴.

### 6.4 Adagrad
$$\theta \leftarrow \theta - \frac{\eta}{\sqrt{\sum g^2 + \epsilon}} g$$
파라미터별 LR. **장점**: sparse feature에 유리. **단점**: LR이 단조감소 → 학습 멈춤.

### 6.5 RMSProp
$$E[g^2] \leftarrow \beta E[g^2] + (1-\beta) g^2$$
Adagrad의 누적을 **이동평균**으로. 학습 멈춤 문제 해소.

### 6.6 Adam (★표준)
Momentum + RMSProp:
$$m \leftarrow \beta_1 m + (1-\beta_1) g$$
$$v \leftarrow \beta_2 v + (1-\beta_2) g^2$$
$$\hat{m} = m/(1-\beta_1^t), \quad \hat{v} = v/(1-\beta_2^t)$$
$$\theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{v}}+\epsilon} \hat{m}$$

**왜 Adam이 기본?** → 대부분 task에서 빨리 수렴, LR 튜닝에 덜 민감.

**왜 그래도 SGD+momentum을 쓰는 사람이 있나?** → ImageNet 같은 task에서 최종 일반화 성능이 더 좋다는 보고. Adam은 sharp minimum으로 가는 경향.

### 6.7 AdamW
Adam의 weight decay를 정확히 분리. **현대 표준** (특히 Transformer).

---

## 7. Learning Rate Schedule

| 방식 | 패턴 | 언제? |
|---|---|---|
| Step decay | 일정 epoch마다 1/n | 단순 |
| Exponential | LR · γ^t | 부드러움 |
| Cosine annealing | 코사인 곡선 | Transformer 표준 |
| Warmup | 작은 LR로 시작 → 본 LR | 큰 모델, Adam과 함께 |
| OneCycle | warmup → 큰 LR → 감소 | 빠른 학습 |

**왜 warmup?** → 학습 초기에 BN 통계, Adam의 second moment 추정이 불안정 → 큰 LR 위험.

---

## 8. Skip Connection / Residual Learning

### 8.1 동기
**왜 깊을수록 train accuracy도 떨어지나?** (degradation problem)
→ 단순 깊이 추가가 오히려 최적화 어렵게 만듦.

### 8.2 ResNet의 핵심
$$h_{l+1} = h_l + F(h_l)$$
- F가 0을 학습하면 identity → 최소한 더 나빠지지 않음.
- gradient가 직접 흐름 → vanishing 완화.

**왜 ResNet이 혁명?** → 100층 이상 학습 가능. 후속 모델(DenseNet, Transformer 등)도 모두 skip connection 사용.

---

## 9. 정규화 위치 (BN vs LN vs GN)

| | 정규화 축 | 사용처 |
|---|---|---|
| **BatchNorm** | batch 차원 | CNN, 큰 batch |
| **LayerNorm** | feature 차원 (sample 내) | RNN, Transformer |
| **GroupNorm** | feature를 그룹으로 묶음 | 작은 batch CNN |
| **InstanceNorm** | sample 내 channel별 | 스타일 변환 |

생각해보라: 왜 Transformer는 LN? → 시퀀스 길이 가변, batch 통계 의존 회피, 자기회귀 추론 시 일관성.

---

## 10. 왜 깊이가 충분히 깊으면 좋은가? (실증)

- ImageNet에서 AlexNet(8) → VGG(19) → GoogLeNet(22) → ResNet(152) → 성능 단조 향상.
- 깊이는 **계층적 추상화**의 수단이지 목적이 아니다. 데이터·계산·정규화가 함께 따라야.

---

## 11. 학습 진단 (왜 안 되는가?)

| 증상 | 원인 후보 | 처방 |
|---|---|---|
| Loss NaN | LR 큼, log(0), unstable softmax | LR↓, log_softmax, eps |
| Gradient norm 폭발 | exploding | clipping, init 점검 |
| Gradient 0 | vanishing, dead ReLU | ReLU→LeakyReLU, He init, ResNet |
| Train↓ Val↑ | overfit | dropout, weight decay, aug |
| Train도 안 떨어짐 | 모델 작음, LR 부적절 | 키우기, LR 탐색 |
| BN 추론 불안정 | eval 모드 안 함 | model.eval() |

---

## 12. 실제 적용 시 표준 stack (2020s)

- 활성: ReLU / GELU
- Init: He / Xavier
- Norm: BN (CNN) / LN (Transformer)
- Optimizer: AdamW + cosine + warmup
- Regularization: dropout + weight decay + augmentation
- Skip connection 기본 탑재

---

## 13. 한 줄 요약

- DNN의 핵심 어려움 = **vanishing/exploding gradient**.
- ReLU + He init + BN + ResNet의 조합으로 **깊이 학습 가능**.
- Dropout은 **암묵적 ensemble**, BN은 **분포 안정 + 약한 정규화**.
- Adam은 momentum + adaptive LR, AdamW가 현재 표준.
- LR schedule은 **warmup + cosine**이 큰 모델에 유리.
- **모든 기법은 "학습이 잘 되게 + 일반화 잘 되게"라는 두 축에서 본다.**
