# 03. Deep Neural Networks (DNN) — 심층 정리

> **이 문서의 목표**
> 깊은 신경망을 *학습 가능하게 만든 기법들*의 체계. Vanishing gradient를 어떻게 우회하는가, BN/Dropout이 왜 효과적인가, optimizer는 어떤 진화를 거쳤는가, ResNet의 skip connection이 왜 혁명이었는가 — 이 질문들에 *수식과 직관 모두로* 답할 수 있게 만든다.
>
> 이 챕터의 핵심 통찰: **DNN은 사실상 "vanishing gradient를 어떻게 풀 것인가"의 답들의 집합**이다. ReLU, He init, BN, Dropout, Skip connection, Adam — 모두 이 한 문제에 대한 다른 각도의 답이다.

---

## 0. 큰 그림 — "왜 깊으면 안 되는가, 그리고 어떻게 풀었나"

### 0.1 깊이의 약속과 배신

1980년대 backprop이 발명되었을 때 사람들은 곧 깊은 신경망의 시대가 올 것으로 기대했다. 깊이 = 표현 효율 = 더 좋은 모델. 단순한 논리. 하지만 1990~2000년대 내내 깊은 망은 학습이 안 됐다. 3~5층이 한계였다.

이 시기를 두 번째 AI 겨울이라 부른다. 신경망은 SVM, Random Forest 같은 전통 ML 기법에 밀려났다. 깊이가 약속한 표현력은 학습 가능성이라는 벽 앞에서 공허한 약속이 되었다.

이걸 깬 것이 2010년대 일련의 발견. 그 발견들이 무엇이고, 왜 그것들이 깊이의 문제를 풀었는지 — 이게 이 챕터의 본질.

### 0.2 깊이의 두 적: Vanishing & Exploding Gradient

Backprop의 식을 다시 보자:

$$\delta^{(l)} = \prod_{k=l+1}^{L} W^{(k)} \cdot \sigma'(z^{(k-1)})$$

(엄밀하지 않게 표현. 행렬 곱셈과 element-wise multiplication의 혼합)

깊이 $L$에서 출력의 gradient가 입력으로 거슬러 갈 때 매 layer마다 $W \cdot \sigma'$이 곱해진다. 이 product가:

- **< 1이면** $\delta^{(l)}$이 0으로 수렴 → **Vanishing**. 얕은 layer가 학습 안 됨.
- **> 1이면** $\delta^{(l)}$이 폭발 → **Exploding**. NaN, 발산.

이게 깊이의 본질적 문제. **Layer 수에 비례한 곱셈은 본질적으로 불안정**하다.

### 0.3 해결책의 큰 그림 — 다섯 축

깊은 망 학습을 가능케 한 기법들은 다섯 축으로 정리된다:

| 축 | 기법 | 어떻게 vanishing 해결 |
|---|---|---|
| 활성화 | ReLU 계열 | $|\sigma'|$의 곱셈을 1로 (활성 영역에서) |
| 초기화 | He, Xavier | $\|W\|$의 분산을 적절히 |
| 정규화 | BN, LN, GN | 매 layer 활성값 분포 안정 |
| 구조 | Skip connection (ResNet) | 곱셈 대신 덧셈 경로 |
| 최적화 | Adam, momentum | 각 파라미터별 effective LR 조정 |

이 다섯이 동시에 작동해야 깊은 망이 학습 가능. 하나만 빠져도 종종 망함. 그래서 modern deep learning은 이 다섯 모두의 기본 stack을 깐다.

### 0.4 Regularization의 자리

위에서 언급 안 했지만 Regularization (dropout, weight decay 등)도 핵심. 깊은 망은 *capacity가 매우 크다* — overfitting 위험. 학습 가능성은 ReLU+BN+ResNet으로, 일반화는 dropout+weight decay+augmentation으로.

학습 가능성과 일반화는 별개의 차원. 둘 다 챙겨야 깊은 망이 production에 갈 수 있다.

---

## 1. Vanishing Gradient — 수식적 분석

### 1.1 정확한 식

Layer $l$의 delta:

$$\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}$$

Chain rule:

$$\delta^{(l)} = \delta^{(l+1)} \cdot \frac{\partial z^{(l+1)}}{\partial z^{(l)}}$$

$z^{(l+1)} = W^{(l+1)} h^{(l)} + b^{(l+1)} = W^{(l+1)} \sigma(z^{(l)}) + b^{(l+1)}$이므로:

$$\frac{\partial z^{(l+1)}}{\partial z^{(l)}} = W^{(l+1)} \cdot \text{diag}(\sigma'(z^{(l)}))$$

따라서:

$$\delta^{(l)} = \delta^{(l+1)} \cdot W^{(l+1)} \cdot \text{diag}(\sigma'(z^{(l)}))$$

Layer $L$에서 $l$로 거슬러 가면:

$$\delta^{(l)} = \delta^{(L)} \prod_{k=l+1}^{L} W^{(k)} \text{diag}(\sigma'(z^{(k-1)}))$$

이 product의 크기를 추적하는 게 핵심.

### 1.2 Sigmoid에서의 분석

$\sigma'(x) = \sigma(x)(1 - \sigma(x))$. 최대값은 $x=0$에서 $0.25$. 일반적으로:

$$\mathbb{E}[\sigma'(z)] \le 0.25$$

가중치를 적절히 init했다면 $\|W\| \approx 1$. 그럼 매 layer마다 곱이 약 0.25.

10층이면 $0.25^{10} \approx 10^{-6}$. 100층이면 $10^{-60}$. 사실상 0.

이게 sigmoid의 vanishing gradient — 기하급수적 감쇠.

### 1.3 ReLU에서의 분석

ReLU는 $x>0$이면 미분 1, $x<0$이면 0. 활성된 뉴런 비율을 $p$라 하면 평균 미분이 $p$.

He init에서 $p \approx 0.5$. 가중치 분산이 $2/n_{in}$로 적절히 조절되어 있으면 $\|W\| \cdot p$의 평균이 1에 가깝다 (He init이 정확히 이걸 목표로).

그래서 ReLU + He init 조합에서는 gradient가 안정적으로 흐름. 곱이 1 근처라 vanishing/exploding 둘 다 약함.

이게 ReLU+He의 조합이 sigmoid를 제친 본질적 이유.

### 1.4 배치 차원에서의 안정성

위 분석은 single sample 기준. 실제로는 mini-batch에서 각 sample이 다른 활성 패턴을 보임. 평균 효과로 매 layer의 gradient norm이 더 안정적이지만, 본질적 동역학은 같다.

### 1.5 Exploding의 원인

Vanishing의 반대. 매 layer의 곱이 > 1이면 gradient가 폭발. 흔한 원인:

1. **Init이 분산 큼**: $\|W\|$가 1보다 큼. 매 layer마다 누적.
2. **활성화 미분이 큼**: ReLU는 활성 영역에서 정확히 1이지만, 어떤 활성화가 1보다 크면 발산.
3. **RNN의 시간 축**: 같은 행렬을 반복 곱 → 고유값이 1을 넘으면 즉시 폭발.

처방:
- **Gradient clipping**: norm이 임계 넘으면 잘라냄. RNN에 거의 필수.
- **적절한 init**.
- **BN/LN**: 활성값 분포 안정.

---

## 2. Weight Initialization — 시작이 절반

### 2.1 왜 init이 중요한가

학습 시작 시점의 가중치가 학습 동역학에 큰 영향. 잘못된 init은:

- **Vanishing**: 분산이 너무 작으면 gradient도 작음.
- **Exploding**: 분산이 너무 크면 gradient 폭발.
- **Symmetry 안 깨짐**: 모두 0으로 init하면 모든 뉴런이 같은 gradient → 같은 update → 영원히 같음. 단일 뉴런으로 환원.

### 2.2 Xavier (Glorot) Initialization

2010년 Glorot & Bengio. **활성값과 gradient의 분산이 layer를 통과해도 보존**되도록.

Forward 분석: layer $l$의 입력 분산 $\text{Var}(h^{(l-1)}) = 1$ 가정. $z^{(l)} = W h^{(l-1)}$의 분산:

$$\text{Var}(z^{(l)}_i) = n_{in} \cdot \text{Var}(W) \cdot \text{Var}(h^{(l-1)})$$

(가중치와 입력이 독립 가정)

$\text{Var}(z^{(l)}) = 1$ 유지 원하면 $\text{Var}(W) = 1/n_{in}$.

Backward 분석도 비슷: gradient의 분산 보존 원하면 $\text{Var}(W) = 1/n_{out}$.

둘의 절충 (조화평균 같은):

$$\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$$

이게 Xavier (또는 Glorot) initialization. **Sigmoid/Tanh** 같은 0-centered 활성화 가정.

구체적 분포:
- Uniform: $W \sim U(-\sqrt{6/(n_{in}+n_{out})}, +\sqrt{6/(n_{in}+n_{out})})$
- Normal: $W \sim \mathcal{N}(0, 2/(n_{in}+n_{out}))$

### 2.3 He (Kaiming) Initialization

2015년 He et al. **ReLU 가정**으로 Xavier 보정.

ReLU는 음수 절반을 0으로 만든다. 입력의 절반이 0이면 분산도 절반으로 줄어듦. 이걸 보상하려면 가중치 분산을 두 배로:

$$\text{Var}(W) = \frac{2}{n_{in}}$$

이게 He initialization. ReLU 신경망의 표준.

직관: "ReLU가 정보를 절반 잃으니 가중치를 더 크게 시작해서 잃는 만큼 보상."

### 2.4 LeCun Initialization

1998년 LeCun. **SELU**나 **tanh**에 적합.

$$\text{Var}(W) = \frac{1}{n_{in}}$$

He의 절반. SELU(self-normalizing ELU) + LeCun init은 BN 없이도 깊은 망 학습 가능 (실용적으론 BN이 더 안정).

### 2.5 Init 선택의 결정 트리

```
은닉층 활성화?
├── ReLU 계열 → He
├── Tanh, Sigmoid → Xavier
├── SELU → LeCun
└── 모르겠으면 → He (보통 안전)
```

PyTorch의 default는 layer 종류마다 다름. nn.Linear은 Kaiming uniform, nn.Conv2d도 Kaiming uniform. 일반적으로 default가 He인 셈.

### 2.6 Bias initialization

대부분 0으로. 예외:
- **LSTM의 forget gate bias**: 1~2로. "처음엔 정보 보존" 학습 시작점.
- **ReLU 직전 layer**: 작은 양수 (예: 0.01)도 가능 — dying ReLU 약간 막음.

### 2.7 잘못된 init의 신호

| 증상 | 원인 |
|---|---|
| 첫 forward에서 활성값이 모두 0 | Init 분산이 너무 작음 |
| 첫 forward에서 활성값이 NaN/inf | Init 분산이 너무 큼 |
| Gradient 모두 0 | Vanishing init |
| Gradient 모두 NaN | Exploding init |
| 모든 뉴런이 같은 출력 | Symmetry 안 깨짐 (예: 모두 0) |

해결: 활성화에 맞는 init 명시적 적용. PyTorch에서:

```python
for layer in model.modules():
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        nn.init.zeros_(layer.bias)
```

---

## 3. Batch Normalization — 깊이 학습의 게임 체인저

### 3.1 동기 — Internal Covariate Shift

2015년 Ioffe & Szegedy. 깊은 망의 한 가지 문제: **각 layer의 입력 분포가 학습 도중에 변한다**. 윗 layer의 가중치가 update되면 그 layer의 출력 분포가 바뀌고, 다음 layer는 매번 새 분포에 적응해야 한다. 이걸 Internal Covariate Shift (ICS)라 부른다.

이 가설이 BN의 동기였다 — "각 layer의 입력 분포를 강제로 안정화하자."

(주: 후속 연구 [Santurkar et al. 2018]가 BN의 효과는 ICS 완화보다 **loss landscape 평활화**가 본질이라 주장. 정확한 메커니즘은 아직 논쟁 중. 어느 쪽이든 BN이 작동한다는 사실은 같음.)

### 3.2 BN 연산

Mini-batch $\{x_1, ..., x_B\}$에 대해:

$$\mu_B = \frac{1}{B} \sum_i x_i$$
$$\sigma_B^2 = \frac{1}{B} \sum_i (x_i - \mu_B)^2$$
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

순서:
1. **정규화**: 평균 0, 분산 1로.
2. **스케일·시프트**: 학습 가능 파라미터 $\gamma, \beta$로 다시 변환.

왜 마지막 스케일·시프트? 정규화가 표현력을 잃을 수 있어서. 만약 정규화 자체가 좋지 않다면 $\gamma = \sigma_B, \beta = \mu_B$로 되돌릴 수도 있다 (identity 학습 가능).

### 3.3 Train vs Eval 모드

**Train**: 현재 mini-batch의 $\mu_B, \sigma_B^2$ 사용.

**Eval**: 학습 중 누적한 **이동평균** 사용:
$$\mu_{running} \leftarrow \alpha \mu_{running} + (1-\alpha) \mu_B$$

이유:
- Eval 시 batch가 1이거나 분포가 train과 다르면 batch 통계가 noisy.
- 추론에서 deterministic 결과 필요.

PyTorch의 `model.eval()`이 이 모드 전환. 잊으면 추론 결과가 들쭉날쭉.

### 3.4 BN의 효과

(1) **큰 LR 사용 가능**: Loss landscape가 평활해져 큰 LR도 안정. 학습 속도↑.

(2) **Init 민감도 감소**: 잘못된 init도 BN이 매 layer에서 분포를 정정.

(3) **암묵적 정규화**: Mini-batch 통계의 noise가 정규화 효과. Dropout 약간 대체 가능.

(4) **Vanishing gradient 완화**: 활성값 분포가 안정되면 미분도 안정.

이래서 BN은 ResNet 등장 전부터 깊은 망 학습의 핵심 도구였고, ResNet 이후에도 표준.

### 3.5 BN의 한계

(1) **작은 batch에서 약함**: B=2~4면 통계가 매우 noisy. → GroupNorm, LayerNorm으로 교체.

(2) **시퀀스 데이터 처리**: RNN에서는 시퀀스 길이가 가변, BN이 불편. → LayerNorm.

(3) **distributed 학습**: 여러 GPU에서 다른 batch 통계 → SyncBN으로 동기화 필요.

(4) **추론 latency**: BN parameter가 conv에 fold-in 가능하지만, 별도 layer로 두면 약간 느림.

### 3.6 Layer Norm — Transformer의 표준

BN과 정규화 축이 다름. BN은 batch 차원, LN은 feature 차원 (한 sample 내).

$$\hat{x}_i = \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}$$

여기서 $\mu_i, \sigma_i$는 **sample $i$의 모든 feature**에 대한 통계.

**왜 Transformer는 LN?**:
- 시퀀스 길이 가변, batch 통계가 의미 약함.
- 자기회귀 추론 시 batch=1.
- Sample 내 정규화라 batch 무관, 일관성 있음.

### 3.7 GroupNorm

BN의 작은 batch 문제 해결. Channel을 G개 group으로 묶어 group 내에서 정규화.

$G=1$이면 LN과 같음, $G$=channel 수면 InstanceNorm. 보통 $G=32$.

Detection, segmentation처럼 batch가 작은 task에서 BN보다 안정.

### 3.8 BN의 위치 — Conv-BN-ReLU vs Conv-ReLU-BN

ResNet 원본은 Conv → BN → ReLU. 일반적으로 이게 표준.

대안 Conv → ReLU → BN: 활성화 후 정규화. 성능 미미하게 다름.

Pre-activation: BN → ReLU → Conv. ResNet v2의 변종으로 더 좋다는 보고.

---

## 4. Dropout — Ensemble의 신경망 버전

### 4.1 작동 방식

Hinton 2012. 학습 중 각 뉴런을 확률 $p$로 무작위 0으로:

$$h_i' = h_i \cdot m_i, \quad m_i \sim \text{Bernoulli}(1-p)$$

Eval 시 마스킹 없음. 대신 활성값 스케일:

$$h_i' = h_i \cdot (1-p)$$

또는 train에서 inverted dropout — 마스킹 후 $1/(1-p)$로 스케일. Eval에서는 그대로. PyTorch의 default가 이거.

### 4.2 왜 효과가 있는가 — 두 해석

**해석 1 (Ensemble)**: 매 step 다른 sub-network 학습. 가중치 공유된 $2^n$개의 sub-network ($n$ = 뉴런 수)를 동시에 학습. 평가 시 사실상 평균 — exponentially many networks의 ensemble.

**해석 2 (Co-adaptation 방지)**: 한 뉴런이 다른 특정 뉴런에 의존하지 못하게. Robust한 feature 학습.

수학적으로는 **variational inference**의 일종으로 해석되기도 (Bayesian dropout).

### 4.3 Dropout 비율 선택

| Layer 종류 | $p$ |
|---|---|
| FC (분류 head) | 0.5 |
| Conv | 0.1~0.2 |
| Embedding | 0.1 |
| Transformer FFN | 0.1 |
| RNN hidden | 0 (시간 방향 위험) |

Conv가 작은 이유: 이미 weight sharing이 있어 정규화 효과가 내재. FC는 가장 over-param이라 강한 dropout 필요.

### 4.4 BN과 Dropout의 충돌

같은 layer에 둘 다 쓰면 충돌 가능. Dropout이 활성값 분포를 흔들어 BN 통계가 부정확. 처방:

- BN을 먼저, dropout을 후. 또는 dropout만.
- ResNet block에는 dropout 거의 안 씀 (BN이 정규화 충분).
- 분류 head FC에서만 dropout, 본체는 BN.

### 4.5 Variational Dropout (RNN용)

RNN에서 dropout을 시간 방향으로 매번 다른 mask로 적용하면 hidden state 흐름이 끊김. Variational dropout은 한 시퀀스 내내 같은 mask 사용 → 시간적 일관성.

### 4.6 Stochastic Depth — Layer 단위 dropout

ResNet에서 layer 자체를 확률적으로 skip. 깊이 매우 깊을 때 정규화 효과.

### 4.7 DropConnect, DropBlock 등 변종

- **DropConnect**: 활성값이 아니라 가중치를 mask.
- **DropBlock**: Conv에서 인접 영역을 함께 mask (단일 픽셀이 아니라 patch).

특정 task에서 약간 우위. 표준은 여전히 Dropout.

---

## 5. Weight Decay — L2 Regularization

### 5.1 정의

Loss에 가중치 크기 penalty 추가:

$$L_{total} = L_{data} + \lambda \|\theta\|^2$$

Update에서:
$$\theta \leftarrow \theta - \eta \nabla L = \theta - \eta(\nabla L_{data} + 2\lambda \theta)$$
$$= (1 - 2\eta\lambda)\theta - \eta \nabla L_{data}$$

매 step 가중치를 $(1 - 2\eta\lambda)$ 배로 줄임. "decay"라 부르는 이유.

### 5.2 왜 효과가 있는가

(1) **함수의 부드러움**: 큰 가중치는 input의 작은 변화에 큰 출력 변화. 가중치 작으면 부드러움 → 일반화.

(2) **Effective capacity 감소**: 같은 architecture라도 weight decay 강하면 표현 가능한 함수가 좁아짐.

(3) **Bayesian 해석**: $\theta$에 가우시안 prior 두는 것과 등가 (MAP estimation).

### 5.3 강도 ($\lambda$) 선택

| 모델 / 데이터 | $\lambda$ |
|---|---|
| 작은 모델, 작은 데이터 | 1e-3 ~ 1e-4 |
| 큰 모델, 큰 데이터 | 1e-4 ~ 1e-5 |
| Transformer | 1e-2 ~ 1e-1 (AdamW) |

Adam에서 weight decay를 정확히 적용하려면 AdamW (다음 섹션).

### 5.4 L1 vs L2

**L1**: $\lambda \|\theta\|_1$. 일부 가중치를 정확히 0으로 → sparse. Feature selection.

**L2**: $\lambda \|\theta\|_2^2$. 부드럽게 줄임. 일반적.

L1이 sparse를 만드는 이유: penalty의 gradient가 일정 (sign($\theta$))이라 0 근처에서 0으로 끌어당김. L2는 0 근처에서 gradient가 작아져 0에 안 도달.

**Elastic Net**: L1 + L2 combined. 둘의 장점 합침.

신경망에서는 거의 항상 L2. L1은 feature selection 의도가 있을 때.

---

## 6. Skip Connection — ResNet의 혁명

### 6.1 동기 — Degradation Problem

2015년 He et al. 깊이를 더 깊게(20 → 50 → 100층) 했을 때 **train accuracy도 떨어지는 현상** 발견. Overfitting이 아니라 **깊이 자체로 학습이 어려워지는 것**. 이걸 "degradation problem"이라 명명.

이론적으로는 50층 망이 20층 망보다 정확도가 낮을 수 없다. 20층의 학습된 함수에 30층의 identity를 더하면 50층이 똑같이 그 함수를 표현. 하지만 SGD가 이 identity를 학습 못 한다. **항등 함수도 학습 어려움**.

### 6.2 Skip Connection의 아이디어

블록의 출력을 **입력에 더한다**:

$$h_{l+1} = h_l + F(h_l)$$

여기서 $F$는 학습할 잔차 함수 (residual). Identity가 default — $F$가 0을 학습하면 자동으로 항등.

### 6.3 왜 효과가 있는가

(1) **Identity가 default**: 학습이 항등 함수에서 시작. 깊이가 더해도 최소한 더 나빠지지 않음.

(2) **Gradient 직접 흐름**:

$$\frac{\partial h_{l+1}}{\partial h_l} = I + \frac{\partial F}{\partial h_l}$$

Identity 항이 있어 gradient가 *덧셈으로* 흐름. 곱셈의 누적이 없으니 vanishing 약함.

깊이 $L$에서:
$$\frac{\partial h_L}{\partial h_l} = \prod_{k=l}^{L-1} \left(I + \frac{\partial F_k}{\partial h_k}\right)$$

각 항에 $I$가 있어 곱이 작아지지 않음. 매우 깊어도 gradient 흐름.

(3) **Optimization landscape 평활**: Skip connection 있으면 loss landscape가 더 부드러워진다는 분석 (Li et al. 2018).

### 6.4 ResNet Block의 구조

```
input x
  ├── Conv 3×3 → BN → ReLU
  ├── Conv 3×3 → BN
  └── + (identity skip)
      └── ReLU
output
```

또는 bottleneck (ResNet-50+):
```
input x
  ├── Conv 1×1 (차원 축소) → BN → ReLU
  ├── Conv 3×3 → BN → ReLU
  ├── Conv 1×1 (차원 복구) → BN
  └── + (identity skip)
      └── ReLU
output
```

1×1 conv로 차원 축소 후 작업, 다시 복구. 같은 RF에 적은 파라미터.

### 6.5 Pre-activation ResNet

ResNet v2 (2016). BN과 ReLU를 identity 경로 앞으로:

```
input x
  ├── BN → ReLU → Conv → BN → ReLU → Conv
  └── + (skip)
output
```

깊이 1000+층까지 학습 가능. 더 안정.

### 6.6 후속 모델들

**DenseNet** (2017): 모든 이전 layer와 연결. Feature reuse 강함, 파라미터 효율.

**Highway Network** (2015, ResNet 동시): gate로 skip 양 조절. 더 일반적이지만 복잡.

**Transformer**: skip connection이 모든 sub-layer에 (multi-head attention, FFN). LN과 짝.

ResNet의 skip connection은 **모든 modern architecture의 핵심 부품**이 됐다.

### 6.7 LSTM과의 본질적 같음

LSTM의 cell state update: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. Forget=1, input=0이면 $c_t = c_{t-1}$ — 정확히 identity skip.

ResNet의 vanishing 해결과 LSTM의 vanishing 해결이 본질적으로 같은 정신. **곱셈 누적을 덧셈 경로로 우회**.

---

## 7. Optimizer의 진화 — SGD에서 AdamW까지

### 7.1 SGD — 기본형

$$\theta \leftarrow \theta - \eta \nabla L$$

단순. Mini-batch에서:
$$\theta \leftarrow \theta - \eta \frac{1}{B} \sum_{i \in \text{batch}} \nabla L_i$$

장점: 단순, noise가 일반화에 도움.
단점: 골짜기에서 진동, 평탄 영역 느림.

### 7.2 SGD + Momentum

$$v_{t+1} = \beta v_t + \nabla L$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

이전 update를 누적. $\beta = 0.9$ default.

직관: 같은 방향이 반복되면 가속, 진동하면 상쇄. **관성**의 비유.

수학적으로: gradient의 지수 가중 평균이 update 방향.

### 7.3 Nesterov Accelerated Gradient

Momentum의 정교한 변종. **미리 가본 위치에서 gradient**:

$$v_{t+1} = \beta v_t + \nabla L(\theta_t - \eta \beta v_t)$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

이론적으로 더 빠른 수렴 (smooth convex case에서 $O(1/T) \rightarrow O(1/T^2)$).

실용적으로 momentum과 비슷, 약간 우위 보고.

### 7.4 Adagrad — 파라미터별 LR

$$G_t = G_{t-1} + g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t$$

자주 update되는 파라미터는 LR↓, 드문 건 LR↑. Sparse feature가 많은 NLP에 효과적이었다.

문제: $G$가 단조 증가 → LR이 결국 0. 학습 멈춤.

### 7.5 RMSProp — Adagrad 보완

$$G_t = \beta G_{t-1} + (1-\beta) g_t^2$$

이동평균으로 변경 → 무한 누적 안 됨.

### 7.6 Adam — Momentum + RMSProp

2014년. 현대 표준의 시작:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(1차 moment)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(2차 moment)}$$
$$\hat{m}_t = m_t / (1 - \beta_1^t) \quad \text{(bias correction)}$$
$$\hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Default: $\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$, LR=1e-3.

**왜 bias correction?**: $m_0 = 0$에서 시작하면 초기에 $m_t$가 작음 (과소추정). $1-\beta_1^t$로 나눠 보정. $t$가 크면 보정이 1에 수렴 (효과 없음).

**왜 효과적?**:
- Momentum: 진동 줄임.
- 파라미터별 LR: sparse gradient에 강함.
- Bias correction: 초기 안정성.
- LR 튜닝 부담 적음.

### 7.7 Adam의 문제와 AdamW

원본 Adam에서 weight decay를 loss에 더하면:

$$L_{total} = L_{data} + \lambda \|\theta\|^2$$
$$g_t = \nabla L_{data} + 2\lambda \theta$$

이 $g_t$가 second moment $v_t$로 normalize 되면서 weight decay의 의도가 왜곡. 큰 gradient의 weight는 weight decay가 약하게, 작은 건 강하게.

**AdamW** (2017, Loshchilov & Hutter): weight decay를 분리:

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

Weight decay가 second moment 영향 안 받음 — 정확한 SGD-style decay.

**Transformer 시대 표준**: AdamW + warmup + cosine decay.

### 7.8 그 밖의 변종

**RAdam**: Adam의 초기 variance 문제 해결.
**AdaBelief**: variance 추정을 더 정확히.
**LAMB**: 큰 batch (4096+) 전용.
**LARS**: 같은 정신, 다른 form.
**Lion**: 2023, sign-only update. 메모리 효율.

표준은 여전히 AdamW. 특수한 상황에 변종.

### 7.9 Optimizer 선택의 결정 트리

```
모델 / task?
├── ImageNet CNN → SGD+momentum (일반화 우위)
├── Transformer → AdamW + warmup + cosine
├── 작은 데이터·복잡 loss → Adam (안전)
├── 강화학습 → RMSProp / Adam
├── 큰 batch (4096+) → LARS / LAMB
└── 모르겠으면 → AdamW
```

---

## 8. Learning Rate Schedule

### 8.1 왜 schedule?

고정 LR은 sub-optimal. 초기 큰 LR로 빠르게 움직이고, 후반 작은 LR로 정밀하게.

### 8.2 Step Decay

전통적. Epoch 30, 60, 90마다 LR을 1/10:

```python
if epoch in [30, 60, 90]:
    lr *= 0.1
```

단순, 효과적. ImageNet ResNet 표준.

### 8.3 Exponential Decay

$$\eta_t = \eta_0 \cdot \gamma^t$$

부드러움. $\gamma$ 약 0.95 정도.

### 8.4 Cosine Annealing

코사인 곡선:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\pi t/T))$$

부드럽게 감소, 끝에 거의 0. **Transformer 표준**.

이론적으로 부드러운 schedule이 일반화에 유리하다는 경험적 보고.

### 8.5 Warmup

학습 초기에 0에서 본 LR로 점진 증가:

```
LR(t) = base_LR * t / warmup_steps  (t < warmup_steps)
LR(t) = base_LR  (t >= warmup_steps)
```

**왜 warmup?**:
- BN의 running 통계가 초기에 부정확.
- Adam의 second moment 추정이 noisy.
- 큰 LR로 시작하면 발산 위험.

큰 모델, 큰 LR, Adam 사용시 사실상 필수.

### 8.6 Cosine + Warmup

가장 흔한 조합. Warmup으로 부드럽게 시작, cosine으로 부드럽게 끝.

```
lr = base_lr * min(t / warmup, 0.5 * (1 + cos(pi * (t - warmup) / (T - warmup))))
```

### 8.7 OneCycle LR

Smith. Warmup → 큰 LR → 감소를 한 사이클에. 매우 빠른 학습 (super-convergence).

### 8.8 Cyclic LR

LR을 주기적으로 키웠다 줄였다. 평탄 영역 탈출에 도움.

### 8.9 Schedule 선택

| 모델 | Schedule |
|---|---|
| ImageNet CNN | Step decay |
| Transformer | Cosine + warmup |
| 빠른 prototype | OneCycle |
| 일반 | Cosine + warmup (안전) |

---

## 9. Loss Landscape — 학습 동역학의 시각화

### 9.1 Loss Surface

가중치 공간의 각 점에 loss 값을 매긴 표면. 매우 고차원이지만 2D로 시각화 시도가 있다 (Li et al. 2018).

특징:
- Local minima 多 (이론상). 단 saddle point가 더 흔하다.
- "Sharp minima" vs "Flat minima"의 trade-off.
- Skip connection이 landscape를 부드럽게.

### 9.2 Sharp vs Flat Minima

**Sharp minima**: 가중치 살짝 변하면 loss 큰 변화. Train data에 완벽히 fit하지만 일반화 약함.

**Flat minima**: 가중치 변해도 loss 안정. 일반화 잘 됨.

**SGD의 implicit bias**: SGD는 flat minima로 가는 경향 (noise 덕에). Adam은 sharp minima로 가는 경향. 그래서 Adam의 일반화가 약간 손해라는 분석.

### 9.3 학습 단계

학습 동역학을 단계로 보면:

1. **Early stage** (~10% 진행): Loss 빠르게 감소. 큰 features 학습.
2. **Middle stage** (~80%): 점진적 개선. 미세한 features.
3. **Late stage** (~10%): 정체 또는 미세 개선. Flat minima 탐색.

LR schedule이 이 단계와 매칭. Warmup → 큰 LR (early) → 감소 (mid/late).

---

## 10. 정규화 종합 — 상호작용

### 10.1 정규화 기법 전체

| 기법 | 작용점 | 강도 (default) |
|---|---|---|
| L2 / Weight decay | 가중치 | 1e-4 |
| L1 | 가중치 | 1e-5 |
| Dropout | 활성값 | 0.1~0.5 |
| BN / LN | 활성값 분포 | 자동 |
| Augmentation | 입력 | task별 |
| Label smoothing | 라벨 | 0.1 |
| Mixup / CutMix | 입력+라벨 | 0.2~1.0 |
| Stochastic depth | layer | 0.0~0.5 |
| Early stopping | 학습 시간 | patience 5~10 |

### 10.2 보완성

대부분 보완적. 같이 쓰면 효과 합. 단 충돌하는 경우:

- BN + Dropout: 분산 충돌. 위치 신중히.
- 강한 augmentation + 강한 dropout: underfit.
- Mixup + label smoothing: 둘 다 라벨 부드럽게 → 과도.

### 10.3 데이터 크기에 따른 강도

| 데이터 | 정규화 강도 |
|---|---|
| < 10k | Strong: dropout 0.5, WD 1e-3, strong aug |
| 10k~100k | Medium |
| 100k~10M | Weak: dropout 0.1, WD 1e-4, weak aug |
| > 10M | Minimal: dropout 0, WD 1e-2, label smooth |

큰 데이터에서는 데이터 자체가 정규화. 외부 정규화 약하게.

---

## 11. 학습 진단 — 무엇이 잘못되었나

### 11.1 Loss NaN

| 원인 | 처방 |
|---|---|
| LR 큼 | 1/10 |
| log(0) | log_softmax, eps |
| Init 분산 큼 | He / Xavier |
| Mixed precision overflow | Loss scaling |
| 데이터 NaN | 입력 점검 |
| Gradient explosion | Clipping |

### 11.2 학습 정체 (Plateau)

| 원인 | 처방 |
|---|---|
| Vanishing | ReLU + He + BN + Skip |
| LR 작음 | 키움 |
| LR schedule 잘못 | 점검 |
| Saddle point | Adam, momentum |
| 데이터 정규화 없음 | 입력 정규화 |

### 11.3 Overfit

| 처방 | 강도 |
|---|---|
| Early stopping | 첫 단계 |
| Dropout 늘림 | 0.3 → 0.5 |
| Weight decay 늘림 | 1e-4 → 1e-3 |
| Augmentation 강화 | task별 |
| 데이터 늘림 | 항상 좋음 |

### 11.4 Underfit

| 처방 | 강도 |
|---|---|
| 모델 키움 | depth/width |
| LR 점검 | LR range test |
| 정규화 줄임 | dropout↓, WD↓ |
| 학습 더 | epoch↑ |

### 11.5 BN/Dropout train mode로 evaluation

증상: val 결과가 매번 다름, 또는 train보다 val이 좋음.

처방: `model.eval()` + `with torch.no_grad():` 항상.

### 11.6 Distribution Shift

증상: val 좋은데 deploy에서 망함.

처방: domain adaptation, 정기 재학습, drift monitoring.

---

## 12. 면접 단골 Q&A

### Q1. Vanishing gradient 왜 발생하고 어떻게 해결?
"Backprop의 chain rule이 매 layer마다 $W \cdot \sigma'$의 곱셈을 누적해서 발생합니다. 곱이 < 1이면 지수적 감쇠. Sigmoid는 $\sigma' \le 0.25$라 10층 만에 $10^{-6}$. 해결책은 다섯 축. (1) 활성화 — ReLU로 활성 영역에서 미분 1. (2) Init — He init으로 분산 보존. (3) BN — 분포 안정. (4) Skip connection — 곱셈 누적 대신 덧셈 경로. (5) Optimizer — Adam의 파라미터별 LR. 현대 deep learning은 이 다섯 모두 기본 stack."

### Q2. BN이 왜 효과가 있나?
"세 가지 해석. (1) 원래 동기 — internal covariate shift 완화. 매 layer 입력 분포 안정. (2) 후속 분석 — loss landscape 평활화. 큰 LR 사용 가능, 학습 빠름. (3) 부수 효과 — mini-batch 통계의 noise가 약한 정규화. 그리고 init에 덜 민감해짐. 단점은 작은 batch에서 부정확, 시퀀스 데이터에 불편 → GroupNorm/LayerNorm으로 보완."

### Q3. Dropout이 왜 ensemble처럼 작동?
"매 step 다른 sub-network를 학습. n개 뉴런이면 가중치 공유된 $2^n$개 sub-network를 동시에 학습. 평가 시 모든 뉴런 사용 + 활성값 스케일 = 사실상 평균 — exponentially many networks의 ensemble 근사. 또 co-adaptation 방지 효과 — 한 뉴런이 다른 특정 뉴런에 의존 못함. Bayesian 관점으론 variational inference."

### Q4. ResNet의 skip connection이 왜 깊이 학습을 가능하게?
"두 측면. (1) Identity가 default — F가 0을 학습하면 자동 항등. 깊이 추가가 최소한 더 나빠지지 않음 (degradation problem 해결). (2) Gradient 직접 흐름 — $\partial h_{l+1}/\partial h_l = I + \partial F$. Identity 항이 있어 gradient가 덧셈으로 흐름. 곱셈 누적이 없어 vanishing 약함. 100+층 학습 가능. LSTM의 cell state도 본질적으로 같은 정신."

### Q5. Adam이 SGD보다 좋은가?
"Task에 따라. Adam은 빠른 수렴 + LR 튜닝 부담 적음 → 작은 데이터, 복잡 loss에 안전. 단 ImageNet 같은 task에서 SGD+momentum이 약간 더 일반화 잘한다는 보고 — Adam이 sharp minima로 가는 경향이라는 분석. AdamW가 weight decay 정확히 처리해서 Transformer 시대 표준. 결론: Transformer면 AdamW, ImageNet 분류면 SGD, 그 외엔 둘 다 시도."

### Q6. He init의 분산이 2/n_in인 이유?
"ReLU 가정. ReLU가 음수 절반을 0으로 만들어 분산이 절반으로 줄어듦. 이걸 보상하려면 가중치 분산을 두 배. Xavier는 sigmoid/tanh 가정 (대칭)이라 1/n_in의 두 배인 2/(n_in+n_out)이 적절. He는 ReLU 보정으로 2/n_in. Forward 분석 + backward 분석의 절충."

### Q7. Warmup이 왜 필요?
"학습 초기 통계의 noise 때문. (1) BN의 running 평균·분산이 초기에 부정확. (2) Adam의 second moment 추정이 noisy (bias correction이 있어도 완벽 안 됨). (3) 큰 LR로 시작하면 이 부정확한 통계 때문에 발산 위험. 작은 LR로 시작해 통계 안정화 → 본 LR로 점진 증가. 큰 모델 + Adam 조합에 사실상 필수."

### Q8. AdamW가 Adam과 다른 점?
"Weight decay 처리. 원본 Adam에서 weight decay를 loss에 더하면 그 항도 second moment $v_t$로 normalize되어 의미 왜곡. AdamW는 weight decay를 update 시점에 별도로 적용 — gradient를 normalize하고 그 다음 weight decay 곱한 가중치 빼기. SGD-style의 정확한 weight decay. Transformer 시대 표준."

### Q9. BN과 Dropout 같이 쓸 때 주의?
"분산 변동 충돌. Dropout이 활성값에 noise를 더하면 BN의 batch 통계가 흔들림. 처방: (1) 위치 신중히 — 보통 conv → BN → ReLU → (dropout). (2) ResNet block에는 dropout 거의 안 씀 (BN이 정규화 충분). (3) 분류 head FC에서만 dropout. (4) 대안 — dropout 대신 stochastic depth (layer 자체 skip)."

### Q10. Layer Norm vs Batch Norm?
"정규화 축이 다름. BN은 batch 차원 (같은 채널의 다른 sample 평균), LN은 feature 차원 (한 sample의 모든 채널 평균). BN은 batch 통계에 의존 → 작은 batch에서 noisy, 시퀀스에서 불편. LN은 sample 내 정규화 → batch 무관, 시퀀스 일관. Transformer가 LN 사용하는 이유: 시퀀스 길이 가변, 자기회귀 추론 시 batch=1, sample 내 정규화로 일관성. CNN은 BN, RNN/Transformer는 LN."

### Q11. Mixed Precision Training이 왜 빠르고 위험?
"빠른 이유: fp16 연산이 fp32보다 2~8배 빠름 (NVIDIA Tensor Core). 위험: fp16의 dynamic range가 좁음 ($10^{-5} \sim 10^{4}$ 정도). Gradient나 activation이 그 범위 밖이면 0이 되거나 inf. 처방: (1) Loss scaling — loss를 큰 수로 곱한 후 backward, gradient 받고 다시 나눔. Underflow 방지. (2) Master weights를 fp32로 유지. (3) 일부 연산 (LayerNorm, softmax)은 fp32로 강제. PyTorch의 `torch.cuda.amp.autocast`가 자동 처리."

### Q12. Skip connection이 LSTM의 cell state와 본질적으로 같다고?
"네. 둘 다 *덧셈으로 정보 전파* 정신. ResNet: $h_{l+1} = h_l + F(h_l)$. LSTM: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. Forget=1, input=0이면 정확히 identity skip. Backward에서 gradient가 곱셈 누적 대신 덧셈 경로로 흐름 → vanishing 우회. Identity 정보가 자동 보존되고, 변환 함수(F 또는 input 곱)는 잔차만 학습. 같은 동기, 다른 자리."

---

## 13. 생각해보라 — 단락 답안

**Q. 왜 ReLU만으로는 깊이가 충분히 학습 안 되나?**

ReLU는 vanishing의 한 측면(활성화 미분)만 해결. 다른 측면이 있다. (1) 가중치 초기화가 잘못되면 활성값 폭발 또는 0. (2) 매 layer의 활성값 분포가 학습 도중 흔들림 → 학습 동역학 불안정. (3) Layer 매우 깊으면 (50+) ReLU의 dying이 누적, 깊은 layer가 죽기 시작. (4) Skip connection 없으면 항등 함수도 학습 어려움 (degradation problem).

이래서 ReLU만으로는 5~10층 정도가 한계. 50+층 가려면 He init + BN + ResNet의 조합 필요. 각 기법이 vanishing의 다른 측면을 해결하므로 합쳤을 때만 매우 깊은 학습 가능.

**Q. Adam과 SGD의 implicit bias는 무엇이 다른가?**

SGD의 mini-batch noise는 *각 step의 gradient가 임의로 perturb*되는 효과. 이 noise가 sharp minima를 회피하고 flat minima를 선호하는 경향. Flat minima는 가중치 변화에 둔감 → 일반화 잘 됨.

Adam은 second moment로 gradient를 normalize. 큰 gradient는 작게, 작은 건 크게. 이러면 모든 방향에서 effective LR이 비슷 → 더 빠른 수렴이지만 noise가 작아져서 sharp minima로 갈 가능성. 또 second moment 자체가 sharp 영역의 신호를 줄임.

이 차이가 실용에선 작아 보일 수 있지만 ImageNet 같은 큰 task에서 SGD+momentum이 약 1% 정도 더 좋은 일반화. 그래서 SOTA 모델 학습은 종종 SGD가 선택. 단 Transformer는 AdamW가 표준 — 큰 모델에서 SGD는 hyperparameter 매우 민감.

**Q. 왜 BN이 RNN에서 잘 작동 안 하는가?**

세 이유. (1) **시퀀스 길이 가변** — BN은 mini-batch 통계가 필요한데 시퀀스마다 길이 다르면 통계 일관성 약함. (2) **시간 축 정규화 모호** — 시점별로 다른 분포일 수 있는데 모든 시점을 같이 정규화하면 정보 손실. 시점별로 따로면 통계가 매우 noisy. (3) **자기회귀 추론** — 한 토큰씩 생성할 때 batch=1, BN의 running 통계와 train 분포 불일치.

Layer Norm은 이걸 모두 회피. Sample 내 정규화라 batch 독립, 시점 독립, 자기회귀에 일관. 그래서 Transformer/RNN의 표준이 LN.

**Q. ResNet의 깊이가 사실은 ensemble처럼 작동한다는 분석?**

Veit et al. 2016이 "Residual Networks Behave Like Ensembles of Relatively Shallow Networks"라는 분석. ResNet의 forward pass를 보면 각 block을 통과하느냐 skip하느냐의 $2^L$개 경로가 있고, 출력은 그 모든 경로의 합.

만약 학습된 ResNet에서 일부 block을 제거해도 성능이 잘 안 망가짐 — 다른 경로들이 보완. 이게 단순 깊은 망과 다른 점 (단순 망은 한 layer 빼면 망함).

함의: ResNet의 깊이는 "단일 매우 깊은 함수"가 아니라 "다양한 깊이 sub-network의 ensemble"로 해석 가능. 이게 일반화 성능에도 도움.

**Q. 왜 weight decay가 dropout보다 일반적으로 더 안정적?**

Weight decay는 *결정론적*. 매 step 정확히 같은 양의 정규화. 학습 곡선이 부드럽다. Dropout은 *확률적* — 매 step 다른 sub-network라 gradient noisy. 학습 곡선이 진동.

또 weight decay는 모든 architecture에 일관 적용 가능. Dropout은 RNN의 시간 방향, BN과의 충돌, conv의 작은 비율 등 미묘한 제약.

그래서 default로 weight decay를 켜고, 추가 정규화 필요하면 dropout 더함. 큰 Transformer에서는 dropout 0이고 weight decay만 사용하는 경우 多.

**Q. 왜 큰 batch가 일반화에 안 좋은가?**

여러 가설. (1) **Sharp minima 가설**: 큰 batch는 gradient가 정확 → noise 없음 → sharp minima로 빠짐. 작은 batch의 noise가 flat minima로 이끔. (2) **Update 횟수 감소**: 같은 epoch에서 큰 batch는 update 횟수 적음 → 모델이 충분히 *움직이지* 못함. (3) **암묵적 정규화 약화**: SGD noise의 정규화 효과 줄어듦.

이 셋의 조합. Linear scaling rule (batch×k면 LR×k)로 어느 정도 보완 가능하지만 한계 있음. 매우 큰 batch (4096+)에선 LARS/LAMB 같은 전용 optimizer 필요.

실용적으론 batch 256~1024가 sweet spot.

**Q. 왜 LayerNorm이 Pre-Norm 위치가 더 안정적인가?**

Transformer의 LN 위치 두 가지:
- **Post-Norm** (원조): `x + LN(SubLayer(x))`
- **Pre-Norm**: `x + SubLayer(LN(x))`

Pre-Norm이 학습 안정성 좋다는 보고. 이유: skip 경로가 LN 거치지 않음. Identity mapping이 더 직접적. Gradient flow 더 안정.

Post-Norm은 모든 sublayer 후 정규화 → variance 흔들림이 누적. Pre-Norm은 매 sublayer 입력만 정규화.

GPT-3 등 큰 모델에선 Pre-Norm이 표준. Original Transformer는 Post-Norm이지만 더 깊은 모델에선 Pre-Norm으로 갈아탔다.

---

## 14. 한 줄 요약 (시험 직전)

- DNN의 핵심 어려움 = **vanishing/exploding gradient**, chain rule의 곱셈 누적.
- 해결의 다섯 축 = **활성화(ReLU) + Init(He) + 정규화(BN) + 구조(Skip) + Optimizer(Adam)**.
- **BN의 효과** = ICS 완화 + landscape 평활 + 약한 정규화. Train/eval mode 다름.
- **Dropout** = ensemble 근사. BN과 짝 신중히.
- **Weight decay (L2)** = 가중치 부드러움. AdamW가 정확한 처리.
- **ResNet의 skip** = 곱셈 누적 대신 덧셈 경로. Identity가 default. LSTM cell state와 본질 같음.
- **Adam vs SGD** = 빠른 수렴 vs 더 좋은 일반화. AdamW가 Transformer 표준.
- **LR schedule** = warmup + cosine이 큰 모델에 안전. Step decay는 ImageNet 표준.
- **He init**의 2/n_in = ReLU의 분산 손실 보정.
- 모든 기법이 *동시에* 작동해야 깊은 학습 가능. 하나만 빠져도 종종 망함.
