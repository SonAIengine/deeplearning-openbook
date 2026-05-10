# 02. Feedforward Neural Networks (FNN / MLP) — 심층 정리

> **이 문서의 목표**
> 신경망의 가장 기본 형태인 FNN을 통해 "왜 비선형이 필수인지", "왜 깊이가 도움이 되는지", "backprop이 정말 무엇을 하는지", "Universal Approximation Theorem이 무엇을 약속하고 무엇을 약속하지 않는지"를 한 단계 깊게 이해한다. 이 챕터는 모든 후속 챕터(CNN, RNN, Transformer)의 토대다.

---

## 0. 큰 그림 — 신경망이란 무엇인가

### 0.1 신경망 = 함수 근사기

신경망의 본질은 매우 단순하다 — **합성된 비선형 함수**.

$$f(x) = \sigma_L(W_L \sigma_{L-1}(W_{L-1} \cdots \sigma_1(W_1 x + b_1) \cdots) + b_L)$$

선형 변환 $Wx + b$와 비선형 함수 $\sigma$를 번갈아 합성한다. 그 위에 학습 가능한 파라미터 $\theta = \{W_l, b_l\}_{l=1}^L$를 데이터로 조정한다.

겉으로는 수식이 복잡해 보이지만 본질은 **변환의 합성**이다. 입력을 한 단계씩 변환해 나가다 보면 결국 출력이 된다. 각 단계는 자기 역할이 있다 — 어떤 layer는 edge를 학습하고, 어떤 layer는 모양을 학습하고, 어떤 layer는 의미를 학습한다.

### 0.2 왜 "feedforward"인가

신호가 입력에서 출력으로 **한 방향**으로 흐른다 — 이게 feedforward의 의미. 같은 layer 안에서 또는 뒤로 가는 연결이 없다.

이 단순한 구조 제약은 두 가지 의미를 가진다. 첫째, **메모리가 없다**. 입력 $x_t$의 처리는 입력 $x_{t-1}$의 처리와 완전히 독립적이다. 시간적 순서가 의미 있는 데이터 (시퀀스)에서는 이게 약점이 되어 RNN이 등장한다. 둘째, **계산이 명확하다**. 정해진 layer 순서로 한 번 forward pass하면 출력이 나온다. 학습도 backward pass로 한 번에 끝난다.

FNN(또는 MLP, Multi-Layer Perceptron)은 신경망의 가장 단순한 형태이고, 다른 모든 신경망(CNN, RNN, Transformer)은 FNN에 *구조적 제약*(inductive bias)을 더한 변종으로 이해할 수 있다. CNN = FNN + weight sharing across space. RNN = FNN + weight sharing across time. Transformer = FNN + self-attention.

### 0.3 신경망은 왜 잘 동작하는가 — 두 단계 설명

표면적으로는 간단하다 — "함수 근사기인데 표현력이 좋다." 그런데 이 답은 사실 두 단계를 모은 것이다.

**1단계 (표현력)**: 충분히 큰 신경망은 임의의 연속함수를 근사할 수 있다 (Universal Approximation Theorem). 이건 1989년 Cybenko, Hornik이 증명. 이 정리는 신경망의 *최대 잠재력*에 대한 진술이다.

**2단계 (학습 가능성)**: 그 잠재력을 *데이터로 끌어낼 수 있다*. 즉 적절한 hyperparameter로 SGD를 돌리면 진짜 좋은 함수에 수렴한다. 이건 훨씬 더 어려운 진술이고, 수학적으로 충분히 이해되지 않은 영역이다 — neural tangent kernel, feature learning theory 등이 활발한 연구.

UAT는 1단계만 보장한다. 2단계는 별개. 그래서 "1 hidden layer로 모든 함수 근사 가능 (UAT)" 말은 맞지만, "그래서 1 hidden layer 신경망이 항상 충분"은 틀린다. 이론과 실용의 갭을 의식해야 한다.

---

## 1. Perceptron — 1957년의 시작과 한계

### 1.1 단일 퍼셉트론

Frank Rosenblatt의 1957년 모델. 입력 $x$, 가중치 $w$, 임계값 $b$:

$$
y =
\begin{cases}
1 & \text{if } w^T x + b > 0 \\[2pt]
0 & \text{otherwise}
\end{cases}
$$

기하학적으로 이건 **초평면(hyperplane)으로 입력 공간을 둘로 나누는 분류기**다. $w$가 초평면의 법선 벡터, $b$가 평면의 위치.

학습 규칙도 단순하다. 틀린 sample $(x, y)$에 대해:

$$w \leftarrow w + \eta(y - \hat{y}) x$$

옳게 분류하면 update 없음. 잘못 분류하면 가중치를 정답 방향으로 살짝 옮김.

이 단순한 모델이 1958년 New York Times에 "전자 두뇌의 탄생"으로 보도됐다. 사람들은 곧 기계가 사람처럼 학습할 거라 기대했다. 그러나 한계가 곧 드러났다.

### 1.2 XOR 문제 — 분야의 첫 번째 겨울

Marvin Minsky와 Seymour Papert가 1969년 책 "Perceptrons"에서 단일 퍼셉트론은 **선형 분리 가능한** 문제만 풀 수 있음을 증명했다. 가장 단순한 비선형 문제가 XOR:

| $x_1$ | $x_2$ | $y$ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

(0,0)과 (1,1)이 한 클래스, (0,1)과 (1,0)이 다른 클래스. 평면 위에 점 4개 그려보면 직선 하나로 두 클래스를 가르는 게 기하학적으로 불가능.

수학적으로도 보자. 만약 $w_1 x_1 + w_2 x_2 + b > 0$이 (0,1)과 (1,0)에는 참이고 (0,0)과 (1,1)에는 거짓이어야 한다면:
- $w_2 + b > 0$ (from (0,1))
- $w_1 + b > 0$ (from (1,0))
- $b \le 0$ (from (0,0))
- $w_1 + w_2 + b \le 0$ (from (1,1))

위 둘을 더하면 $w_1 + w_2 + 2b > 0$. 아래 둘을 더하면 $w_1 + w_2 + b \le 0$이고 $b \le 0$이라 $w_1 + w_2 + b - b \le -b$, 즉 $w_1 + w_2 \le -b$. 모두 합쳐보면 모순. 즉 단일 퍼셉트론으로 XOR 못 푼다는 게 산술적으로 증명된다.

이 발견이 신경망 분야의 첫 번째 겨울을 가져왔다. 1970년대 자금과 관심이 빠지면서 분야가 거의 멈췄다.

### 1.3 해결책 — 은닉층 + 비선형

이걸 푸는 방법은 단순하다 — **층을 하나 추가하고 비선형을 넣자**. XOR을 푸는 한 가지 신경망:

```
x_1, x_2 → h_1 = AND(x_1, NOT x_2)
        → h_2 = AND(NOT x_1, x_2)
        → y = OR(h_1, h_2)
```

이게 정확히 XOR. 한 층에서는 안 되지만 두 층 + 비선형이면 된다. 이 통찰이 multilayer perceptron (MLP)의 동기.

문제는 학습 알고리즘. 단일 퍼셉트론의 학습 규칙은 출력이 라벨과 직접 비교 가능해서 단순했다. 은닉층이 생기면 **은닉 뉴런의 정답이 무엇인지 모른다** — 이걸 *credit assignment problem*이라 부른다. 어느 가중치가 잘못한 건지 어떻게 알 수 있나?

답: **backpropagation** (1986, Rumelhart-Hinton-Williams). 출력에서의 오차를 chain rule로 거슬러 올라가며 각 가중치에 책임을 분배. 이 알고리즘이 신경망의 두 번째 부흥을 가져왔다.

---

## 2. Multi-Layer Perceptron (MLP) 구조

### 2.1 기본 구조

$L$개의 layer를 가진 MLP:

$$h^{(0)} = x$$
$$z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}, \quad h^{(l)} = \sigma(z^{(l)}) \quad (l = 1, ..., L-1)$$
$$\hat{y} = g(z^{(L)})$$

여기서:
- $h^{(l)}$: layer $l$의 hidden activation
- $z^{(l)}$: layer $l$의 pre-activation (활성화 전)
- $W^{(l)}, b^{(l)}$: 가중치 행렬과 bias
- $\sigma$: 비선형 활성화 함수 (보통 ReLU)
- $g$: 출력층 활성화 (분류면 softmax, 이진이면 sigmoid, 회귀면 linear)

핵심: 각 layer는 **선형 변환 + 비선형 변환**의 두 단계로 구성. 두 단계가 항상 짝지어진다.

### 2.2 왜 layer를 쌓는가 — 표현의 계층적 구조

신경망의 layer를 표현 학습의 시각으로 보면:

| Layer 깊이 | 학습되는 표현 (CNN 기준) |
|---|---|
| Layer 1 | edge, color blob, oriented gradient |
| Layer 2 | texture, simple shape |
| Layer 3 | object part (eye, wheel, leaf) |
| Layer 4+ | full object, semantic concept |

이게 **계층적 추상화** (hierarchical abstraction). 깊은 layer는 얕은 layer가 학습한 feature를 조합해 더 복잡한 표현을 만든다.

이론적으로도, 같은 함수를 표현할 때 깊은 신경망은 얕은 신경망보다 **지수적으로 적은 파라미터**로 가능한 경우가 있다 (Hastad의 parity 함수 결과 등). 그래서 깊이는 단순히 더 많은 파라미터가 아니라 **더 효율적인 표현**의 수단이다.

이래서 같은 1M 파라미터를 너비로 키우는 것보다 깊이로 키우는 게 (보통) 유리하다.

### 2.3 입력·출력 layer의 역할

**입력 layer**: 단순히 입력 데이터를 받아들이는 자리. 학습 가능한 파라미터가 보통 없음 (단, 임베딩 layer가 있으면 학습됨).

**출력 layer**: task에 따라 다르다.
- *회귀*: 마지막 linear layer, 출력 = 예측값.
- *이진 분류*: 마지막에 sigmoid, 출력 = 양성 확률.
- *다중 분류 K개*: 마지막에 softmax, 출력 = K 클래스 확률 분포.
- *Multi-label*: 클래스마다 sigmoid 따로.

출력층 활성화와 loss는 짝이다. Sigmoid + BCE, softmax + CE, linear + MSE.

### 2.4 차원 흐름의 직관

전형적인 MLP:

```
입력 (784,) → linear → (512,) → ReLU → (512,)
            → linear → (256,) → ReLU → (256,)
            → linear → (128,) → ReLU → (128,)
            → linear → (10,)  → softmax → (10,)
```

각 layer가 차원을 변환한다. 줄어들 수도, 늘어날 수도, 같을 수도 있다. 분류에서는 보통 점진적으로 차원을 줄여 마지막에 클래스 수에 도달.

차원 변경에는 의미가 있다. 줄어드는 단계는 **압축**, 즉 본질만 남기는 단계. 늘어나는 단계는 **풀어 헤치기**, 즉 더 많은 표현 채널 확보. 어느 시점에 어느 방향으로 갈지가 architecture 설계의 일부.

---

## 3. 활성화 함수 — 비선형이 핵심

### 3.1 비선형이 없으면 깊이가 의미 없다

이건 신경망 이해의 가장 중요한 단일 사실이다. **비선형 활성화가 없으면 깊은 망이 단일 layer 선형 모델로 환원된다**.

증명: 활성화 없는 두 layer는

$$f(x) = W_2 (W_1 x + b_1) + b_2 = W_2 W_1 x + (W_2 b_1 + b_2) = W' x + b'$$

여기서 $W' = W_2 W_1$, $b' = W_2 b_1 + b_2$는 단일 선형 변환. 100층 1000층 쌓아도 마찬가지 — 모든 곱이 하나의 행렬 곱으로 합쳐진다.

따라서 비선형 활성화의 자리는 **표현력의 원천**이다. 비선형을 통해 신경망은 임의의 비선형 함수를 합성할 수 있게 된다. UAT의 본질도 여기에 있다 — sigmoid든 ReLU든 비선형이면 충분히 많은 뉴런으로 임의 함수 근사 가능.

### 3.2 Sigmoid — 시작이자 한계

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

미분: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$.

**그래프**: S자 형태. $x \rightarrow -\infty$이면 0, $x \rightarrow \infty$이면 1, $x = 0$이면 0.5.

**역사**: 신경망 초창기부터 표준이었다. 생물학적 뉴런의 firing rate와 비슷하다는 동기. 또 (0, 1) 출력이 확률처럼 해석 가능.

**장점**:
- 출력 범위가 (0, 1)로 깔끔.
- 미분 가능, 부드러움.
- 확률적 해석.

**치명적 단점들**:

(1) **Vanishing gradient**: $\sigma'(x)$의 최대값이 0.25 (x=0에서). $|x| > 4$이면 $\sigma'(x) < 0.02$. 깊은 망에서 backprop으로 gradient를 chain rule로 곱하면 매 layer마다 0.25 이하 곱. 10층이면 $0.25^{10} \approx 10^{-6}$. Gradient가 사라져 학습이 안 된다.

(2) **출력이 0-centered가 아니다**: 항상 양수. 다음 layer의 입력이 항상 양수면 그 layer의 가중치 update가 모두 같은 부호 → zigzag pattern. 학습 효율 손해.

(3) **Exp 연산 비용**: 단순 max나 곱셈에 비해 exp이 비싸다. 큰 batch에서는 무시할 수 없는 오버헤드.

이 셋 합쳐서, sigmoid는 깊은 망의 hidden layer로 사실상 사용 안 한다. Output layer (이진 분류) 또는 LSTM의 gate처럼 (0, 1) 비율이 의미 있는 자리에만 한정적으로.

### 3.3 Tanh — sigmoid의 0-centered 버전

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

미분: $\tanh'(x) = 1 - \tanh^2(x)$.

**그래프**: S자, 출력 (-1, 1).

**Sigmoid 대비 장점**: 0-centered. 미분 최대값 1.0 (x=0에서). 깊은 망에서 sigmoid보다 vanishing 약함.

**여전한 단점**: vanishing gradient. $|x| > 2$이면 $\tanh'(x) < 0.07$.

**현재 위치**: RNN의 hidden state에는 여전히 표준. 그 외에는 ReLU가 거의 대체.

### 3.4 ReLU — 깊은 망의 혁명

$$\text{ReLU}(x) = \max(0, x)$$

미분: $x > 0$이면 1, $x < 0$이면 0, $x = 0$이면 정의 안 됨 (실용적으로 0 또는 1로 처리).

**그래프**: 음수에선 0, 양수에선 y=x 직선.

**왜 혁명이었나** (2012, AlexNet):

(1) **Vanishing 거의 없음**: 활성된 뉴런(x>0)에서 gradient = 1. 깊이가 깊어도 곱이 1이라 그대로 흐름. Sigmoid의 $0.25^L$ 대신 그냥 1.

(2) **계산이 매우 빠름**: max 연산만. exp 같은 거 없음.

(3) **Sparse activation**: 절반 정도의 뉴런이 0을 출력. 이게 sparse representation으로 이어져 종종 일반화에 도움.

(4) **생물학적 동기**: 실제 뉴런도 발화 임계 이상에서만 활동. 부드러운 sigmoid보다 ReLU가 사실 더 비슷.

**단점 — Dying ReLU**: 큰 음수 gradient가 한 번 들어가서 가중치가 음의 영역으로 가면, 그 뉴런의 입력은 항상 음수, 출력은 항상 0, gradient도 항상 0이라 더 이상 학습 안 됨. 그 뉴런은 **죽었다(dead)**. 이게 누적되면 모델 capacity 손해.

처방:
- *He initialization*: ReLU 가정 분산 2/n_in.
- *Leaky ReLU*: 음수에서도 작은 gradient.
- *Learning rate를 너무 크게 안 하기*.

ReLU는 2010년대 깊은 신경망 부흥의 핵심 요인 중 하나다.

### 3.5 ReLU의 변종들

**Leaky ReLU**: $\text{LReLU}(x) = \max(\alpha x, x)$, $\alpha = 0.01$ 기본. 음수에서도 작은 gradient $\alpha$. Dying ReLU 완화.

**PReLU (Parametric ReLU)**: $\alpha$를 학습. 데이터로부터 음수 기울기 결정.

**ELU (Exponential Linear Unit)**: 음수에서 $\alpha(e^x - 1)$. 0-centered + 부드러움.

**SELU (Scaled ELU)**: 특정 분산 가정에서 self-normalizing 성질. BN 없이도 깊이 학습 가능 (실용적으론 BN이 더 안정).

**GELU (Gaussian Error Linear Unit)**: $x \cdot \Phi(x)$, $\Phi$는 가우시안 누적분포. 부드러운 ReLU. **Transformer의 표준**.

**Swish/SiLU**: $x \cdot \sigma(x)$. EfficientNet 등에서 사용.

각각의 변종은 ReLU의 한계(dying, 0-centered 안 됨)를 해결하려는 시도. 실용적으로는 GELU가 Transformer 시대 표준이고, 일반 CNN/MLP에서는 여전히 ReLU가 default.

### 3.6 Softmax — 분류 출력의 표준

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**역할**: $K$차원 logit 벡터를 합이 1인 확률 분포로 변환.

**왜 단순 정규화 ($z_i / \sum z_j$) 안 되나?** Logit은 음수일 수 있고, 단순 정규화는 음수 처리가 안 됨. Exp이 모든 logit을 양수로 만들고, 큰 logit을 더 강조 (smooth max).

**Temperature**: $\text{softmax}(z/T)$. $T < 1$이면 더 sharp (argmax에 가까움), $T > 1$이면 더 uniform. 생성 모델에서 sampling 다양성 조절에 사용.

**수치적 안정성**: $z$가 매우 크면 $e^z$가 overflow. 표준 트릭은 max를 빼는 것:
$$\text{softmax}(z)_i = \frac{e^{z_i - \max z}}{\sum_j e^{z_j - \max z}}$$

수학적으로 같은 값이지만 overflow 방지. 모든 신경망 라이브러리에서 자동 적용.

### 3.7 활성화 선택의 사고 흐름

```
은닉층?
├── 일반 CNN/MLP → ReLU
├── Transformer → GELU
├── RNN의 hidden → Tanh
├── LSTM의 gate → Sigmoid (0~1 비율)
└── Dying ReLU 우려 → Leaky ReLU / ELU

출력층?
├── 회귀 → Linear
├── 이진 분류 → Sigmoid
├── 다중 분류 → Softmax
└── Multi-label → 클래스별 Sigmoid
```

이 흐름을 외우면 활성화 선택은 거의 자동.

---

## 4. Forward Propagation — 데이터가 어떻게 흐르는가

### 4.1 한 layer의 forward

$l$번째 layer에서:

$$z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$$
$$h^{(l)} = \sigma(z^{(l)})$$

행렬 곱 + bias + 비선형. 이게 한 layer의 모든 것.

차원 추적:
- $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$
- $h^{(l-1)} \in \mathbb{R}^{n_{l-1}}$
- $b^{(l)} \in \mathbb{R}^{n_l}$
- $z^{(l)}, h^{(l)} \in \mathbb{R}^{n_l}$

### 4.2 Mini-batch에서의 forward

실제로는 한 sample이 아니라 배치 $X \in \mathbb{R}^{B \times n_0}$로 forward:

$$Z^{(l)} = H^{(l-1)} (W^{(l)})^T + b^{(l)} \quad (\text{broadcast})$$
$$H^{(l)} = \sigma(Z^{(l)})$$

크기: $H^{(l)} \in \mathbb{R}^{B \times n_l}$. 한 번에 B개 sample을 처리.

GPU에서는 이 행렬 곱이 매우 효율적이다. CUDA의 cuBLAS, cuDNN이 최적화되어 있다.

### 4.3 출력에서 loss

마지막 layer 출력 $\hat{Y}$와 정답 $Y$로 loss 계산:
- 회귀: $L = \frac{1}{B} \sum_i (y_i - \hat{y}_i)^2$
- 분류: $L = -\frac{1}{B} \sum_i \sum_c y_{i,c} \log \hat{y}_{i,c}$

이 scalar loss를 최소화하는 게 학습 목표.

### 4.4 의사 코드

```python
def forward(X, weights):
    H = X
    for W, b in zip(weights[:-1], biases[:-1]):
        Z = H @ W.T + b
        H = relu(Z)
    Z_out = H @ weights[-1].T + biases[-1]
    Y_hat = softmax(Z_out)  # 또는 sigmoid, linear
    return Y_hat
```

이게 forward의 모든 것. 깊이만 다를 뿐 본질은 한 줄짜리 변환의 반복.

---

## 5. Backpropagation — 학습의 엔진

### 5.1 동기 — 어떻게 모든 가중치를 update하나

목표: $\frac{\partial L}{\partial W^{(l)}}$를 모든 $l$에 대해 계산. 그 후 SGD로 update:

$$W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}$$

**naive 방법**: 각 가중치를 살짝 바꾸고 loss 차이로 numerical gradient. 가중치가 1M개면 1M번 forward. 비현실적.

**Backprop의 통찰**: chain rule을 똑똑하게 사용하면 **forward 1번 + backward 1번**으로 모든 gradient 계산. 이게 50년대부터 알려진 자동미분(automatic differentiation)의 정신.

### 5.2 Chain rule 복습

$L = L(\hat{y}(z(W, h)))$일 때:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial W}$$

각 항을 따로 계산해서 곱한다. 이걸 layer마다 반복하면 모든 gradient.

### 5.3 한 layer의 backward — 4개 핵심 식

Layer $l$의 입력 $h^{(l-1)}$, 가중치 $W^{(l)}, b^{(l)}$, pre-activation $z^{(l)}$, 출력 $h^{(l)}$.

먼저 **delta** (오차 신호)를 정의:

$$\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}$$

이건 "layer $l$의 pre-activation을 살짝 바꾸면 loss가 얼마나 변하는가"의 정량화.

**식 1 — 출력층의 delta**:

분류면 $\delta^{(L)} = \hat{y} - y$ (softmax+CE의 마법). 일반적으로:

$$\delta^{(L)} = \nabla_{\hat{y}} L \odot g'(z^{(L)})$$

(g는 출력층 활성화)

**식 2 — delta의 backprop**:

$$\delta^{(l)} = ((W^{(l+1)})^T \delta^{(l+1)}) \odot \sigma'(z^{(l)})$$

직관: 다음 layer의 오차를 가중치를 거꾸로 통해 전파, 이번 layer의 활성화 미분으로 변환.

**식 3 — bias gradient**:

$$\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$

(bias는 z에 그냥 더해지므로 직접)

**식 4 — weight gradient**:

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (h^{(l-1)})^T$$

직관: layer의 입력과 오차의 외적.

이 네 식이 backprop의 모든 것이다. 한 번 외워두면 평생 쓴다.

### 5.4 식 2의 유도 — 왜 그렇게 되나

Layer $l$에서:
$$z^{(l+1)} = W^{(l+1)} h^{(l)} + b^{(l+1)} = W^{(l+1)} \sigma(z^{(l)}) + b^{(l+1)}$$

Chain rule:
$$\frac{\partial L}{\partial z^{(l)}} = \frac{\partial z^{(l+1)}}{\partial z^{(l)}} \cdot \frac{\partial L}{\partial z^{(l+1)}}$$

여기서:
- $\frac{\partial L}{\partial z^{(l+1)}} = \delta^{(l+1)}$
- $\frac{\partial z^{(l+1)}}{\partial z^{(l)}} = \frac{\partial z^{(l+1)}}{\partial h^{(l)}} \cdot \frac{\partial h^{(l)}}{\partial z^{(l)}} = W^{(l+1)} \cdot \text{diag}(\sigma'(z^{(l)}))$

(엄밀히는 행렬 미분. $h^{(l)} = \sigma(z^{(l)})$의 Jacobian이 대각 행렬이라 element-wise 곱으로 표현)

조합하면:
$$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$$

이게 식 2.

### 5.5 식 4의 유도

$$z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$$

Element-wise: $z^{(l)}_i = \sum_j W^{(l)}_{ij} h^{(l-1)}_j + b^{(l)}_i$.

$$\frac{\partial z^{(l)}_i}{\partial W^{(l)}_{ij}} = h^{(l-1)}_j$$

Chain rule:
$$\frac{\partial L}{\partial W^{(l)}_{ij}} = \delta^{(l)}_i \cdot h^{(l-1)}_j$$

행렬 형태:
$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (h^{(l-1)})^T$$

### 5.6 Backprop 알고리즘 전체

```python
# Forward (저장하면서)
H[0] = X
for l in range(1, L+1):
    Z[l] = W[l] @ H[l-1] + b[l]
    H[l] = sigma(Z[l]) if l < L else g(Z[l])
loss = L(H[L], Y)

# Backward
delta[L] = grad_L(H[L]) * g_prime(Z[L])
for l in range(L-1, 0, -1):
    delta[l] = (W[l+1].T @ delta[l+1]) * sigma_prime(Z[l])

# Gradients
for l in range(1, L+1):
    grad_W[l] = delta[l] @ H[l-1].T
    grad_b[l] = delta[l]

# Update
for l in range(1, L+1):
    W[l] -= eta * grad_W[l]
    b[l] -= eta * grad_b[l]
```

이게 SGD 한 step의 전체 과정. 라이브러리 (PyTorch, TensorFlow)는 이걸 자동으로 해준다 — autograd가 forward 그래프를 기록하고, `loss.backward()` 한 줄로 모든 gradient 계산.

### 5.7 자동미분과 backprop의 관계

**Backprop은 자동미분의 특수한 경우**다. 자동미분은 임의의 미분 가능 함수의 gradient를 chain rule로 자동 계산하는 일반 기법. 신경망의 backprop은 그중 특별히 효율적인 form이다.

자동미분에는 두 mode가 있다:
- **Forward mode**: 입력에서 출력으로. 입력 차원이 작을 때 효율 (예: 입력 1, 출력 多).
- **Reverse mode (= backprop)**: 출력에서 입력으로. **출력 차원이 작을 때** 효율 (예: 입력 1M, 출력 1).

신경망은 입력 多, 출력 (loss) 1 — reverse mode가 압도적으로 유리. 그래서 backprop이 표준.

면접 질문: "Numerical gradient와 backprop 차이?" 답은 "Numerical은 각 파라미터마다 forward 1번씩. Backprop은 전체 forward 1번 + backward 1번. 1M 파라미터면 1M배 차이."

### 5.8 Vanishing / Exploding Gradient 진단

식 2를 다시 보자: $\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$.

깊이 $l$에서 $\delta^{(l)}$의 크기를 추적하면:

$$\|\delta^{(l)}\| \approx \|W^{(l+1)}\| \cdot \|\sigma'(z^{(l)})\| \cdot \|\delta^{(l+1)}\|$$

매 layer마다 $\|W\| \cdot |\sigma'|$이 곱해진다. 이 product가:
- < 1이면 $\delta$가 0으로 → **vanishing**
- > 1이면 $\delta$가 폭발 → **exploding**

Sigmoid의 $|\sigma'| \le 0.25$, 가중치 $\|W\| \approx 1$ 가정하면 매 layer마다 0.25배 → 10층이면 $10^{-6}$. 학습 정지.

ReLU는 활성된 영역에서 $|\sigma'| = 1$, 비활성에서 0. 활성된 뉴런 비율이 50%면 평균 0.5 정도 — sigmoid보다 덜 vanish.

이게 ReLU + 적절한 init (He init, $\text{Var}(W) = 2/n_{in}$)이 깊은 망 학습을 가능하게 한 핵심 이유.

처방:
- **활성화**: ReLU 계열
- **Init**: He / Xavier
- **BN**: 활성화 분포 안정화
- **Skip connection**: gradient 직접 흐름 보장 (ResNet)

이건 다음 챕터(03_DNNs)에서 자세히.

---

## 6. Loss와 출력층의 짝 — 다시

### 6.1 매칭 표

| 출력층 활성화 | Loss | 해석 |
|---|---|---|
| Linear | MSE / MAE / Huber | 회귀. 가우시안/라플라스 noise 가정 |
| Sigmoid | Binary CE | 이진 분류. Bernoulli 가정 |
| Softmax | Categorical CE | 다중 분류. Categorical 가정 |
| Sigmoid (per class) | per-class BCE | Multi-label |

### 6.2 왜 Sigmoid + MSE는 안 좋은가

수학적으로 보자. Sigmoid 출력 $\hat{y} = \sigma(z)$, MSE loss:

$$L = \frac{1}{2}(y - \hat{y})^2$$

Gradient:
$$\frac{\partial L}{\partial z} = -(y - \hat{y}) \cdot \sigma'(z) = -(y - \hat{y}) \cdot \hat{y}(1-\hat{y})$$

문제: $\hat{y}$가 0 또는 1 근처면 $\hat{y}(1-\hat{y}) \approx 0$. 매우 자신 있게 틀려도 (예: $\hat{y} = 0.999$인데 $y = 0$) gradient가 $0.999 \cdot 0.001 = 0.001$로 작음. 학습이 거의 안 됨.

반면 sigmoid + BCE:

$$L = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$$

$$\frac{\partial L}{\partial z} = \hat{y} - y$$

(유도 생략, 깔끔하게 떨어짐)

자신 있게 틀리면 (예: $\hat{y} = 0.999, y = 0$) gradient = 0.999로 큼. 빠르게 교정.

이래서 분류는 항상 CE다.

### 6.3 Softmax + CE의 마법 다시

$$L = -\sum_c y_c \log \hat{y}_c, \quad \hat{y}_c = \frac{e^{z_c}}{\sum_k e^{z_k}}$$

Logit $z_c$에 대한 gradient:

$$\frac{\partial L}{\partial z_c} = \hat{y}_c - y_c$$

유도 (스케치):
$$\frac{\partial \hat{y}_c}{\partial z_j} = \hat{y}_c (\delta_{cj} - \hat{y}_j)$$

$$\frac{\partial L}{\partial z_j} = -\sum_c y_c \cdot \frac{1}{\hat{y}_c} \cdot \hat{y}_c (\delta_{cj} - \hat{y}_j) = -\sum_c y_c (\delta_{cj} - \hat{y}_j)$$

$$= -y_j + \hat{y}_j \sum_c y_c = \hat{y}_j - y_j$$

(마지막에 $\sum_c y_c = 1$ 사용)

이 깔끔함이 softmax + CE를 분류의 표준으로 만든 이유.

---

## 7. Universal Approximation Theorem — 약속과 한계

### 7.1 정리의 진술

**UAT (Cybenko 1989, Hornik 1991)**: Compact 영역에서 임의의 연속함수 $f$에 대해, 1 hidden layer + 충분히 많은 뉴런 + 적절한 비선형 활성화로 $\epsilon$ 정확도로 근사 가능. 즉:

$$\forall \epsilon > 0, \exists N, W, b, c \text{ s.t. } \left| f(x) - \sum_{i=1}^N c_i \sigma(w_i^T x + b_i) \right| < \epsilon, \forall x \in K$$

### 7.2 무엇을 약속하나

표현력. 신경망은 *원리적으로* 임의의 연속함수를 근사할 수 있다. 다른 말로, 신경망의 함수 클래스는 *연속함수 전체*의 dense subset이다.

이게 신경망이 그토록 다재다능한 이유의 이론적 근거. 이미지 분류든 음성 인식이든 번역이든, 풀려는 함수가 연속함수면 충분히 큰 신경망은 그걸 표현할 수 있다.

### 7.3 무엇을 약속하지 않나

여러 가지를 약속하지 않는다.

**(1) 학습 가능성**: UAT는 "그런 가중치가 *존재*한다"고 말한다. SGD로 그 가중치에 *수렴할* 수 있다고 말하지 않는다. 1 hidden layer로 모든 함수 근사 가능하지만, 실제로 SGD가 그 가중치를 찾는 건 거의 불가능. 학습이라는 동역학과 표현력은 별개.

**(2) 효율성**: 1 hidden layer로 표현 가능한 가중치 수가 *지수적*일 수 있다. Parity 함수 같은 건 1 layer로 $2^d$ 뉴런이 필요하지만 깊은 망으로는 $O(d)$. 깊이 없이 표현은 가능하지만 비현실적인 크기.

**(3) 일반화**: UAT는 train set에 fit하는 함수의 존재를 말한다. 그 함수가 test set에서 잘 동작한다고 말하지 않는다. 함수가 train data를 정확히 외워도 일반화는 별개 문제.

**(4) 활성화의 종류**: Sigmoid 가정에서 증명됐지만, 다른 비선형 (ReLU, tanh)에서도 비슷한 결과. 단 *어떤 비선형*인지는 본질적이다. 비선형 없으면 (선형 활성화) UAT 안 됨.

### 7.4 그래서 왜 deep?

UAT만 보면 "1 layer로 충분"이라 결론 내릴 수 있다. 실제로는 압도적으로 deep이 좋다. 왜?

**(1) 표현 효율**: 같은 함수를 deep으로 더 적은 파라미터로. Hastad의 결과는 parity 함수 같은 특정 함수는 shallow에서 지수적 vs deep에서 다항적.

**(2) 학습 용이성**: SGD가 deep을 잘 학습. shallow 큰 망은 학습이 어려운 경우 多.

**(3) Hierarchical representation**: 깊은 layer가 얕은 layer의 feature를 합성 → 자연 데이터의 계층적 구조를 잘 잡음.

**(4) Inductive bias**: Deep + 적절한 architecture (CNN의 sharing 등)가 자연 데이터에 맞는 강한 prior.

이래서 1989년의 UAT는 신경망의 잠재력을 보장했지만, 실제로 깊은 신경망의 시대가 오기까지 20년이 더 걸렸다 — 학습 가능성, 데이터, 연산이 다 따라붙어야 했다.

---

## 8. FNN의 한계 — 왜 CNN/RNN이 필요한가

### 8.1 파라미터 폭발

이미지 분류에 FNN을 쓰면? 224×224×3 = 150,528 입력. 첫 hidden layer 1000 뉴런이면 가중치 150M개. 한 layer에. 메모리·연산 폭발.

CNN은 weight sharing으로 해결. 3×3 conv 커널 256채널이면 가중치 9×256 = 2,304개. 6만 배 적음.

### 8.2 위치 정보 미활용

FNN은 입력 픽셀의 위치를 모른다. 픽셀 $x_{10,20}$과 $x_{50,60}$을 같은 가중치로 다른 비중으로 처리하지만, **같은 패턴이 다른 위치에 있어도 따로 학습**한다. 즉 "왼쪽 위 코너의 고양이 눈"과 "오른쪽 아래의 고양이 눈"을 별도로 인식해야 함.

이게 CNN의 동기 — translation equivariance. 같은 패턴을 어디서나 같은 응답으로.

### 8.3 가변 길이 처리 불가

FNN의 입력 차원은 고정. 시퀀스 데이터(텍스트, 음성)는 길이가 가변 → FNN 못 씀.

해결책 (1): **Sliding window**로 고정 길이만큼 자르기. 단 long-range 정보 잃음.
해결책 (2): **RNN** — 가변 길이 처리. 시간 축 weight sharing.

### 8.4 Inductive bias 부재

FNN은 **거의 inductive bias가 없다**. 입력 차원 사이의 어떤 구조도 가정 안 함. 이건 양날의 검:
- 장점: 매우 일반적, 어떤 데이터에도 적용 가능.
- 단점: 데이터의 본래 구조를 활용 못 함. 비효율.

CNN의 국소성·sharing, RNN의 시간 sharing, Transformer의 attention — 모두 데이터 구조에 맞춘 inductive bias.

### 8.5 그래도 FNN의 자리

FNN이 망한 건 아니다. 여러 자리에 여전히 사용:

- **표 데이터**: feature 간 위치/시간 의존이 없음. FNN이 자연.
- **Embedding 후의 후처리**: 추천 시스템에서 임베딩한 벡터를 받아 분류/회귀.
- **다른 모델의 일부**: CNN의 마지막 FC layer, Transformer의 FFN 부분.
- **Baseline**: 어떤 새 task든 FNN baseline부터.

이래서 FNN은 신경망의 *기본형*이고, CNN/RNN/Transformer는 FNN에 *구조적 제약*을 더한 변종이라는 시각이 정확하다.

---

## 9. 학습 실패 패턴 — FNN에서 자주 보는 것들

### 9.1 Train loss 안 떨어짐 (underfit)

| 원인 | 처방 |
|---|---|
| 모델이 너무 작음 | depth/width 키움 |
| 비선형 부족 (또는 부적절) | 활성화 점검 |
| LR 부적절 | LR range test |
| 입력 정규화 안 됨 | 평균 0, 분산 1로 |
| Init 잘못 | He / Xavier |
| Output-loss mismatch | softmax+CE 등 짝 확인 |

가장 흔한 원인은 **입력 정규화 안 함** (의외로 많이 발생). 이미지 [0, 255] 그대로 넣으면 학습 안 되거나 매우 느림. [-1, 1] 또는 [0, 1]로.

### 9.2 Train ↓ Val ↑ (overfit)

| 처방 | 강도 |
|---|---|
| Early stopping | 가장 단순, 항상 first |
| Dropout | 0.3~0.5 |
| Weight decay | 1e-3 ~ 1e-5 |
| Data augmentation | task별 |
| 모델 크기 줄임 | 마지막 수단 |
| 더 많은 데이터 | 항상 좋음, 비용 高 |

### 9.3 NaN / 발산

| 원인 | 처방 |
|---|---|
| LR 너무 큼 | 1/10 |
| log(0) | log_softmax, BCEWithLogitsLoss |
| Init 잘못 (분산 폭발) | 적절한 init |
| 데이터에 NaN/inf | 데이터 점검 |
| Mixed precision overflow | Loss scaling |

### 9.4 모든 출력 같음

| 원인 | 처방 |
|---|---|
| Init이 모두 0 | Symmetry 깨는 init |
| Dying ReLU 누적 | LeakyReLU, He init |
| 출력층 학습 안 됨 | LR, 출력층 init 점검 |

### 9.5 Loss 떨어지는데 accuracy 안 오름

| 원인 | 처방 |
|---|---|
| 클래스 불균형 | weighted loss, focal |
| Threshold 부적절 | 0.5 외 시도 |
| Loss와 metric 괴리 | metric 직접 최적화 시도 |

---

## 10. 면접 단골 Q&A

### Q1. 비선형 활성화가 없으면?
"전체 망이 단일 선형 변환으로 환원됩니다. $W_2(W_1 x + b_1) + b_2 = W'x + b'$. 100층 1000층 쌓아도 logistic regression 수준의 표현력. XOR도 못 풀어요. 비선형이 신경망의 표현력의 원천이고, UAT의 본질도 비선형에 있습니다."

### Q2. ReLU vs Sigmoid?
"세 측면입니다. (1) Vanishing — sigmoid는 미분 최대 0.25라 깊은 망에서 gradient 사라짐, ReLU는 활성 영역에서 1로 잘 흐름. (2) 계산 — ReLU는 max만 하면 되어 매우 빠름. (3) Sparse — ReLU는 절반이 0 출력으로 sparse representation. 단점은 dying ReLU지만 LeakyReLU나 He init으로 완화. 그래서 hidden layer는 ReLU가 거의 default."

### Q3. Softmax + CE가 분류 표준인 이유?
"두 가지입니다. (1) 확률적 — softmax는 logit을 합 1인 분포로, CE는 categorical distribution의 negative log-likelihood. 통계적으로 자연스러운 선택. (2) Gradient의 마법 — softmax+CE의 logit gradient가 정확히 (예측 - 정답)으로 깨끗하게 떨어집니다. saturation 영역에서도 gradient 살아 있어 학습이 잘 됩니다."

### Q4. Backprop이 왜 효율적인가?
"Chain rule을 출력에서 입력 방향으로 적용해서 gradient를 재사용합니다. 1번의 forward + 1번의 backward로 모든 가중치의 gradient를 동시 계산. Numerical gradient는 가중치마다 forward 1번씩 필요해서 1M 가중치면 1M배 느립니다. Backprop은 자동미분의 reverse mode 특수 case로, 출력이 scalar(loss)인 신경망에 매우 적합합니다."

### Q5. Universal Approximation Theorem의 의미?
"표현력 보장 정리입니다. 1 hidden layer + 충분한 뉴런 + 비선형이면 임의 연속함수를 임의 정확도로 근사 가능. 단 약속 안 하는 게 많아요. (1) 학습 가능성 — 그런 가중치가 존재하는 것과 SGD로 찾는 것은 별개. (2) 효율성 — 1 layer로 표현하려면 뉴런이 지수적으로 필요할 수 있음. (3) 일반화 — train fit과 test 성능은 다름. 그래서 실용적으로는 deep이 압도적으로 유리합니다."

### Q6. 왜 신경망에 깊이가 필요한가?
"네 가지 이유입니다. (1) 표현 효율 — 같은 함수를 적은 파라미터로. parity 함수는 1 layer에 지수적, deep에 다항적. (2) 학습 용이성 — SGD가 deep을 잘 학습. (3) Hierarchical representation — 자연 데이터의 계층 구조와 잘 맞음. (4) Inductive bias — 적절한 architecture 위에서 강한 prior. 그래서 UAT의 1-layer 보장이 실용에선 deep으로 갑니다."

### Q7. Dying ReLU 어떻게 막나?
"세 가지입니다. (1) He initialization — ReLU 가정 분산 2/n_in으로 분산 폭발 방지. (2) LR을 너무 크게 안 하기 — 큰 음수 gradient가 가중치를 음의 영역으로 보내서 dying 발생. (3) LeakyReLU 사용 — 음수에서도 0.01x 정도의 작은 gradient 흐름. 큰 모델에서는 dying이 누적되면 effective capacity가 줄어들기 때문에 의식적으로 모니터링."

### Q8. 출력층 활성화와 loss를 짝지어야 하는 이유?
"수치적·통계적 정합성 때문입니다. Sigmoid+BCE, Softmax+CE는 logit gradient가 (예측-정답)으로 깨끗하게 떨어집니다. 잘못 짝지으면 (예: Softmax+MSE) gradient가 saturation 영역에서 0에 가까워져 학습 안 됨. 또 통계적으로도 각 활성화는 특정 분포 가정 (sigmoid=Bernoulli, softmax=categorical, linear=gaussian)에 대응되고, 짝의 loss는 그 분포의 NLL입니다."

### Q9. FNN으로 이미지 분류 안 하는 이유?
"세 가지입니다. (1) 파라미터 폭발 — 224×224×3 입력, 1000 hidden 첫 layer만 150M 가중치. (2) 위치 정보 미활용 — 같은 패턴이 다른 위치에 있어도 별도 학습. (3) Translation 약함 — 입력 살짝 이동하면 출력 크게 변함. CNN은 weight sharing + 국소성으로 이 셋 모두 해결. 그래서 이미지엔 거의 항상 CNN."

### Q10. Backprop의 4개 핵심 식이 뭐였나?
"(1) 출력층 delta = $\nabla_{\hat{y}} L \odot g'(z^{(L)})$. (2) Delta 역전파 = $((W^{(l+1)})^T \delta^{(l+1)}) \odot \sigma'(z^{(l)})$. (3) Bias gradient = delta 그대로. (4) Weight gradient = delta와 입력의 외적 = $\delta^{(l)} (h^{(l-1)})^T$. 이 네 식이 backprop의 모든 것. layer 종류와 무관하게 같은 패턴."

---

## 11. 생각해보라 — 단락 답안

**Q. 왜 모든 layer가 같은 활성화를 쓰는가? 다르게 쓰면?**

이론적으로는 layer마다 다른 활성화를 쓸 수 있다. 실용적으로는 거의 모든 모델이 같은 활성화를 쓴다. 이유는 (1) **단순함** — architecture 설계 단순화. (2) **이론적 분석** — UAT 같은 정리가 동질적 활성화 가정. (3) **실증적으로 별 차이 없음** — 다양한 활성화 mixing이 큰 이득 없음. 단 *출력층*은 task에 따라 달라야 함 (sigmoid/softmax/linear). 또 LSTM의 gate는 sigmoid, hidden은 tanh로 자리에 따라 다른 활성화. 즉 "기능적으로 다른 자리는 다른 활성화"가 합리적. 일반 hidden은 같은 활성화로.

**Q. Backprop과 인간 뇌의 학습?**

흥미로운 이론적 질문. 인간 뇌는 *역방향 신호 전파*가 본질적이지 않다는 게 신경과학의 합의 — synapse는 단방향 정보 전달이 강하고, 뇌는 backprop처럼 정확한 chain rule을 계산하지 않는 것으로 보인다. 그럼 어떻게 학습할까? 가설은 (1) **Hebbian learning** — "함께 활성된 뉴런은 연결이 강화" (cells that fire together wire together). 단순한 local rule. (2) **Predictive coding** — 뇌가 예측을 하고 오차를 받아 update. (3) **Feedback alignment** — 정확한 backprop이 아닌 random feedback도 surprisingly 잘 동작. 신경과학 + ML 융합의 활발한 연구. 결론: backprop은 학습이라는 일반적 문제의 *한 풀이*이지 *유일한 풀이*가 아님.

**Q. 왜 weight sharing이 신경망의 모든 변종에서 핵심인가?**

CNN의 공간 sharing, RNN의 시간 sharing, Transformer의 토큰 sharing — 모두 같은 정신. **같은 함수가 다른 위치/시간에서 동일하게 작동해야 한다는 inductive bias**. 자연 데이터는 이 성질을 자주 가진다 — 고양이 눈은 어디에 있든 고양이 눈, "the"는 문장 어디에 와도 "the". 이 prior를 모델 구조에 강제하면 (1) 파라미터 효율, (2) 학습 효율 (한 위치 학습이 모든 위치 적용), (3) generalization 향상 (보지 못한 위치도 자동 처리). FNN이 sharing 없이도 표현력은 있지만, 이 효율성을 못 누려서 자연 데이터엔 비효율적. Sharing은 단순한 최적화 트릭이 아니라 데이터의 구조와 모델의 구조를 정렬하는 깊은 설계 원리.

**Q. Backprop이 vanishing gradient에서 약한 이유?**

식 자체가 곱셈의 누적이기 때문. $\delta^{(l)} = \prod_k W^{(k)} \sigma'(\cdot)$ — 매 layer마다 곱이 들어간다. 이 곱이 < 1 이면 지수적으로 줄어들고, > 1이면 폭발. 곱셈 구조는 매우 sensitive — 작은 변화가 누적되면 큰 차이. 이게 **chain rule의 본질적 약점**. 해결책들:
- *덧셈 구조 만들기*: ResNet의 skip connection. $h_{l+1} = h_l + F(h_l)$이라 $\partial h_{l+1}/\partial h_l = I + \partial F$ — identity 항이 있어 gradient가 직접 흐름.
- *분포 안정화*: BN으로 매 layer 통계 안정 → $|\sigma'|$의 평균이 안정.
- *Gradient 직접 흐름 구조*: LSTM의 cell state — additive update로 vanishing 우회.

이 모든 후속 연구가 chain rule의 곱셈 누적을 어떻게 *덧셈으로 바꾸느냐*의 변주. 그래서 ResNet과 LSTM이 본질적으로 같은 정신.

**Q. UAT가 보장하는 1 hidden layer 신경망을 실제로 만들어 본다면?**

이론은 가능하지만 실용적으론 거의 불가능에 가깝다. 예시: ImageNet 분류 함수를 1 hidden layer로 표현하려면 뉴런이 몇 개 필요할까? 정확한 수는 모르지만 *지수적*으로 많을 것. ResNet50은 25M 파라미터로 76% 정확도. 같은 함수를 1 layer로 표현하려면 $10^{20}$ 정도일 수도 있다 (추측). 메모리·연산 모두 비현실적.

또 학습이 안 된다. 1 layer 큰 망의 loss landscape는 매우 험난해서 SGD가 좋은 minimum에 안 간다. 그래서 UAT는 *함수 클래스의 표현력* 보장이지 *학습 가능 모델의 보장*이 아니다. 이론과 실용의 gap이 신경망 연구의 큰 부분이다.

**Q. 왜 자동미분이 산업의 표준이 됐나?**

(1) **개발자 효율**: Chain rule을 손으로 적는 건 큰 모델에서 사실상 불가능. 자동미분이 라이브러리화되면서 모델 정의가 forward만 적으면 되는 단계로 단순화. PyTorch의 `loss.backward()` 한 줄.
(2) **버그 감소**: 사람이 적은 미분 식은 항상 버그 위험. 자동미분은 정확.
(3) **빠른 prototyping**: 새 architecture 시도가 forward 함수만 바꾸면 됨. RL, 메타학습 같은 복잡한 학습 흐름도 가능.
(4) **GPU 최적화**: cuBLAS, cuDNN이 자동미분 backbone과 연결되어 매우 빠름.
(5) **확장성**: 미분 가능한 모든 연산이 자동으로 학습 가능. attention 같은 복잡한 연산도 자동 처리.

자동미분이 없었다면 딥러닝의 폭발도 없었을 것. 알고리즘만큼이나 도구가 분야를 만든다.

---

## 12. 한 줄 요약 (시험 직전)

- **신경망 = 선형 변환 + 비선형 활성화의 합성**. 비선형 빼면 선형 모델로 환원.
- **ReLU가 표준** — vanishing 약함, 빠름, sparse. Dying ReLU는 He init + LeakyReLU.
- **Backprop의 4개 식**: 출력 delta, delta 역전파, bias gradient, weight gradient. Chain rule의 효율적 적용.
- **Softmax + CE의 logit gradient = 예측 - 정답**, 깔끔. 분류의 표준.
- **UAT**: 표현력 보장하지만 학습 가능성·효율성·일반화는 별개.
- **왜 deep**: 표현 효율, 학습 용이, 계층 구조, inductive bias.
- **FNN의 한계**: 파라미터 폭발, 위치 정보 미활용, 가변 길이 불가 → CNN/RNN의 동기.
- **Vanishing gradient**의 본질 = chain rule의 곱셈 누적. 해결 = additive 구조 (ResNet, LSTM).
