# 02. Feedforward Neural Networks (FNN / MLP) — 오픈북 정리

> 핵심: **퍼셉트론이 왜 안 되었고, 왜 layer를 쌓고 비선형을 넣으면 풀리는가?**

---

## 0. 큰 그림

FNN = "입력 → 은닉층(여러 개) → 출력"으로 신호가 **한 방향**으로 흐르는 망. 순환 X, 가중치 sharing X.

**왜 FNN이 ML의 기본?** → 임의의 함수를 근사할 수 있는 가장 일반적 형태(UAT). CNN/RNN도 결국 FNN의 **구조적 제약** 버전.

---

## 1. Perceptron — "왜 한계가 있었나?"

### 1.1 구조
$$y = \text{sign}(w^T x + b)$$
선형 분리 가능한 문제만 학습 가능.

### 1.2 XOR 문제 (★시험 단골)
| x₁ | x₂ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

XOR는 **선형 분리 불가능** → 단일 퍼셉트론으로 못 푼다 (Minsky & Papert, 1969).

생각해보라: 왜 선형 분리 불가능? → 두 점(0,0)·(1,1) vs (0,1)·(1,0)을 직선 하나로 가르는 건 기하적으로 불가능.

### 1.3 해결: 층을 쌓고 비선형 활성화
은닉층 1개 + 비선형 → XOR 풀린다. **이것이 MLP가 등장한 이유**.

---

## 2. MLP 구조 — "왜 layer를 쌓는가?"

$$h^{(1)} = \sigma(W^{(1)} x + b^{(1)})$$
$$h^{(2)} = \sigma(W^{(2)} h^{(1)} + b^{(2)})$$
$$\hat{y} = W^{(L)} h^{(L-1)} + b^{(L)}$$

각 layer는 **표현(representation)을 변환**하는 역할:
- 1층: low-level feature
- 2층: feature의 조합
- 깊을수록 추상적 표현

**왜 깊을수록 좋은가?** (이론적)
- Shallow는 같은 함수를 표현하려면 **지수적으로 많은** 뉴런 필요 (parity 함수 등).
- 계층적 합성으로 **효율적 표현**.

**왜 깊을수록 어려운가?** (실용)
- Vanishing gradient (DNN 챕터에서 자세히).

---

## 3. Activation Function — "왜 비선형이 필수인가?" (★핵심)

비선형이 없으면:
$$W^{(2)}(W^{(1)} x + b^{(1)}) + b^{(2)} = W' x + b'$$
→ 아무리 쌓아도 **선형 모델 1개**와 같다. 깊이의 의미가 사라진다.

### 3.1 종류 비교 (★시험 단골 표)

| Activation | 식 | 출력 범위 | 장점 | 단점 |
|---|---|---|---|---|
| **Sigmoid** | 1/(1+e⁻ˣ) | (0,1) | 확률 해석 | **vanishing gradient**, 0-centered 아님, exp 비용 |
| **Tanh** | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | (−1,1) | 0-centered | vanishing gradient 여전 |
| **ReLU** | max(0,x) | [0,∞) | 빠름, vanishing 약함 | **dying ReLU** (x<0에서 gradient 0) |
| **Leaky ReLU** | max(αx, x) | ℝ | dying ReLU 완화 | α 튜닝 |
| **ELU/SELU** | x>0: x, x≤0: α(eˣ−1) | (−α,∞) | smooth, 평균 0 근처 | exp 비용 |
| **GELU** | x·Φ(x) | ℝ | smooth, Transformer 표준 | 비용 |
| **Softmax** | eˣⁱ/Σeˣⱼ | 확률분포 | 분류 출력 | 출력층 전용 |

**왜 ReLU가 표준?**
1. 양의 영역에서 gradient = 1 → vanishing 약함.
2. 0 출력을 통한 **sparse activation**.
3. 계산이 단순(max 연산).
4. 생물학적 관찰과 부합(rate coding 일부 유사).

**Dying ReLU**: 큰 음수 gradient로 w가 한 번 크게 음수로 가면 그 뉴런은 항상 0 출력 → 영원히 학습 안됨.
→ Leaky ReLU, He init으로 완화.

생각해보라: 왜 sigmoid가 vanishing gradient를 일으키나? → σ'(x) = σ(x)(1−σ(x))는 max 0.25. 깊은 망에서 gradient를 곱하면 0.25ⁿ → 0.

---

## 4. Forward Propagation

각 층에서:
$$z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}, \quad h^{(l)} = \sigma(z^{(l)})$$

출력층에서 loss 계산:
- 회귀: MSE
- 이진 분류: sigmoid + BCE
- 다중 분류: softmax + CE

---

## 5. Backpropagation — "왜 chain rule인가?"

목표: $\partial L / \partial W^{(l)}$를 계산해 SGD로 갱신.

### 5.1 핵심 아이디어
연쇄 미분(chain rule)을 **재사용** — 출력층에서 입력층 방향으로 한 번만 forward, 한 번만 backward.

$$\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}} = \delta^{(l)} \cdot (h^{(l-1)})^T$$

여기서 $\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$

**왜 이게 효율적?** → naive하게 모든 경로를 따로 계산하면 지수적, chain rule 재사용으로 다항.

### 5.2 backprop의 4개 핵심 식
1. 출력층 오차: $\delta^{(L)} = \nabla_{\hat{y}} L \odot \sigma'(z^{(L)})$
2. 역전파: $\delta^{(l)} = ((W^{(l+1)})^T \delta^{(l+1)}) \odot \sigma'(z^{(l)})$
3. bias gradient: $\partial L / \partial b^{(l)} = \delta^{(l)}$
4. weight gradient: $\partial L / \partial W^{(l)} = \delta^{(l)} (h^{(l-1)})^T$

생각해보라: backprop과 numerical gradient 차이? → numerical은 O(파라미터 수)번 forward, backprop은 단 1번. 자동미분 = backprop의 일반화.

---

## 6. Loss Function의 매칭 (왜 짝이 정해져 있나?)

| 출력층 | Loss | 이유 |
|---|---|---|
| Linear | MSE | 회귀, 가우시안 가정 |
| Sigmoid | BCE | 이진, MLE 등가, gradient 깔끔 |
| Softmax | CE | 다중, MLE, gradient = (ŷ − y) |

**왜 sigmoid+MSE는 안 되나?** → sigmoid 포화 영역에서 σ' ≈ 0이라 gradient 죽음. CE는 σ'가 분모에서 상쇄되어 살아있다.

---

## 7. Universal Approximation Theorem — "이론적 보장"

**진술**: 은닉층 1개 + 충분한 뉴런 + 적절한 비선형이면 임의의 연속함수를 ε 정확도로 근사 가능.

**의미**:
- 표현력은 1-layer로도 충분 (이론적으로).
- 단, **얼마나 많은 뉴런 필요?** → 지수적일 수 있다.
- **어떻게 학습?** → UAT는 학습 가능성을 보장하지 않음.

**그래서 왜 deep?** → 같은 함수를 **훨씬 적은 파라미터**로 표현 가능 + 학습이 더 잘됨(empirically).

---

## 8. FNN의 한계 — "왜 CNN/RNN이 따로 필요?"

| 데이터 | FNN의 문제 | 필요한 구조적 제약 |
|---|---|---|
| 이미지 | 픽셀 수 = 입력 수 → 파라미터 폭발. 위치 이동에 약함. | **Local connectivity, weight sharing** → CNN |
| 시퀀스(텍스트, 음성) | 길이가 가변. 순서 정보 못 살림. | **Recurrent / attention** → RNN, Transformer |
| 그래프 | 노드 순서 임의. | GNN |

**핵심 통찰**: 데이터의 **구조적 사전지식(inductive bias)**을 모델 구조로 인코딩하면, 같은 데이터로 더 잘 학습된다. FNN은 inductive bias가 거의 없어 일반적이지만 비효율적.

---

## 9. 학습 실패 패턴 (왜 안 되는가?)

| 증상 | 원인 | 처방 |
|---|---|---|
| Train loss 안 떨어짐 | 모델 작음/비선형 부족/LR 부적절 | 모델 키움, ReLU, LR 조정 |
| Train ↓ Val ↑ | overfit | regularization, early stop, 데이터↑ |
| 처음부터 NaN | LR 큼, init 부적절, log(0) | LR↓, 적절한 init, eps 추가 |
| 모든 출력 같음 | Dead neuron, init 문제 | He init, LeakyReLU |

---

## 10. 실제 적용 시 체크리스트

1. 입력 정규화 (평균 0, 분산 1) — 안 하면 학습 매우 느림.
2. 적절한 init (He for ReLU, Xavier for tanh).
3. Mini-batch SGD + Adam.
4. Validation으로 early stop.
5. 출력층-loss 매칭 확인.

---

## 11. 한 줄 요약

- **왜 비선형 활성화?** → 없으면 깊이가 의미 없음, 선형 모델로 환원.
- **왜 ReLU?** → vanishing 약함, 빠름, sparse.
- **왜 backprop?** → chain rule 재사용으로 효율적 미분.
- **왜 deep?** → 같은 표현을 적은 파라미터로 + 학습 잘됨.
- **UAT의 의미와 한계**: 표현력 OK, 학습 가능성은 별개.
- **FNN 한계**: 구조적 사전지식 부족 → 이미지·시퀀스에 비효율 → CNN/RNN 등장.
