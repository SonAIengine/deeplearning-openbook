# 09. Architecture & Hyperparameter Decisions — "왜 이 숫자, 이 구조?" 심층

> **이 문서의 핵심 사고**: 모든 디자인 선택에는 *이유*가 있어야 한다. "depth=12로 했어요"가 답이 아니라, "왜 12, 그리고 16/8과 비교해 어떻게 다른가, 어떤 신호로 그 결정을 했는가"가 답.
>
> 결정 = **양 끝 비교 + 신호 진단 + Trade-off 명시**. 이 세 단계가 모든 hyperparameter 결정의 골격.

---

## 0. 사고법 — 결정의 근거를 답하기

### 0.1 4축 분석 패턴

설계 결정은 항상 다음 4축으로:

1. **무엇을 결정?**: 정확한 hyperparameter 또는 architecture 선택.
2. **양 끝 비교**: 작게/크게의 양 극단에서 무엇이 변하나.
3. **어떤 신호로 판단?**: Loss 곡선, gradient norm, val 성능 등.
4. **Trade-off의 본질**: 무엇과 무엇 사이의 균형인가.

이 네 축을 답할 수 있으면 hyperparameter 결정이 *체계적*. 시험·면접에서도 강력.

### 0.2 결정의 우선순위

모든 결정이 같은 영향을 주진 않음. 일반적 민감도 순서:

| 순위 | Hyperparameter | 영향도 |
|---|---|---|
| 1 | Learning Rate | 가장 큼 |
| 2 | Batch Size | 큼 (LR과 짝) |
| 3 | Optimizer | 큼 |
| 4 | Init | 중간 (BN 있으면 작음) |
| 5 | Activation | 중간 |
| 6 | Architecture (depth/width) | 중간 |
| 7 | Regularization 강도 | 작음~중간 |
| 8 | LR Schedule | 작음~중간 |

이 우선순위에 따라 *시간 투자*. LR 결정에 가장 많은 시간, schedule은 default 시도 후 조정.

### 0.3 본 챕터의 흐름

§1~5: 가장 민감한 결정
§6~10: 중간 민감도
§11~15: 일반 가이드
§16: 결정 매트릭스
§17: 면접 Q&A
§18: 생각해보라

---

## 1. Learning Rate — 가장 민감한 단일 결정

### 1.1 양 끝 비교

**LR 너무 큼**:
- 첫 epoch부터 NaN 또는 발산.
- Loss가 무한히 진동.
- BN 통계 흔들림.
- 학습 시작도 못함.

**LR 너무 작음**:
- 매우 느린 학습. 100 epoch에 진전 미미.
- Local minimum/saddle point에 갇힘.
- Plateau에서 못 빠져나감.
- Sub-optimal solution.

### 1.2 LR Range Test (Smith 2017)

LR 결정의 표준 절차:

1. LR을 매우 작게 시작 (예: $10^{-7}$).
2. 매 mini-batch마다 LR을 *지수적으로 증가*.
3. Loss 추적.
4. Loss가 *급격히 떨어지는 구간* 찾음.
5. 그 구간의 1/10 정도가 적절한 시작 LR.

```python
# 의사 코드
lr = 1e-7
losses = []
while lr < 10:
    train_one_batch(lr)
    losses.append(loss)
    lr *= 1.1  # 매 step 10% 증가
plot(lr, losses)
# 가파르게 떨어지는 곳의 1/10
```

이게 가장 신뢰할 수 있는 LR 시작점 결정 방법.

### 1.3 일반 가이드 (default 값)

| Optimizer | 시작 LR |
|---|---|
| SGD + momentum | 0.1 (CV), 0.01 (NLP) |
| Adam | $10^{-3}$ (일반) |
| AdamW (Transformer) | $10^{-4} \sim 5 \times 10^{-4}$ |
| AdaFactor (큰 모델) | $10^{-3}$ |

이게 *시작점*이지 *최종*이 아님. LR Range Test 또는 grid search로 조정.

### 1.4 신호로 진단

**LR 너무 큼의 신호**:
- 첫 epoch부터 NaN.
- Train loss가 진동만 하고 안 떨어짐.
- Gradient norm이 매우 큼 (1e3+).

**LR 너무 작음의 신호**:
- Loss가 매우 천천히 떨어짐.
- Gradient norm 작음 (1e-3 이하).
- 100 epoch 후에도 train accuracy 낮음.

**적절한 LR의 신호**:
- 첫 1~2 epoch에서 큰 loss 감소.
- 안정적 수렴.
- Val accuracy가 의미 있는 성장.

### 1.5 LR Schedule

고정 LR은 sub-optimal. 학습 진행에 따라 조정.

**Step decay**: 30, 60, 90 epoch마다 1/10. 단순, ImageNet 표준.
**Cosine annealing**: 부드러운 감소. Transformer 표준.
**Warmup**: 0에서 본 LR로 점진 증가. BN/Adam 안정.
**OneCycle**: warmup → 큰 LR → 감소.

**Warmup이 왜 필요한가** — 큰 모델·Adam에선 학습 초기 통계가 noisy. 큰 LR로 시작하면 발산 위험. 작게 시작해 통계 안정 후 키움.

### 1.6 답안 골격

> "LR이 가장 민감한 hyperparameter. 결정 절차: (1) LR Range Test로 적절 구간 찾기 — LR을 점진적 증가하며 loss 가파르게 떨어지는 구간의 1/10. (2) Optimizer별 default 시작점 — Adam 1e-3, SGD 0.1, AdamW (Transformer) 1e-4. (3) Schedule 적용 — cosine + warmup이 큰 모델 표준, step decay가 ImageNet 표준. (4) 신호로 진단 — NaN이면 1/10, 너무 느리면 키움."

---

## 2. Batch Size — LR과 짝

### 2.1 양 끝 비교

**작은 batch (16~64)**:
- Gradient noise 多 → flat minima 선호 → 일반화 ↑
- BN 통계 noisy
- GPU 비효율 (작은 행렬 곱)
- 진동 큼
- 학습 수동적 안정

**큰 batch (1024~)**:
- Gradient noise 적음 → sharp minima 위험
- BN 통계 정확
- GPU 효율 ↑
- 진동 적음
- 일반화 약간 손해

### 2.2 LR과의 관계 — Linear Scaling Rule

Batch×k면 LR×k. 이유: 같은 epoch에서 update 횟수 1/k → 한 번 update를 k배 크게.

```
Original: batch=256, LR=0.1
Scaled: batch=1024, LR=0.4
```

단 매우 큰 batch (4096+)에서는 깨짐. LARS, LAMB 같은 전용 optimizer.

### 2.3 신호로 진단

**작은 batch가 좋다는 신호**:
- 큰 batch에서 val accuracy 떨어짐 (sharp minima).
- BN 통계 안정 (이미 충분한 batch).

**큰 batch가 필요한 신호**:
- BN 통계 매우 noisy (batch < 8).
- GPU 활용 < 50%.
- 메모리 충분.

### 2.4 일반 가이드

| 메모리 / GPU | Batch |
|---|---|
| 매우 작은 (8GB) | 16~32 |
| 일반 (16GB) | 64~256 |
| 큰 (32GB+) | 512~1024 |
| Multi-GPU | 1024~4096 |
| 매우 큰 cluster | 4096+ (LARS) |

선택 후 LR linear scaling. Sweet spot 보통 256~1024.

### 2.5 답안 골격

> "Batch size는 LR과 짝. 작은 batch는 noise → flat minima → 일반화 ↑, 단 GPU 비효율. 큰 batch는 정확한 gradient + GPU 효율, 단 일반화 약간 손해. 표준 sweet spot 256~1024. 변경 시 linear scaling rule (batch×k면 LR×k). 매우 큰 batch (4096+)에선 LARS/LAMB 전용 optimizer."

---

## 3. Optimizer 선택

### 3.1 Optimizer별 비교

| Optimizer | 장점 | 단점 | 언제? |
|---|---|---|---|
| **SGD** | 단순, 메모리 적음 | 진동, 평탄 영역 느림 | 거의 안 씀 |
| **SGD + Momentum** | 진동 줄임, 빠름 | LR 튜닝 부담 | ImageNet CNN |
| **Adam** | 빠른 수렴, LR 튜닝 쉬움 | Sharp minima 경향 | 일반 default |
| **AdamW** | Adam + 정확한 weight decay | 약간 더 복잡 | Transformer 표준 |
| **AdaFactor** | 메모리 매우 적음 | 약간 느림 | 매우 큰 모델 |
| **LARS** | 매우 큰 batch 가능 | 복잡 | batch 4096+ |
| **LAMB** | LARS의 Adam version | 복잡 | batch 4096+ |
| **Lion** (2023) | 메모리 적음, sign-only | 비교적 새로움 | 큰 모델 (실험) |

### 3.2 결정 트리

```
어떤 task / 모델?
├── ImageNet CNN → SGD+momentum (일반화 약간 우위)
├── Transformer → AdamW + warmup + cosine
├── 작은 데이터·복잡 loss → Adam (안전)
├── 강화학습 → RMSProp / Adam
├── 큰 batch (4096+) → LARS / LAMB
└── 모르겠음 → AdamW
```

### 3.3 Adam vs SGD — 실증

여러 보고:
- ImageNet 분류: SGD+momentum이 약 0.5~1% 더 나은 일반화.
- NLP Transformer: AdamW 압도. SGD는 매우 어려움.
- 작은 데이터: Adam이 안전.
- 강화학습: Adam/RMSProp.

이유:
- SGD의 noise가 implicit regularization → flat minima.
- Adam의 sharp minima 경향 → 일반화 약간 손해.
- 단 Adam은 빠른 수렴 + LR 튜닝 부담 적음.

### 3.4 답안 골격

> "Task와 데이터에 따라. (1) ImageNet CNN — SGD+momentum 약간 우위 (일반화). (2) Transformer — AdamW 표준 (warmup + cosine). (3) 작은 데이터 — Adam 안전. (4) 큰 batch (4096+) — LARS/LAMB. 결정 신호: Adam에서 일반화 gap 크면 SGD 시도, SGD에서 학습 정체면 Adam 시도. 둘 다 시도 후 val로 결정이 정석."

---

## 4. Initialization

### 4.1 Init 종류와 활성화 매칭

| Init | 분산 | 활성화 |
|---|---|---|
| **Xavier (Glorot)** | $2/(n_{in}+n_{out})$ | Tanh, Sigmoid (대칭) |
| **He (Kaiming)** | $2/n_{in}$ | ReLU 계열 |
| **LeCun** | $1/n_{in}$ | SELU |

### 4.2 He init이 2/n_in인 이유

ReLU는 음수 절반을 0으로 만듦. 분산이 절반으로 줄어듦. 이걸 보상하려면 가중치 분산을 두 배 → $2/n_{in}$.

### 4.3 Bias init

대부분 0. 예외:
- **LSTM forget gate**: 1~2로. 정보 보존 시작점.
- **ReLU 직전**: 작은 양수 (0.01)도 가능 — dying ReLU 약간 막음.

### 4.4 신호 — 잘못된 init

| 증상 | 원인 |
|---|---|
| 첫 forward 활성값 모두 0 | 분산 너무 작음 |
| 첫 forward 활성값 NaN/inf | 분산 너무 큼 |
| Gradient 모두 0 | Vanishing init |
| Gradient 모두 NaN | Exploding init |
| 모든 뉴런 같은 출력 | Symmetry 안 깨짐 (모두 0) |

### 4.5 BN과의 관계

BN이 활성값 분포를 강제 정규화 → init 영향 줄어듦. 그래도 적절한 init은 학습 안정성에 도움.

매우 깊은 망 (50+) + BN: He init이 안전한 default.

### 4.6 답안 골격

> "활성화에 맞춰 결정. ReLU → He, Tanh/Sigmoid → Xavier, SELU → LeCun. He의 분산 $2/n_{in}$은 ReLU의 분산 손실 보정. Bias 0이 default, LSTM forget만 1~2. BN/LN 있으면 init 영향 작아지지만 적절한 init이 학습 시작 안정성에 중요."

---

## 5. Activation Function 선택

### 5.1 결정 트리

```
출력층?
├── 이진 분류 → Sigmoid (+ BCE)
├── 다중 분류 → Softmax (+ CE)
├── 회귀 → Linear (+ MSE)
└── Multi-label → 클래스별 Sigmoid

은닉층?
├── 일반 CNN/MLP → ReLU
├── Transformer → GELU
├── RNN의 hidden → Tanh
├── LSTM의 gate → Sigmoid (0~1 비율)
├── Dying ReLU 우려 → LeakyReLU
└── 매우 큰 모델 → GELU/Swish
```

### 5.2 신호로 진단

**Dying ReLU**:
- 많은 뉴런이 항상 0 출력.
- Layer activation의 분산이 매우 작음.
- 처방: He init, LeakyReLU.

**비대칭 출력**:
- 출력이 모두 양수 또는 음수.
- 다음 layer에 zigzag 학습.
- 처방: 0-centered 활성화 (Tanh, GELU).

### 5.3 GELU vs ReLU

GELU: $x \cdot \Phi(x)$ — 부드러운 ReLU.

GELU 장점:
- 부드러움 → 학습 동역학 안정.
- 음수 영역에서 작은 gradient → dying 방지.
- Transformer에서 약간 우위 보고.

GELU 단점:
- 계산 약간 비싼.
- ReLU만큼 sparse 안 됨.

ReLU vs GELU의 trade-off는 작음. 큰 차이 없으면 ReLU (단순), Transformer에서는 GELU (관행).

### 5.4 답안 골격

> "출력층은 task에 따라 (sigmoid/softmax/linear). 은닉층은 ReLU가 default — vanishing 약함, sparse, 빠름. Transformer는 GELU 관행. RNN hidden은 tanh (exploding 약함). LSTM gate은 sigmoid (0~1 비율 의미). Dying ReLU 우려면 LeakyReLU."

---

## 6. Architecture — Depth vs Width

### 6.1 양 끝 비교

**Depth 多 (Width 적음)**:
- 계층적 추상화 자연.
- Vanishing 위험 (skip 없으면).
- 학습 어려움 (실증).
- Parameter 효율 ↑ (같은 함수 적은 파라미터).
- 일반화 약간 좋음.

**Width 多 (Depth 적음)**:
- 표현 풍부 (한 layer에 많은 feature).
- 학습 안정 (vanishing 적음).
- 병렬 친화적.
- Parameter 비효율 (같은 함수 더 많은 파라미터).

### 6.2 같은 파라미터 예산에서

이론·실증 모두: **Depth가 약간 우위**. 단:
- Depth 너무 깊으면 학습 어려움 (skip 없으면).
- 데이터 적으면 width가 안정.

EfficientNet의 compound scaling: depth, width, resolution을 *균형* 있게.

### 6.3 결정 가이드

| 상황 | 권장 |
|---|---|
| 자연 데이터 (이미지, 텍스트) | Depth 多 (with skip) |
| 표 데이터 | Depth 적음, width 多 |
| 작은 데이터 | Depth 적음 (overfit 방지) |
| 매우 큰 데이터 | 둘 다 키움 |

### 6.4 답안 골격

> "Depth는 계층적 추상화에 자연, parameter 효율 ↑. 너무 깊으면 학습 어려움 (skip 없으면). Width는 표현 풍부, 학습 안정. 같은 파라미터 예산에서 depth가 약간 우위. EfficientNet식 compound scaling — depth/width/resolution 균형. 자연 데이터엔 depth, 표 데이터·작은 데이터엔 width 우선."

---

## 7. Regularization 강도

### 7.1 데이터 크기에 따른 강도

| 데이터 크기 | Dropout | Weight Decay | Augmentation | Label Smoothing |
|---|---|---|---|---|
| < 10K | 0.5 | 1e-3 | Strong | 0.1 |
| 10K~100K | 0.3 | 1e-4 | Medium | 0.1 |
| 100K~10M | 0.1 | 1e-4 | Mild | 0.1 |
| > 10M | 0~0.1 | 1e-2 | Mild | 0.1 |

큰 데이터 + 약한 정규화. 작은 데이터 + 강한 정규화.

### 7.2 신호로 진단

**Train ↓ Val ↑ (큰 gap)**:
- Overfit. 정규화 강하게.
- Dropout 0.3 → 0.5.
- Weight decay 1e-4 → 1e-3.
- Augmentation 추가.

**Train ↑ Val ↑ (둘 다 높음)**:
- Underfit. 정규화 약하게.
- Dropout 줄임.
- 모델 키움.

**Train ≈ Val (작은 gap)**:
- 좋은 균형. 그대로 또는 약간 강화.

### 7.3 정규화 종합

여러 정규화 같이:
- L2 + Dropout + Augmentation: 가장 흔한 조합.
- BN과 Dropout: 충돌 가능, 위치 신중.
- Mixup + Label smoothing: 보완적.

### 7.4 답안 골격

> "데이터 크기에 따라. 작으면 강하게 (dropout 0.5, WD 1e-3, strong aug), 크면 약하게 (dropout 0.1, WD 1e-4, mild aug). 매우 큰 데이터엔 dropout 0 + label smoothing. 신호: Train ↓ Val ↑면 강화, 둘 다 높으면 약화. L2 + dropout + augmentation 조합이 흔함."

---

## 8. LR Schedule 결정

### 8.1 옵션과 사용처

| Schedule | 사용처 |
|---|---|
| **Step decay** | ImageNet CNN 표준. 30/60/90 epoch에 1/10 |
| **Cosine annealing** | Transformer 표준. 부드러운 감소 |
| **Warmup + cosine** | 큰 모델, AdamW |
| **OneCycle** | 빠른 prototype |
| **Cyclic LR** | 평탄 영역 탈출 |
| **ReduceLROnPlateau** | Val loss 정체 시 자동 |

### 8.2 Warmup의 이유

학습 초기 통계 noise:
- BN running 평균·분산 부정확.
- Adam의 second moment noisy.

큰 LR로 시작 시 발산 위험. 작게 시작 → 통계 안정 → 점진 증가.

큰 모델 + Adam에선 사실상 필수. Warmup steps: 보통 총 학습의 5~10%.

### 8.3 답안 골격

> "Schedule이 sub-optimal 학습 동역학 보정. 표준: ImageNet CNN은 step decay (30/60/90 epoch 1/10), Transformer는 warmup + cosine. Warmup이 BN/Adam 통계 안정 시간 제공 — 큰 모델 필수. 신호: val loss plateau면 LR decay, 진동이면 LR 줄임."

---

## 9. Normalization 위치

### 9.1 결정 가이드

| 모델 / 상황 | Norm |
|---|---|
| 큰 batch CNN | **BatchNorm** |
| 작은 batch (detection, seg) | **GroupNorm** |
| RNN | **LayerNorm** |
| Transformer | **LayerNorm** (Pre-Norm) |
| 스타일 변환 | **InstanceNorm** |

### 9.2 BN vs LN

**BN**: batch 차원 정규화. 큰 batch에서 효과적, 작은 batch에서 부정확.

**LN**: feature 차원 정규화 (한 sample 내). Batch 무관.

Transformer가 LN인 이유:
- 시퀀스 길이 가변 → BN 통계 부정확.
- 자기회귀 추론 시 batch=1.
- Sample 내 정규화로 일관성.

### 9.3 Pre-Norm vs Post-Norm (Transformer)

**Post-Norm** (원조): `x + LN(SubLayer(x))`
**Pre-Norm**: `x + SubLayer(LN(x))`

Pre-Norm이 더 안정 — skip 경로가 LN 거치지 않음. Identity flow 직접. 큰 모델에서 표준.

### 9.4 답안 골격

> "Normalization 종류는 데이터·모델에 따라. 큰 batch CNN은 BN, 작은 batch는 GN, RNN/Transformer는 LN. Transformer가 LN인 이유: 시퀀스 길이 가변 + 추론 batch=1 → batch 통계 부정확, sample 내 정규화로 일관. Pre-Norm이 Post-Norm보다 안정 — skip이 직접 흐름."

---

## 10. Loss와 Output 매칭

### 10.1 매칭 표

| 출력 활성화 | Loss |
|---|---|
| Linear | MSE / MAE / Huber |
| Sigmoid | BCE |
| Softmax | CE |
| Sigmoid (per class) | per-class BCE (multi-label) |

### 10.2 보조 loss

- **Label smoothing**: CE 보강. 큰 모델 일반화 ↑.
- **Focal loss**: 클래스 불균형. 어려운 sample에 가중.
- **Dice loss**: Segmentation. 작은 영역에 강함.
- **Triplet loss**: 메트릭 학습. 얼굴 인식, 검색.
- **Contrastive loss**: 자기지도 학습.

### 10.3 결정 신호

- 클래스 불균형 심하면 → Focal.
- 큰 모델 일반화 → Label smoothing.
- Segmentation 작은 영역 → Dice.
- 메트릭 학습 → Triplet.

### 10.4 답안 골격

> "Output 활성화와 loss는 짝. Sigmoid+BCE, Softmax+CE, Linear+MSE — gradient가 (예측-정답)으로 깔끔. 보조 loss로 보강 — focal (불균형), label smoothing (일반화), dice (segmentation). Loss 선택은 task와 데이터 특성에 의존."

---

## 11. 데이터 vs 모델 — 어디에 투자

### 11.1 신호로 판단

| 상황 | 다음 행동 |
|---|---|
| Train도 안 떨어짐 | **모델 키움** (capacity 부족) |
| Train ↓ Val ↑ | **데이터 늘림** (overfit, capacity 충분) |
| 둘 다 OK 더 원함 | **둘 다** scaling |
| 데이터 한계 | **외부 데이터** / pretrained |

### 11.2 Scaling Law 직관

OpenAI의 scaling law 분석 (2020): 데이터·연산·파라미터를 균형 있게 키울 때 성능 멱법칙으로 향상.

핵심 통찰:
- 데이터만 늘려도 한계.
- 모델만 키워도 한계.
- 셋 다 균형이 sweet spot.

EfficientNet의 compound scaling이 이걸 CNN에 적용. GPT의 scaling이 LLM에.

### 11.3 답안 골격

> "신호 기반. Train도 못 맞추면 capacity 부족 → 모델 키움. Train ↓ Val ↑면 overfit → 데이터 늘리거나 정규화. 데이터·모델·연산을 균형 있게 scaling이 sweet spot (scaling law). 1순위 부족 자원에 투자가 효율적."

---

## 12. Early Stopping 기준

### 12.1 Patience 선택

Val loss가 N epoch 연속 개선 없으면 멈춤. N = 5~20.

너무 작은 N: 노이즈에 민감, 너무 일찍 멈춤.
너무 큰 N: overfit 진행 후 멈춤.

### 12.2 신호 모니터링

| Val 곡선 | 처방 |
|---|---|
| 부드럽게 감소 | 충분히 더 |
| 진동 | LR 줄이거나 patience 늘림 |
| 급격히 상승 | 즉시 멈춤, 정규화 강화 |

### 12.3 Best vs Last

Early stopping 후 두 가지 선택:
- Last checkpoint: 마지막 모델.
- Best val checkpoint: 최고 val 성능 모델.

표준은 best. Last는 hyperparameter overfit 가능.

### 12.4 답안 골격

> "Val loss 정체 시 멈춤. Patience 5~20. Best val checkpoint 사용 (last 아님). 신호: val 진동이면 LR 줄임, 급격 상승이면 즉시 멈춤 + 정규화. Early stop은 시간을 통한 정규화 — 가장 단순하고 효과적인 정규화 중 하나."

---

## 13. Hyperparameter Search

### 13.1 옵션 비교

| 방법 | 장점 | 단점 |
|---|---|---|
| **Grid search** | 단순, 재현 가능 | 차원 폭발, 비효율 |
| **Random search** | 효율 (Bergstra 2012) | 단순 |
| **Bayesian (Optuna)** | 효율 ↑↑ | 복잡, 의존성 |
| **PBT (Population)** | 동적, 최강 | 비용 매우 큼 |
| **Hyperband / SHA** | 빠른 종료 | 복잡 |

### 13.2 우선순위

가장 민감한 것부터:
1. LR
2. Batch size
3. Optimizer + scheduler
4. Architecture (depth, width)
5. Regularization 강도
6. Init

각 단계에서 다른 것 고정하고 한 변수만 바꿈.

### 13.3 좋은 baseline

논문의 default 시작점이 좋음. 그 위에서 조정.

### 13.4 답안 골격

> "Random search > grid search (대부분 case, Bergstra 증명). Bayesian (Optuna)이 비싼 모델에 효율. 우선순위: LR > batch > optimizer > architecture > regularization > init. 가장 민감한 것부터 시도. 좋은 baseline (논문 default)에서 시작."

---

## 14. Architecture 선택 — 처음부터

### 14.1 절차

1. **데이터 분석**: 형태, 크기, 라벨 분포, 길이 분포.
2. **Baseline**: 작고 빠른 모델 (logistic, GBM, 작은 CNN).
3. **Ablation**: 한 컴포넌트씩 추가/제거.
4. **Scaling**: 효과 보이면 키움.
5. **Regularization 조정**.

### 14.2 흔한 패턴

**이미지 분류**:
- ResNet50 + CE + SGD/Adam + step decay
- 데이터 적으면 transfer learning

**텍스트 분류**:
- BERT/DistilBERT + AdamW + warmup + cosine
- 또는 TextCNN baseline

**시계열**:
- LSTM/GBM/TCN baseline
- 비교하고 결정

### 14.3 답안 골격

> "절차: 데이터 분석 → Baseline → Ablation → Scaling → Regularization. Baseline은 항상 단순한 것부터. 복잡한 모델이 baseline 못 이기면 무언가 잘못. 흔한 패턴: 이미지 ResNet+CE+SGD, NLP BERT+AdamW+warmup."

---

## 15. 실수하기 쉬운 결정

### 15.1 자주 하는 실수

(1) **LR 안 튜닝**: Default 값 그대로 — 종종 sub-optimal.
(2) **Batch size 큰 것이 좋다고 가정**: 일반화 손해.
(3) **Adam이 항상 좋다고 가정**: SGD가 ImageNet에서 우위.
(4) **Init 무시**: BN 있어도 적절한 init이 학습 안정.
(5) **Schedule 안 함**: 큰 모델에서 sub-optimal 수렴.
(6) **Regularization 안 조정**: 데이터 크기에 안 맞음.
(7) **Eval mode 안 부름**: 추론 비결정성.
(8) **Validation split 부적절**: leakage, 신뢰 못함.

### 15.2 체크리스트

학습 시작 전:
- [ ] 입력 정규화
- [ ] Output-loss 매칭
- [ ] Init이 활성화에 맞음
- [ ] LR Range Test
- [ ] Validation split (그룹/시간 leak 없음)
- [ ] Tiny dataset overfit 가능 확인
- [ ] BN/Dropout train/eval mode

이 8개를 통과하면 최소한 시작은 안전.

---

## 16. 결정 매트릭스 — 데이터 크기별

| 결정 | 작은 (<10K) | 중간 (10K~100K) | 큰 (100K~10M) | 매우 큼 (>10M) |
|---|---|---|---|---|
| 모델 크기 | 작게 | 중간 | 크게 | 매우 크게 |
| Optimizer | Adam (안전) | Adam/AdamW | SGD/AdamW | AdamW |
| LR | 1e-3 (Adam) | 1e-3 ~ 1e-4 | 1e-4 | 1e-4 + warmup |
| Schedule | Step or constant | Cosine | Cosine + warmup | Cosine + warmup |
| Batch | 32~64 | 64~256 | 256~1024 | 1024~ |
| Regularization | Strong | Medium | Weak | Minimal |
| Dropout | 0.5 | 0.3 | 0.1 | 0~0.1 |
| Weight decay | 1e-3 | 1e-4 | 1e-4 | 1e-2 |
| Augmentation | Strong | Medium | Mild | Mild |
| Pretrained | 거의 필수 | 권장 | 옵션 | from scratch도 OK |

---

## 17. 면접 단골 Q&A

### Q1. LR 결정 절차?
"LR Range Test가 표준. LR을 매우 작게 (1e-7) 시작 → 매 step 지수적 증가 → loss 추적. 가파르게 떨어지는 구간의 1/10이 적절 시작점. 그 후 schedule 적용 — Transformer면 cosine + warmup, ImageNet이면 step decay. 신호: NaN이면 1/10, 너무 느리면 키움. 가장 민감한 hyperparameter."

### Q2. Batch size 키우면 LR도 키워야 하는가?
"Linear scaling rule. Batch×k면 LR×k. 이유는 같은 epoch에서 update 횟수 1/k → 한 update를 k배 크게 해서 보상. 단 매우 큰 batch (4096+)에선 깨짐 — LARS/LAMB 전용 optimizer. 일반 sweet spot은 256~1024."

### Q3. Adam이 SGD보다 항상 좋은가?
"아니. Adam은 빠른 수렴 + LR 튜닝 부담 적음 → 일반 default 좋음. 단 ImageNet 같은 task에서 SGD+momentum이 약 1% 정도 더 일반화 잘함 — Adam이 sharp minima 경향이라는 분석. AdamW가 weight decay 정확 처리해서 Transformer 표준. 결국 task와 데이터에 따라."

### Q4. Warmup이 필요한 이유?
"학습 초기 통계 noise 때문. BN running 평균·분산이 부정확, Adam의 second moment 추정 noisy. 큰 LR로 시작하면 이 부정확한 통계가 발산 유발. 작게 시작 → 통계 안정 시간 제공 → 점진 증가. 큰 모델 + Adam에서 사실상 필수. Warmup steps는 보통 총 학습의 5~10%."

### Q5. Depth vs Width 어느 쪽?
"같은 파라미터 예산에서 depth가 약간 우위 — 계층적 추상화 자연, parameter 효율 ↑. 단 너무 깊으면 vanishing (skip 없으면). EfficientNet의 compound scaling: depth/width/resolution 균형. 자연 데이터 (이미지, 텍스트)는 depth 우선, 표 데이터·작은 데이터는 width 안정."

### Q6. Regularization 강도 결정?
"데이터 크기에 따라. 작으면 강하게 (dropout 0.5, WD 1e-3, strong aug), 크면 약하게. 신호: Train ↓ Val ↑면 강화, 둘 다 높으면 약화. 큰 모델 + 큰 데이터엔 dropout 거의 안 쓰고 weight decay + label smoothing만. 데이터 자체가 정규화."

### Q7. He init의 분산이 2/n_in인 이유?
"ReLU 가정. ReLU가 음수 절반을 0으로 만들어 분산이 절반으로 줄어듦. 이걸 보상하려면 가중치 분산을 두 배 → $2/n_{in}$. Xavier는 sigmoid/tanh 가정 (대칭)으로 $2/(n_{in}+n_{out})$. 활성화에 안 맞는 init은 첫 forward부터 vanishing/exploding."

### Q8. BN과 Dropout 같이 쓸 때?
"분산 변동 충돌. Dropout이 활성값 noise → BN 통계 흔들림. 처방: (1) 위치 신중 — conv → BN → ReLU → (dropout). (2) ResNet block에는 dropout 거의 안 씀 (BN 충분). (3) 분류 head FC에서만 dropout. (4) 또는 둘 중 하나만. (5) BN 대신 LN (덜 영향)."

### Q9. Learning Rate Schedule 어떻게?
"Optimizer와 함께 결정. ImageNet CNN: step decay (30/60/90 epoch에 1/10). Transformer: cosine + warmup. 큰 모델·Adam엔 warmup 필수. ReduceLROnPlateau는 자동 — val loss 정체 시 감소. Cosine이 step보다 부드러워서 일반화 약간 좋다는 보고."

### Q10. Hyperparameter Search 방법?
"Random > Grid (Bergstra 2012). 대부분 hyperparameter는 sensitivity 다름 — random이 효율적. Bayesian (Optuna)은 비싼 모델에 유리. 우선순위: LR → batch → optimizer → architecture → regularization. 가장 민감한 것부터. 좋은 baseline (논문 default)에서 시작."

### Q11. 데이터 vs 모델 어디에 투자?
"신호 기반. Train도 못 맞추면 capacity 부족 → 모델 키움. Train ↓ Val ↑면 overfit → 데이터 늘림 또는 정규화. 데이터·모델·연산을 균형 있게 (scaling law). 1순위 부족 자원에 투자. ImageNet 모델 1억→2억 키워도 데이터 100만 그대로면 수익 체감."

### Q12. Architecture 선택 절차?
"5단계. (1) 데이터 분석 — 형태·크기·분포. (2) Baseline — 단순한 것부터 (logistic, GBM, 작은 CNN). (3) Ablation — 한 컴포넌트씩 추가/제거. (4) Scaling — 효과 보이면 키움. (5) Regularization 조정. Baseline 못 이기면 무언가 잘못. 단순에서 복잡으로 점진적."

---

## 18. 생각해보라 — 단락 답안

**Q. 왜 LR이 가장 민감한가?**

LR이 *학습 동역학의 step size*. 너무 크면 1차 Taylor 근사 깨짐 → 발산. 너무 작으면 진전 없음 또는 local에 갇힘. 다른 hyperparameter는 *질적*으로 다른 영향 (모델 크기, 정규화 등)이지만 LR은 *학습 자체의 가능성*을 결정.

수학적: SGD의 수렴 정리는 LR이 적절 범위에 있을 때만 보장. Convex case에서 $\eta < 1/L$ ($L$ = Lipschitz 상수). 비convex에서는 더 미묘 — 너무 크면 발산, 너무 작으면 sub-optimal.

실증: ImageNet 모델의 정확도가 LR에 매우 민감. 1e-1 vs 1e-2 사이에서 큰 차이. 다른 hyperparameter 동일해도 LR만 잘못이면 SOTA 못 미침.

**Q. 왜 큰 batch가 일반화에 안 좋은가?**

여러 가설:

(1) **Sharp minima 가설** (Keskar 2016): 큰 batch는 gradient noise 적음 → flat minima로 가는 implicit bias 약함 → sharp minima에 빠짐. Sharp minima는 가중치 변화에 민감 → 일반화 약함.

(2) **Update 횟수 감소**: 같은 epoch에서 큰 batch는 update 횟수 적음. 모델이 *충분히 움직이지* 못함.

(3) **암묵적 정규화 약화**: SGD noise 자체가 정규화. 큰 batch는 이 효과 약함.

이 셋 합쳐서 큰 batch가 일반화에 sub-optimal. Linear scaling rule + warmup으로 어느 정도 보완 가능하지만 한계 있음.

매우 큰 batch (4096+) 학습은 LARS/LAMB 같은 전용 optimizer. ImageNet 모델 1만 batch에 1.5시간 학습 가능 (Goyal 2017).

**Q. Init이 BN 있어도 중요한 이유?**

BN이 활성값 분포를 강제 정규화 → init 영향 *상당히* 줄임. 단 *완전 제거*는 아님:

(1) **첫 forward의 학습 신호**: Init이 잘못이면 첫 forward의 활성값 분포가 매우 비정상 → BN의 첫 mini-batch 통계도 비정상 → 학습 시작이 어려움.

(2) **BN의 running 통계 누적**: 학습 진행 중 running 평균이 update. 초기에 잘못된 분포면 이 누적도 오래 잘못.

(3) **BN 없는 layer**: 모든 layer에 BN 있는 게 아님. 출력 layer, attention 등.

그래서 BN 있어도 He/Xavier 같은 적절한 init이 학습 안정성에 도움. 단 BN 없을 때만큼 critical은 아님.

**Q. Cosine vs Step decay 차이는 작아 보이는데 왜 cosine으로 가나?**

세 이유:

(1) **부드러움**: Step decay는 *급격한* LR 변화 (1/10). 학습 동역학에 충격. Cosine은 부드러움 → 안정.

(2) **이론적 분석**: Cosine annealing이 *flat minima*로 이끈다는 분석. SGD 동역학과 잘 어울림.

(3) **실증**: 큰 모델·Transformer에서 cosine이 약간 일관되게 우위. 차이는 작지만 (0.5% 정도) 일관됨.

또 cosine은 *총 epoch 수*를 미리 정해야 한다는 약점. ReduceLROnPlateau는 동적이지만 schedule이 deterministic 아님.

**Q. 왜 Transformer는 LayerNorm을 Pre-Norm 위치?**

원조 Transformer는 Post-Norm: `x + LN(SubLayer(x))`. Pre-Norm: `x + SubLayer(LN(x))`.

차이:
- Post-Norm: skip 경로가 LN을 *통과*. 정규화가 모든 정보에 적용.
- Pre-Norm: skip 경로가 LN을 *통과 안 함*. Identity가 더 직접.

Pre-Norm의 장점:
- Identity flow 보장 (ResNet과 비슷한 정신).
- 학습 안정 — 매우 깊은 모델 (96+층) 학습 가능.
- Warmup 부담 약간 줄어듦.

GPT-3, T5, modern Transformer는 모두 Pre-Norm. 원조는 Post-Norm이지만 *결과적으로 Pre-Norm이 표준*.

**Q. 왜 SGD가 ImageNet에서 SGD가 일반화 우위인가?**

여러 분석:

(1) **Implicit bias**: SGD의 mini-batch noise가 *flat minima*로 향하는 경향. Adam은 second moment로 normalize해서 noise 효과 줄어듦 → sharp minima 경향.

(2) **Adaptive LR의 부작용**: Adam이 차원별로 LR 자동 조정. 일부 차원이 매우 빠르게, 일부는 느리게. 이게 모델 capacity 비대칭 사용으로 일반화 손해 가능.

(3) **Weight decay 처리**: Adam의 weight decay가 second moment 영향 받음. 부정확. AdamW가 해결하지만 여전히 SGD만큼 안정 아님.

이래서 ImageNet 같은 큰 데이터·long training task에서 SGD+momentum이 *일관되게* 약간 우위. Transformer NLP에서는 AdamW가 압도 — 큰 모델 학습 안정성이 중요. Task별 다른 답.

**Q. 왜 Bayesian hyperparameter optimization이 표준이 안 됐나?**

장점은 분명: efficient search. 단 단점이 큼:

(1) **복잡성**: Optuna 같은 라이브러리 의존. 학습 코드와 통합 부담.

(2) **메타 hyperparameter**: Bayesian opt 자체에 hyperparameter (acquisition function, kernel 등). 그 조정이 또 부담.

(3) **Discrete 처리 어려움**: Activation 종류, layer 수 같은 이산 변수에 약함.

(4) **시간**: 충분한 trial이 있어야 좋은 결과. 100 trial로 부족한 경우 多.

(5) **Random search가 충분히 강력**: Bergstra 2012의 보고 — random이 grid보다 효율, Bayesian보다 약간 못하지만 단순.

실무에서: 빠르게 prototype에 random, SOTA 추구에 Bayesian (또는 PBT), 매우 큰 모델은 단일 학습이 비싸서 모든 방법이 어려움.

---

## 19. 한 줄 요약

- 모든 결정은 **신호로 진단** + **이유로 답**.
- 우선순위: **LR > batch > optimizer > init > activation > depth/width > regularization**.
- **LR Range Test**가 가장 좋은 LR 결정 방법.
- **Linear scaling rule**: batch×k면 LR×k.
- **Adam vs SGD**: Transformer AdamW, ImageNet SGD+momentum.
- **Warmup**이 큰 모델·Adam에 사실상 필수.
- **Depth가 약간 우위** (with skip), width는 안정.
- **데이터 크기에 비례**한 정규화 강도.
- **He init**이 ReLU 표준, **Xavier**가 sigmoid/tanh.
- **데이터 vs 모델 vs 연산**의 균형 (scaling law).
- **Validation split**이 leakage 없이 정직하게.
- **Tiny dataset overfit**으로 코드 정상성 첫 확인.
