# 11. Comparison & Decision — "X냐 Y냐" 심층

> **이 문서의 핵심 사고**: 같은 자리를 다투는 옵션들의 *본질적 차이*와 *결정 기준*. 시험·면접에서 "왜 X 대신 Y?" 질문에 즉답할 수 있도록.
>
> **3축 비교 패턴**:
> 1. **메커니즘적으로 무엇이 다른가** (구조의 차이)
> 2. **어떤 inductive bias / 가정이 다른가**
> 3. **언제 무엇을 선택?** (상황 적합도)
>
> 모든 비교를 이 3축으로 답하면 체계적이다.

---

## 0. 비교 사고법

### 0.1 비교의 함정

흔한 답: "X는 빠르고 Y는 정확". 너무 막연. *왜* 빠르고 *왜* 정확한지를 답해야 의미.

좋은 답: "X는 메커니즘 A로 인해 [상황]에서 빠름. Y는 메커니즘 B로 인해 [상황]에서 정확. 따라서 [내 task]에선 [결정]."

3축 분석이 이런 답을 만드는 framework.

### 0.2 본 챕터의 흐름

§1: 4대 Architecture (FNN/CNN/RNN/Transformer) — 가장 큰 비교
§2: 시퀀스 모델 (LSTM/GRU/Attention)
§3: Normalization (BN/LN/GN/IN)
§4: Optimizer (SGD/Adam/AdamW)
§5: Regularization (L1/L2/Dropout/...)
§6: Activation
§7: Loss
§8: Initialization
§9: Pooling vs Strided Conv
§10: Output 설계 (분류·회귀·multi-label)
§11: Batch Size 작·크
§12: From scratch vs Transfer
§13: Decision Cheatsheet
§14: 면접 Q&A
§15: 생각해보라

---

## 1. 4대 Architecture: FNN vs CNN vs RNN vs Transformer

### 1.1 메커니즘 비교

| | FNN | CNN | RNN | Transformer |
|---|---|---|---|---|
| 핵심 연산 | 행렬 곱 | Convolution | Recurrent | Self-attention |
| 정보 전파 | Layer 사이 | 공간 sliding | 시간 sequential | Token 간 동시 |
| Parameter sharing | 없음 | 공간 축 | 시간 축 | 토큰 간 동등 |
| 계층 구조 | Layer 깊이 | 작은 RF → 큰 RF | 시간 누적 | Layer 깊이 |

### 1.2 Inductive Bias

| | FNN | CNN | RNN | Transformer |
|---|---|---|---|---|
| 핵심 prior | 거의 없음 | 국소성 + 평행이동 | 시간 순서 + 시간 sharing | 거의 없음 (PE만) |
| 데이터 가정 | 일반 | 격자 | 시퀀스 | 시퀀스/뭐든 |
| 거리 의존성 | 모든 거리 동등 | 깊이 따라 RF↑ | 거리 따라 약함 | 거리 무관 |

### 1.3 학습·메모리 특성

| | FNN | CNN | RNN | Transformer |
|---|---|---|---|---|
| 병렬 | ✓ | ✓ | ✗ | ✓✓ |
| 메모리 | $O(N)$ | $O(N)$ | $O(T \cdot d)$ | $O(T^2 \cdot d)$ |
| 데이터 효율 | 보통 | 좋음 | 보통 | 많이 필요 |
| Long-range | 가능 (큰 모델) | 깊이 필요 | 약함 | 강함 |

### 1.4 결정 트리

```
데이터가 무엇인가?
├── 격자 (이미지) 
│   ├── 데이터 적음~중 → CNN (ResNet/EfficientNet)
│   └── 데이터 매우 많음 → ViT
├── 시퀀스 (텍스트, 음성, 시계열)
│   ├── 짧음 + 데이터 적음 → CNN(1D), LSTM/GRU
│   ├── 길음 + long-range → Transformer
│   └── 매우 큰 데이터 → Transformer
├── 그래프 → GNN
├── 표 → MLP / GBM / TabTransformer
└── 그 외 → 데이터 구조 분석 후 결정
```

### 1.5 "왜 NLP가 RNN→Transformer로 갔나"

세 압도적 이유:

(1) **병렬화**: RNN은 시점 t가 t-1 의존. GPU 활용 못함. Transformer는 모든 시점 동시 → GPU 거의 100% 활용.

(2) **Long-range**: LSTM도 100~200 step 어려움. Transformer는 한 layer에서 거리 무관 직접.

(3) **Scaling**: 모델·데이터 키울수록 성능 멱법칙으로 향상. RNN은 한계 빨리 옴.

이 셋이 GPT-3 (175B parameters), BERT 등 foundation model 시대를 가능하게.

### 1.6 "왜 이미지에선 CNN이 여전히 강한가"

CNN의 inductive bias가 자연 이미지에 강한 prior:
- 국소성: 가까운 픽셀 강한 상관.
- 평행이동: 같은 패턴 어디 있든 같은 의미.

이 prior 덕에 작은 데이터 (ImageNet 정도)에서 CNN이 ViT 우위.

ViT는 매우 큰 데이터 (JFT-300M, 3억 장)에서 prior 없이 학습한 패턴이 CNN 추월. ImageNet은 그 임계 미만.

ConvNeXt (2022)가 CNN에 modern training 기법 흡수해 ViT와 거의 동급.

### 1.7 답안 골격

> "FNN은 inductive bias 거의 없음, CNN은 국소성·평행이동, RNN은 시간 순서·시간 sharing, Transformer는 거의 없음. 데이터 구조와 매칭이 효율: 격자 → CNN, 시퀀스 → RNN/Transformer. NLP가 Transformer로 간 이유: 병렬 + long-range + scaling. 이미지는 CNN이 여전 (작은~중간 데이터), ViT는 매우 큰 데이터에서 우위."

---

## 2. 시퀀스: LSTM vs GRU vs Attention/Transformer

### 2.1 메커니즘 비교

| | LSTM | GRU | Attention |
|---|---|---|---|
| State | Cell + Hidden | Hidden | (직접 query-key) |
| Gate 수 | 3 + candidate | 2 (update, reset) | (없음, soft selection) |
| 정보 흐름 | 시간 sequential | 시간 sequential | 모든 시점 동시 |

### 2.2 Vanishing 완화 메커니즘

**LSTM**: Cell state의 덧셈 update. $\partial c_t/\partial c_{t-1} = f_t$ (scalar 곱).

**GRU**: Update gate의 덧셈. 본질적으로 LSTM과 같은 정신.

**Attention**: 시간 chain 자체를 우회 — 거리 무관 직접 연결.

### 2.3 비교

| | LSTM | GRU | Attention |
|---|---|---|---|
| 표현력 | 약간 더 | 약간 덜 | 매우 강함 |
| 파라미터 | 많음 | LSTM의 75% | task에 따라 |
| 병렬 | ✗ | ✗ | ✓ |
| Long-range | 보통 | 보통 | 강함 |
| 메모리 | $O(T \cdot d)$ | $O(T \cdot d)$ | $O(T^2 \cdot d)$ |

### 2.4 결정 가이드

```
시퀀스 처리?
├── 데이터 적고 시퀀스 짧음 → GRU (단순, 빠름)
├── 매우 긴 의존성, 큰 데이터 → Transformer
├── 중간, RNN 꼭 써야 → LSTM
├── 메모리 제약 (모바일) → GRU
└── NLP 큰 모델 → Transformer
```

### 2.5 답안 골격

> "LSTM은 3 gate + cell/hidden 분리, GRU는 2 gate + hidden 단일. 거의 동급 성능. Attention은 chain 우회 — 거리 무관, 병렬, 강력. 단 메모리 $O(T^2)$. NLP 큰 모델은 Transformer 표준. RNN/LSTM/GRU는 메모리 제약, 작은 데이터, 단순 시계열에 자리. 둘 사이 (LSTM vs GRU)는 거의 동급, 둘 다 시도가 정석."

---

## 3. Normalization: BN vs LN vs GN vs IN

### 3.1 정규화 축

| | 정규화 차원 | 의미 |
|---|---|---|
| **BatchNorm** | batch (channel별) | 같은 채널의 다른 sample끼리 |
| **LayerNorm** | feature (sample 내) | 한 sample의 모든 channel |
| **GroupNorm** | feature 그룹 | 채널 그룹별 |
| **InstanceNorm** | sample 내 channel별 | 채널마다 따로 |

### 3.2 사용처

| 모델 / 상황 | 표준 |
|---|---|
| 큰 batch CNN | BN |
| 작은 batch (detection, seg) | GN |
| RNN | LN |
| Transformer | LN |
| 스타일 변환 | IN |

### 3.3 BN의 장단점

**장점**:
- 큰 batch에서 매우 효과적.
- 약한 정규화 효과 (mini-batch noise).
- 큰 LR 사용 가능.

**단점**:
- 작은 batch (< 8)에서 통계 부정확.
- Train/eval mode 다름.
- 시퀀스 데이터에 불편.
- 분산 학습 시 동기화 부담 (SyncBN).

### 3.4 LN이 Transformer 표준인 이유

(1) **시퀀스 길이 가변**: BN의 batch 통계가 시퀀스마다 다름.
(2) **자기회귀 추론**: batch=1.
(3) **Sample 내 정규화**: batch 무관, 일관성.

### 3.5 답안 골격

> "정규화 축이 다름. BN은 batch (큰 batch에 효과), LN은 feature (sample 내 일관), GN은 feature 그룹 (작은 batch에 안전), IN은 channel별 (스타일). Transformer가 LN인 이유: 시퀀스 길이 가변 + 자기회귀 batch=1 → BN 부정확. CNN 큰 batch는 BN, 작은 batch는 GN."

---

## 4. Optimizer: SGD vs Momentum vs Adam vs AdamW

### 4.1 메커니즘

| | 메커니즘 | 특징 |
|---|---|---|
| **SGD** | 단순 update | noise, 진동 |
| **SGD + Momentum** | 관성 누적 | 진동 줄임, 빠름 |
| **NAG** | 미리 가본 위치 gradient | 더 빠른 수렴 |
| **Adagrad** | 파라미터별 LR (누적) | sparse feature |
| **RMSProp** | Adagrad의 이동평균 | 안정 |
| **Adam** | Momentum + RMSProp | 빠른 수렴, 표준 |
| **AdamW** | Adam + 정확한 weight decay | Transformer 표준 |
| **LAMB / LARS** | 큰 batch 전용 | 대규모 분산 |

### 4.2 Adam vs SGD

**Adam 장점**:
- 빠른 수렴.
- LR 튜닝 부담 적음.
- 다양한 task에 안정.

**SGD 장점**:
- ImageNet에서 약 1% 더 일반화.
- Sharp minima 회피 (implicit bias).

**언제 무엇을**:
- ImageNet CNN: SGD+momentum.
- Transformer NLP: AdamW.
- 작은 데이터: Adam (안전).
- 큰 batch (4096+): LARS/LAMB.

### 4.3 답안 골격

> "Adam은 빠른 수렴 + LR 튜닝 부담 적음 → 일반 default. SGD+momentum은 ImageNet에서 약 1% 더 일반화 (Adam이 sharp minima 경향). AdamW가 weight decay 정확 처리해서 Transformer 표준. Task와 데이터에 따라: ImageNet은 SGD, Transformer는 AdamW, 작은 데이터는 Adam. 둘 다 시도가 정석."

---

## 5. Regularization: L1 vs L2 vs Dropout vs Early Stop

### 5.1 메커니즘 차이

| | 작용점 | 효과 |
|---|---|---|
| **L1 (Lasso)** | 가중치 sparsity | 일부 정확히 0 → feature selection |
| **L2 (Ridge / Weight Decay)** | 가중치 크기 | 부드럽게 줄임 |
| **Dropout** | 활성값 마스킹 | 암묵적 ensemble |
| **Early stopping** | 학습 시간 | 시간을 통한 정규화 |
| **Augmentation** | 입력 분포 | 분포 확장 |
| **Label smoothing** | 라벨 분포 | 과신 방지 |
| **Mixup / CutMix** | 입력+라벨 | 더 강력한 정규화 |
| **Stochastic depth** | layer | layer 자체 skip |

### 5.2 결정

| 상황 | 1순위 |
|---|---|
| 일반 default | L2 (weight decay) |
| Sparse feature 원함 | L1 |
| FC layer 많음 | + Dropout 0.3~0.5 |
| 작은 데이터 | + Strong augmentation |
| 큰 모델 | + Label smoothing |
| Vision (이미지) | + Mixup/CutMix |
| 매우 깊은 망 | + Stochastic depth |

### 5.3 보완성

대부분 보완적. 같이 쓰면 효과 합.

단 충돌:
- BN + Dropout: 분산 충돌.
- Strong aug + Strong dropout: underfit.
- Mixup + Label smoothing: 둘 다 라벨 부드럽게 → 과도.

### 5.4 답안 골격

> "각자 다른 축에서 작용. L2는 가중치 부드러움 (default), L1은 sparsity, Dropout은 ensemble, Early stop은 시간, Augmentation은 분포 확장, Label smoothing은 calibration. 보통 L2 + Dropout + Augmentation 조합. 큰 모델·큰 데이터엔 Dropout 줄이고 Label smoothing/Mixup 추가. 데이터 크기에 비례한 정규화 강도."

---

## 6. Activation: ReLU vs LeakyReLU vs GELU vs Sigmoid vs Tanh

### 6.1 비교표

| | 식 | 미분 최대 | 0-centered | 사용처 |
|---|---|---|---|---|
| **Sigmoid** | $1/(1+e^{-x})$ | 0.25 | ✗ | 출력층 (이진), LSTM gate |
| **Tanh** | $(e^x-e^{-x})/(e^x+e^{-x})$ | 1 | ✓ | RNN hidden |
| **ReLU** | $\max(0,x)$ | 1 (활성) | ✗ | CNN/MLP 표준 |
| **LeakyReLU** | $\max(\alpha x, x)$ | 1 (활성) | ✗ | Dying ReLU 우려 |
| **ELU** | $x>0: x, x \le 0: \alpha(e^x - 1)$ | 1 (활성) | 거의 ✓ | 부드러움 |
| **GELU** | $x \cdot \Phi(x)$ | ≤ 1 | 거의 ✓ | Transformer 표준 |
| **Swish/SiLU** | $x \cdot \sigma(x)$ | ≤ 1 | 거의 ✓ | EfficientNet |

### 6.2 결정 트리

```
출력층?
├── 이진 분류 → Sigmoid + BCE
├── 다중 분류 → Softmax + CE
├── 회귀 → Linear + MSE
└── Multi-label → 클래스별 Sigmoid + BCE

은닉층?
├── 일반 CNN/MLP → ReLU
├── Transformer → GELU
├── RNN의 hidden → Tanh
├── LSTM의 gate → Sigmoid
└── Dying ReLU 우려 → LeakyReLU
```

### 6.3 답안 골격

> "출력층은 task에 따라. 은닉층은 ReLU가 default — vanishing 약함, sparse, 빠름. Transformer는 GELU 관행 — 부드러운 ReLU, 약간 우위. RNN hidden은 tanh — exploding 약함. LSTM gate는 sigmoid — 0~1 비율 의미. Dying ReLU 우려면 LeakyReLU."

---

## 7. Loss: MSE vs MAE vs Huber / CE vs Focal

### 7.1 회귀

| | 특징 | 언제? |
|---|---|---|
| **MSE** | 가우시안 가정, 큰 오차 민감 | 일반 |
| **MAE** | 라플라스 가정, 이상치 robust | 이상치 多 |
| **Huber** | MSE+MAE 절충 | 안전 default |

### 7.2 분류

| | 특징 | 언제? |
|---|---|---|
| **CE** | 표준, MLE 등가 | 일반 |
| **Focal** | 어려운 sample 가중 | 클래스 불균형 |
| **Label Smoothing CE** | 과신 방지 | 큰 모델 |

### 7.3 결정

```
회귀?
├── 데이터 깨끗 → MSE
├── 이상치 多 → MAE/Huber
└── 안전 default → Huber

분류?
├── 클래스 균형 → CE
├── 클래스 불균형 → Focal 또는 weighted CE
└── 큰 모델 일반화 → CE + Label smoothing
```

### 7.4 답안 골격

> "회귀: 일반 MSE, 이상치 多 MAE/Huber. 분류: 표준 CE, 클래스 불균형 Focal, 큰 모델 일반화 Label smoothing. Loss와 출력 활성화는 짝 (Sigmoid+BCE, Softmax+CE, Linear+MSE) — gradient (예측-정답)으로 깔끔."

---

## 8. Initialization: Xavier vs He vs LeCun

### 8.1 비교

| | 분산 | 활성화 가정 |
|---|---|---|
| **Xavier (Glorot)** | $2/(n_{in}+n_{out})$ | Tanh, Sigmoid |
| **He (Kaiming)** | $2/n_{in}$ | ReLU 계열 |
| **LeCun** | $1/n_{in}$ | SELU |

### 8.2 He의 분산이 두 배인 이유

ReLU가 음수 절반을 0으로 → 분산 절반 손실. 보상 위해 가중치 분산 두 배.

### 8.3 답안 골격

> "활성화에 맞춰. ReLU → He, Tanh/Sigmoid → Xavier, SELU → LeCun. He의 $2/n_{in}$은 ReLU의 분산 손실 보정. 잘못 매칭 시 첫 forward부터 vanishing/exploding. BN/LN 있으면 영향 줄지만 적절 init 여전히 학습 안정에 도움."

---

## 9. Pooling vs Strided Conv

### 9.1 비교

| | Pooling | Strided Conv |
|---|---|---|
| 학습 | 비학습 | 학습 가능 |
| 표현력 | 단순 (max/avg) | 복잡 변환 |
| 메모리 | 적음 | 약간 더 |
| 현대 | GAP만 | Down-sampling 표준 |

### 9.2 현대 트렌드

- ResNet 등 modern CNN: stride conv가 down-sampling 표준.
- Max pool은 첫 layer 후 한 번 정도.
- 마지막엔 GAP (FC 대체, 파라미터 절약).

### 9.3 답안 골격

> "Pooling은 hand-crafted (max/avg), strided conv는 학습된 down-sampling. 현대는 stride conv가 표준 — 학습된 down-sampling이 hand-crafted보다 유연. Max pool은 단순함의 장점이 있어 edge device에서 사용. 분류 마지막에 GAP — 파라미터 절약."

---

## 10. Output 설계 — 분류·회귀·multi-label

### 10.1 매칭

| Task | 마지막 layer | Loss |
|---|---|---|
| 회귀 (실수) | Linear | MSE/MAE/Huber |
| 이진 분류 | Sigmoid | BCE |
| 다중 분류 (mutually exclusive) | Softmax | Categorical CE |
| Multi-label (동시 여러 라벨) | 클래스별 Sigmoid | per-label BCE |
| 순위 (검색·추천) | (다양) | Pairwise loss, NDCG-based |
| 분포 예측 | Mixture density | NLL |

### 10.2 흔한 실수

(1) **다중 분류에 Sigmoid (per class)**: 확률 합 1 안 됨 → multi-label로 잘못 됨.

(2) **Multi-label에 Softmax**: 확률 분배 강제 → 동시 라벨 예측 안 됨.

(3) **회귀에 Sigmoid**: 출력 범위 제한.

### 10.3 답안 골격

> "Output 활성화와 loss는 짝. 분류 vs 회귀 구분 + 분류 내에서 mutually exclusive vs multi-label 구분 중요. 다중 분류엔 softmax+CE, multi-label엔 클래스별 sigmoid+BCE. Loss와 활성화 잘못 짝지으면 학습 동역학 망가짐."

---

## 11. Batch Size: 작 vs 큼

### 11.1 비교

| | 작 (16~64) | 큼 (1024+) |
|---|---|---|
| Gradient | Noisy | 정확 |
| 일반화 | 좋음 (flat minima) | 약간 손해 (sharp) |
| 학습 속도 | 느림 (per epoch) | 빠름 |
| BN 통계 | Noisy | 정확 |
| GPU 효율 | 낮음 | 높음 |

### 11.2 결정

```
GPU 메모리?
├── 작음 (8GB) → 16~64
├── 일반 (16GB) → 64~256
├── 큼 (32GB+) → 256~1024
├── Multi-GPU → 1024~4096
└── 매우 큰 cluster → 4096+ (LARS)
```

LR linear scaling rule: batch×k면 LR×k.

### 11.3 답안 골격

> "작은 batch는 noise 효과로 일반화 우위, 큰 batch는 GPU 효율 + 안정. Sweet spot 256~1024. 변경 시 linear scaling rule. 매우 큰 batch (4096+)는 LARS/LAMB. ImageNet 학습은 256~512가 표준, 큰 모델은 1024~4096."

---

## 12. Train from Scratch vs Transfer vs Foundation

### 12.1 데이터 크기에 따른 결정

| 데이터 | 권장 |
|---|---|
| < 1k | Pretrained + linear probing (FC만) |
| 1k~10k | Pretrained + fine-tune 마지막 몇 layer |
| 10k~100k | Pretrained + full fine-tune (작은 LR) |
| 100k+ | from scratch도 가능, 그래도 pretrained 종종 우위 |

### 12.2 도메인 차이의 영향

**ImageNet과 비슷** (자연 사진): pretrained 매우 효과적.
**다른 도메인** (의료, 만화, 위성): 효과 줄지만 random init보다 우위. 깊은 layer까지 fine-tune.
**완전 다른 modality** (음성, 텍스트): 직접 transfer 어려움. Self-supervised pretrain은 modality 무관 도움.

### 12.3 답안 골격

> "데이터 매우 적으면 (< 1k) pretrained의 freeze + linear FC. 적음 (1k~10k)이면 pretrained + 부분 fine-tune. 많음 (100k+)이면 from scratch도 OK. 도메인 차이 클수록 깊은 layer까지 fine-tune. Self-supervised pretrain (BERT, MAE)이 foundation model 시대 답."

---

## 13. Decision Cheatsheet — 빠른 결정

### 13.1 모델 선택

| 데이터 | 권장 모델 |
|---|---|
| 이미지 (작은~중간 데이터) | ResNet/EfficientNet + transfer |
| 이미지 (매우 큰 데이터) | ViT |
| 짧은 텍스트 분류 | TextCNN, BERT fine-tune |
| 긴 문서 이해 | Transformer (Longformer 등) |
| 음성 인식 | CNN + Transformer (Whisper) |
| 시계열 단기 | TCN, LSTM, GBM |
| 시계열 장기 | Transformer (Informer) |
| 표 데이터 | GBM (XGBoost), 또는 TabTransformer |
| 그래프 | GCN/GAT |
| 추천 | Two-tower, embedding+MLP |
| 이상 탐지 | AE, Isolation Forest, GBM |
| 이미지 생성 | Diffusion |
| 텍스트 생성 | Transformer (decoder-only) |

### 13.2 Hyperparameter 빠른 선택

| 결정 | 일반 default |
|---|---|
| Optimizer | AdamW (Transformer), SGD+momentum (CNN) |
| LR | 1e-3 (Adam), 0.1 (SGD) — Range Test로 조정 |
| Batch | 256 (메모리 따라) |
| Schedule | Cosine + warmup |
| Init | He (ReLU), Xavier (tanh) |
| Norm | BN (CNN 큰 batch), LN (Transformer/RNN), GN (작은 batch) |
| Activation | ReLU (일반), GELU (Transformer) |
| Regularization | WD 1e-4 + Dropout 0.1~0.5 |
| Augmentation | task별 (이미지: flip, crop, color) |

---

## 14. 면접 단골 Q&A

### Q1. CNN, RNN, Transformer 비교?
"Inductive bias: CNN은 국소성·평행이동, RNN은 시간 순서·시간 sharing, Transformer는 거의 없음 (PE만). 정보 흐름: CNN은 RF 깊이로 확장, RNN은 시간 sequential, Transformer는 모든 시점 동시. 병렬: Transformer 매우 잘, CNN 잘, RNN 못. 데이터: CNN과 RNN은 prior로 효율, Transformer는 많이 필요. 이미지엔 CNN, NLP엔 Transformer가 표준 (작은 데이터엔 RNN도)."

### Q2. LSTM vs GRU?
"거의 동급. LSTM은 3 gate + cell/hidden 분리, GRU는 2 gate + hidden 단일. 파라미터 GRU가 ~25% 적음, 학습 약간 빠름. Task별 차이 — 데이터 적고 빠라야 하면 GRU, 매우 긴 의존성 + 큰 데이터면 LSTM 약간 우위. 실무에선 둘 다 시도해 val로 결정."

### Q3. BN vs LN — 언제 무엇?
"정규화 축이 다름. BN은 batch (큰 batch에 효과), LN은 feature (sample 내 일관). Transformer가 LN인 이유: 시퀀스 길이 가변 + 자기회귀 batch=1 → BN의 batch 통계 부정확. CNN 큰 batch는 BN 표준, 작은 batch는 GN."

### Q4. Adam vs SGD — 어느 쪽?
"Adam은 빠른 수렴 + LR 튜닝 부담 적음 (default 좋음). SGD+momentum은 ImageNet에서 약 1% 더 일반화 (Adam이 sharp minima 경향). AdamW가 weight decay 정확 처리해서 Transformer 표준. 결국 task에 따라."

### Q5. L1 vs L2 정규화?
"L1은 가중치 sparsity (일부 정확히 0) — feature selection 효과. L2는 가중치 부드럽게 줄임 — 일반 default. 신경망에선 거의 항상 L2 (weight decay). L1은 해석성·sparsity 의도일 때. Elastic Net (L1+L2)으로 둘 합칠 수 있음."

### Q6. Dropout vs Weight Decay?
"다른 축. Dropout은 활성값 마스킹 → 암묵적 ensemble. WD는 가중치 크기 → 함수 부드러움. 보통 같이 사용 (보완적). BN과 Dropout은 충돌 가능 → 위치 신중. 큰 Transformer는 dropout 줄이고 WD 강화 추세."

### Q7. He init이 ReLU에 표준인 이유?
"ReLU가 음수 절반을 0으로 만들어 분산 절반 손실. He의 $2/n_{in}$이 이를 보상 — 가중치 분산 두 배. Xavier ($2/(n_{in}+n_{out})$)는 sigmoid/tanh 대칭 가정. ReLU에 Xavier 쓰면 분산 부족 → vanishing 위험."

### Q8. Pooling vs Strided Conv?
"Pooling은 hand-crafted (max/avg), strided conv는 학습된 down-sampling. 현대는 stride conv가 표준 (ResNet) — task-적응적. Max pool은 단순함의 장점 (edge device). 분류 마지막엔 GAP가 표준 — 파라미터 절약, overfit 적음."

### Q9. ViT vs CNN 어느 쪽?
"데이터에 따라. CNN의 국소성 inductive bias가 자연 이미지에 강한 prior → 작은 데이터 (ImageNet 정도)에서 우위. ViT는 prior 없어 매우 많은 데이터 (JFT-300M)에서 우위. ConvNeXt가 CNN에 modern training 흡수해 ViT와 거의 동급."

### Q10. AdamW가 Adam과 다른 점?
"Weight decay 처리. Adam에서 L2 항을 loss에 더하면 그 항도 second moment $v_t$로 normalize되어 의미 왜곡. AdamW는 weight decay를 update 시점에 별도 적용 — gradient를 normalize하고 그 다음 weight decay 곱한 가중치 빼기. SGD-style의 정확한 weight decay. Transformer 시대 표준."

### Q11. 표 데이터엔 왜 GBM이 신경망보다 강한가?
"트리의 분할 기반 학습이 표 데이터에 강함. (1) Feature value 임의 분할로 비선형 자연. (2) 트리 분기로 feature 상호작용. (3) Scale 무관. (4) 이상치·결측 robust. (5) Feature 수에 잘 적응. 신경망은 inductive bias가 표 데이터에 안 맞음 — 컬럼 순서 무관, 위치 의존 없음."

### Q12. 큰 batch가 일반화에 안 좋은 이유?
"세 가설. (1) Sharp minima — gradient noise 적어 flat minima 회피 못함. (2) Update 횟수 감소 — 같은 epoch에서 model이 충분히 안 움직임. (3) 암묵적 정규화 약화 — SGD noise의 정규화 효과. Linear scaling rule + warmup으로 어느 정도 보완, 매우 큰 batch (4096+)는 LARS/LAMB 필요."

---

## 15. 생각해보라 — 단락 답안

**Q. 왜 한 가지 *universal* architecture가 모든 task의 답이 안 되나?**

각 데이터 종류는 *본래의 구조*를 갖는다. 이미지는 2D 격자, 텍스트는 1D 시퀀스, 그래프는 임의 위상. 각 architecture는 특정 구조에 맞는 inductive bias.

CNN의 국소성은 이미지에 자연. 텍스트에선 의미 약함. RNN의 시간 순서는 시퀀스에 자연. 이미지에선 어색함.

*Universal architecture*가 가능하려면:
- 모든 data structure에 적응 가능한 prior.
- 또는 prior 없이 매우 큰 데이터로 학습.

Transformer가 후자에 가까움 — 모든 task (이미지, 텍스트, 음성, 단백질)에 적용. 단 데이터·연산 매우 많이 필요. 작은 task엔 specialized model이 효율.

미래는 *foundation model* — 매우 큰 사전학습 모델이 다양한 task에 fine-tune. 사실상 universal에 가까움. 단 표 데이터 등 일부에선 여전히 specialized (GBM).

**Q. 왜 *비교*가 ML에서 그토록 중요한가?**

ML의 발전은 *각 단계의 한계가 다음 단계의 동기*가 되는 진화. 이 진화를 이해하려면 비교가 필수:

- Perceptron의 XOR 한계 → MLP.
- MLP의 vanishing → ReLU + ResNet.
- RNN의 long-range → LSTM.
- LSTM의 sequential → Transformer.

각 새 기법의 가치는 *직전 기법의 한계*와의 비교에서 명확. "Transformer가 좋다"는 의미 약함. "Transformer가 RNN의 long-range·병렬 한계를 [어떤 메커니즘으로] 풀었다"가 명확.

면접·시험·실무 모두 비교 사고가 핵심.

**Q. 왜 *결정*에 이렇게 많은 trade-off가 있는가?**

ML의 모든 결정은 *제약된 자원*에서의 trade-off. 자원:
- 데이터 양.
- 연산 (시간, GPU).
- 메모리.
- 모델 capacity.
- 시간 (개발 일정).

각 hyperparameter는 이 자원들 사이의 trade-off:
- 큰 batch: GPU 효율 ↑, 일반화 약간 ↓.
- 깊이 多: 표현력 ↑, 학습 어려움 ↑.
- 강한 정규화: overfit 줄임, underfit 위험.

ML 엔지니어의 일은 *주어진 제약*에서 *최선의 trade-off*. 정답은 없고 *상황별 best*. 그래서 비교 사고가 결정 사고로 이어짐.

**Q. *Foundation Model*의 부상이 비교의 본질을 바꾸는가?**

부분적으로 yes. Foundation model 시대의 변화:

(1) **Architecture 비교의 평준화**: 매우 큰 모델은 architecture 차이 작아짐 (ConvNeXt vs ViT). 데이터·연산이 dominant.

(2) **Pretraining의 압도적 가치**: from scratch는 거의 없어짐. Pretrained의 fine-tune이 표준.

(3) **Specialized vs Generic**: Foundation model은 generic, 특정 task는 specialized 여전 우위 (표 데이터의 GBM 등).

(4) **Cost의 변화**: 큰 모델 학습이 매우 비싸 → 비교 자체가 어려움 (한 번 학습에 수백만 달러).

여전 비교의 본질은 같음 — *trade-off의 분석*. 단 axes가 약간 변함. 데이터 효율, 학습 비용, 추론 latency 등이 더 중요.

**Q. *Domain knowledge*가 비교 결정에 어떻게 영향?**

순수 ML 지식만으론 부족. Domain knowledge가:

(1) **Prior 강화**: 의료에선 patient-level split, 시계열엔 forward chaining 등.

(2) **Metric 설계**: 분야별 표준 metric (mAP, BLEU, NDCG).

(3) **Failure mode 인식**: 사기 탐지의 concept drift, 의료의 distribution shift.

(4) **Feature engineering**: 표 데이터에서 도메인 특화 feature.

좋은 ML 엔지니어는 ML 기술 + domain knowledge의 결합. 어느 한 쪽만으론 production에 부족.

**Q. 왜 *ablation study*가 비교의 표준 도구?**

비교의 어려움은 *변수 통제*. 두 모델 비교 시 다른 모든 것 동일해야 의미. Ablation:

(1) **한 컴포넌트씩 제거/추가**: A vs A+X, A+X vs A+X+Y. 각 단계가 X의 가치 측정.

(2) **계산 가능한 비교**: 막연한 "X가 좋다"가 아닌 "X가 [이 setup에서] [이 정도] 기여".

(3) **Trade-off 명확**: X 추가의 비용 (parameter, time) 정량.

(4) **재현 가능**: 명확한 setup → 다른 연구자 검증 가능.

논문에서 ablation table이 거의 필수. 면접에서도 "X와 Y 비교"가 ablation 사고의 적용. 이 챕터의 모든 비교가 사실 ablation 사고의 형태.

---

## 16. 한 줄 요약

- **3축 비교**: 메커니즘 + bias + 상황 적합도.
- **4대 architecture**: FNN(general)/CNN(grid)/RNN(seq)/Transformer(scale). 데이터·길이·양으로 결정.
- **LSTM vs GRU**: 거의 동급, 둘 다 시도.
- **BN vs LN**: 큰 batch CNN BN, RNN/Transformer LN.
- **Adam vs SGD**: Transformer AdamW, ImageNet SGD+momentum.
- **L2 + Dropout + Augmentation** 표준 정규화 조합.
- **ReLU/GELU/Tanh**: hidden 활성화 자리별.
- **He/Xavier**: ReLU/tanh init.
- **Stride conv**가 pooling보다 modern. **GAP**가 FC head 표준.
- **Output-Loss 매칭**: Sigmoid+BCE, Softmax+CE, Linear+MSE.
- **Batch 256~1024** sweet spot, linear scaling rule.
- **Pretrained**가 거의 항상 도움.
- 모든 결정 = *제약된 자원*에서의 trade-off. 정답 없음, 상황별 best.
