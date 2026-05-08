# 11. Comparison & Decision — "X냐 Y냐"

> 같은 자리를 다투는 옵션들의 **본질적 차이**와 **결정 기준**.
> 시험·면접에서 "왜 X 대신 Y?" 질문에 즉답할 수 있도록.

---

## 0. 사고 패턴 — "비교는 항상 3축으로"

비교는 막연한 장단점 나열이 아니라 **3개 축**에서 답한다:
1. **메커니즘적으로 무엇이 다른가** (구조의 차이)
2. **어떤 inductive bias / 가정이 다른가**
3. **언제 무엇을 선택?** (데이터·문제·예산 기준)

---

## 1. 4대 Architecture: FNN vs CNN vs RNN vs Transformer (★ 단골)

### 1.1 한 표 비교

| | FNN | CNN | RNN | Transformer |
|---|---|---|---|---|
| Inductive bias | 거의 없음 | 국소성·평행이동·계층성 | 시간 순서·시간 sharing | 거의 없음 (PE만) |
| 파라미터 sharing | 없음 | 공간 축 | 시간 축 | 토큰 간 동등 (attention) |
| 입력 크기 | 고정 | 가변(주로 fixed) | 가변 | 가변 (T² 메모리 주의) |
| 병렬화 | 가능 | 가능 | 어려움 | 매우 잘됨 |
| Long-range | 가능(파라미터 많이) | 깊이 필요 | 약함 | 강함 |
| 데이터 요구 | 보통 | 보통 | 보통 | 많음 |
| 적합 데이터 | 표/일반 | 격자(이미지) | 시퀀스 | 시퀀스/뭐든(데이터 多) |

### 1.2 결정 트리

```
데이터가 무엇인가?
├── 격자(이미지) → CNN (or Vision Transformer if 데이터 多)
├── 시퀀스(텍스트, 음성, 시계열)
│   ├── 짧고 데이터 적음 → CNN(1D), LSTM/GRU
│   ├── 길고 long-range 필요 → Transformer
│   └── 매우 큰 데이터·연산 → Transformer
├── 그래프 → GNN
├── 표 → MLP / GBM / TabTransformer
└── 그 외 → 데이터 구조에 맞춰 모델 설계
```

### 1.3 "왜 NLP가 RNN→Transformer로?"
- 병렬화: RNN은 순차, GPU 활용 못함.
- Long-range: LSTM도 수천 step 어렵.
- Scaling: 데이터·파라미터 늘릴수록 Transformer가 더 잘 따라감.

### 1.4 "왜 이미지에선 CNN이 여전히 강한가?"
- 강력한 inductive bias로 데이터 적을 때 우위.
- ViT는 데이터 매우 많을 때 (수억 장) CNN을 이김.
- ConvNeXt는 Transformer 디자인을 CNN에 역수입.

---

## 2. LSTM vs GRU vs Attention/Transformer

### 2.1 핵심 차이
| | LSTM | GRU | Attention |
|---|---|---|---|
| State | cell + hidden | hidden | 직접 query-key |
| Gate | 3 + candidate | 2 | (없음, soft selection) |
| 파라미터 | 많음 | LSTM의 ~75% | task에 따라 |
| 병렬 | ✗ | ✗ | ✓ |
| Long-range | 보통 | 보통 | 강함 |

### 2.2 언제?
- 데이터 적고 시퀀스 짧음 → **GRU** (단순, 빠름).
- 매우 긴 의존성, 큰 데이터, 연산 충분 → **Transformer**.
- 중간, RNN을 꼭 써야 → LSTM.

### 2.3 "LSTM과 GRU 중 무엇을?"
- 실무: **둘 다 시도해보고 val 결과로 결정**. 보통 거의 동급.
- 시험 답: GRU가 단순하고 빠르나 long·복잡 task에서는 LSTM이 약간 우위 보고도 있음.

---

## 3. Normalization: BN vs LN vs GN vs IN

### 3.1 정규화 축

| | 정규화하는 축 | 의미 |
|---|---|---|
| **BatchNorm** | batch 차원 (channel별) | 같은 채널의 다른 샘플끼리 정규화 |
| **LayerNorm** | feature 차원 (sample 내) | 한 샘플의 모든 channel 정규화 |
| **GroupNorm** | feature를 그룹으로 묶음 | 채널 그룹별 정규화 |
| **InstanceNorm** | sample 내 channel별 | 채널마다 따로 정규화 |

### 3.2 어디에 쓰나?

| 모델 | 표준 |
|---|---|
| 큰 batch CNN | **BN** |
| 작은 batch CNN | **GN** |
| RNN, Transformer | **LN** |
| 스타일 변환 | **IN** |

### 3.3 왜 Transformer는 LN?
- 시퀀스 길이 가변 → batch 통계 부정확.
- 자기회귀 추론 시 batch=1.
- 한 sample 내에서 정규화하므로 batch와 무관.

### 3.4 왜 작은 batch에서 BN이 안 좋은가?
- batch=2, 4면 통계 추정이 매우 noisy.
- detection, segmentation처럼 메모리 큰 task에서 흔한 상황 → GN 사용.

---

## 4. Optimizer: SGD vs Adam vs AdamW

### 4.1 본질
| | 메커니즘 | 특징 |
|---|---|---|
| **SGD** | 단순 update | noise 많음, 평탄 영역 느림 |
| **SGD + Momentum** | 관성 | 진동 줄임 |
| **Adam** | momentum + adaptive LR | 빠른 수렴, LR 튜닝 쉬움 |
| **AdamW** | Adam + 정확한 weight decay | 현대 표준 (특히 Transformer) |

### 4.2 언제?
- ImageNet CNN: **SGD + momentum**이 종종 더 나은 일반화 (보고됨).
- Transformer: **AdamW + warmup + cosine** (사실상 표준).
- 작은 데이터·복잡한 loss landscape: Adam이 안전.
- 최적화가 정말 중요한 RL: 종종 RMSProp.

### 4.3 "Adam이 항상 최선이 아닌 이유?"
- sharp minima로 가는 경향 → 일반화 약간 손해 가능.
- second moment 추정의 편향 (warmup 없으면 초반 불안정).

---

## 5. Regularization: L1 vs L2 vs Dropout vs Early Stop

### 5.1 메커니즘 차이
| | 작용점 | 효과 |
|---|---|---|
| **L1 (Lasso)** | 가중치 sparsity | 일부 정확히 0 → feature selection |
| **L2 (Ridge)** | 가중치 크기 | 부드럽게 줄임 |
| **Dropout** | 활성값 마스킹 | 암묵적 ensemble |
| **Early stopping** | 학습 시간 | 최적 시점 멈춤 |
| **Augmentation** | 입력 분포 | 분포 확장 |
| **Label smoothing** | 라벨 분포 | 과신 방지 |

### 5.2 언제?
- 해석성 중요 + sparse feature 원함 → **L1**.
- 일반 딥러닝 → **L2 (weight decay)** 기본.
- FC layer 많은 모델 → **Dropout 0.3~0.5**.
- 데이터 적음 → **Augmentation + Early stop**.
- 분류 → **Label smoothing** (특히 큰 모델).

### 5.3 같이 쓸 수 있나?
- 가능. 대부분 보완적.
- 단 BN + Dropout은 같이 쓸 때 조심.

---

## 6. Activation: ReLU vs LeakyReLU vs GELU vs Sigmoid/Tanh

### 6.1 비교
| | 식 | 어디서? |
|---|---|---|
| **Sigmoid** | 1/(1+e⁻ˣ) | 출력층(이진), gate (LSTM) |
| **Tanh** | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | RNN의 hidden |
| **ReLU** | max(0,x) | 일반 CNN/MLP 표준 |
| **LeakyReLU** | max(αx,x) | dying ReLU 우려 시 |
| **GELU** | x·Φ(x) | Transformer 표준 |
| **Swish/SiLU** | x·σ(x) | EfficientNet 등 |

### 6.2 결정
- 일반 분류 CNN/MLP → **ReLU** (단순, 빠름).
- Transformer → **GELU**.
- RNN의 hidden → **Tanh** (전통).
- LSTM의 gate → **Sigmoid** (의미: 0~1 비율).
- 출력층:
  - 이진 분류 → **Sigmoid + BCE**.
  - 다중 분류 → **Softmax + CE**.
  - 회귀 → **Linear + MSE**.

---

## 7. Loss: MSE vs MAE vs Huber / CE vs Focal

### 7.1 회귀
| | 특징 |
|---|---|
| **MSE** | 큰 오차에 민감, 가우시안 가정 |
| **MAE** | 이상치 robust, 0에서 미분 불가 |
| **Huber** | 둘의 절충 |

→ 이상치 많으면 MAE/Huber, 일반은 MSE.

### 7.2 분류
| | 특징 |
|---|---|
| **CE (Cross-Entropy)** | 표준. MLE 등가 |
| **Focal Loss** | 어려운 sample에 가중. 클래스 불균형 |
| **Label Smoothing CE** | 과신 방지, 일반화↑ |

→ 일반은 CE, 불균형 심하면 Focal.

---

## 8. Initialization: Xavier vs He vs LeCun

### 8.1 가정
| | 분산 | 가정한 활성화 |
|---|---|---|
| **Xavier (Glorot)** | 2/(n_in + n_out) | tanh, sigmoid |
| **He (Kaiming)** | 2/n_in | ReLU 계열 |
| **LeCun** | 1/n_in | SELU |

### 8.2 결정
- ReLU → **He**.
- Tanh/Sigmoid → **Xavier**.
- SELU → **LeCun**.
- 잘못 매칭하면 vanishing/exploding 빨리 발생.

---

## 9. Pooling vs Strided Conv

| | Pooling | Strided Conv |
|---|---|---|
| 학습 | 비학습 | 학습 가능 |
| 표현력 | 약 | 강 |
| 메모리 | 적음 | 약간 더 |
| 현대 | 마지막 GAP에서만 | down-sampling 표준 |

→ 현대는 strided conv가 표준. 마지막에 GAP.

---

## 10. Regression vs Classification 출력 설계

| Task | 마지막 layer | Loss |
|---|---|---|
| 회귀 (실수) | Linear | MSE/MAE/Huber |
| 이진 분류 | Sigmoid | BCE |
| 다중 분류 | Softmax | CE |
| Multi-label | 각 라벨 Sigmoid | per-label BCE |
| 순위 | Rank loss | Pairwise loss |
| 분포 예측 | Mixture density | NLL |

생각해보라: 다중 분류에서 sigmoid를 클래스마다 쓰면? → 확률 합이 1이 안 되어 multi-label이 됨. 반대로 multi-label에서 softmax 쓰면 확률 분배가 강제됨.

---

## 11. Batch Size 작게 vs 크게

| | 작은 batch (16~64) | 큰 batch (1024+) |
|---|---|---|
| 일반화 | 좋음 (noise 효과) | 약간 떨어짐 |
| 학습 속도(per epoch) | 느림 | 빠름 |
| 수렴 안정성 | noisy | 안정 |
| GPU 효율 | 낮음 | 높음 |
| BN 통계 | 부정확 | 정확 |

**결정**:
- 메모리 한계 + GPU 효율 → 큰 batch.
- 일반화 우선 → 작은~중간 batch (32~256).
- 큰 batch + LR scaling rule (linear warmup) 사용.

---

## 12. Train from Scratch vs Transfer Learning vs Foundation Model

| 데이터 | 권장 |
|---|---|
| 매우 적음 (< 1000) | Pretrained + linear probing |
| 적음 (~10k) | Pretrained + fine-tuning |
| 많음 (100k+) | Train from scratch도 가능, 그래도 pretrained가 종종 우위 |
| 매우 많음 + 도메인 특이 | from scratch 또는 domain-specific pretrain |

---

## 13. Decision Cheatsheet — "이 상황에는 무엇을?"

| 상황 | 권장 |
|---|---|
| 이미지 분류, 데이터 中 | **CNN (ResNet/EfficientNet)** + transfer learning |
| 이미지 분류, 데이터 매우 多 | **Vision Transformer** |
| 짧은 텍스트 분류 | **TextCNN** 또는 BERT fine-tuning |
| 긴 문서 이해 | **Transformer (Longformer 등)** |
| 음성 인식 | **CNN+Transformer (Whisper 식)** |
| 시계열 단기 예측 | **TCN, LSTM, 또는 트리** |
| 시계열 장기 예측 | **Transformer (Informer)** |
| 표 데이터 | **GBM (XGBoost)** 또는 TabTransformer |
| 그래프 | **GCN/GAT** |
| 추천 | **embedding + MLP, two-tower** |
| 이상 탐지 | **Autoencoder, Isolation Forest** |
| 생성 (이미지) | **Diffusion** |
| 생성 (텍스트) | **Transformer (decoder-only)** |

---

## 14. "왜 X 대신 Y?" 빈출 질문 골격

> "X와 Y는 같은 자리에서 다른 [메커니즘 / 가정]을 사용한다.
> [상황 A]에선 X가 [이유]로 우위이고,
> [상황 B]에선 Y가 [이유]로 우위이다.
> 따라서 [내 task에선 X/Y]를 선택한다."

이 틀을 외워두면 거의 모든 비교 질문에 답 가능.

---

## 15. "생각해보라" 확장

1. **Vision Transformer가 ImageNet에서 데이터 적을 땐 CNN보다 못한 이유?** → CNN의 inductive bias가 강한 prior 역할. 데이터 적으면 prior가 큰 도움.
2. **Adam이 sharp minima로 가는 경향이 있다면 왜 여전히 표준?** → 빠른 수렴, LR 튜닝 부담 적음, 대부분 task에서 충분.
3. **L2와 weight decay는 같은가?** → SGD에선 동등, Adam에선 다름 (AdamW가 정확한 weight decay).
4. **ResNet이 Inception보다 표준이 된 이유?** → skip connection의 단순함 + 깊이 한계 돌파 + 후속 연구의 호환성.
5. **GBM이 표 데이터에서 딥러닝을 자주 이기는 이유?** → 트리는 feature 간 상호작용을 자연스럽게 학습, 차원 무관, 데이터 적어도 동작. 딥러닝은 inductive bias 부족 + 데이터 요구.

---

## 16. 한 줄 요약

- 모든 비교는 **3축**(메커니즘, bias, 상황)으로.
- "왜 X 대신 Y?" → "다른 메커니즘 + 다른 상황 적합도".
- **모델·loss·norm·optimizer·activation은 서로 짝**이 있다 (예: ReLU+He, Transformer+LN+AdamW+GELU).
- 결정 트리를 외우는 게 아니라 **선택 기준**(데이터, 길이, 양, bias)을 답할 수 있어야.
