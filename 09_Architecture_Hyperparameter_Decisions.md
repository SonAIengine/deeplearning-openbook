# 09. Architecture & Hyperparameter Decisions — "왜 이 숫자, 이 구조?"

> 모든 디자인 선택에는 **이유가 있어야 한다**.
> "depth=12로 했어요"가 답이 아니라, "왜 12, 그리고 16/8과 비교해 어떻게 다른가"가 답.

---

## 0. 사고법 — "결정의 근거"

설계 결정은 항상 다음 4개 축으로 답한다:
1. **무엇을 결정?**
2. **숫자를 키우면 / 줄이면 어떻게?** (양 끝)
3. **어떤 신호로 판단?** (val loss, 학습 곡선)
4. **trade-off의 본질**

---

## 1. 깊이(depth) vs 너비(width)

### 1.1 같은 파라미터 예산이라면?

| 깊이 多 | 너비 多 |
|---|---|
| 계층적 추상화 | 표현 풍부 |
| Vanishing 위험 | 수평적 |
| 학습 어려움 (skip 필요) | 학습 안정 |
| 일반화 잘 됨 (실증적) | 과적합 위험 |

### 1.2 결정 기준
- **계층적 구조가 의미 있는 데이터** (이미지, 자연어) → 깊이가 효율.
- **단순한 함수 근사** → 너비 1~2층로 충분.
- 연산·메모리 한계 있으면 → 너비가 병렬 친화적.

### 1.3 실증
- ResNet: 깊이 50~152가 ImageNet sweet spot.
- EfficientNet: depth/width/resolution 균형 scaling.
- 너비만 키우면 일정 지점에서 수익 체감.

생각해보라: 왜 단순 너비 증가는 깊이 증가만큼 좋지 않은가? → 계층적 합성이 표현 효율을 지수적으로 증가시키는 반면, 너비는 다항적.

---

## 2. Batch Size

### 2.1 작게 vs 크게

| 작음 (16~64) | 큼 (512+) |
|---|---|
| Gradient noise 多 | 정확한 gradient |
| 느림 per epoch | 빠름 per epoch |
| 일반화 ↑ (flat minima) | 일반화 ↓ (sharp minima) |
| BN 통계 부정확 | BN 통계 정확 |
| GPU 비효율 | GPU 효율 |

### 2.2 결정
- 메모리 ↔ batch ↔ LR은 한 묶음.
- 큰 batch 쓰면 **LR linear scaling rule**: batch×k면 LR×k.
- 매우 큰 batch (4096+)는 LARS/LAMB 같은 전용 optimizer.

### 2.3 신호
- val accuracy가 큰 batch일수록 낮아지면 → batch를 줄이거나 LR 보정.
- BN 통계 noise가 학습을 흔들면 → GN/LN으로 교체 또는 batch 키움.

---

## 3. Learning Rate

### 3.1 가장 중요한 단일 하이퍼파라미터.
- 너무 크면 발산.
- 너무 작으면 느림 / local에 갇힘.

### 3.2 결정 절차
1. **LR Range Test** (Smith): LR을 점진적으로 키우며 loss 관찰. 급격히 떨어지는 구간의 1/10 정도가 출발점.
2. 그 LR로 일정 epoch 학습.
3. **Warmup**: 큰 모델·Adam에서 초반 작은 LR로 시작.
4. **Schedule**: Step decay / Cosine / OneCycle.

### 3.3 일반 가이드
| Optimizer | 시작 LR |
|---|---|
| SGD + momentum | 0.1 (CV), 0.01 (NLP) |
| Adam | 1e-3 (일반), 1e-4 (Transformer) |
| AdamW (Transformer) | 1e-4 ~ 5e-4 + warmup + cosine |

### 3.4 신호
- 첫 epoch부터 NaN → LR 너무 큼.
- 수십 epoch 학습해도 안 떨어짐 → 너무 작거나 모델 부족.
- val loss가 진동 → LR 줄일 시점 (decay).

---

## 4. Learning Rate Schedule

### 4.1 옵션
- **Constant**: 단순. 좋은 결과 어려움.
- **Step decay**: epoch 30, 60, 90에서 1/10. 전통.
- **Cosine annealing**: 부드럽게 감소. Transformer 표준.
- **Warmup**: 큰 모델 필수.
- **OneCycle**: warmup → 큰 LR → 감소. 빠른 학습.

### 4.2 결정
- ImageNet ResNet → step decay.
- Transformer/큰 모델 → warmup + cosine.
- 빠른 prototype → OneCycle.

### 4.3 왜 warmup?
- 학습 초기에 BN, Adam의 second moment 통계가 noisy.
- 큰 LR이 위험 → 작게 시작해 안정화 후 키움.

---

## 5. Activation Function 선택

### 5.1 결정 트리
```
출력층?
├── 이진 분류 → Sigmoid
├── 다중 분류 → Softmax
├── 회귀 → Linear
└── Multi-label → 클래스별 Sigmoid

은닉층?
├── 일반 CNN/MLP → ReLU
├── Transformer → GELU
├── RNN의 hidden → Tanh
├── LSTM gate → Sigmoid (의미: 0~1)
└── Dying ReLU 우려 → LeakyReLU
```

### 5.2 신호
- 많은 뉴런이 항상 0 출력 → dying ReLU → He init or LeakyReLU.
- 출력이 비대칭 → 0-centered activation 필요.

---

## 6. Optimizer 선택

### 6.1 결정 트리
```
모델 / 데이터?
├── 큰 Transformer / NLP → AdamW + warmup + cosine
├── ImageNet CNN → SGD+momentum (일반화 우위)
├── 작은 데이터·복잡 loss → Adam
├── 강화학습 → RMSProp / Adam
└── 큰 batch (4096+) → LARS / LAMB
```

### 6.2 LR과 짝
- SGD: 0.1 + momentum 0.9.
- Adam: 1e-3 (Transformer 1e-4).

### 6.3 신호
- Adam에서 일반화 gap 크면 → SGD 시도.
- SGD에서 학습 정체 → momentum 확인, LR 재탐색.

---

## 7. Initialization

### 7.1 결정
| 활성화 | Init |
|---|---|
| ReLU | He (Kaiming) |
| Tanh/Sigmoid | Xavier (Glorot) |
| SELU | LeCun |

### 7.2 신호
- 학습 초기에 활성값이 모두 0 또는 모두 큰 값 → init 잘못.
- gradient norm이 매우 작거나 큼 → init 잘못.

### 7.3 Bias init
- 일반적으로 0.
- LSTM forget gate bias = 1~2 (정보 보존 시작).
- ReLU 직전이라면 작은 양수도 가능 (dead 뉴런 방지).

---

## 8. Regularization 강도 선택

### 8.1 일반 가이드
| 데이터 크기 | 권장 |
|---|---|
| 작음 (<10k) | dropout 0.5, weight decay 1e-4, strong aug |
| 中 (10k~100k) | dropout 0.3, weight decay 1e-4, aug |
| 큼 (100k~) | dropout 0.1, weight decay 1e-4, mild aug |
| 매우 큼 | dropout 0, weight decay 1e-2, label smooth |

### 8.2 신호
- Train ↑ Val ↑ → underfit, regularization 줄임.
- Train ↓ Val ↑ → overfit, regularization 늘림.
- Train ≈ Val 둘 다 높음 → 모델 키움.

---

## 9. Normalization 선택

### 9.1 결정
| 모델 / 상황 | 권장 |
|---|---|
| 큰 batch CNN | BatchNorm |
| 작은 batch (detection, seg) | GroupNorm |
| RNN | LayerNorm |
| Transformer | LayerNorm |
| 스타일 변환 | InstanceNorm |
| Self-normalizing 원함 | SELU + LeCun init (BN 없이) |

### 9.2 어디에 두나?
- Pre-Norm vs Post-Norm.
- Transformer는 **Pre-Norm**이 더 안정 (대부분 현대 구현).
- CNN은 conv → BN → ReLU가 표준.

---

## 10. Loss 선택 (출력층-loss 매칭)

### 10.1 매칭 표
| 출력 활성화 | Loss |
|---|---|
| Linear | MSE/MAE/Huber |
| Sigmoid | BCE |
| Softmax | CE |
| 각 클래스 Sigmoid (multi-label) | per-label BCE |

### 10.2 보조 loss
- **Label smoothing**: CE를 smooth. 큰 모델 일반화↑.
- **Focal loss**: 어려운 sample에 가중. 클래스 불균형.
- **Auxiliary loss**: 깊은 모델에서 중간 head로 부수 loss (vanishing 완화).
- **Contrastive**: 표현 학습 (SimCLR 등).

---

## 11. 데이터 vs 모델 — "어디에 투자?"

### 11.1 신호로 판단
| 상황 | 다음 행동 |
|---|---|
| Train도 안 떨어짐 | 모델 키움, LR 점검 |
| Train ↓ Val ↑ | 데이터/aug 늘림, regularization |
| 둘 다 좋지만 더 원함 | 데이터·모델 동시에 (scaling) |
| 데이터 한계, 모델 충분 | 외부 데이터·전이학습 |

### 11.2 Scaling law 직관
- 일반적으로 **데이터·연산·파라미터**를 균형 있게 키울 때 성능이 멱법칙으로 향상.
- 데이터만 늘려도 한계, 모델만 키워도 한계.
- 1순위 부족 자원이 무엇인지 판단해 거기에 투자.

---

## 12. Early Stopping 기준

### 12.1 patience
- val loss가 N epoch 연속 개선 없으면 멈춤.
- N = 5~20 (task에 따라).

### 12.2 신호
- val loss가 부드럽게 감소 → 충분히 더.
- val loss가 진동 → LR 줄이거나 patience 늘림.
- val loss가 급격히 상승 → 즉시 멈춤, regularization 추가.

---

## 13. Hyperparameter Search 전략

### 13.1 옵션
| 방법 | 장단점 |
|---|---|
| Grid search | 단순, 차원 폭발 |
| Random search | 효율적, 단순 |
| Bayesian (Optuna, Hyperopt) | 효율적, 의존성 ↑ |
| Population-based (PBT) | 동적, 비싼 |

### 13.2 우선순위
중요도 순:
1. LR (가장 민감)
2. Batch size
3. Optimizer
4. Init
5. Activation
6. Architecture (depth, width)
7. Regularization 강도

### 13.3 좋은 baseline
- 잘 알려진 hyperparam (논문 기본값) 시작.
- 한 번에 한 변수만 변경 (ablation).

---

## 14. Architecture 선택 골격 — "이 문제에는?"

### 14.1 절차
1. **데이터 분석**: 형태(2D/시퀀스/표), 크기, 라벨 분포, 길이 분포.
2. **Baseline 모델**: 작고 빠른 모델로 시작.
3. **Ablation**: 한 컴포넌트씩 추가/제거.
4. **Scaling**: 효과 보이면 깊이·너비 키움.
5. **Regularization 조정**: overfit 시 늘림.

### 14.2 흔한 패턴
- 이미지 분류: ResNet50 + CE + SGD/Adam + step decay.
- 텍스트 분류: BERT/Transformer + CE + AdamW + cosine.
- 시계열: LSTM/TCN + MSE + Adam.

---

## 15. 한 표로 — "결정 매트릭스"

| 결정 | 작은 데이터 | 큰 데이터 |
|---|---|---|
| 모델 크기 | 작게 | 크게 |
| Regularization | 강하게 | 약하게 |
| Dropout | 0.5 | 0~0.1 |
| Weight decay | 1e-4 | 1e-2 |
| Augmentation | 강하게 | 약하게 (이미 다양) |
| Optimizer | Adam (안전) | SGD/AdamW |
| Schedule | Step or Cosine | Cosine + warmup |
| Pretrained | 거의 필수 | 옵션 |

---

## 16. "생각해보라" 확장

1. **LR을 epoch 하나에 한 번만 바꾸는 step decay vs 매 step 바꾸는 cosine, 어느 쪽이 좋은가?** → Cosine이 부드럽고 일반화 좋다고 보고됨. Step decay는 단순하고 명확.
2. **Batch size를 키우면 왜 LR을 같이 키워야?** → 같은 epoch 안에서 update 횟수가 줄어드니 한 번 update를 더 크게 해서 보상.
3. **Adam의 LR이 SGD의 LR보다 작은 이유?** → Adam은 자체적으로 LR을 second moment로 normalize하므로 effective LR이 다름.
4. **He init이 ReLU를 위해 분산을 2/n_in으로 정한 이유?** → ReLU가 음수 절반을 0으로 만들어 분산 절반 손실. 보상.
5. **모델 크기를 키울 때 너비와 깊이 중 무엇을 먼저?** → 일반적으로 깊이부터 키우다가 학습이 어려워지면 너비. 또는 EfficientNet 식 균형 scaling.

---

## 17. 한 줄 요약

- 모든 결정은 **신호**(loss 곡선, gradient norm)로 진단하고 **이유**로 답한다.
- LR > Batch > Optimizer > Init > Activation > Depth/Width > Regularization 순으로 민감.
- 데이터·모델·연산은 **균형**, 부족한 자원에 투자.
- Baseline → Ablation → Scaling → Regularization 조정의 흐름.
- "왜 이 숫자?"엔 항상 양 끝(작게/크게)을 비교해서 답한다.
