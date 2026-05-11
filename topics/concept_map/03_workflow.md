# 03. 워크플로우 — 결정들이 서로 어떻게 제약하는가

계보·관통이 *"무엇이 있는가"*였다면 워크플로우는 *"무엇을 먼저 정해야 다음을 정할 수 있는가"*다. 결정의 순서가 잘못되면 *뒤에서 앞을 뒤집어야* 해서 비용이 폭발한다.

---

## 1. 큰 그림 — 5단계 체인

```
[1] EDA            →  [2] 전처리·feature  →  [3] 아키텍처     →  [4] 학습 setup       →  [5] 진단·반복         →  [6] 평가
 분포·결측·이상치     결측 처리·scale·encode    inductive bias       loss·optim·LR·batch     train/val gap → 처방    metric ↔ 도메인 KPI
```

각 단계는 *다음 단계의 선택지*를 제약한다. 한 단계의 결정을 모르면 다음 단계에서 *옵션이 거짓으로 많아 보인다*.

---

## 2. 각 단계의 결정 + 무엇을 제약하는가

### [1] EDA — *데이터의 진실을 알기 전엔 아무것도 못 정한다*

| 알아낼 것 | 다음 단계의 무엇을 결정하나 |
|---|---|
| target 분포 (편향·imbalance) | loss 선택 (CE vs Focal vs Weighted) + sampling 전략 + metric (Acc vs F1/AUC) |
| feature 분포 (skew·heavy tail) | scaling 방법 (standardize vs robust scaler vs log transform) |
| 결측률·결측 패턴 (MCAR/MAR/MNAR) | imputation 전략 + 결측 flag feature 생성 여부 |
| 이상치 (분포 양 끝 / Cook's distance) | loss 선택 (MSE vs Huber vs MAE) + clipping 여부 |
| feature 간 상관 (다중공선성) | feature selection·PCA 필요성 + tree 모델 vs linear 모델 |
| 데이터 type (이미지·텍스트·tab·시계열) | architecture family 자체 |

자세한 EDA 사고법은 [topics/EDA/README.md](../EDA/README.md).

**함정**: EDA 없이 architecture부터 정하면, 나중에 *데이터 보고 architecture 갈아엎는* 비용이 폭발.

---

### [2] 전처리·feature — *모델은 입력 분포를 가정한다*

| 결정 | 무엇이 제약하나 |
|---|---|
| **scaling** (standardize / minmax / robust) | optimizer 수렴 속도, 특히 SGD·Adam의 LR 적절값 |
| **encoding** (one-hot / target / embedding) | 모델 capacity 요구 + 메모리 |
| **imputation** | BN과의 상호작용 — imputed flag가 useful feature가 됨 |
| **augmentation 전략** | regularization 강도 — 강한 aug는 다른 regularization을 줄여도 됨 |
| **outlier 처리** (drop / clip / robust loss) | loss 선택과 직결 |

**함정**: train·val·test에 *동일한* preprocessor를 fit하면 leakage. 항상 train에서 fit, val/test에 transform만.

---

### [3] 아키텍처 선택 — *inductive bias의 매칭*

데이터 type → 아키텍처 후보 1차 결정:

| 데이터 | 1순위 후보 | 2순위 | 언제 2순위? |
|---|---|---|---|
| 이미지 | CNN (ResNet·EfficientNet) | ViT | 데이터 매우 많을 때 |
| 텍스트 | Transformer | RNN/LSTM | 시퀀스 매우 길고 streaming일 때 |
| 시계열 | Transformer / LSTM | TCN (1D CNN) | 짧은 패턴 위주 |
| Tabular | GBDT (XGBoost/LightGBM) | MLP | 데이터 매우 많거나 feature 매우 많을 때 |
| Set / Graph | DeepSet / GNN | — | 자연스러운 구조 |

**제약**: 데이터량과 architecture가 미스매치면 다른 결정으로 보완 불가.
- 작은 데이터 + 강한 inductive bias 아키텍처 (CNN) ✓
- 작은 데이터 + 약한 bias 아키텍처 (Transformer) ✗ → augmentation·pretraining 필수

자세한 매칭은 본문 §7.

---

### [4] 학습 setup — *모두 묶여 있다*

서로 *얽혀 있는* 하이퍼파라미터:

```
batch size   ↔  learning rate    ↔  optimizer      ↔  regularization
       ↘            ↓                  ↓                    ↓
       gradient    어떻게 step       Adam이면         weight decay,
       noise        취하나           per-param LR        dropout
```

**얽힘의 규칙**:
- batch size ↑ → gradient noise ↓ → 일반화 약화 → LR도 같이 ↑ (linear scaling rule) + warmup 도입
- Adam 쓰면 LR을 SGD보다 작게 (보통 1e-3 vs 1e-1)
- weight decay는 AdamW로 분리 (Adam의 L2와 다름)
- dropout과 BN을 같이 쓰면 충돌 — 보통은 BN만, 또는 dropout을 매우 작게(0.1 이하)

**LR schedule**:
| 패턴 | 언제 |
|---|---|
| Cosine | 일반 default — 학습 후반 부드러운 감쇄 |
| Step decay | 명시적 단계 (특정 epoch에서 1/10) |
| Warmup + Cosine | 큰 batch / Transformer 표준 |
| ReduceOnPlateau | val loss 정체 시 자동 감쇄 |
| One-cycle | 빠른 학습 (Fastai 스타일) |

자세히는 [topics/hyperparameters/lr_schedule.md](../hyperparameters/lr_schedule.md).

---

### [5] 진단·반복 — *loss curve가 모든 것을 말한다*

| 증상 (train, val) | 진단 | 처방 |
|---|---|---|
| train ↑, val ↑ (둘 다 안 떨어짐) | **underfit** — capacity 부족 또는 LR 잘못 | 모델 크게, LR 검토, regularization 줄임 |
| train ↓, val ↓ (둘 다 떨어지지만 격차 작음) | 정상 학습 진행 중 | 계속 |
| train ↓, val ↗ (격차 벌어짐) | **overfit** | regularization ↑, augmentation ↑, dropout, early stopping, 데이터 ↑ |
| train ↓, val 진동 | val 데이터 너무 작거나 noisy | val set 키우거나 k-fold |
| train ↓ 빠르게, val 횡보 | 같은 클래스/feature만 학습 | sampling 균형, loss 가중치 |
| train과 val 모두 NaN | gradient explosion / 잘못된 init | grad clipping, init 점검, LR 낮춤 |
| 학습 안 됨 (loss 미동) | LR 0 / data 잘못 / 그래디언트 끊김 | LR 검증, data flow 검증, requires_grad 확인 |

자세한 진단은 본문 §10.

**핵심 사고**: *gap의 방향*이 진단의 출발. train≈val이면 capacity 문제, train↘val↗면 generalization 문제.

---

### [6] 평가 — *metric이 도메인 KPI와 맞는가*

| 도메인 상황 | 적절한 metric | 이유 |
|---|---|---|
| 클래스 균형 + 오분류 비용 동일 | Accuracy | 자연스러운 정답률 |
| 클래스 불균형 | Precision / Recall / F1 | majority 편향 막기 |
| 의료·사기 탐지 (FN이 매우 위험) | Recall + threshold 낮춤 | 놓치지 않는 게 우선 |
| 스팸 분류 (FP가 짜증) | Precision + threshold 높임 | 잘못 잡지 않기 우선 |
| 등급 매기기 (랭킹) | AUC / NDCG | 순서 자체의 품질 |
| 회귀 (스케일 의미) | RMSE / MAE / MAPE | 차이 크기 직접 측정 |
| 다중라벨 | Macro/Micro F1 | 클래스별 평균 vs 전체 평균 |

**함정**:
- 단일 metric으로 결정하지 말 것 — confusion matrix와 함께 보기.
- val/test 분리가 임의면 metric도 임의 — split이 도메인 (시계열은 시간 순, sample들 간 누수 없는지) 반영해야.
- Acc 99%인데 실제로는 majority class만 맞추는 경우 (imbalance) — 항상 *baseline 비교* (다수 클래스로 모두 찍은 정확도와 비교).

---

## 3. 결정 다이어그램 — Overfit 진단 → 처방

가장 자주 만나는 시나리오의 체인:

```
val loss > train loss → overfit 확실

    ↓
원인 1: 데이터가 작다
    ├─ 처방 A: augmentation 강화 (이미지: rotate/flip/crop/mixup, 텍스트: dropout/back-translation)
    ├─ 처방 B: pretrained 모델로 fine-tune
    └─ 처방 C: 데이터 수집 (가능하면)

원인 2: 모델이 크다
    ├─ 처방 A: depth/width 줄임 (가장 직접적)
    ├─ 처방 B: weight decay ↑ (L2 또는 AdamW)
    └─ 처방 C: dropout 도입 또는 ↑

원인 3: 학습이 길다
    ├─ 처방 A: early stopping
    └─ 처방 B: LR 낮춰 학습 안정시키고 step 줄임

원인 4: regularization 부족
    ├─ 처방 A: label smoothing
    └─ 처방 B: BN 도입 (CNN classification에서 효과)
```

**조합의 원칙**:
- *한 번에 하나씩* 바꿔서 효과 확인 (ablation 사고)
- *augmentation*은 거의 항상 켜둠 (cost 없음)
- *weight decay*는 default 1e-4 – 1e-2 범위로 항상 켜기
- 위 4개 다 해도 안 되면 데이터·label 자체 의심 (noisy label, leakage 등)

---

## 4. 결정 다이어그램 — Class Imbalance 처방

데이터 EDA에서 imbalance 발견 시:

```
imbalance 정도 (1:10 미만 / 1:10–1:100 / 1:100 초과)
       ↓
┌─────────────────────────────────────────────────┐
│ 약함 (1:10 미만)                                  │
│ → metric만 F1/AUC로 바꿈, 모델은 그대로            │
├─────────────────────────────────────────────────┤
│ 중간 (1:10 – 1:100)                              │
│ → Weighted CE (클래스 빈도 역수 가중치)              │
│ → 또는 oversampling (SMOTE 등) — train만          │
│ → val set은 *반드시* 원래 분포로 (실제 환경 반영)     │
├─────────────────────────────────────────────────┤
│ 심각 (1:100 초과)                                 │
│ → Focal Loss (γ=2 기본)                          │
│ → 또는 anomaly detection으로 framing 자체 변경     │
│ → minority class oversampling + majority under   │
│ → threshold 후처리 (decision boundary 조정)        │
└─────────────────────────────────────────────────┘
```

자세한 task별 가이드는 본문 §13 (이상탐지·추천 등).

---

## 5. 결정 다이어그램 — 작은 데이터셋 대응

데이터가 1k – 10k 정도로 작을 때:

```
[1] Pretrained model 사용 가능한가?
       Yes → Transfer learning (대부분 1순위)
       No  → 아래로

[2] Augmentation을 강하게 줄 수 있는가?
       Yes → 강한 aug + 작은 모델
       No  → 아래로

[3] Strong inductive bias 아키텍처를 쓸 수 있는가?
       Yes → CNN (이미지) / Tree (tabular)
       No (Transformer 등 약한 bias) → 더 작은 데이터 분류 → 비추천

[4] Cross-validation으로 정직하게 평가하는가?
       항상 Yes — 작은 데이터에서 single split은 운에 좌우됨
```

**함정**: 작은 데이터에 큰 모델 + 약한 regularization → 거의 확실히 overfit. *모델 크기를 데이터 크기에 맞추는 것*이 가장 안전.

---

## 한 줄 메타 통찰

> *"단계별 결정을 따로 외우는 게 아니라, **앞 단계의 결정이 뒤 단계의 옵션을 어떻게 깎는가**를 외워야 한다. 작업 순서가 곧 사고 순서다 — EDA부터 metric까지 한 사슬."*
