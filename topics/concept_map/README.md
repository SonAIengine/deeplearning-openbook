# 개념 연결도 — 용어를 계보·관통·워크플로우로 묶기

용어를 나열하는 사전은 외울 거리만 늘린다. 같은 자료를 **계보·관통·워크플로우**의 세 시선으로 다시 묶으면, 각 용어가 *왜 등장했고 무엇을 해결했는지*가 한 흐름으로 잡힌다.

이 문서가 도와주려는 사고 패턴 한 줄:

> "이 개념은 *어떤 한계*에서 태어났고, *무엇을 해결*했으며, *어떤 새 함정*을 만들었는가?"

세 개의 관점은 서로 보완한다.

| 관점 | 무엇을 묻는가 | 정리 자료 |
|---|---|---|
| 1. **계보 (Lineage)** | 시간 순으로 — 다음 단계가 *이전의 무엇*을 해결했나? | [01_lineages.md](01_lineages.md) |
| 2. **관통 (Transversal)** | 하나의 문제를 *여러 관점*에서 동시에 푼다 — 같은 표적, 다른 무기 | [02_transversal.md](02_transversal.md) |
| 3. **워크플로우 (Workflow)** | 결정들이 *서로 어떻게 제약*하는가 — EDA → … → 진단의 사슬 | [03_workflow.md](03_workflow.md) |

## 한눈에 — 계보 6개

각 행은 *"이전의 한계 → 다음의 해결"* 사슬을 하나의 흐름으로 본다.

| 계보 | 흐름 | 통찰 한 줄 |
|---|---|---|
| **Loss (회귀)** | MSE → MAE → Huber | "outlier에 얼마나 너그러운가"가 축. δ로 조절. |
| **Loss (분류)** | CE → Focal → LabelSmooth | "어떤 sample에 집중·어떤 자신감 막을지"가 축. |
| **Activation** | Sigmoid → tanh → ReLU → GELU/Swish | vanishing → zero-center → 살아있는 gradient → 부드러움. |
| **Optimizer** | SGD → Momentum → AdaGrad → RMSProp → Adam → AdamW | 조부(가속) + 적응(per-param LR) + 분리(weight decay). |
| **Normalization** | BN → LN → IN → GN | 어느 축으로 평균·분산을 잡는가의 차이. |
| **CNN 아키텍처** | LeNet → AlexNet → VGG → Inception → ResNet → DenseNet → EfficientNet | 깊이 → 연산 → gradient → 재사용 → 균형 scaling. |
| **Sequence model** | RNN → LSTM/GRU → bi-RNN → seq2seq → Attention → Transformer | vanishing → context bottleneck → 병렬성. |

## 한눈에 — 관통 4개

같은 문제를 여러 무기로 동시에 푼다. 한 무기만 알면 다른 무기가 *덤덤한 대체*로 보이지만, 모아 보면 *상보적 portfolio*임이 드러난다.

| 관통 주제 | 같은 표적 | 다른 무기들 |
|---|---|---|
| **Vanishing gradient** | gradient가 0으로 죽지 않게 | 초기화(He/Xavier) · BN · ReLU · ResNet skip · LSTM gate |
| **Inductive bias 매칭** | 데이터 구조와 아키텍처의 짝짓기 | CNN(공간 locality) · RNN/Transformer(순서) · MLP/DeepSet(no structure) |
| **Bias-Variance** | underfit ↔ overfit의 다이얼 | depth · width · regularization · 데이터량 · ensembling 모두 BV로 환원 |
| **Regularization** | overfit 막기 | L1/L2(파라미터) · Dropout(구조) · BN(신호) · Aug(데이터) · EarlyStop(학습) · LabelSmooth(라벨) |

## 한눈에 — 워크플로우 사슬

```
EDA → 전처리·feature → architecture 선택 → 학습 setup → 진단·반복 → 평가
 ↓        ↓                 ↓                  ↓             ↓            ↓
분포·결측 결측처리·scale    inductive bias    loss·optim·    train/val    metric
이상치   정규화·encoding   data와 매칭      LR·batch       gap → 처방   ↔ 도메인
```

각 단계의 결정은 *다음 단계의 선택지를 제약*한다. 예:
- 결측 imputation 방식 → BN과의 상호작용 (imputed flag 정보가 신호로 갈 수 있나?)
- 데이터가 작다 → architecture를 가볍게 + augmentation 강하게 → batch도 작아짐 → BN 대신 GN
- 클래스 imbalance → loss 선택(Focal/Weighted CE) + sampling 전략 + metric(Acc → F1/AUC)

자세한 결정 흐름은 [03_workflow.md](03_workflow.md).

## 사용법

- **시험 직전**: 이 hub 페이지만 외워도 "왜·비교" 질문의 80%가 풀린다.
- **개념이 헷갈릴 때**: 해당 계보 페이지로 가서 *위·아래 이웃*을 함께 본다 (한 칸 차이가 의미 차이).
- **새 task 설계 시**: workflow 사슬로 결정 순서를 확인하고, 각 단계의 옵션을 계보 표에서 짚는다.
