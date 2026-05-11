# 02. 관통 — 한 문제를 여러 무기로

계보는 *"시간 축"*이었다. 관통은 *"같은 표적에 어떤 무기들이 모이는가"*다. 무기 하나만 외우면 "이게 또 이거랑 비슷하네"로 끝나지만, 모아 보면 *각 무기가 다른 면을 공격*하는 portfolio임이 보인다.

---

## 1. Vanishing Gradient — 한 문제, 다섯 관점

**문제**: 깊은 네트워크에서 backprop이 진행될수록 gradient가 곱셈 누적으로 작아져 초기 layer가 학습되지 않는다.

다섯 무기가 *서로 다른 지점*에서 이 문제를 친다:

| 관점 | 무기 | 어디를 치는가 | 한계 |
|---|---|---|---|
| **초기 신호 자체** | Xavier / He init | 입력·출력 분산이 같아지도록 가중치 초기 분포를 설계 — *학습 시작점*을 vanishing 안 일어나는 자리에 놓음 | 학습 도중 분포가 흔들리면 무력 |
| **활성 함수의 미분** | ReLU | 양수 영역 gradient = 1로 *곱셈 누적이 무해* | 음수 영역에서 dead — LeakyReLU·GELU 등으로 보완 |
| **layer별 신호 분포** | BN (또는 LN/GN) | 매 layer마다 평균·분산을 재정렬 — *분포 drift*를 직접 막음 | small batch에서 통계 불안정 |
| **gradient의 우회로** | ResNet skip connection $y = F(x) + x$ | gradient가 $+x$ 경로로 *곱셈을 통째로 우회* — 깊이가 곱셈을 늘려도 합산 경로는 그대로 | residual 안의 학습 정체 가능성 |
| **시퀀스의 시간 축** | LSTM/GRU gate | 시간 방향 곱셈 누적을 *cell state의 덧셈*으로 우회 — 매 step gradient가 보존 | gate 학습 자체의 부담, 파라미터 4배 |

**통찰**: 같은 적("gradient 소실")에 *공간 축*(BN, ResNet, init)과 *시간 축*(LSTM)이 각각 대응. 한 가지만 쓰는 게 아니라 **함께** 쓴다 — 가령 깊은 CNN은 He init + ReLU + BN + skip을 *전부* 사용한다.

> 시험에서 "vanishing 어떻게 해결?"이 나오면, 한 무기만 적지 말고 *"신호 시작 / 활성 / 분포 / gradient 경로 / 시간 축"* 다섯 관점을 모두 짚어주는 것이 안전한 답.

---

## 2. Inductive Bias 매칭 — 데이터 구조 ↔ 아키텍처

**문제**: 모든 데이터에 같은 모델을 쓰면 *데이터 구조의 사전 정보*를 버리는 셈. 어느 구조에 어떤 inductive bias를 매칭할까.

| 데이터 구조 | 가정 | 매칭되는 아키텍처 | bias의 정체 |
|---|---|---|---|
| **공간 locality + translation invariance** | 인접 픽셀이 의미 단위, 위치가 변해도 같은 객체 | CNN | 같은 weight를 sliding (공간 sharing), pooling으로 위치 약불변 |
| **순서가 의미** | 앞 token이 뒤 token에 영향 | RNN / Transformer | RNN: 순차 hidden state. Transformer: positional encoding + self-attention |
| **권력 관계가 없는 set** | 순서 무관, 원소들 간 관계만 | Deep Sets / Graph NN | sum/mean aggregation (순서 불변) |
| **그래프 구조** | 노드 간 edge로 영향 전파 | GCN / GAT | message passing (이웃 노드와의 교환) |
| **구조가 없음 (tabular)** | feature가 i.i.d.한 의미 | MLP / GBDT | bias 없음 — 데이터로 모든 걸 배우거나, tree로 split |

**비대칭 매칭의 함정** — 예시:
- 이미지를 RNN으로? → locality bias 버림. 가능하지만 비효율.
- 텍스트를 CNN으로? → 가능 (TextCNN). 하지만 long-range 의존 못 잡음.
- 작은 데이터에 ViT? → CNN의 locality bias 없이 attention만으로 배워야 하는데 데이터 부족 → 성능 ↓.

**통찰**: *"데이터의 구조 = 아키텍처의 bias"*가 잘 맞을수록 같은 데이터로 더 빨리, 더 잘 배운다. 데이터가 압도적으로 많으면 bias가 약해도 다 배울 수 있지만 (ViT가 ImageNet-21k 이상에서 CNN 추월), *데이터가 작을수록 bias 매칭이 결정적*. 자세한 매트릭스는 본문 §7.

---

## 3. Bias-Variance — 모든 결정의 중심 다이얼

거의 모든 하이퍼파라미터·아키텍처 선택은 결국 *bias↔variance* 슬라이더 위에 있다. 한 다이얼을 돌리면 다른 다이얼이 반대로 움직인다.

```
        ┌────────────────────────────────┐
underfit│  bias 높음                       │overfit
        │                                  │
        │  ◀────  variance ────▶          │
        │  낮음 (모델 단순)   높음 (모델 복잡) │
        └────────────────────────────────┘
```

| 결정 다이얼 | 어느 쪽으로? | 효과 |
|---|---|---|
| **깊이·너비 ↑** | variance ↑, bias ↓ | 표현력 확보 |
| **regularization (L1/L2, dropout) ↑** | variance ↓, bias 살짝 ↑ | overfit 억제 |
| **데이터량 ↑** | variance ↓, bias 영향 미미 | 가장 안전한 방향 |
| **augmentation ↑** | variance ↓ | 데이터를 합성으로 늘림 |
| **early stopping** | variance ↓ | 모델 capacity는 그대로, 학습 step만 제한 |
| **앙상블** | variance ↓ | 여러 모델 평균 — 가장 깔끔한 variance 감소 |
| **batch size ↑** | gradient noise ↓ → 일반화 약간 떨어질 수 있음 | 대형 batch는 regularizer 추가 필요 |
| **learning rate ↑** | gradient noise ↑ → 약한 regularization 효과 | 너무 크면 발산 |

**중요한 비대칭**: bias를 줄이는 방향은 표현력을 *키워*야 한다 (모델 ↑, feature engineering). variance를 줄이는 방향은 *옵션이 많다* (데이터, regularization, ensembling). 그래서 *underfit 진단이 더 어렵고, overfit 진단·처방이 더 다양*하다.

**통찰**: 면접에서 "X를 늘리면?"이 나오면 자동으로 *"bias가 어떻게 / variance가 어떻게 / 같이 어떻게 따라오는가"*를 답할 수 있어야 한다. 예: dropout↑ → variance↓ (regularization) + bias↑ (활성 일부 죽임) → train acc 약간 ↓, val acc ↑.

---

## 4. Regularization — 같은 적, 여섯 면

**문제**: overfit (train↓ val↑) 막기. 같은 적이지만 *공격할 수 있는 면이 6개*다.

| 면 | 무기 | 어느 부분을 친다 | 직관 |
|---|---|---|---|
| **파라미터 크기** | L1 / L2 weight decay | weight magnitude에 페널티 | "큰 weight = 한 feature에 과의존" → 작게 유지 |
| **구조의 redundancy** | Dropout | 매 step random subset만 사용 | "모델 안의 ensemble" — 어느 뉴런도 필수가 아니게 |
| **신호 분포** | BN (+ ensemble effect from batch) | sample들의 통계로 정규화 | 다른 sample이 잡음 역할 → variance 감소 |
| **데이터 다양성** | Data Augmentation | 같은 sample을 변형 (rotate, crop, mixup) | "데이터를 늘리는 대신 변형으로" |
| **학습 시간** | Early Stopping | val loss 증가 시점에서 정지 | 모델 capacity는 그대로 두고 *학습 step만* 줄임 |
| **라벨의 자신감** | Label Smoothing | one-hot 1.0 → 0.9, 0 → 0.025 | "정답을 너무 확신하지 마라" — calibration + 약한 regularization |

**조합의 효과 — 비직관적 사례**:
- BN + Dropout 같이 쓰면 충돌 가능 (BN의 train/eval 통계와 dropout 마스크 상호작용). 보통은 BN만 쓰거나 dropout을 BN 뒤에.
- L2 + Adam은 Adam에서 weight decay가 *adaptive LR에 의해 왜곡*됨 → AdamW로 분리하는 게 정석.
- Augmentation + Label Smoothing은 *과한 부드러움*을 만들 수 있어 둘 다 강하게 쓰면 underfit.

**통찰**: regularization은 *"한 종류로 끝내는 게 아니라 portfolio로"*. 보통은 weight decay (L2 or AdamW) + dropout(작게) + augmentation(주력) + early stopping(안전망)을 *동시에* 가볍게 켠다. 어느 하나를 매우 강하게 쓰는 것보다 *여러 면을 살짝씩*이 안정적.

---

## 한 줄 메타 통찰

> *"하나의 적(vanishing, overfit, bias-variance)을 풀 때 무기가 하나 있다고 만족하지 마라. 다른 면에서 같은 적을 치는 두 번째·세 번째 무기를 함께 알아야 — 실전에서 한 무기가 무력한 상황을 만난다."*
