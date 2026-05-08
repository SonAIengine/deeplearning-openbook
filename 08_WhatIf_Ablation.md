# 08. What-If / Ablation — "이거 빼면 어떻게 되나"

> 모든 컴포넌트의 **역할은 그것을 제거했을 때 드러난다**.
> "왜 BN이 필요?"라는 질문엔 **"BN 없으면 어떻게 망가지는지"**로 답한다.

---

## 0. 사고법 — "ablation으로 이해"

각 항목은 같은 4단계로 답한다:
1. **어떤 컴포넌트?**
2. **빼면 무엇이 달라지나?** (구체적 현상)
3. **왜 그렇게 되나?** (메커니즘)
4. **그러면 어떻게 보완 가능?**

---

## 1. 비선형 활성화 함수를 빼면? (★★ 단골)

### 빼면
모든 layer가 선형 → 깊이가 의미 없음.

### 왜?
$f(x) = W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2) = W' x + b'$
→ **선형 모델 1개**와 등가.

### 의미
- 100층이든 1000층이든 **표현력은 logistic regression 수준**.
- XOR 같은 단순 비선형 문제도 풀지 못함.

### 보완
- 비선형 자체가 핵심이라 **보완 불가**. 무조건 넣어야.

---

## 2. ReLU 대신 Sigmoid를 깊은 망에서 쓰면?

### 빼면 (= Sigmoid로 대체)
- 학습이 매우 느려지거나 안 됨.
- 깊이 10층 이상에서 사실상 학습 불가.

### 왜?
- σ'(x) ≤ 0.25.
- backprop에서 gradient를 매 층마다 곱 → 0.25^L → 0 (**vanishing**).
- 출력이 0~1, 0-centered 아님 → 다음 층 입력에 bias.

### 보완
- ReLU/LeakyReLU/GELU로 교체.
- 또는 BN + Skip connection 강화 (그래도 한계).
- LSTM의 gate처럼 sigmoid가 의미 있는 자리에만 한정.

---

## 3. Batch Normalization을 빼면?

### 빼면
- 학습이 매우 느려지고 LR을 크게 못 씀.
- 깊은 망(50+층)에서 발산 또는 학습 정체 빈번.
- 초기화에 매우 민감.

### 왜?
- 층마다 입력 분포가 학습 진행에 따라 흔들림 (internal covariate shift).
- Loss landscape가 거칠어 큰 LR이 위험.

### 보완
- **LayerNorm** (Transformer/RNN).
- **GroupNorm / InstanceNorm** (작은 batch CNN).
- **Self-normalizing**: SELU + LeCun init.
- **WeightStandardization** + GN.
- 가장 안전: ResNet style + GN/LN.

생각해보라: BN을 빼고 LR을 줄이면 학습은 가능. 단 매우 느림. **현대 표준에서 BN/LN은 거의 필수.**

---

## 4. Pooling을 빼면? (CNN에서)

### 빼면
- 공간 차원이 줄지 않음 → 메모리·연산 부담↑.
- Receptive field 키우기 어려움.
- Translation invariance 약화.

### 왜?
- Pooling이 down-sampling 역할.
- Max pooling이 두드러진 특징을 골라 약한 invariance 부여.

### 보완
- **Stride conv**로 down-sampling (현대 트렌드 — ResNet의 strided conv).
- **Dilated conv**로 RF 확장 (segmentation에 유용).
- 마지막에 **Global Average Pooling**으로 공간 → 1.

생각해보라: 최근에는 max pooling보다 strided conv가 표준. 왜? → 학습된 down-sampling이 hand-craft보다 유연.

---

## 5. Dropout을 빼면? (over-parameterized 시대에)

### 빼면
- 작은 데이터에서 overfit 위험↑.
- 큰 데이터·강한 augmentation에서는 의외로 차이 작음.

### 왜?
- Dropout = 암묵적 ensemble.
- 데이터가 많으면 ensemble의 효과가 줄어듦 (이미 다양한 데이터).

### 보완
- **Data augmentation** (Mixup, CutMix, RandAugment).
- **Weight decay** (L2).
- **Label smoothing**.
- **Stochastic Depth** (residual 일부를 학습 중 끔).

생각해보라: 큰 Transformer에서는 dropout 비율 낮거나 0 (대신 데이터·augmentation에 의존). 이 trade-off를 이해해야.

---

## 6. Skip Connection (ResNet)을 빼면?

### 빼면
- 30+층에서 **degradation**: train accuracy도 떨어짐.
- Vanishing gradient로 학습 멈춤.
- 100+층은 사실상 학습 불가.

### 왜?
- Plain deep net은 항등 함수도 학습 어려움.
- gradient가 곱해져 소멸.

### 보완
- **Highway Network** (게이트 형태의 skip).
- **DenseNet** (모든 이전 층과 연결).
- **Skip connection 자체를 빼고 깊게 가는 건 비현실적**.

생각해보라: ResNet 이전(VGG)은 19층이 한계. ResNet 이후 152, 1000층까지. **skip은 깊은 학습의 본질**.

---

## 7. LSTM의 각 gate를 빼면? (★ 단골)

### 7.1 Forget gate 없애면 (f=1 고정)
- 모든 과거를 누적 → cell state가 폭발하거나 노이즈로 가득.
- 새 정보 추가만 가능, 잊기 불가.
- 긴 시퀀스에서 정보 포화.

### 7.2 Input gate 없애면 (i=1 고정)
- 모든 candidate를 항상 cell에 추가.
- 노이즈에 매우 약함.
- "선택적 기록"이 안 됨.

### 7.3 Output gate 없애면 (o=1 고정)
- cell state 전체가 항상 외부로 노출.
- task-무관한 정보까지 출력으로 흘러 학습 신호 흐려짐.
- GRU에는 output gate가 없음 (실제로 큰 손실 아님).

### 7.4 Forget + Input 합치기 (i = 1 - f)
- 파라미터 절약, 표현력 약간 감소.
- 실제로 GRU가 이 형태 → 거의 동급 성능.

### 핵심
- **forget가 가장 중요**. 없으면 LSTM이 망가짐.
- input과 output은 합치거나 제거해도 어느 정도 작동 (GRU가 증명).

---

## 8. RNN에서 가중치 sharing(시간 축) 없으면?

### 빼면
- 각 시점마다 다른 가중치 → 시퀀스 길이만큼 파라미터 폭발.
- 가변 길이 처리 불가.
- **사실 RNN이 아니라 깊은 FNN**.

### 핵심
- 시간 축 sharing이 RNN의 정의이자 본질.
- 빼면 RNN의 모든 장점이 사라진다.

---

## 9. CNN의 weight sharing(공간 축) 없으면?

### 빼면 (= Locally Connected Network)
- 각 위치마다 다른 커널.
- 파라미터 폭증.
- **Translation equivariance 상실** — 이미지가 살짝 움직이면 결과 변함.
- 데이터 효율 급락.

### 단 한 가지 장점
- 위치별 특화된 패턴 (얼굴 인식의 specific 위치) 가능.
- 일부 face recognition에서 사용된 적 있음 (현대엔 거의 안 씀).

### 결론
- "왜 sharing 하겠냐?"의 답 = **이거 빼면 모든 장점이 사라진다**.

---

## 10. Attention만, RNN 없이? (Transformer의 동기)

### Transformer는 RNN 없이 attention만으로 시퀀스 처리.

### 빼면 (= attention 없이 RNN만)
- Long-range 약화 (LSTM의 한계 그대로).
- 병렬화 못함.

### 추가하면 (= RNN 없이 attention만)
- Transformer.
- 모든 시점을 동시 처리 → 병렬 + long-range.
- 단 위치 정보가 자동으로 안 들어가므로 **PE 필요**.

생각해보라: PE 없이 Transformer를 시퀀스에 쓰면? → bag-of-tokens가 됨. 순서 무관. NLP에선 망함.

---

## 11. Positional Encoding을 빼면? (Transformer)

### 빼면
- 순서 정보 완전 소실.
- "He hit me" = "Me hit he" (구분 불가).
- 시퀀스 task에서 큰 성능 저하.

### 보완
- 학습 가능 PE (BERT 식).
- Sinusoidal PE (원조 Transformer).
- Relative PE (T5, RoPE).

### 의미
- Transformer의 **거의 유일한 inductive bias**가 PE.
- PE 설계가 long-range generalization에 큰 영향.

---

## 12. Optimizer의 Momentum 빼면? (SGD)

### 빼면
- 골짜기에서 진동.
- 평탄 영역에서 느림.

### 왜?
- 매 step gradient만 보고 움직임 → 노이즈에 흔들림.
- 같은 방향을 누적하면 가속, 반대 방향이면 상쇄.

### 보완
- Adam의 m이 사실 momentum.
- Nesterov로 더 정확한 momentum.

---

## 13. Weight Decay 빼면?

### 빼면
- 가중치가 발산할 수 있음 (특히 BN 없을 때).
- 일반화 약화.

### 왜?
- 명시적 정규화 항이 없으면 모델이 train data에 과적합.

### 보완
- Dropout, augmentation으로 일부 보완.
- 큰 모델에선 weight decay가 가장 안정적이고 효과적.

---

## 14. Mini-batch 대신 full-batch (전체 데이터로 한 번에)?

### 빼면 (= full batch GD)
- gradient noise 사라짐 → **일반화 약화**.
- Sharp minima에 빠짐.
- 메모리·연산 부담.

### 왜?
- SGD의 noise가 flat minima로 이끔 → 더 좋은 일반화.
- 또 mini-batch가 GPU 메모리 + 병렬에 최적.

### 보완
- 큰 batch + LR scaling (linear scaling rule).
- LARS, LAMB optimizer (큰 batch 전용).

생각해보라: 왜 큰 batch가 일반화에 안 좋은가는 여전히 활발한 연구 주제. 적절한 batch가 noise와 안정성의 sweet spot.

---

## 15. Train/Validation 분리 안 하면?

### 빼면
- Hyperparameter 선택을 어디서? → test에서 함 → **간접 overfit**.
- 모델 선택이 잘못되어 실세계 성능 추정 부정확.

### 의미
- "Test set은 한 번만 본다"는 원칙의 이유.
- val 없으면 신뢰할 수 있는 모델 선택 불가.

---

## 16. 한 표로 정리 — "이거 빼면"

| 컴포넌트 | 빼면 발생하는 일 | 핵심 메커니즘 |
|---|---|---|
| 비선형 활성화 | 선형 모델로 환원 | 함수 합성이 합쳐짐 |
| ReLU (→sigmoid) | Vanishing | 미분 작음 |
| BN | LR 작게, 학습 느림 | 분포 흔들림 |
| Pooling | 메모리·RF 부담 | down-sample 부재 |
| Dropout | 작은 데이터에서 overfit | ensemble 효과 사라짐 |
| Skip connection | 30+층 학습 불가 | gradient 곱 |
| Forget gate | 정보 포화 | 망각 불가 |
| 시간 sharing | RNN 본질 상실 | 파라미터 폭발 |
| 공간 sharing | translation equiv 상실 | 패턴이 위치 종속 |
| Positional Encoding | 순서 소실 | bag-of-tokens |
| Momentum | 진동, 느림 | gradient noise |
| Weight decay | 가중치 발산 | 명시적 정규화 부재 |
| Mini-batch | sharp minima | gradient noise 부재 |
| Val split | 간접 overfit | 평가 정직성 부재 |

---

## 17. "생각해보라" 확장

1. **BN과 Dropout을 같이 쓸 때 자주 충돌하는 이유?** → 분산 통계가 dropout의 변동에 영향. 보통 BN 먼저, dropout 뒤(또는 둘 중 하나).
2. **GRU가 LSTM에서 output gate를 뺀 것과 같은 정신인데, 왜 큰 성능 차이 없나?** → output gate의 역할이 다른 곳(다음 층 처리)에서 보상됨.
3. **ReLU도 gradient 0인 영역(x<0)이 있는데 왜 vanishing이 덜 심한가?** → 활성된 뉴런의 절반(평균)에서 gradient=1. sigmoid는 모든 곳에서 < 0.25.
4. **Skip connection을 RNN에 적용할 수 있나?** → 가능. Residual RNN, highway RNN 등이 있고 깊은 RNN 학습에 도움.
5. **Inception의 1×1 conv를 빼면?** → 채널 차원 축소가 없어져 파라미터·연산 폭증.

---

## 18. 한 줄 요약

- **각 컴포넌트의 가치 = 그것을 제거했을 때의 손해**.
- "왜 X가 필요한가"의 가장 강력한 답은 "**X 없으면 이렇게 망가진다**".
- 교수님 단골: "이거 빼면 어떻게 되나?" → 이 챕터의 표를 그대로 사용.
