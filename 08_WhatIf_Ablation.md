# 08. What-If / Ablation — "이거 빼면 어떻게 되나" 심층

> **이 문서의 핵심 사고**:
> 모든 컴포넌트의 가치는 그것을 *제거했을 때 드러난다*. "왜 BN이 필요?" 는 "BN이 무엇을 하나"로 답하기보다 "**BN 없으면 어떻게 망가지는지**"로 답하는 게 더 명료하다.
>
> 이 챕터는 24개의 컴포넌트를 다음 4단계 패턴으로 분석:
> 1. 제거의 정의
> 2. 무엇이 망가지나 (구체적 현상)
> 3. 왜 그렇게 되나 (수식·메커니즘)
> 4. 보완 가능성과 한계

---

## 0. 사고법 — Ablation으로의 이해

### 0.1 왜 ablation 사고?

ML의 모든 기법은 *어떤 문제의 답*으로 등장했다. 그 문제가 무엇인지를 가장 명확히 보여주는 방법은 *그 기법을 빼는 것*. 그러면 원래 문제가 다시 드러난다.

예: "BN이 무엇인가" → "각 layer의 활성값을 정규화" — 정의. 이걸로는 *왜 필요*한지 모름.
"BN을 빼면" → "큰 LR로 학습 발산", "초기화에 민감", "깊은 망 수렴 안 함" — 명확.

이 사고가 면접·시험·실무에서 모두 강력. *컴포넌트의 가치 = 그것의 부재가 만드는 문제*.

### 0.2 4단계 분석 패턴

각 컴포넌트를 다음으로:

1. **정의·제거**: 그 컴포넌트가 무엇이고, "제거"가 뭐를 의미하나.
2. **현상**: 빼면 무엇이 일어나나 — 구체적 학습 동역학·성능.
3. **메커니즘**: 왜 그렇게 되나 — 수식·이론.
4. **보완**: 다른 기법으로 우회 가능한가, 한계는?

이 패턴이 외워지면 어떤 새 ablation question에도 체계적 답.

### 0.3 다루는 24개 컴포넌트

기본 (1~6):
1. 비선형 활성화
2. ReLU (→ Sigmoid)
3. Batch Normalization
4. Pooling
5. Dropout
6. Skip Connection (ResNet)

LSTM/RNN (7~10):
7. Forget Gate
8. Input Gate
9. Output Gate
10. 시간 축 Sharing

CNN (11~13):
11. 공간 축 Sharing
12. 1×1 Conv
13. Pretraining

Transformer (14~16):
14. Attention
15. Positional Encoding
16. Skip + LN in Transformer

Optimizer/학습 (17~21):
17. Momentum
18. Weight Decay
19. Mini-batch
20. Learning Rate Schedule
21. He Init (→ random uniform)

평가/실무 (22~24):
22. Validation Set
23. Train/Eval Mode
24. Gradient Clipping

---

## 1. 비선형 활성화 함수를 빼면 (★ 단골)

### 1.1 정의·제거

비선형 활성화 ($\sigma$, ReLU 등) 제거 = identity 함수로 대체. 즉:
$$h^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$$

(no $\sigma$).

### 1.2 무엇이 망가지나

**모든 layer가 합쳐져 단일 선형 변환**:

$$f(x) = W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2) = W' x + b'$$

100층 1000층 쌓아도 logistic regression 수준의 표현력. **XOR도 못 풀음**.

### 1.3 왜?

행렬 곱이 *associative*. 두 선형 변환의 합성은 다시 선형. 깊이의 의미 사라짐.

UAT의 본질: 비선형이 있어야 임의 함수 근사 가능. 비선형 빼면 표현력이 *연결되지 않은 차원만큼*으로 제한.

### 1.4 보완

**보완 불가**. 비선형 자체가 신경망의 본질. 무조건 넣어야.

단 비선형의 *종류*는 선택 가능 (ReLU, GELU, Tanh 등).

---

## 2. ReLU 대신 Sigmoid를 깊은 망에서

### 2.1 제거의 의미

Hidden layer 활성화를 ReLU에서 sigmoid로 바꿈.

### 2.2 무엇이 망가지나

- **학습이 매우 느려짐**. 깊이 10층 이상에서 사실상 불가.
- 첫 epoch에서 train loss가 거의 안 떨어짐.
- Gradient norm이 깊은 layer에서 매우 작음.

### 2.3 왜? — Vanishing Gradient

Sigmoid의 미분 $\sigma'(x) = \sigma(x)(1 - \sigma(x))$의 최대값 0.25 (x=0).

Backprop의 chain rule:
$$\delta^{(l)} \propto \prod_{k=l+1}^{L} W^{(k)} \cdot \sigma'(z^{(k-1)})$$

10층에서 $0.25^{10} \approx 10^{-6}$. 100층에서 $10^{-60}$. 사실상 0.

ReLU는 활성된 영역에서 미분 1. 깊이가 깊어도 곱이 1 근처라 vanishing 약함.

### 2.4 보완

- **ReLU 계열 활성화**: 표준 답.
- **BN**: 활성값 분포 안정 → vanishing 약화.
- **Skip connection (ResNet)**: gradient 직접 흐름.
- **LayerNorm**: 비슷.

이 셋의 조합으로 sigmoid도 깊은 망 학습 *어느 정도* 가능. 단 ReLU가 훨씬 안전.

생각해보라: LSTM의 gate에는 왜 sigmoid? 답: gate는 0~1 비율을 표현해야 함. Saturation이 *기능*이지 버그가 아님. 그리고 cell state의 덧셈 구조가 vanishing 우회.

---

## 3. Batch Normalization을 빼면 (★)

### 3.1 제거의 의미

각 layer 후의 BN을 제거. Conv → ReLU 식으로 바로.

### 3.2 무엇이 망가지나

- **큰 LR을 못 씀**. LR 0.01도 자주 발산.
- **초기화에 매우 민감**. 잘못된 init으로 학습 발산.
- **깊은 망 (50+) 학습 매우 어려움**.
- 학습 속도 매우 느림 (작은 LR + 많은 epoch).

### 3.3 왜?

(1) **Internal covariate shift**: 각 layer 입력 분포가 학습 도중 변동. 다음 layer가 매번 새 분포에 적응해야 함.

(2) **Loss landscape가 거침**: BN은 landscape을 평활화. BN 없으면 landscape이 매우 sharp + curvy → 큰 LR 위험.

(3) **활성값 분포 폭주/소멸**: 매 layer에서 활성값 분산이 폭발 또는 0으로. 학습 불안정.

### 3.4 보완

- **GroupNorm/LayerNorm**: BN 대체. 작은 batch나 RNN에 더 적합.
- **Self-normalizing**: SELU + LeCun init. BN 없이 자동 정규화 (실용적으론 BN이 더 안정).
- **WeightStandardization** + GN.
- **적절한 init** (He/Xavier).
- **작은 LR + 많은 epoch + warmup**: 시간 부담 多.

가장 안전: ResNet style + LN/GN. 현대 deep learning에서 BN/LN/GN 등 정규화는 거의 필수.

---

## 4. Pooling을 빼면 (CNN에서)

### 4.1 제거의 의미

Conv layer 사이의 max/avg pool 제거. 또는 stride conv도 함께 (down-sampling 제거).

### 4.2 무엇이 망가지나

- **공간 차원 안 줄어듦** → 메모리 폭발.
- **Receptive field 키우기 어려움** (깊이만으로).
- **Translation invariance 약화** (강한 rough invariance 부재).

### 4.3 왜?

Pooling이 *공간 차원 축소* + *invariance 부여*의 두 역할. 빼면 둘 다 잃음.

메모리: 224×224 입력이 224×224 그대로 깊이 따라 → 메모리 100배 폭발.

RF: 깊이만으론 RF 매우 천천히 증가. Pooling이 RF를 한 번에 두 배.

### 4.4 보완

- **Stride conv**: 학습 가능 down-sampling. 현대 표준 (ResNet 식).
- **Dilated conv**: RF를 파라미터 그대로 키움. Segmentation에 유용.
- **GAP**: 마지막에 공간 → 1로 압축.

현대 CNN은 max pool 거의 안 씀. 첫 layer 후 한 번 정도. 마지막엔 GAP.

생각해보라: 왜 stride conv가 max pool 대체했나? 학습된 down-sampling이 hand-crafted (max)보다 유연. 단 max pool은 단순함의 장점이 있어 edge device에서 여전히 사용.

---

## 5. Dropout을 빼면 (Over-parameterized 시대)

### 5.1 제거의 의미

학습 중 dropout layer 제거. 또는 $p=0$.

### 5.2 무엇이 망가지나 (case별)

**작은 데이터**: 강한 overfit. Train loss는 0에 가까운데 val loss 폭발.

**큰 데이터·강한 augmentation**: 의외로 차이 적음. 대신 약간 느린 수렴.

**큰 Transformer (modern)**: 거의 같음 (이미 dropout=0.1 또는 0).

### 5.3 왜?

Dropout = 암묵적 ensemble. 각 step 다른 sub-network → variance 감소.

데이터 충분하면 데이터 자체가 정규화. Augmentation도 비슷한 효과 (서로 다른 sample이 사실상 다양한 sub-network의 학습을 시뮬).

큰 모델 + 작은 데이터에서는 dropout이 *결정적*. 큰 데이터에선 보조.

### 5.4 보완

- **Data augmentation**: Mixup, CutMix, RandAugment.
- **Weight decay**: 가중치 부드러움.
- **Label smoothing**: 라벨 부드러움.
- **Stochastic depth**: layer 자체 skip.
- **데이터 늘림**: 가장 좋음.

현대 큰 모델은 dropout 거의 안 씀 — 위 다른 정규화로 충분.

---

## 6. Skip Connection (ResNet)을 빼면 (★)

### 6.1 제거의 의미

ResNet block에서 + identity 빼고 plain stack:
```
이전: h_{l+1} = h_l + F(h_l)
이후: h_{l+1} = F(h_l)
```

### 6.2 무엇이 망가지나

- **30+층에서 train accuracy도 떨어짐** (degradation problem).
- **Vanishing gradient 강하게**.
- **100+층은 학습 사실상 불가**.

### 6.3 왜?

(1) **Identity가 default 아님**: $F$가 0을 학습해도 항등 안 됨. 깊이 추가가 *최소한* 더 나빠지지 않을 보장 사라짐.

(2) **Gradient flow**: 미분 $\partial h_{l+1}/\partial h_l = \partial F/\partial h_l$ — 곱셈 누적 (덧셈 항 없음). Vanishing 강하게.

(3) **Optimization landscape**: Skip 없으면 landscape이 매우 sharp. 학습 어려움.

### 6.4 보완

- **Highway Network**: gate 형태 skip.
- **DenseNet**: 모든 이전 layer와 concat (sum 대신).
- **Skip 없이 깊게는 비현실적**.

ResNet 이전 (VGG): 19층이 한계. 이후: 152, 1000+층 가능. *Skip 없이 매우 깊은 학습 거의 불가능*.

---

## 7. LSTM의 Forget Gate 없애면 (★)

### 7.1 제거의 의미

$f_t = 1$ 고정. 즉 $c_t = c_{t-1} + i_t \odot \tilde{c}_t$.

### 7.2 무엇이 망가지나

- **Cell state 폭발 또는 noise 누적**.
- **컨텍스트 전환 못함**: 새 sentence/주제에서 이전 정보 못 잊음.
- **긴 시퀀스에서 cell이 무한 누적**.

### 7.3 왜?

Forget gate가 *선택적 망각*. 없으면 모든 과거가 영원히 누적. 정보 포화.

수식적으로 cell state norm이 단조 증가 (input gate가 0이 아니면) → 발산.

### 7.4 보완

- **Cell state 노름 정규화**: layer norm 적용.
- **Bounded cell**: tanh 등으로 제한.
- 단 forget gate 자체가 가장 자연스러운 답. **LSTM에 forget이 가장 중요한 gate**.

---

## 8. Input Gate 없애면

### 8.1 제거의 의미

$i_t = 1$ 고정. $c_t = f_t \odot c_{t-1} + \tilde{c}_t$.

### 8.2 무엇이 망가지나

- **모든 candidate가 항상 cell에 추가**. 노이즈에 약함.
- **선택적 기록 불가**. "the", "a" 같은 무관 토큰도 강하게 영향.
- **중요 정보 희석**.

### 8.3 왜?

Input gate가 *새 정보의 비율* 결정. 없으면 모든 시점의 모든 candidate를 같은 가중치로 흡수.

### 8.4 보완

- Forget gate와 묶어 GRU 식 ($i = 1 - f$). 표현력 약간 손해, 단순.
- Forget gate 강화로 약간 보완.

GRU가 이 자리에 있음 — input·forget을 update gate 하나로.

---

## 9. Output Gate 없애면

### 9.1 제거의 의미

$o_t = 1$ 고정. $h_t = \tanh(c_t)$.

### 9.2 무엇이 망가지나

상대적으로 **덜 치명적**. GRU가 사실상 output gate 없는 형태로 잘 작동.

단점:
- Cell의 모든 정보가 외부 노출.
- Task-무관 정보까지 출력.
- 다음 시점 gate 계산이 task-relevant·무관 정보 모두 받음.

### 9.3 왜 덜 치명적?

다음 layer 또는 출력 head가 *어차피* 필요한 정보를 골라낼 수 있음. Output gate가 *미리 골라내는* 것이라, 안 해도 큰 손해 없음.

GRU가 이 자리에 있는 증거 — output gate 없이도 LSTM과 거의 동급.

### 9.4 보완

- 다음 layer (FC, Transformer) 가 정보 선택.
- Pruning post-hoc.

---

## 10. RNN의 시간 축 Sharing 없으면

### 10.1 제거의 의미

각 시점마다 *다른 가중치*:
$$h_t = \tanh(W^{(t)}_{xh} x_t + W^{(t)}_{hh} h_{t-1})$$

### 10.2 무엇이 망가지나

- **파라미터 폭발**: 가중치 수가 시퀀스 길이 $T$에 비례. 길이 100, hidden 100이면 100배 파라미터.
- **가변 길이 처리 불가**: 학습 시 본 길이만 처리 가능.
- **데이터 효율 매우 낮음**: 같은 패턴이 다른 시점에 있어도 별도 학습.

### 10.3 왜?

시간 축 sharing이 RNN의 *정의이자 본질*. 빼면 *RNN이 아니라 깊이 T의 깊은 FNN*. 모든 RNN 장점 사라짐.

### 10.4 보완

- 그냥 RNN 쓰면 됨. Sharing이 핵심.
- *부분 sharing*도 가능 — 처음 몇 시점은 다른 가중치, 후속은 sharing. 단 표준 아님.

---

## 11. CNN의 공간 축 Sharing 없으면

### 11.1 제거의 의미

위치마다 다른 가중치 — Locally Connected Network (LCN).

```
일반 CNN: 모든 위치가 같은 커널 W
LCN: 위치 (i, j)마다 다른 커널 W_{i,j}
```

### 11.2 무엇이 망가지나

- **파라미터 폭증**: 위치 수에 비례. 224×224 conv면 50,000배 더.
- **Translation equivariance 상실**: 같은 패턴이 다른 위치에 있으면 별도 학습.
- **데이터 효율 급락**: 한 위치 학습이 다른 위치에 영향 없음.

### 11.3 왜? — "왜 sharing 하겠냐"의 답

자연 이미지는:
- 같은 패턴이 어디에 있든 같은 의미 (translation invariance).
- 위치별 *완전 독립* 패턴이 거의 없음.

LCN은 이 prior 없이 모든 위치를 *완전 자유*. Capacity 너무 커서 학습 어렵고 일반화 약함.

### 11.4 단 한 자리 — Face Recognition

얼굴 영상은 *위치별 다른 패턴*: 눈은 위쪽, 입은 아래쪽. LCN이 *부분적*으로 의미 있는 case. 일부 face recognition 모델 (DeepFace)에서 사용.

단 일반 자연 이미지에선 CNN이 압도. 표준은 항상 sharing.

### 11.5 답안 골격

> "공간 축 sharing이 CNN의 본질. 빼면 (1) 파라미터 폭증 (50,000배), (2) translation equivariance 상실, (3) 데이터 효율 급락. *Locally Connected Network*가 이 형태. 자연 이미지의 translation invariance prior와 어긋남. Face recognition 등 *위치별 패턴*이 의미 있는 일부에선 의미. 일반은 절대 안 씀."

---

## 12. 1×1 Conv 없애면

### 12.1 제거의 의미

ResNet bottleneck, Inception, MobileNet 등에서 1×1 conv 제거.

### 12.2 무엇이 망가지나

**ResNet-50 bottleneck**:
```
이전: 1×1 (256→64) → 3×3 (64→64) → 1×1 (64→256)
이후 (1×1 제거): 3×3 (256→256)
```

파라미터:
- Bottleneck: 16K + 37K + 16K = 69K
- 직접 3×3: 590K
- **8.6배 증가**

같은 표현력에 *훨씬 비싼*.

**Inception**: 1×1 없으면 5×5 conv가 256채널 그대로 → 매우 비싼.

**MobileNet (depthwise separable)**: pointwise 1×1 conv 없으면 채널 mixing 불가.

### 12.3 왜?

1×1 conv = 공간 보존 + 채널 mixing/축소. 매우 효율적인 *bottleneck* 도구.

### 12.4 보완

- Group conv: 채널을 group으로 묶음. 비슷한 정신.
- 1×1 자체가 너무 효율적이라 거의 항상 사용.

---

## 13. Pretraining 없이 학습 (Transfer 없이)

### 13.1 제거의 의미

ImageNet 사전학습 없이 random init부터 학습.

### 13.2 무엇이 망가지나

**작은 데이터** (1k~10k):
- 정확도 매우 낮음 (50% vs pretrained 90%).
- Overfit 강함.
- 학습 매우 느림.

**중간 데이터** (10k~100k):
- 정확도 약간 손해 (1~5%).
- 학습 시간 ↑.

**매우 큰 데이터** (1M+):
- 차이 거의 없음.

### 13.3 왜?

얕은 layer의 generic feature (edge, texture, color)는 *모든 자연 이미지에 공통*. ImageNet으로 학습한 이 feature가 새 task에 직접 유용.

작은 데이터로는 이 generic feature 학습이 어려움. Pretrained 모델은 이미 학습된 상태에서 task-specific만 fine-tune.

### 13.4 보완

- **Self-supervised pretraining**: SimCLR, MoCo, MAE. 라벨 없는 대량 데이터로.
- **외부 데이터 추가**.
- **Heavy augmentation**: 데이터 효과적 증가.

---

## 14. Self-Attention 없이 (Transformer에서)

### 14.1 제거의 의미

Self-attention block 제거 — FFN만 남김.

### 14.2 무엇이 망가지나

- **토큰 간 정보 흐름 없음**: 각 토큰이 *독립적*으로 처리.
- **Context 무시**: 각 토큰의 의미가 주변 무관하게 결정.
- **사실상 token-wise FFN**: 시퀀스 모델이 아님.

### 14.3 왜?

Attention이 토큰 간 정보 전달의 *유일한 메커니즘*. FFN은 각 토큰에 *독립적으로* 적용. Attention 없으면 시퀀스 정보 없음.

### 14.4 보완

- **Convolution**: 1D conv로 국소 정보. 단 long-range 약함.
- **RNN**: 시퀀스 처리. 단 병렬 안 됨.
- **MLP-Mixer 식**: 토큰 차원에 MLP. Attention 비슷한 효과.

Attention이 Transformer의 본질. 빼면 Transformer가 아님.

---

## 15. Positional Encoding 없으면 (Transformer)

### 15.1 제거의 의미

Token embedding에 PE 더하지 않음.

### 15.2 무엇이 망가지나

- **순서 정보 완전 소실**.
- "He hit me"와 "Me hit he" 구분 못함.
- 사실상 *bag-of-tokens*.
- 시퀀스 task 거의 모두 망함 (분류, 번역 등).

### 15.3 왜?

Self-attention은 *permutation invariant*. 토큰 순서를 바꿔도 같은 출력. 이건 set 처리에는 좋지만 시퀀스에는 치명.

PE가 위치 정보를 *명시적*으로 주입. 없으면 위치 정보 없음.

### 15.4 보완

- **Sinusoidal PE**: 원조 Transformer.
- **Learned PE**: BERT 식.
- **Relative PE**: T5, RoPE. 더 좋은 long-range generalization.
- **ALiBi**: extrapolation 잘 됨.

PE 자체는 어떤 형태든 필요. 빼면 안 됨.

---

## 16. Transformer의 Skip + LN 없으면

### 16.1 제거의 의미

Transformer block에서 residual connection과 LayerNorm 제거.

### 16.2 무엇이 망가지나

- **Vanishing gradient**: 깊이 12층도 학습 어려움.
- **학습 매우 불안정**: 큰 모델일수록 발산 위험.
- **Pre-Norm vs Post-Norm 차이**: Post-Norm은 skip 안에 LN, Pre-Norm은 skip 밖 → Pre-Norm이 더 안정.

### 16.3 왜?

ResNet의 정신과 같음 — skip이 gradient flow 보장. Deep Transformer (수십 층)는 skip 없이 학습 거의 불가.

LN이 활성값 분포 안정 → 큰 LR 사용 가능, 학습 빠름.

### 16.4 보완

- 둘 다 빼면 사실상 학습 불가.
- 일부만 빼면 일부 모델은 작동 (얕은 모델).

현대 Transformer는 skip + Pre-LN이 표준.

---

## 17. Optimizer Momentum 없으면 (Pure SGD)

### 17.1 제거의 의미

SGD + momentum → SGD only. $v$ 누적 없이 매 step gradient만.

### 17.2 무엇이 망가지나

- **골짜기에서 진동**: 폭이 좁은 방향으로 왔다갔다.
- **평탄 영역 매우 느림**: gradient 작은 영역에서 진전 거의 없음.
- **수렴 매우 느림** 전반적.

### 17.3 왜?

Momentum이 이전 update의 *관성*. 같은 방향이 반복되면 가속, 반대 방향이면 상쇄. 진동 줄이고 평탄 영역 가속.

Pure SGD는 매 step의 noisy gradient만 → 진동 큼.

### 17.4 보완

- **Adam**: 자체적 momentum + adaptive LR. Default로 좋음.
- **NAG**: 더 정교한 momentum.

순수 SGD는 거의 안 씀. SGD+momentum이 *최소 표준*.

---

## 18. Weight Decay 없으면

### 18.1 제거의 의미

Loss에 L2 항 없음. $\lambda = 0$.

### 18.2 무엇이 망가지나

- **가중치 발산 가능**: 특히 BN 없을 때.
- **일반화 약화**: train·val gap 커짐.
- **함수가 sharp**: 작은 input 변화에 큰 출력 변화.

### 18.3 왜?

가중치 크기 제약 없으면 큰 가중치를 학습할 수 있음. 큰 가중치 = sharp 함수 = 일반화 약함.

수학적으로: $\theta \leftarrow \theta - \eta \nabla L$. Decay 항 없으면 $\theta$ 크기에 제약 없음. SGD의 implicit bias도 weight decay와 함께 작동할 때 더 효과적.

### 18.4 보완

- Dropout: 부분적 보완. 단 weight decay와 다른 메커니즘.
- Augmentation: 데이터 다양성으로 정규화.
- Early stopping: 시간 정규화.

큰 모델에서 weight decay가 가장 안정적·효과적. AdamW가 표준.

---

## 19. Mini-batch 없이 (Full Batch GD)

### 19.1 제거의 의미

전체 데이터로 한 번에 gradient. Batch size = N (전체).

### 19.2 무엇이 망가지나

- **Sharp minima로 빠짐**: 일반화 약화.
- **메모리 폭발**: 큰 데이터에선 GPU 메모리 부족.
- **학습 느림**: epoch당 update 1번.
- **Local optima 탈출 못함**: noise 없어 stuck.

### 19.3 왜?

Mini-batch SGD의 noise가 *implicit regularization*. Sharp minima 회피, flat minima 선호. Full batch는 이 noise 없음.

또 GPU는 일정 크기의 batch에 최적화. 매우 큰 batch는 GPU 효율 떨어짐.

### 19.4 보완

- **큰 batch + LR scaling**: linear scaling rule. 단 매우 큰 batch (4096+)는 LARS/LAMB.
- **Gradient noise 추가** (인공적): 비표준.

Mini-batch가 거의 universal. Full batch는 매우 작은 데이터에서만.

---

## 20. LR Schedule 없이 (고정 LR)

### 20.1 제거의 의미

학습 내내 같은 LR. Step decay, cosine 등 없음.

### 20.2 무엇이 망가지나

- **수렴 sub-optimal**: 초반 큰 LR로 빨리 → 후반 작은 LR로 정밀, 이 패턴 잃음.
- **종종 진동**: 후반에서도 큰 LR이라 minimum 근처에서 진동.
- **학습 불안정**: 초반 큰 LR이 발산 위험.

### 20.3 왜?

학습 단계마다 적절한 LR 다름. 초반: 큰 LR로 큰 gradient 영역 탐색. 후반: 작은 LR로 minimum 근처 정밀화.

고정 LR은 이 변화를 못 따라감.

### 20.4 보완

- 주기적 LR 조정 (수동).
- ReduceLROnPlateau: val loss 정체 시 자동 감소.

표준은 cosine + warmup (Transformer) 또는 step decay (ImageNet CNN).

---

## 21. He Init 없이 (Random Uniform)

### 21.1 제거의 의미

표준 정규분포 또는 균등분포로 단순 init. He나 Xavier 안 함.

### 21.2 무엇이 망가지나

- **첫 forward에서 활성값 0 또는 폭발**: 분산이 적절히 조절 안 됨.
- **Gradient vanishing 또는 exploding**.
- **학습 매우 어려움**.

### 21.3 왜?

He init이 ReLU의 *분산 손실*을 보상 ($Var(W) = 2/n_{in}$). 단순 init은 분산이 깊이 따라 폭발 또는 0.

### 21.4 보완

- 활성화에 맞는 init: He (ReLU), Xavier (tanh/sigmoid).
- BN: 활성값 분포 강제 안정 → init 영향 줄임.
- Skip connection: gradient flow 보장.

BN과 skip이 init 영향을 많이 줄이지만, 적절한 init은 여전히 학습 안정성에 중요.

---

## 22. Validation Set 없이

### 22.1 제거의 의미

Train·test만. Hyperparameter는 test로 결정.

### 22.2 무엇이 망가지나

- **Test에 간접 overfit**: hyperparameter를 test로 반복 선택하면 test가 사실상 val.
- **실세계 성능 추정 부풀림**: 보고된 test 성능 ≠ 실제 deploy 성능.
- **모델 선택 신뢰 못함**.

### 22.3 왜?

Hyperparameter 선택은 *결정 과정*. 한 번이 아닌 여러 번 시도하고 best 선택. 이 과정에서 사용한 데이터에 *간접 overfit* 발생.

Val이 hyperparameter 결정용, test는 *최종 평가용 단 한 번*. 분리해야 신뢰 가능.

### 22.4 보완

- 데이터 매우 적으면 cross-validation. K개 fold로 통계적 신뢰.
- Nested CV: hyperparameter 튜닝과 평가를 fold로 분리.

---

## 23. Train/Eval Mode 구분 없이

### 23.1 제거의 의미

`model.eval()` 안 부르고 inference. 또는 BN/Dropout이 train mode 그대로.

### 23.2 무엇이 망가지나

- **Inference 결과 들쭉날쭉**: BN의 mini-batch 통계가 매번 다름.
- **Dropout이 마스킹 적용**: 매번 다른 sub-network → 비결정.
- **추론 정확도 감소**.
- **혼란 디버깅**: 같은 입력에 다른 출력.

### 23.3 왜?

BN과 Dropout이 train과 eval에서 *다르게 작동*. Train: 학습용 동작 (mini-batch 통계, 마스킹). Eval: deterministic·안정 동작 (running 통계, no mask).

`model.eval()`이 이 모드 전환.

### 23.4 보완

- 무조건 `model.eval()` 호출.
- 또는 LayerNorm 사용 (train·eval 동일).
- GroupNorm도 동일.

이건 *반드시* 지켜야 하는 실수 방지 사항.

---

## 24. Gradient Clipping 없이 (RNN/Transformer)

### 24.1 제거의 의미

Gradient norm이 임계 넘어도 그대로 update.

### 24.2 무엇이 망가지나

**RNN**:
- Exploding gradient → loss NaN, 발산.
- 학습 매우 빈번하게 깨짐.

**Transformer**:
- 보통 LayerNorm + warmup으로 어느 정도 안정.
- 단 큰 모델에서 가끔 NaN.

### 24.3 왜?

RNN은 같은 행렬의 거듭제곱 → exploding 본질적 위험.

Transformer는 attention 자체에서 큰 값 가능 (특히 학습 초기).

### 24.4 보완

- Gradient clipping (norm 1~5).
- LR 줄임 — 단 학습 매우 느려짐.
- Init 점검 — 분산 조절.

RNN 학습엔 거의 필수. Transformer도 안전 차원에서 사용.

---

## 25. 종합 표 — "이거 빼면"

| 컴포넌트 | 빼면 발생 |
|---|---|
| 비선형 활성화 | 선형 모델로 환원, 깊이 의미 사라짐 |
| ReLU (→sigmoid) | Vanishing, 깊은 학습 불가 |
| BN | 큰 LR 못 씀, init 민감, 깊은 학습 어려움 |
| Pooling | 메모리 폭발, RF 부담 |
| Dropout | 작은 데이터 overfit |
| Skip connection | 30+층 학습 불가, vanishing 강함 |
| Forget gate | Cell 폭발, 컨텍스트 못 잊음 |
| Input gate | 노이즈에 약함, 선택적 기록 불가 |
| Output gate | 덜 치명적 (GRU 증명) |
| 시간 sharing | RNN 본질 상실, 파라미터 폭발 |
| 공간 sharing | Translation equiv 상실, 파라미터 폭증 |
| 1×1 conv | Bottleneck 비용 폭발 |
| Pretraining | 작은 데이터에서 큰 손해 |
| Self-attention | 토큰 간 정보 흐름 없음 |
| Positional Encoding | 순서 소실, bag-of-tokens |
| Skip + LN (Transformer) | 깊은 학습 불가, 발산 |
| Momentum | 진동, 평탄 영역 느림 |
| Weight decay | 가중치 발산, 일반화 약화 |
| Mini-batch | Sharp minima, 메모리 폭발 |
| LR schedule | 수렴 sub-optimal |
| He init | 첫 forward부터 vanishing/exploding |
| Validation | Test에 간접 overfit |
| Eval mode | Inference 들쭉날쭉 |
| Gradient clipping | RNN 발산, Transformer 가끔 NaN |

---

## 26. "생각해보라" 확장

**Q. BN과 Dropout을 같은 자리에 같이 쓰면?**

분산 변동 충돌 가능. Dropout이 활성값에 noise를 더하면 BN의 batch 통계가 흔들림. 결과:
- Train의 BN 통계가 dropout mask에 따라 변동 → 학습 noise.
- Eval에서 running 통계와 train 통계 mismatch 더 커짐.

처방:
- 위치 신중 (보통 conv → BN → ReLU → dropout 또는 dropout 없음).
- ResNet block엔 dropout 거의 안 씀 (BN으로 정규화 충분).
- 또는 dropout 대신 stochastic depth.
- 또는 BN 대신 LN (이 mismatch 적음).

**Q. GRU가 LSTM의 output gate를 뺀 형태와 비슷한데, 왜 큰 성능 차이 없나?**

Output gate의 역할이 *다른 곳*에서 보상되기 때문:
- 다음 layer (Transformer block의 FFN, RNN의 다음 timestep) 가 정보 선택.
- 분류 head가 task-relevant 정보만 사용.

Output gate가 *미리 골라내는* 일이라, 안 해도 후속 처리가 결국 골라냄. 그래서 큰 손실 없음.

이게 GRU의 디자인 철학 — output gate 제거 + forget·input 합침. 단순화로도 거의 동급.

**Q. ReLU도 gradient 0 영역이 있는데 왜 vanishing 덜 심한가?**

ReLU의 절반 (음수 영역)은 gradient 0. 단 활성된 (양수) 영역은 gradient 1.

평균: 정상 분포 가정에서 ~50% 활성. 활성된 뉴런의 gradient는 1.

Sigmoid는 *모든 곳*에서 미분 < 0.25, 평균 더 작음.

깊이 N에서:
- Sigmoid: $0.25^N \rightarrow 0$ 빠르게.
- ReLU: 활성 비율 $p$의 $p^N$. $p = 0.5$이면 $0.5^N$ — 여전히 줄지만 sigmoid의 $0.25^N$보다 느림.

또 He init이 분산을 보정해서 활성된 뉴런의 *수를 적절히 유지*. 결과적으로 vanishing 매우 약함.

**Q. Skip connection을 RNN에 적용할 수 있나?**

가능. **Highway RNN, Residual RNN**:
$$h_t = h_{t-1} + F(x_t, h_{t-1})$$

또는:
$$h_t = (1-g_t) h_{t-1} + g_t F(x_t, h_{t-1})$$

(gate 형태)

장점:
- Vanishing 완화 (LSTM과 비슷한 정신).
- 깊은 RNN (수직 방향 layer stack) 학습 가능.

단 LSTM이 이미 cell state로 비슷한 일을 함. Residual RNN의 효과는 *수직 방향*에서 더 두드러짐.

**Q. Inception의 1×1 conv를 빼면?**

채널 차원 축소가 없어져 파라미터·연산 폭증. 5×5 conv를 256채널 그대로면:
- 1×1 (256→32) 후 5×5 (32→64) 후 1×1 (64→128): 적은 파라미터.
- 직접 5×5 (256→128): 5 × 5 × 256 × 128 = 819K. 매우 많음.

Inception module은 다양한 RF (1×1, 3×3, 5×5)의 *효율적 결합*을 1×1 bottleneck으로 가능하게. 빼면 비효율.

**Q. Self-attention 대신 RNN 또는 1D conv를 Transformer에?**

Transformer의 핵심은 attention. RNN으로 대체하면:
- 병렬 안 됨 (Transformer 본질의 한 축 잃음).
- Long-range는 RNN보다 attention이 강함.

1D conv로 대체:
- ConvS2S (Conv Sequence to Sequence) — Facebook 2017. RNN보다 빠름, Transformer 이전 시도.
- 단 long-range 약함 (RF 한계).
- Transformer가 결국 압도.

대체는 가능하나 *각각 다른 trade-off*. Transformer의 장점 (병렬 + long-range)을 유지하려면 attention이 답.

**Q. Validation 없이 cross-validation만 쓰면?**

가능. 작은 데이터에서 표준 방법.
- K-fold CV: 데이터를 K 등분. 매 fold가 한 번씩 val.
- Hyperparameter 결정.

단 마지막 *test* 평가도 fold로 하면 hyperparameter 튜닝과 평가가 *같은 데이터*. → Nested CV (outer loop 평가, inner loop 튜닝).

표준은 train/val/test 분리 + (선택적으로) train·val 부분에 CV. 또는 매우 작은 데이터에서만 nested CV.

---

## 27. 한 줄 요약

- **각 컴포넌트의 가치 = 그것을 제거했을 때의 손해**.
- **비선형** 빼면 선형 모델 환원 — 깊이 의미 사라짐.
- **BN** 빼면 큰 LR 못 씀, 깊은 학습 어려움.
- **Skip** 빼면 30+층 학습 불가, 항등이 default 아님.
- **Forget gate**가 LSTM에서 가장 중요. **Output gate**는 덜 치명적 (GRU 증명).
- **Sharing** (시간 또는 공간) 빼면 본질이 사라짐 — 파라미터 폭발 + 효율 손실.
- **PE** 없으면 Transformer는 bag-of-tokens.
- **Weight decay** 빼면 일반화 약화 + 가중치 발산 위험.
- **Validation** 빼면 test 간접 overfit.
- **Eval mode** 안 부르면 BN/Dropout 비결정성 → 추론 망가짐.
- **Gradient clipping** 없으면 RNN 발산 빈번.
- 사고 패턴: 제거 → 현상 → 메커니즘 → 보완. 어떤 컴포넌트든 이 4단계로 이해.
