# 12. Big Picture & FAQ — "전체 흐름 + 시험 직전"

> 모든 챕터를 하나의 큰 그림으로 묶고, 단골 질문 30개에 답한다.
> **시험 시작 5분 전, A4 1장만 본다면 이거.**

---

## 0. 큰 그림 — "딥러닝의 발전사 = 한계 극복의 역사"

각 기술은 **직전 한계의 답**으로 등장했다.

```
Perceptron (1958)
  ↓ 한계: XOR 선형 분리 불가
MLP + Backprop (1986)
  ↓ 한계: 깊으면 vanishing
[발전 정체기]
  ↓
2006: Pre-training (Hinton)
  ↓ 한계: 이미지 처리 비효율
CNN (LeNet 1998 → AlexNet 2012)
  ↓ 한계: 깊이의 degradation
ResNet (2015) — skip connection
  ↓ 한계: 시퀀스 처리
RNN
  ↓ 한계: vanishing, long-range
LSTM/GRU (1997 / 2014) — gating
  ↓ 한계: 병렬화 불가, 여전한 long-range
Attention (2014) → Transformer (2017)
  ↓ 한계: 데이터·연산 폭발
LLM (GPT, BERT 2018~) — scaling + pretrain
```

각 화살표 = "왜 이게 등장?" = 면접·시험 단골.

---

## 1. 한 화면 빅 픽처 표

| 시대 | 모델 | 해결 | 새 한계 |
|---|---|---|---|
| 1958 | Perceptron | 선형 분리 | XOR |
| 1986 | MLP + Backprop | XOR | Vanishing |
| 1998 | LeNet | 이미지 inductive bias | 작은 규모 |
| 2012 | AlexNet | Deep + GPU + ReLU + Dropout | Depth degradation |
| 2014 | VGG | 작은 커널 깊게 | 더 깊으면 학습 불가 |
| 2014 | GoogLeNet | 다중 receptive + 1×1 | depth 한계 |
| 2015 | ResNet | Skip connection | Long-range (text) |
| 1997 | LSTM | RNN의 vanishing | 병렬 불가 |
| 2014 | GRU | LSTM 단순화 | 여전한 long-range |
| 2014 | Attention | RNN의 long-range | RNN과 결합 시 순차 |
| 2017 | Transformer | 완전 병렬 + long-range | 데이터·연산 폭증 |
| 2018~ | BERT, GPT | Pretrain + scaling | 자원, hallucination |

---

## 2. 단골 Q&A 30선 (★ 시험·면접 직격)

### A. 기초

**Q1. 머신러닝과 딥러닝의 차이는?**
→ ML은 데이터로 함수를 학습하는 일반 개념, 딥러닝은 다층 신경망 기반의 ML. 딥러닝은 feature를 자동 학습 (representation learning).

**Q2. Bias-Variance trade-off의 의미는?**
→ 단순 모델은 bias 高/variance 低 (underfit), 복잡 모델은 bias 低/variance 高 (overfit). 둘의 합을 최소화하는 sweet spot이 좋은 일반화.

**Q3. Validation과 Test의 차이?**
→ Val은 hyperparameter 선택과 early stopping용, Test는 최종 평가용 (한 번만). Val로 반복 모델 선택하면 val에 간접 overfit.

### B. FNN

**Q4. 비선형 활성화가 없으면?**
→ 모든 layer가 합쳐져 선형 모델 1개. 깊이가 의미 없음. XOR도 못 풀음.

**Q5. ReLU가 표준이 된 이유?**
→ 양수 영역 미분 1 → vanishing 약함. 단순 max 연산. Sparse activation. 빠른 학습.

**Q6. Dying ReLU란? 어떻게 막나?**
→ 큰 음수 gradient로 뉴런이 영구 0 출력. Leaky ReLU, He init으로 완화.

**Q7. Backprop이 왜 효율적?**
→ Chain rule을 재사용. Forward 1번, backward 1번에 모든 gradient. Naive 계산은 지수적.

**Q8. Universal Approximation Theorem의 의미와 한계?**
→ 1 hidden layer로 임의 연속함수 근사 가능. 단 뉴런 수가 지수적일 수 있고, 학습 가능성은 보장 안 함. 그래서 deep이 실용적.

### C. DNN

**Q9. Vanishing/Exploding gradient가 왜 발생?**
→ Backprop에서 gradient가 층마다 곱해짐. < 1 곱이면 0으로 (vanishing), > 1이면 발산 (exploding).

**Q10. BN이 왜 효과 있나?**
→ Internal covariate shift 완화 + loss landscape 평활화 + 약한 정규화. 큰 LR 사용 가능, 깊은 망 학습 가능.

**Q11. Dropout이 ensemble처럼 작동하는 이유?**
→ 매 step 다른 sub-network 학습. 학습된 모델은 여러 sub-network의 평균.

**Q12. ResNet의 skip connection이 왜 깊은 학습을 가능하게?**
→ $h_{l+1} = h_l + F(h_l)$의 미분은 $I + \partial F/\partial h_l$. Identity 항 덕에 gradient가 직접 흐름. F가 0을 학습하면 최소한 더 나빠지지 않음.

**Q13. Adam이 SGD보다 항상 나은가?**
→ 아님. Adam은 빨리 수렴하지만 sharp minima로 가는 경향 → 일반화 약함. ImageNet CNN에선 SGD+momentum이 종종 우위. AdamW가 현대 표준.

**Q14. Warmup이 필요한 이유?**
→ 학습 초기에 BN 통계, Adam의 second moment가 noisy. 큰 LR이 위험. 작게 시작해 안정화 후 키움.

### D. CNN

**Q15. 이미지에 FNN을 안 쓰는 이유?**
→ 파라미터 폭발 (224×224×3 입력 시 첫 층 1억+). 위치 변경에 약함 (translation 안 됨).

**Q16. Weight sharing을 왜 하나? (★ 단골)**
→ 1) 파라미터 효율, 2) Translation equivariance (같은 패턴은 어디서든 같은 응답), 3) 데이터 효율 (한 위치 학습이 모든 위치 적용).

**Q17. Conv는 equivariance, pooling은 invariance — 차이?**
→ Equivariance는 입력 이동에 출력도 같이 이동. Invariance는 입력 이동에 출력 변화 없음. Pooling/GAP가 위치 무관성을 부여.

**Q18. 1×1 conv는 왜 쓰나?**
→ 채널 차원만 섞음 (공간 그대로). 차원 축소 (bottleneck), 계산량 감소, 채널 간 비선형 결합.

**Q19. 3×3 conv를 깊이 쌓는 이유? (VGG)**
→ 같은 receptive field를 더 적은 파라미터·더 많은 비선형으로 달성. 7×7 1개 = 49C², 3×3 3개 = 27C².

**Q20. Transfer learning이 왜 작동?**
→ 얕은 층은 도메인 무관한 일반 visual feature (edge, texture). 깊은 층만 task-specific. 그래서 얕은 층 freeze + 깊은 층 fine-tune 가능.

### E. RNN/LSTM

**Q21. RNN이 arbitrary length를 어떻게 처리?**
→ 시간 축 가중치 공유. 같은 셀을 시간에 따라 반복 적용. 길이에 따라 파라미터 변하지 않음. Hidden state가 과거의 압축 요약.

**Q22. RNN에서 vanishing이 더 심한 이유?**
→ 같은 가중치 행렬이 시간 축에 거듭제곱. FNN은 층마다 다른 행렬 → 우연히 균형 가능. RNN은 한쪽으로 폭주 또는 소멸 거의 보장.

**Q23. LSTM이 vanishing을 어떻게 완화?**
→ Cell state의 미분이 forget gate (scalar) → 행렬 곱이 아닌 element-wise. Forget=1 학습 시 gradient 그대로 흐름. ResNet skip과 본질 같음.

**Q24. LSTM의 각 gate의 역할?**
→ Forget: 과거 cell 중 무엇을 버릴까. Input: 새 정보 중 무엇을 쓸까. Output: cell의 어느 부분을 노출할까.

**Q25. LSTM vs GRU 어느 쪽?**
→ 거의 동급. 데이터 적고 빨라야 하면 GRU. 매우 긴 의존성 + 큰 데이터에서 LSTM이 약간 우위 보고. 실무에선 둘 다 시도.

**Q26. 이미지를 RNN으로 학습하려면?**
→ 픽셀(또는 patch)을 raster/Hilbert scan으로 1D 시퀀스화. PixelRNN 식. CNN의 2D 공간 inductive bias를 잃어 매우 비효율. 분류엔 부적절, 자기회귀 생성에 의미.

### F. Transformer

**Q27. Transformer가 RNN을 대체한 이유?**
→ 1) 완전 병렬화 (GPU 활용), 2) Long-range 직접 (거리 무관), 3) Scaling이 잘 됨. 단 데이터·연산 多 필요, O(n²) 메모리.

**Q28. Self-attention의 핵심?**
→ 모든 토큰이 모든 토큰을 동시에 본다. Query·Key·Value의 dot product로 가중치. 거리 무관 정보 전달.

**Q29. Positional Encoding이 왜 필요?**
→ Self-attention은 순서 무관. PE가 위치 정보를 명시적으로 주입해야 시퀀스 task 가능.

### G. 일반 사고

**Q30. "이거 빼면 어떻게 되나"의 위력?**
→ 모든 컴포넌트의 가치는 그것을 제거했을 때 드러난다. 비선형 빼면 선형 환원, BN 빼면 학습 어려움, skip 빼면 30층 한계 — 이런 답들이 컴포넌트의 본질.

---

## 3. 주제 간 연결 다이어그램

```
                ┌── 표현력 ──→ 깊이가 좋다
        FNN ────┤
                └── 학습 어려움 ──→ Vanishing 문제
                                  ↓
        ┌── ReLU + He init (시작 단계 처방)
DNN ────┤── BN (분포 안정)
        │── Skip connection (gradient 직접)
        └── Adam, AdamW + warmup (optimizer 발전)
                                  ↓
        ┌── Weight sharing (파라미터)
CNN ────┤── Locality (국소 패턴)
        │── Pooling (invariance)
        └── 깊이 → ResNet
                                  ↓
        ┌── 시간 sharing (가변 길이)
RNN ────┤── Vanishing 더 심함
        │── BPTT, gradient clip
        └── LSTM/GRU (gating)
                                  ↓
        ┌── 병렬화 한계
LSTM ───┤── Long-range 한계
        └── Attention 등장
                                  ↓
        ┌── Self-attention only
Transformer ────┤── Position Encoding
                ├── 데이터·연산 많이
                └── LLM 시대
```

---

## 4. A4 1장 — "시험 직전 5분"

### 핵심 공식
- **GD update**: $\theta \leftarrow \theta - \eta \nabla L$.
- **CE+softmax gradient**: $\hat{y} - y$ (★기억하기 좋음).
- **Conv output**: $\lfloor (W + 2p - k)/s \rfloor + 1$.
- **LSTM cell**: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$.
- **Self-attention**: $\text{Att}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k}) V$.

### 짝짓기
- ReLU + He init.
- Sigmoid + BCE.
- Softmax + CE.
- BN (CNN, big batch), LN (Transformer/RNN), GN (small batch).
- AdamW + warmup + cosine (Transformer).
- SGD + momentum (ImageNet CNN, 일반화 우위).

### 단골 답안
- "왜 X 빼면 안 되나?" = ablation으로 답.
- "왜 sharing?" = 파라미터·equivariance·데이터 효율.
- "Arbitrary length 어떻게?" = 시간 축 sharing.
- "LSTM 왜 vanishing 완화?" = forget gate scalar 곱.
- "왜 Transformer가 RNN 대체?" = 병렬 + long-range + scaling.

### 실수 방지
- BN/Dropout `model.eval()`.
- Output-Loss 매칭.
- 데이터 정규화 train만으로 fit.
- Gradient clipping (RNN, Transformer).
- LR > Batch > Optimizer > Init 순으로 민감.

---

## 5. 마지막 사고 — "교수님처럼 답하기"

질문이 오면 항상 다음 5단계로:

1. **무엇을 묻나?** (정의? 메커니즘? 비교? 설계?)
2. **구조적 답변** (어떻게 작동? 어떤 식?)
3. **왜 그렇게?** (메커니즘적 이유)
4. **다른 옵션과 비교** (이거 대신 저거면?)
5. **실용적 함의** (언제 쓰고 언제 안 쓰나)

**정의를 묻는 게 아니라 사고의 흐름을 보고 싶어한다.**
"이건 X입니다"가 아니라 "X는 Y의 한계를 풀기 위해 Z 메커니즘을 사용하는데, A와 비교하면 B 상황에서 우위, C 상황에선 D"가 좋은 답.

---

## 6. 응용·면접 질문 보너스 12선

1. ImageNet으로 학습한 ResNet을 의료 영상에 적용하려면? → Transfer learning, 마지막 FC 교체, 작은 LR로 fine-tuning. 도메인 차이 크면 깊은 층도 학습.
2. 텍스트 감성 분류를 가장 빠르게 prototype하려면? → TF-IDF + Logistic / TextCNN baseline → 필요 시 BERT.
3. 시계열 단기 예측이 잘 안 될 때 첫 시도? → 데이터 정규화, lag feature, baseline (MA, AR), 그 다음 LSTM/TCN.
4. 클래스 불균형 (양성 1%) 분류는? → Class weight, focal loss, oversampling, 평가는 PR-AUC.
5. GAN 학습이 자주 collapse하는 이유와 처방? → mode collapse, equilibrium 어려움 → spectral norm, gradient penalty (WGAN-GP).
6. 추천 시스템 cold start 문제 해결? → Content-based feature, transfer learning, meta-learning.
7. 이상 탐지에 supervised vs unsupervised? → 라벨 거의 없을 때 unsupervised (Autoencoder, Isolation Forest), 라벨 있으면 supervised (Focal loss).
8. 모델이 deploy 후 성능 하락하면? → Distribution shift 의심. 모니터링 + 정기 재학습.
9. Knowledge distillation의 핵심? → 큰 teacher의 soft label로 작은 student 학습. Soft target에 dark knowledge.
10. Self-supervised learning이 왜 중요한가? → 라벨 없는 대량 데이터로 표현 학습. Pretrain → fine-tune의 기초.
11. Attention의 시간 복잡도 O(n²)을 줄이는 방법? → Sparse attention, linear attention, FlashAttention, sliding window.
12. Mixed precision training이 왜 빠르고 위험한가? → fp16의 빠른 연산, 단 동적 범위 좁음 → loss scaling 필요. Overflow/underflow 주의.

---

## 7. 한 줄 요약 (최종)

- 딥러닝 = **각 시대의 한계를 극복하는 기술의 누적**.
- 모든 답은 "**왜 이 구조? + 빼면 어떻게? + 다른 옵션과 비교?**"로 정리.
- 정의 외우는 것보다 **사고의 흐름**.
- "이미지를 RNN으로?"처럼 cross-architecture 질문에는 **inductive bias 매칭**으로 답.
- 시험 직전엔 본 챕터의 표 + Q&A 30개만으로 충분.
