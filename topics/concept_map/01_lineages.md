# 01. 계보 — 시간 순으로 본 6대 패밀리

각 계보는 *"한 단계 → 다음 단계가 이전의 어떤 한계를 해결했는가"*의 흐름이다. 표의 마지막 칸은 *해결되며 새로 생긴 함정·하이퍼파라미터*.

---

## 1. Loss 계보 — 회귀편

**문제**: 잔차(예측 − 정답)를 *어떻게 벌점화* 할 것인가. 핵심 축은 "outlier에 얼마나 너그러울까".

| 단계 | 수식 핵심 | 이전의 한계 해결 | 새로 생긴 함정 |
|---|---|---|---|
| **MSE** | 잔차의 제곱 평균 | 첫 출발 — 미분 부드럽고 optimizer 친화 | 큰 잔차가 제곱되어 폭주 → outlier에 학습이 끌려다님 |
| **MAE** | 잔차의 절댓값 평균 | outlier가 1차로만 들어와 robust | $r=0$에서 미분 불연속 → optimizer가 진동 |
| **Huber** | $\|r\| \le \delta$이면 제곱, 아니면 선형 | 작은 잔차는 부드럽고 큰 잔차는 robust — 두 장점 결합 | 새 하이퍼파라미터 $\delta$ — outlier 경계를 사람이 정해야 |

**통찰**: MSE↔MAE는 *"평균 vs 중앙값"*과 같은 구도. Huber는 둘 사이의 슬라이더. 자세한 기하학적 직관은 [topics/regression_loss/](../regression_loss/README.md).

---

## 2. Loss 계보 — 분류편

**문제**: 확률 출력으로 정답을 얼마나 못 맞췄나를 정보이론적으로 측정. 핵심 축은 *"어느 sample에 집중하고 어떤 자신감을 막을지"*.

| 단계 | 핵심 아이디어 | 이전의 한계 해결 | 새로 생긴 함정 |
|---|---|---|---|
| **BCE / CE** | $-\log p_y$ — 정답 클래스 확률의 음의 로그 | 잘 맞춘 클래스는 빠르게 학습 무시, 못 맞춘 클래스에 집중 | class imbalance 시 majority만 학습됨 (minority의 절대 contribution 부족) |
| **Weighted CE** | 클래스별 가중치 $w_y$ 곱 | 빈도 역수 가중 → minority 신호 증폭 | 가중치 결정의 임의성 (역수? 제곱근 역수? heuristic) |
| **Focal** | $(1-p_y)^\gamma \cdot \text{CE}$ — easy sample down-weight | 같은 클래스 안에서도 *easy vs hard* 구분 → hard에 집중 | $\gamma$ 튜닝, $\gamma$ 크면 학습 늦어짐 |
| **Label Smoothing** | 정답 $1.0 \to 1-\epsilon$, 오답 $0 \to \epsilon/(K-1)$ | overconfident 방지 → calibration 개선 + 약한 regularization | 너무 부드러우면 정확도 손실, $\epsilon$ 튜닝 |
| **Contrastive / Triplet** | 같은 클래스끼리 끌어당기고 다른 클래스끼리 밀어냄 | 카테고리가 무한 또는 미정의일 때 (얼굴 인증, retrieval) | negative mining 난도, margin 튜닝 |

**통찰**: CE→Focal→LabelSmooth는 *"전체 loss → sample별 가중 → label 자체 부드럽게"*로 추상화 레벨이 한 단계씩 올라간다. Contrastive는 "분류"라는 framing 자체를 바꾸는 분기.

---

## 3. Activation 계보

**문제**: 비선형성 도입 + gradient 흐름 유지. 축은 *"vanishing 안 하면서 부드럽게"*.

| 단계 | 식 | 이전의 한계 해결 | 새로 생긴 함정 |
|---|---|---|---|
| **Step** | $\mathbb{1}[x>0]$ | 퍼셉트론 시대 첫 nonlinearity | 미분 불가능 → backprop 자체가 안 됨 |
| **Sigmoid** | $1/(1+e^{-x})$ | 미분 가능 + 확률로 해석 가능 | (1) 양 끝에서 gradient ≈ 0 → vanishing  (2) zero-centered가 아님 → zigzag update |
| **tanh** | $\tanh(x)$, 범위 $[-1,1]$ | zero-centered → 학습 효율 ↑ | 여전히 양 끝 vanishing |
| **ReLU** | $\max(0, x)$ | 양수 영역 gradient = 1 → vanishing 거의 없음, 계산 매우 빠름 | (1) dead ReLU — 음수만 받으면 영원히 0  (2) zero-centered 아님 |
| **LeakyReLU / PReLU** | $\max(\alpha x, x)$ | 음수 영역에서도 작은 gradient → dead 방지 | $\alpha$ 선택 (LeakyReLU 고정 / PReLU 학습) |
| **ELU** | $x$ if $x>0$ else $\alpha(e^x-1)$ | 음수에서 saturation → mean activation 0에 가까움 (BN 같은 효과) | exp 계산 — ReLU보다 느림 |
| **GELU / Swish** | $x \cdot \Phi(x)$ / $x \cdot \sigma(x)$ | 부드러운 ReLU — gradient가 0에서 연속적으로 변함, Transformer 표준 | exp/Φ 계산 비용 |

**통찰**: 계보의 축은 두 개로 갈린다 — (a) *gradient를 죽이지 않으면서* (vanishing 해결), (b) *수학적으로 부드럽게* (학습 안정). ReLU가 (a)를 압도적으로 해결했고, 이후는 (a)를 유지하면서 (b)를 보충하는 흐름.

---

## 4. Optimizer 계보

**문제**: gradient 신호로 어떻게 step을 취할 것인가. 축은 *"속도(누적) + 적응(파라미터별 LR) + 정규화(weight decay)의 단계적 도입"*.

| 단계 | 핵심 추가 | 이전의 한계 해결 | 새로 생긴 함정 |
|---|---|---|---|
| **SGD** | $\theta \leftarrow \theta - \eta g$ | 첫 출발 — 단순 | noisy gradient, ravine·saddle에서 진동·정체 |
| **Momentum** | $v \leftarrow \beta v + g$, $\theta \leftarrow \theta - \eta v$ | 이전 gradient를 누적 → 진동 감쇄, ravine 통과 가속 | $\beta$ 튜닝 (보통 0.9). overshoot 가능 |
| **Nesterov** | gradient를 *미래 위치*에서 평가 | momentum의 overshoot 완화 (lookahead) | 미세한 차이 — 실용 효과는 task마다 |
| **AdaGrad** | $g$의 제곱합으로 per-parameter LR 조절 | 자주 업데이트되는 param의 LR을 자동으로 줄임 (희소 feature에 유리) | 누적합이 단조 증가 → LR이 0으로 수렴, 후반 학습 정체 |
| **RMSProp** | 누적을 *exponential moving average*로 (옛 값 forget) | AdaGrad의 LR 소멸 해결 | $\rho$ 튜닝 |
| **Adam** | Momentum + RMSProp 결합 (1차·2차 모멘트 동시) | 두 흐름의 장점 합산 → 기본값 robust → 사실상 default | weight decay 처리 방식이 L2로 들어가 generalization 손실 |
| **AdamW** | weight decay를 gradient와 *분리*해 적용 | Adam의 generalization 손상 회복 | 메모리 2 buffer/param (Adam과 동일) |
| **Lion** | sign(momentum) 기반 업데이트 | 메모리 절반 (1 buffer/param) + 더 큰 batch에서 안정 | 새로움 — 검증된 task 범위 좁음 |

**통찰**: 세 흐름이 *순차적으로 더해진* 구조. (1) **속도 누적** (Momentum 계열), (2) **파라미터별 적응** (Ada 계열), (3) **정규화 분리** (Weight decay 계열). Adam은 (1)+(2), AdamW는 (1)+(2)+(3).

---

## 5. Normalization 계보

**문제**: 학습 중 각 layer의 입력 분포가 흔들리는 internal covariate shift를 어떻게 잡을까. 축은 *"어느 축으로 평균·분산을 계산하는가"*.

배치 텐서의 모양이 `(N, C, H, W)` (이미지) 또는 `(N, T, C)` (시퀀스)라 할 때:

| 단계 | 정규화 축 | 강점 | 한계·새 함정 |
|---|---|---|---|
| **BN** | batch+spatial $(N, H, W)$ → 채널별 평균·분산 | CNN classification 표준. regularization 효과 (다른 sample이 잡음 역할) | small batch에서 통계 불안정. RNN·variable-length에 어색. inference 시 running stat 사용 → train/inference gap |
| **LN** | feature 전체 $(C, H, W)$ 또는 $(C,)$ → sample별 정규화 | batch size 무관. RNN·Transformer 표준 | image classification에서는 BN보다 살짝 약함 |
| **IN** | spatial $(H, W)$ → sample × 채널별 | style 정보 (sample-wise) 제거 — style transfer에 사용 | classification 신호도 같이 제거 → 일반 task에 부적합 |
| **GN** | 채널을 그룹화해 $(C_g, H, W)$ | small batch에서도 안정 (BN의 batch 의존 해소) | 그룹 수 $G$ 튜닝 |

**통찰**: "어느 차원을 함께 묶어 평균·분산을 잡는가"의 단순한 선택지 차이. *데이터 모양 + batch 크기*가 자연스럽게 선택을 결정한다. (큰 batch CNN → BN, 시퀀스/Transformer → LN, 작은 batch CNN → GN, style → IN)

---

## 6. CNN 아키텍처 계보

**문제**: 깊이를 어떻게 늘리고 그 안에서 어떤 *효율·gradient·재사용* 트릭을 발견하는가.

| 세대 | 모델 | 핵심 추가 | 이전의 한계 해결 | 새 함정 |
|---|---|---|---|---|
| 1세대 | **LeNet (1998)** | conv + pool + FC 조합 | 손글씨 분류 — CNN이라는 framework 자체 | 깊이 한계, ReLU 전이라 학습 어려움 |
| 2세대 | **AlexNet (2012)** | ReLU + Dropout + GPU + ImageNet 규모 | 깊이 + 학습 노하우 (vanishing 완화) → ImageNet 우승 | 파라미터 매우 큼 |
| 3세대 | **VGG (2014)** | 3×3 conv를 *순수히 깊게* | "큰 conv 한 번 = 작은 conv 여러 번" → 깊이의 단순 power 입증 | 파라미터·연산 폭발 (138M) |
| 3.5세대 | **GoogLeNet / Inception** | 1×1 conv로 dim reduce + 다양한 size 병렬 (inception module) | 깊이·연산을 분리 → 파라미터 효율 | 모듈 설계 복잡 |
| 4세대 | **ResNet (2015)** | skip connection ($y = F(x) + x$) | depth가 100+가 되어도 학습 가능 — vanishing gradient 결정적 해결 | residual block 설계 (pre-act vs post-act 등 미세 결정) |
| 4.5세대 | **DenseNet** | 모든 이전 layer를 concat | feature reuse → 파라미터 적게 같은 성능 | 메모리 — concat이 메모리에 잔존 |
| 5세대 | **EfficientNet** | depth·width·resolution을 *동시에 균형 scaling* | 세 차원 중 하나만 늘리는 비효율 해결 — compound scaling | scaling 계수 검색 (NAS) 비용 |
| 6세대 | **ViT / Swin** | conv 없이 patch attention | locality bias를 *학습으로 발견* (큰 데이터에서) | 작은 데이터에서는 inductive bias 부족으로 성능 떨어짐 |

**통찰**: 흐름은 *"깊이 → 효율 → gradient 흐름 → 재사용 → 균형 → bias 자체를 학습"*으로 추상화 레벨이 한 단계씩 올라간다. ResNet의 skip connection은 단순한 트릭이 아니라 *"identity를 학습 안 해도 되게 만든"* 결정적 inductive bias 변경.

---

## 7. Sequence Model 계보

**문제**: 임의 길이의 시퀀스를 어떻게 처리하고, long-range 의존을 어떻게 잡을까.

| 단계 | 핵심 | 이전의 한계 해결 | 새 함정 |
|---|---|---|---|
| **Vanilla RNN** | hidden state $h_t = \sigma(W h_{t-1} + U x_t)$ | 임의 길이 시퀀스 처리 가능 (parameter sharing) | vanishing/exploding gradient → long-range 못 잡음 |
| **LSTM** | cell state + 3 gate (forget, input, output) | gate가 gradient를 곱셈으로 보존 → long-range 가능 | 파라미터 4배, 학습 느림 |
| **GRU** | 2 gate (update, reset) | LSTM과 비슷한 성능 + 더 가벼움 | 미세 task에서 LSTM이 살짝 우세 |
| **bi-RNN** | 좌→우 + 우→좌 hidden state concat | 양방향 context (token classification에 필수) | inference 시 전체 시퀀스 필요 (online 불가) |
| **seq2seq (RNN encoder-decoder)** | encoder → fixed-size context → decoder | 입출력 길이 다른 task (번역) 가능 | fixed-size context bottleneck — 긴 입력의 정보가 압축 손실 |
| **Attention** | decoder가 각 step마다 encoder의 *모든* hidden state에 weighted sum | context bottleneck 해소 — 긴 입력도 직접 참조 | encoder/decoder 모두 RNN이라 *순차성*은 그대로 |
| **Transformer** | attention만 — recurrence 제거, positional encoding으로 순서 주입 | 병렬 학습 가능 (각 token이 독립적으로 attention 계산) | $O(n^2)$ 메모리 — 매우 긴 시퀀스 부담 |

**통찰**: 세 흐름의 합성. (1) **장기 의존성** (gate → attention), (2) **양방향성** (bi-RNN, full self-attention), (3) **병렬성** (Transformer가 결정적으로 추가). 각 추가가 *이전의 핵심 한계 하나*를 푼다.

---

## 한 줄 메타 통찰

> *"모든 계보는 '이전이 풀고 남긴 한계'를 다음이 푸는 사슬이다. 새 기법을 외울 때, **'이게 무엇을 해결했나'와 '이게 새로 만든 함정은'**의 두 칸을 함께 메모하면 평생 안 잊는다."*
