# 10. Training Diagnosis — "왜 학습이 안 되는가" 심층

> **이 문서의 핵심 사고**:
> 학습 실패는 모두 다른 얼굴을 하지만 *원인은 패턴*이 있다. 증상 → 원인 가설 → 빠른 검증 → 처방의 흐름으로 빠르게 진단.
>
> ML 엔지니어의 실무는 *모델 설계*보다 *디버깅*이 더 큰 비중. 이 챕터는 그 디버깅의 체계를 만든다.

---

## 0. 진단 사고법

### 0.1 4단계 흐름

학습 문제를 만나면:

1. **증상 정확한 관찰**: Loss 곡선, gradient norm, 예측 분포, 활성값 분포.
2. **가설 좁히기**: 어디서 발생? data / model / optimizer / loss / code.
3. **빠른 검증**: 작은 ablation으로 원인 격리.
4. **처방**: 원인에 맞는 fix.

### 0.2 정찰 도구 — 무엇을 봐야 하나

**기본 모니터링** (모든 학습에):
- Train loss curve
- Val loss curve
- Train accuracy / metric
- Val accuracy / metric

**디버깅용 추가**:
- Gradient norm (per layer)
- Activation 분포 (per layer, histogram)
- Weight 분포 (per layer)
- Learning rate (schedule 적용 후)
- BN running 통계

PyTorch + TensorBoard 또는 Weights & Biases가 표준.

### 0.3 첫 번째 디버깅 — Tiny Dataset Overfit

새 모델·코드 첫 sanity check:

1. 데이터 10~100개만 추출.
2. Validation 없이 train loss 0에 도달 시도.
3. 도달 못하면 **모델·코드·loss에 버그**.

이건 매우 빠른 반복 — 30초~수 분. 이걸 통과 못하면 큰 데이터 학습은 무의미.

### 0.4 본 챕터의 흐름

§1~5: 자주 발생하는 5가지 큰 패턴
§6~10: BN/Dropout, leakage, 시간 leak, 클래스 불균형
§11~13: RNN, Transformer, 큰 모델 특화
§14~16: 디버깅 도구·체크리스트
§17~18: 면접 Q&A, 생각해보라

---

## 1. Loss NaN — 즉시 진단

### 1.1 원인 매트릭스

| 원인 | 신호 | 처방 |
|---|---|---|
| LR 너무 큼 | 첫 epoch부터 NaN | LR 1/10 |
| log(0) | 분류에서 발생 | log_softmax, BCEWithLogitsLoss |
| 나눗셈 0 | Adam, BN | eps 추가 |
| Init 분산 큼 | 첫 forward에서 NaN | He/Xavier |
| Mixed precision overflow | fp16 사용 시 | Loss scaling |
| 데이터에 NaN | 입력 자체에 NaN/inf | 데이터 점검 |
| Gradient explosion | RNN, 깊은 망 | Clipping |

### 1.2 디버깅 절차

```
1. 작은 batch (1~4)로 forward만 실행.
   → 출력에 NaN인가?
   Yes → init/loss/데이터 문제.
   No → 다음 단계.

2. Backward 1번 실행.
   → Gradient norm 출력 (per layer).
   매우 큼? → exploding.
   매우 작음? → vanishing (NaN 아님).
   NaN? → loss 계산 문제.

3. Loss 분해.
   → Cross-entropy를 logits·log_softmax 단계로.
   → 어디서 NaN 발생하나.

4. 데이터 한 sample씩 forward.
   → NaN 발생하는 sample 식별.
   → 그 sample의 입력 출력 점검.

5. LR 1/10.
   → 여전히 NaN이면 다른 원인.
   → 안전해지면 LR 문제.

6. Gradient clipping 추가.
```

### 1.3 자주 발생하는 NaN 패턴

(1) **Sigmoid + BCE의 log(0)**:
```python
loss = -(y * log(p) + (1-y) * log(1-p))
```
$p = 0$ 또는 $p = 1$이면 log(0) = $-\infty$. 처방: `BCEWithLogitsLoss` 사용 — sigmoid + log를 *수치 안정* 형태로.

(2) **Softmax의 overflow**:
```python
exp(z_i) / sum(exp(z_j))
```
$z$가 매우 크면 exp overflow. 처방: 모든 라이브러리는 자동 — max를 빼고 계산. 직접 구현하지 말 것.

(3) **Mixed precision의 underflow**:
fp16의 dynamic range 좁음. Gradient가 매우 작으면 0이 됨. 처방: loss scaling — loss를 큰 수로 곱한 후 backward, gradient 받고 다시 나눔.

### 1.4 답안 골격

> "원인 후보 7개: LR 큼, log(0), 0 나눗셈, init 분산 큼, mixed precision overflow, 데이터 NaN, gradient explosion. 디버깅: 작은 batch forward → backward → loss 분해 → 데이터 점검 → LR 1/10 → clipping 추가. 가장 흔한 원인은 LR과 log(0). BCEWithLogitsLoss와 log_softmax가 안전 form."

---

## 2. Vanishing Gradient — 학습 정체

### 2.1 신호

- 첫 epoch부터 train loss 안 떨어짐.
- 깊은 layer의 gradient norm이 매우 작음 ($10^{-7}$ 이하).
- 얕은 layer만 학습되고 깊은 layer는 random.
- Activation 분포가 매우 좁거나 0 근처.

### 2.2 원인 매트릭스

| 원인 | 처방 |
|---|---|
| Sigmoid/Tanh 깊게 | ReLU 계열 |
| 잘못된 init | He / Xavier |
| BN/LN 없음 | 추가 |
| Skip connection 없음 | ResNet 식 |
| LSTM gate 없음 | GRU/LSTM 사용 |

### 2.3 진단 코드

```python
def check_grad_norms(model):
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f"{name}: {p.grad.norm().item():.2e}")
```

깊을수록 작아지면 vanishing. 첫 layer가 $10^{-8}$이고 마지막이 $10^{-2}$이면 매우 심각.

### 2.4 처방 우선순위

1. **활성화 점검**: ReLU? GELU? 그 외면 ReLU로.
2. **Init 점검**: He (ReLU), Xavier (tanh).
3. **BN/LN 추가**.
4. **Skip connection** (ResNet 식 또는 deep RNN).
5. **LR 점검**: 너무 작으면 진전 없음.

### 2.5 답안 골격

> "신호: 깊은 layer gradient $10^{-7}$ 이하, train loss 정체. 진단: per-layer gradient norm. 처방 우선순위: (1) ReLU 활성화 + He init, (2) BN/LN, (3) Skip connection. 매우 깊은 망 (50+층)은 이 셋 모두 필요. LSTM/GRU도 vanishing 완화 (덧셈 구조)."

---

## 3. Exploding Gradient — 갑작스런 발산

### 3.1 신호

- 학습 진행 중 갑자기 loss 폭증.
- Gradient norm이 $10^3$, $10^6$ 이상.
- NaN으로 끝남.
- 특히 RNN/LSTM에서 자주.

### 3.2 원인

(1) **LR 너무 큼**: 매 step gradient를 크게 update.
(2) **RNN의 같은 행렬 거듭제곱**: $W_{hh}$의 고유값 > 1.
(3) **Init 분산 큼**: 시작부터 큰 가중치.
(4) **데이터의 outlier**: 매우 큰 입력 값.

### 3.3 처방

**Gradient clipping** — 표준 답:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

작동: gradient의 norm이 임계값 넘으면 비례 축소.

```
g_clipped = g if ||g|| < c else c * g / ||g||
```

매우 단순하지만 매우 효과적. RNN/Transformer 학습에 사실상 필수.

```python
# RNN/LSTM 학습 표준 패턴
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

### 3.4 답안 골격

> "신호: gradient norm $10^3+$, 갑자기 NaN. 원인: LR 큼, RNN의 행렬 거듭제곱, init 분산 큼, 데이터 outlier. 처방: gradient clipping (norm 1~5). RNN/Transformer 표준. LR 줄임 + init 점검 보조."

---

## 4. Overfit — Train ↓ Val ↑

### 4.1 신호

- Train loss 계속 감소 → 0에 가까움.
- Val loss 일정 epoch 후 증가.
- Train·val gap 큼 (예: train acc 99% vs val acc 70%).

### 4.2 처방 우선순위

(1) **Early stopping**: 가장 단순. Val loss 정체 또는 증가 시 멈춤.

(2) **Dropout**: 0.3 → 0.5.

(3) **Weight decay**: 1e-4 → 1e-3.

(4) **Data augmentation**: task별 적절한 변형.

(5) **Label smoothing**: CE에 정규화 (0.1 정도).

(6) **Mixup / CutMix**: 입력+라벨 혼합. 강력.

(7) **모델 크기 줄임**: 마지막 수단.

(8) **외부 데이터** / pretrained: 항상 도움.

### 4.3 신호로 처방 강도

| 증상 | 처방 강도 |
|---|---|
| 작은 gap (5%) | Augmentation 추가 |
| 중간 gap (10~20%) | Dropout, weight decay 강화 |
| 큰 gap (20%+) | 모델 줄임, 데이터 늘림, 강한 정규화 |

### 4.4 답안 골격

> "Train ↓ Val ↑이 overfit 신호. 처방 우선순위: early stopping → dropout → weight decay → augmentation → label smoothing → mixup → 모델 줄임. Gap 크기에 비례한 처방 강도. 데이터 추가가 가장 좋음. 큰 모델 + 큰 데이터 시대엔 augmentation + label smoothing이 자주 충분."

---

## 5. Underfit — 둘 다 못 맞춤

### 5.1 신호

- Train loss도 안 떨어짐.
- Val도 마찬가지.
- 둘 다 sub-optimal.

### 5.2 원인 매트릭스

| 원인 | 처방 |
|---|---|
| 모델 너무 작음 | depth/width 키움 |
| LR 부적절 | LR Range Test |
| 비선형 부족 | 활성화 점검 |
| Init 잘못 | He/Xavier |
| 데이터 정규화 없음 | 입력 정규화 (mean 0, std 1) |
| Output-loss mismatch | Sigmoid+BCE 등 짝 점검 |
| 정규화 너무 강함 | Dropout, weight decay 줄임 |
| 학습 부족 | epoch 늘림 |

### 5.3 가장 흔한 원인 — 입력 정규화

이미지 [0, 255] 그대로 입력 → 학습 안 되거나 매우 느림. 처방:
```python
input = (input - mean) / std  # mean, std는 학습 데이터 통계
```

ImageNet 표준: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] (RGB).

### 5.4 디버깅 — Tiny Dataset Overfit

10개 sample로 train loss 0 도달 가능한가?
- Yes → 모델·loss 정상. 큰 데이터에서 underfit이면 hyperparameter 조정.
- No → 코드/loss/data에 버그.

이게 매우 빠른 sanity check.

### 5.5 답안 골격

> "둘 다 stuck이 underfit. 원인 7가지: 모델 작음, LR 잘못, 활성화/init 부적절, 데이터 정규화 안됨, output-loss mismatch, 정규화 강함, 학습 부족. 가장 흔한 건 입력 정규화 안 함. Tiny dataset overfit 가능한지가 첫 sanity check — 안 되면 코드 버그."

---

## 6. BN/Dropout Mode 실수

### 6.1 신호

- Val 결과가 매번 다름.
- Train보다 Val이 더 좋음 (이상).
- 같은 입력에 다른 출력.

### 6.2 원인

`model.eval()` 안 부름. BN과 Dropout이 train mode 그대로 동작:
- BN: mini-batch 통계 사용 (running 아님).
- Dropout: 마스킹 적용.

### 6.3 처방

```python
model.eval()
with torch.no_grad():
    output = model(input)
```

이게 표준 inference 패턴. 항상.

또는 LayerNorm 사용 (train·eval 동일 동작).

### 6.4 작은 batch에서 BN 통계 부정확

batch=2~4면 통계 매우 noisy. 처방:
- **GroupNorm**: BN 대체. Detection, segmentation 표준.
- **Batch 키움**: 메모리 허락 시.
- **SyncBN**: multi-GPU에서 batch 통계 동기화.

### 6.5 답안 골격

> "Val 결과 들쭉날쭉이면 model.eval() 안 부른 가능성. BN의 mini-batch 통계와 dropout의 마스킹이 inference에서 비결정성. 처방: model.eval() + torch.no_grad() 표준. 작은 batch BN은 GroupNorm으로. 자기회귀 추론 (batch=1)도 BN 부정확 → LayerNorm."

---

## 7. 데이터 누수 (Data Leakage)

### 7.1 신호

- Val 성능 비현실적으로 높음.
- Production deploy에서 성능 급락.
- Val·test 성능 갭 작음.

### 7.2 흔한 leakage 형태

(1) **전처리 leakage**:
정규화 통계를 train+test 합쳐 fit. 표준화의 mean·std가 test 정보 포함.

```python
# 잘못된 방식
all_data = np.concat([train, test])
mean, std = all_data.mean(), all_data.std()  # leakage!
train = (train - mean) / std
test = (test - mean) / std

# 올바른 방식
mean, std = train.mean(), train.std()
train = (train - mean) / std
test = (test - mean) / std  # train 통계로 적용
```

(2) **Group leakage**:
같은 환자/사용자/제품의 sample이 train과 test 양쪽에. 모델이 ID 학습 → val에서 잘 보이지만 새 sample에선 망함.

(3) **Time leakage**:
시계열에 random split. 미래로 과거를 예측하는 비현실적 task.

(4) **Target leakage**:
미래 정보가 feature에 포함. 예: "지난 7일 평균"이 *오늘 포함*이면 target 정보 누수.

(5) **Selection bias**:
학습 데이터 수집 자체가 편향. 예: 활성 사용자만 데이터. Non-active 사용자에게 모델 적용하면 망함.

### 7.3 처방

(1) **Train만으로 통계 fit**, test에 적용만.

(2) **Group-aware split**: patient-level, user-level, product-level.

(3) **Time-based split**: 시계열은 forward chaining.

(4) **Feature engineering 점검**: 미래 정보 사용 안 하나 확인.

(5) **모델·feature 가능성 분석**: 너무 좋은 성능이면 의심.

### 7.4 자주 잡히는 패턴

데이터 분석 시 다음 점검:
- 같은 ID가 train·val·test에 모두 있나?
- 시간 순서가 보존되나?
- 정규화 통계가 train만으로?
- Feature가 미래 정보 포함?

이 4개를 통과하면 leakage 위험 大幅 감소.

### 7.5 답안 골격

> "Leakage 형태 5가지: 전처리, group, time, target, selection bias. 신호: val 성능이 비현실적, production에서 급락. 처방: (1) train만으로 통계 fit, (2) group-aware split, (3) time-based split, (4) feature 미래 정보 점검, (5) 너무 좋으면 의심. 의료·금융 등 분야별 표준 split 패턴 사용."

---

## 8. Distribution Shift

### 8.1 신호

- Val 좋은데 deploy에서 망함.
- 시간 지날수록 성능 하락.
- 새 데이터 type에서 무력.

### 8.2 종류

| 종류 | 정의 | 예시 |
|---|---|---|
| **Covariate shift** | 입력 분포 변화 | 다른 병원 X-ray |
| **Label shift** | 라벨 분포 변화 | 새 사기 수법 |
| **Concept drift** | $P(Y|X)$ 자체 변화 | 사용자 선호 변경 |
| **Selection bias** | 수집 자체 편향 | 활성 사용자만 |

### 8.3 처방

(1) **Domain adaptation**: 적은 target 데이터로 fine-tuning.

(2) **Online learning**: 실시간 데이터로 점진 update.

(3) **정기 재학습**: 매월/분기.

(4) **Drift monitoring**:
- 입력 분포 통계 추적 (mean, std, percentile).
- 예측 분포 추적.
- 실제 outcome (가능하면) 추적.
- KL divergence 등으로 분포 차이 정량.

(5) **Robust 모델**: domain adversarial training, mixup, augmentation.

### 8.4 답안 골격

> "Distribution shift 4종: covariate, label, concept drift, selection bias. 신호: 시간 따라 성능 하락. 처방: domain adaptation, online learning, 정기 재학습, drift monitoring (입력·예측 분포 추적). 의료·금융 등 분포 자주 변하는 분야에 critical."

---

## 9. 클래스 불균형

### 9.1 신호

- 다수 클래스 accuracy 99%, 소수 0%.
- F1 / recall 매우 낮음.
- 모델이 *다수 클래스만 출력*.

### 9.2 처방 다섯 축

(1) **Loss 측면 — Class Weighting**:
```python
loss = nn.CrossEntropyLoss(weight=class_weights)
```
소수 클래스 가중. 보통 inverse frequency.

(2) **Loss 측면 — Focal Loss**:
$$L_{focal} = -(1-\hat{y})^\gamma \log \hat{y}$$
어려운 sample (잘못 예측)에 가중. $\gamma = 2$.

(3) **Sampling — Oversampling**:
소수 클래스 복제. SMOTE는 합성.

(4) **Sampling — Undersampling**:
다수 클래스 줄임. 정보 손실 위험.

(5) **Threshold 조정**:
Default 0.5 대신 비즈니스 비용에 맞게. Sigmoid output을 0.3 또는 0.7 등.

### 9.3 평가 metric 변경

Accuracy 무의미. 사용:
- **PR-AUC**: 클래스 불균형 강함.
- **F1**: P/R 조화평균.
- **Recall**: FN 비용 큰 task.
- **Specificity**: 의학 통계.

### 9.4 답안 골격

> "처방 다섯 축: (1) class weighting in loss, (2) focal loss (어려운 sample), (3) oversampling/SMOTE, (4) undersampling, (5) threshold 조정. 평가는 PR-AUC, F1, recall (accuracy 무의미). 비즈니스 비용 (FN vs FP)에 따라 우선순위. 의료는 recall, 사기 탐지는 PR-AUC."

---

## 10. Loss와 Metric 괴리

### 10.1 신호

- Loss는 떨어지는데 metric (accuracy 등) 안 오름.
- 목표 metric 정체.

### 10.2 원인

(1) **Loss와 metric이 다름**: CE 줄여도 F1 안 오를 수 있음.

(2) **클래스 불균형**: accuracy는 안 오르지만 다수 클래스에 overfit.

(3) **Threshold 부적절**: 이진 분류에서.

(4) **Metric calibration 불일치**: 모델 출력이 잘 calibrated 아님.

### 10.3 처방

- Confusion matrix 출력.
- 클래스별 P/R/F1 분석.
- Loss를 metric에 더 가까운 form으로 — Focal, weighted CE.
- Threshold 조정 (PR curve 분석).
- 모델 calibration (Platt scaling, temperature).

### 10.4 답안 골격

> "Loss ≠ Metric. Loss 떨어지는데 accuracy 안 오르면 (1) 클래스 불균형, (2) threshold 부적절, (3) calibration 문제. Confusion matrix와 클래스별 P/R/F1 분석. Focal loss로 어려운 sample 강조 또는 threshold 비즈니스 비용에 맞게."

---

## 11. RNN 특화 진단

### 11.1 자주 발생

| 증상 | 처방 |
|---|---|
| Gradient NaN | Clipping (norm 1) |
| Long-term 정보 손실 | LSTM/GRU/Attention |
| Exposure bias (생성) | Scheduled sampling |
| Memory 폭발 | Truncated BPTT |
| 학습 매우 느림 | LR 재탐색, batch packing |

### 11.2 디버깅 절차

```
1. 짧은 시퀀스 (10 step)로 학습 가능?
   No → 모델/loss 버그.
   Yes → 다음.

2. 점진적으로 길이 늘림 (50, 100, 500).
   어디서 망가지나?
   100 step에서 망가지면 → vanishing/exploding.

3. Gradient norm을 *시점별*로 출력.
   초기 시점 gradient가 0에 가까우면 vanishing.
   너무 크면 exploding.

4. LSTM/GRU로 변경, gradient clipping 추가.
```

### 11.3 답안 골격

> "RNN 디버깅: 짧은 시퀀스부터, 점진적으로 길이 늘림. Gradient norm을 시점별로 추적. NaN이면 clipping 1.0, vanishing이면 LSTM/GRU. Long-term이 안 잡히면 attention 추가. Memory 폭발은 truncated BPTT (K=32~64)."

---

## 12. Transformer 특화 진단

### 12.1 자주 발생

| 증상 | 처방 |
|---|---|
| Warmup 없으면 학습 발산 | Warmup 추가 |
| Attention 폭발 | Scaled dot-product (1/√d_k) |
| Long-range 못 잡음 | Position encoding 점검 |
| Memory $O(T^2)$ 폭발 | FlashAttention, sparse attention |
| Pre-Norm vs Post-Norm | Pre-Norm이 안정 |

### 12.2 흔한 함정

**Warmup 없는 Adam**:
큰 모델 + Adam → 초반 second moment noisy → 큰 LR 위험. 처방: 5~10% warmup.

**Position Encoding 잘못**:
긴 시퀀스에서 PE의 generalization 약함. RoPE, ALiBi가 더 robust.

**Attention의 Numerical 불안정**:
$QK^T/\sqrt{d_k}$의 scale 빼면 큰 값으로 softmax saturated. 항상 scaled.

### 12.3 답안 골격

> "Transformer 디버깅: warmup 없으면 큰 모델 발산 (Adam의 second moment 통계). Scaled dot-product 필수. Long-range 약하면 PE 점검 (RoPE 등). Memory 폭발은 FlashAttention. Pre-Norm 사용 (Post보다 안정). 큰 모델은 위 모두 default 적용."

---

## 13. 큰 모델 학습 디버깅

### 13.1 메모리 OOM (Out of Memory)

처방:
- **Gradient checkpointing**: 중간 activation 재계산.
- **Mixed precision**: fp16/bf16.
- **Smaller batch + accumulation**: gradient accumulation.
- **Model parallelism**: 여러 GPU에 분산.
- **DeepSpeed ZeRO**: optimizer state 분산.

### 13.2 매우 느린 학습

- GPU 활용률 (`nvidia-smi`) — 50% 이하면 데이터 loading bottleneck.
- DataLoader의 `num_workers` 키움.
- Data prefetch.
- 작은 모델로 baseline 후 키움.

### 13.3 분산 학습 동기화

Multi-GPU에서:
- BN → SyncBN.
- Gradient: AllReduce (모든 GPU 동기화).
- Optimizer state: ZeRO로 분산.

### 13.4 답안 골격

> "큰 모델 디버깅: OOM은 gradient checkpointing/mixed precision/accumulation. 느린 학습은 GPU 활용률 점검 — 데이터 loading bottleneck이면 num_workers/prefetch. 분산은 SyncBN + AllReduce + ZeRO. 표준 framework (PyTorch DDP, DeepSpeed) 사용 권장."

---

## 14. 디버깅 체크리스트

### 14.1 학습 시작 전

- [ ] 입력 정규화 (mean 0, std 1)
- [ ] Output-loss 매칭 확인
- [ ] BN/Dropout train/eval mode 확인
- [ ] Init이 활성화에 맞음 (He for ReLU)
- [ ] Gradient clipping (RNN/Transformer)
- [ ] LR이 적절 (LR Range Test)
- [ ] Val loss 모니터
- [ ] Tiny dataset (10 sample) overfit 가능 확인
- [ ] Validation split이 leakage 없음 (group/time)
- [ ] 메트릭이 task에 적절 (accuracy 아니라 F1 등)

### 14.2 학습 중 모니터링

- Train loss curve
- Val loss curve
- Gradient norm (per layer)
- Learning rate (schedule 적용 후)
- 가능하면 activation 분포

### 14.3 학습 안 될 때

- [ ] 첫 batch 입출력 직접 확인 (NaN, 분포)
- [ ] Loss를 분해 (loss components 모두 의미)
- [ ] Gradient norm을 layer별로 출력
- [ ] LR 1/10 / 10× 시도
- [ ] 작은 모델로 baseline부터
- [ ] 데이터 누수 점검
- [ ] Tiny dataset overfit 시도

### 14.4 모범 디버깅 워크플로

```
1. Tiny dataset overfit (sanity check).
2. 작은 데이터 + 작은 모델로 빠른 iteration.
3. 점진적으로 모델·데이터 키움.
4. 매 단계에서 expected behavior 확인.
5. 망가지면 직전 변경으로 롤백.
```

---

## 15. "이런 증상엔 이걸" — 빠른 매트릭스

| 증상 | 1순위 처방 | 2순위 |
|---|---|---|
| NaN | LR 1/10 | Gradient clip, log_softmax |
| Vanishing | ReLU + He init | BN, ResNet skip |
| Exploding | Gradient clip | LR↓, init 점검 |
| Train ↓ Val ↑ | Early stop | Dropout, aug, weight decay |
| Train과 Val 둘 다 못 | LR 점검 | 모델 키움, 정규화 줄임 |
| BN train/eval 미스 | model.eval() | GN/LN 교체 |
| 클래스 불균형 | Class weight | Focal loss, resample |
| Plateau | LR decay | Scheduler 점검 |
| Train ≪ Val | model.eval() | 데이터 누수 점검 |
| Memory OOM | Gradient checkpoint | Mixed precision, accumulate |

---

## 16. 면접 단골 Q&A

### Q1. 학습 디버깅의 첫 단계?
"Tiny dataset overfit이 첫 sanity check. 데이터 10개로 train loss 0 도달 가능한지. 도달 못하면 모델·loss·코드에 버그. 매우 빠른 반복으로 큰 데이터 시작 전에 정상성 확인. 통과하면 hyperparameter 튜닝 단계로."

### Q2. Loss NaN 가장 흔한 원인?
"LR이 너무 큼 (1/10 시도)이 가장 흔함. 다음으로 log(0) — sigmoid+BCE 직접 구현 시. BCEWithLogitsLoss와 log_softmax는 수치 안정. Mixed precision의 underflow도 흔함 — loss scaling. 데이터에 NaN/inf 섞임도. 디버깅: 작은 batch forward → backward → loss 분해 단계로."

### Q3. Vanishing gradient 진단?
"Gradient norm을 per layer 출력. 깊은 layer가 $10^{-7}$ 이하면 vanishing. 처방 우선순위: ReLU + He init → BN/LN → Skip connection (ResNet). LSTM/GRU도 vanishing 완화 (덧셈 구조). Sigmoid 깊은 망에선 거의 항상 발생."

### Q4. Overfit 처방 우선순위?
"Early stopping (가장 단순) → Dropout → Weight decay → Augmentation → Label smoothing → Mixup → 모델 줄임. Gap 크기에 비례한 강도. 가장 좋은 답은 데이터 늘림. 큰 모델 + 큰 데이터 시대엔 augmentation + label smoothing이 자주 충분."

### Q5. Train도 안 떨어지면?
"Underfit 신호. 원인 7가지: 모델 작음, LR 잘못, 활성화/init 부적절, 데이터 정규화 안됨, output-loss mismatch, 정규화 강함, 학습 부족. 가장 흔한 건 입력 정규화 안 함. Tiny dataset overfit 가능한지로 코드 정상성 먼저 확인."

### Q6. Train과 Val 결과 다를 때?
"Train < Val (val이 더 좋음)이면 model.eval() 안 부른 가능성. BN의 mini-batch 통계와 dropout의 마스킹이 inference에서 비결정성. Val ≪ Train (큰 gap)이면 overfit. Val이 train보다 일관되게 좋으면 데이터 split 문제."

### Q7. Data Leakage 흔한 형태?
"5가지. (1) 전처리 통계를 train+test 합쳐 fit. (2) Group leak — 같은 ID가 train·test 모두. (3) Time leak — 시계열에 random split. (4) Target leak — feature에 미래 정보. (5) Selection bias — 수집 자체 편향. 신호는 비현실적 val 성능, deploy에서 급락. 그룹/시간 단위 split이 핵심."

### Q8. 클래스 불균형 처방?
"5축. (1) Class weight in loss, (2) Focal loss, (3) Oversampling/SMOTE, (4) Undersampling, (5) Threshold 조정. 평가는 accuracy 아닌 PR-AUC, F1, recall. 비즈니스 비용 (FN vs FP)에 따라 우선순위. 의료는 recall, 사기는 PR-AUC."

### Q9. Distribution Shift 발견했을 때?
"4종 구분: covariate (입력), label (라벨), concept (P(Y|X) 변경), selection bias. 처방: domain adaptation, online learning, 정기 재학습, drift monitoring (입력·예측·outcome 분포). 의료 (병원 간), 금융 (시간 변화)에 critical."

### Q10. RNN 학습 디버깅?
"짧은 시퀀스부터 시작, 점진적으로 길이 늘림. 어디서 망가지나 식별. Gradient norm을 시점별 추적. NaN이면 clipping 1.0, vanishing이면 LSTM/GRU 대체, long-term 안 잡히면 attention 추가. Memory 폭발은 truncated BPTT."

### Q11. Transformer 학습 발산?
"Warmup 없는 큰 모델 + Adam이 가장 흔한 원인. Adam의 second moment가 초반 noisy → 큰 LR이 발산. 처방: warmup 5~10%. 또 scaled dot-product (1/√d_k) 필수, attention saturated 방지. Pre-Norm이 Post-Norm보다 안정."

### Q12. Tiny Dataset Overfit이 왜 첫 sanity check?
"학습 코드·loss·모델 정상성 빠르게 확인. 10 sample은 30초~수 분에 학습 가능. Train loss 0 도달 못하면 모델·loss·data에 명백한 버그. 도달하면 큰 데이터로 진행. 큰 데이터 학습은 오래 걸리므로 그 전에 sanity check가 효율적."

---

## 17. 생각해보라 — 단락 답안

**Q. Gradient의 *분포*가 학습 동역학에서 의미하는 것?**

Gradient norm이 layer별로 어떻게 분포하느냐가 학습 상태의 신호:

(1) **균일한 norm**: 가장 이상적. 모든 layer가 동등하게 학습.

(2) **얕은 layer만 큼**: vanishing — 깊은 layer 학습 안 됨. 처방: skip, BN.

(3) **깊은 layer만 큼**: 흔치 않음. 보통 출력 layer가 큰 gradient. 약간 비정상.

(4) **첫 layer가 매우 큼**: input 정규화 안 됨 가능. 입력 분포 점검.

(5) **변동 매우 큼**: 학습 불안정. LR 줄임, gradient clip.

이 분포를 실시간 모니터링이 디버깅의 가장 강력한 도구. TensorBoard 또는 W&B의 histogram tab.

**Q. 왜 *Tiny Dataset Overfit*이 그토록 강력한 sanity check인가?**

세 이유:

(1) **정상 모델은 무조건 외울 수 있음**: 10 sample의 capacity 요구는 어떤 신경망도 충분. 만약 못 외우면 모델이 *해당 task를 표현 못함*이거나 *학습 자체가 망가짐*.

(2) **빠른 반복**: 큰 데이터 학습은 시간 단위, tiny는 초~분. 100번의 디버깅 사이클이 가능.

(3) **다양한 버그 노출**: data loading, loss 계산, optimizer 설정, gradient flow, BN/Dropout mode — 모든 것의 정상성 확인.

세 시나리오 분석:
- Loss 0 도달: 모든 컴포넌트 정상. 하이퍼파라미터 조정 단계로.
- Loss 정체: 모델 표현력 부족 또는 vanishing/exploding.
- Loss 진동: LR 큼 또는 코드 버그.
- NaN: log(0), 분산 폭발.

**Q. *Drift Detection*의 어려움?**

이론은 단순 — 학습 데이터 분포와 deploy 데이터 분포 비교. 실무 어려움:

(1) **Ground truth 지연**: 라벨이 늦게 도착 (사기 confirm은 며칠/주). 즉시 drift 감지 어려움.

(2) **다차원 drift**: 100차원 feature 모두 모니터링? 어떤 차원이 critical?

(3) **자연 변동 vs drift**: 일별 매출 변동은 자연. 어디서 drift로 판단?

(4) **Feedback loop**: 모델 결정이 데이터 생성에 영향. 사기 모델이 막은 거래는 confirm 안 됨 → 그 분포가 학습 데이터에서 누락.

처방:
- 입력 분포 통계 추적 (mean, percentile, KL).
- 예측 confidence 분포.
- 실제 outcome (가능하면).
- Window-based 비교 (최근 N일 vs base period).
- ML 모델로 "drift 분류기" 학습.

이게 production ML 엔지니어의 큰 일.

**Q. 왜 *Class Weighting*만으로는 클래스 불균형 충분히 해결 안 되나?**

Class weighting은 loss에 가중. 다수 클래스 sample에 작은 가중, 소수에 큰 가중. 직관적 답.

단점:
(1) **여전히 다수 클래스에 dominant exposure**: Mini-batch에서 다수 클래스 sample 수가 압도. Gradient signal이 다수 클래스 패턴에 끌림.

(2) **Calibration 왜곡**: 가중된 loss는 모델 출력을 진짜 확률로부터 멀게.

(3) **Hyperparameter**: 가중치 비율을 어떻게 정하나? Inverse frequency? Squared? 데이터마다 다름.

이래서 다른 처방과 *결합*이 표준:
- Class weighting + Focal loss: 어려운 sample에도 가중.
- Class weighting + Resampling: 균형 데이터.
- 평가 metric을 PR-AUC: accuracy 함정 회피.

각각이 *다른 측면*을 잡으므로 결합이 효과적.

**Q. *디버깅의 능력*이 ML 엔지니어의 핵심 역량인 이유?**

이론 모델 학습은 "X 데이터에 Y 모델 적용". 실무는 *그 사이*에 무수한 함정:

- 데이터 누수 (다양한 형태)
- Hyperparameter 부적절
- 코드 버그 (data loader, loss 계산, mode 전환)
- 라벨 noise
- 분포 변화
- 클래스 불균형
- BN/Dropout 실수
- Optimizer instability
- Memory OOM

이 모든 함정을 *체계적으로 진단·해결*하는 능력이 실무 ML의 차이를 만든다. 박사 ML 연구자도 코드 버그로 한 달 잃을 수 있음. 좋은 ML 엔지니어는 *증상 → 가설 → 검증 → 처방*의 흐름을 빠르게.

이 챕터의 모든 내용이 그 능력의 토대. 실무에서 매일 사용.

**Q. *Loss Curve*의 패턴 해석이 진단에 가장 빠른 길인가?**

Yes. Loss curve는 *학습 동역학의 단일 가장 풍부한 신호*. 하나의 그래프에서:

- 첫 epoch의 큰 감소: 모델이 데이터를 *어느 정도* 외움 (정상).
- 점진적 감소: 학습 진행 (정상).
- 진동: LR 큼.
- Plateau: LR 작음 또는 model 부족.
- 갑작스런 점프: gradient explosion.
- NaN: 즉시 진단.
- Train ↓ Val ↑: overfit.
- Train ↑ Val ↑ stuck: underfit.

이 모든 것을 한 그래프에서 읽음. TensorBoard/W&B를 매번 봐야 함. 좋은 ML 엔지니어는 *loss curve를 보자마자 무엇이 잘못되었는지* 70% 정도 진단.

물론 다른 신호도 (gradient norm, activation 분포)와 합쳐야 정확. 단 loss curve가 가장 빠른 첫 진단 단계.

**Q. *Reproducibility*가 디버깅에 어떻게 영향?**

같은 코드·같은 데이터·같은 hyperparameter가 *다른 결과*면 디버깅 매우 어려움.

원인:
(1) **Random seed**: weight init, data shuffle, dropout mask.
(2) **Non-deterministic ops**: 일부 cuDNN 연산이 비결정.
(3) **Floating point**: 다른 reduction 순서로 약간 다른 값.
(4) **Multi-GPU**: communication 순서 차이.

처방:
```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
random.seed(42)
```

단 deterministic은 약간 느림 (cudnn benchmark 끔). 디버깅 중에만 사용.

또 *여러 seed로 평균* — 단일 결과의 noise 줄이기. 논문 reporting 표준.

---

## 18. 한 줄 요약

- **모든 학습 실패 = 증상 → 원인 → 처방 패턴**.
- **Tiny dataset overfit**으로 sanity check 첫.
- **Loss NaN**의 가장 흔한 원인 = LR 큼, log(0). 처방: 1/10, log_softmax.
- **Vanishing**: gradient norm per layer 진단. ReLU + He + BN + Skip.
- **Exploding**: gradient clipping 표준.
- **Overfit**: early stop → dropout → WD → aug → mixup.
- **Underfit**: 입력 정규화 점검이 가장 흔한 원인.
- **BN/Dropout**: model.eval() 항상.
- **Leakage**: 그룹·시간 단위 split, train만으로 통계 fit.
- **클래스 불균형**: PR-AUC, focal loss, class weight + threshold.
- **RNN/Transformer 디버깅** 패턴 별도.
- 진단의 핵심 도구 = loss curve + gradient norm + 출력 분포.
