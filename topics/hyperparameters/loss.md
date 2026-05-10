# Loss hyperparameters — Huber δ, Focal γ, Label smoothing α, Triplet margin, Contrastive τ

> 본문 §2 Loss function의 보강. 분포·task 특성에 따라 달라지는 loss 내부 hyperparameter들. 본문에서는 한 줄 요약만 두고 여기서 깊게.

---

## Huber δ

**위치 — 본문 §2.2 회귀 loss**

Huber loss는 작은 오차엔 MSE, 큰 오차엔 MAE인 절충:

$$
L_\delta(y, \hat{y}) =
\begin{cases}
\tfrac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le \delta \\[2pt]
\delta |y - \hat{y}| - \tfrac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

$\delta$가 squared/linear 전환 임계값.

### 표준값

- *Default*: $\delta = 1.0$. 데이터를 표준화(평균 0, 분산 1)했다는 가정 하의 자연스러운 시작점 — 잔차의 표준편차 정도가 squared/linear 전환 임계값.
- *Outlier가 많은 경우*: $\delta$ 작게 (0.3–0.5). 작은 오차 영역이 줄어 더 빠르게 robust(linear) 영역으로 전환.
- *Outlier가 적고 가우시안에 가까운 경우*: $\delta$ 크게 (3–5). 거의 MSE에 가까워져 부드러운 학습.
- *극단*: $\delta \to \infty$이면 MSE, $\delta \to 0$이면 (스케일된) MAE.

### 왜 1이 좋은 시작점인가

데이터를 정규화했다면 잔차의 1 standard deviation 정도가 *normal* 영역, 그 이상은 outlier로 보는 게 통계적으로 자연스럽다. 정규화 안 한 raw scale이면 잔차 분포의 IQR(사분위수 범위) 또는 중앙 절대편차(MAD)에 맞춰 δ를 조정.

### 함정

- δ를 너무 작게 잡으면 *거의 모든 점에 linear 영역 적용 → MAE와 다를 바 없어짐*. Huber의 부드러움 이점이 사라진다.
- δ를 LR과 함께 동시에 바꾸지 말 것 — 두 효과가 섞여 분리가 어렵다.

---

## Focal γ

**위치 — 본문 §2.4 분류 loss 변종**

Focal loss는 클래스 불균형에 강한 분류 loss. 쉽게 맞히는 sample에 가중치를 줄이고 어려운 sample에 집중:

$$L_{\text{focal}} = -(1 - \hat{y})^\gamma \log \hat{y}$$

$\gamma$가 어려운 sample에 얼마나 집중할지 결정.

### 표준값

- *Default*: γ = 2 — RetinaNet 논문의 권장. 잘 맞힌 sample의 weight를 원래의 1% 정도로 낮춤.
- γ = 0 — 일반 cross-entropy와 동일. focal 효과 없음.
- γ = 1 — 약한 focusing. 클래스 불균형이 가벼울 때.
- γ = 5 — 강한 focusing. background 비율이 99.9% 이상으로 극단적일 때.

### 언제 키우고 줄이는가

- 클래스 불균형이 심할수록 γ↑. 의료 segmentation처럼 양성 픽셀이 0.1% 미만이면 γ=2–3.
- 라벨 noise가 많으면 γ↓ 또는 0 — 어려운 sample에 집중하면 라벨이 잘못된 sample도 외워버려 일반화 손해.
- 보통 *α-balanced focal loss*로 같이 씀: $-\alpha (1-\hat{y})^\gamma \log\hat{y}$. α는 클래스별 가중치, γ는 어려움 가중치.

### 함정

- γ만 키우고 LR을 그대로 두면 effective gradient가 작아져 학습이 매우 느려짐. γ↑ 시 LR도 살짝 올리는 게 보통.
- 라벨 noise + 큰 γ = 잘못 라벨된 sample 외움. 데이터 정제가 우선.

---

## Label smoothing α

**위치 — 본문 §2.4 분류 loss 변종**

정답을 one-hot 대신 약간 smooth하게 — 정답 클래스에 $1-\alpha$, 나머지 K-1개 클래스에 $\alpha/(K-1)$씩 분배.

### 표준값

- *Default*: α = 0.1. 정답 클래스에 0.9, 나머지 K-1개 클래스에 0.1/(K-1)씩 분배.
- α = 0 — hard one-hot. label smoothing 없음.
- α = 0.05 — 약한 smoothing. 데이터가 많고 깨끗할 때.
- α = 0.2 이상 — 강한 smoothing. 라벨 noise가 의심되거나 매우 큰 모델일 때. *그러나 0.3 이상은 거의 안 씀* — 모델이 정답에 60% 정도 확률만 줘도 OK가 되어 정확도가 손해.

### 왜 0.1인가

ImageNet/Transformer 등 대규모 학습에서 0.1이 가장 안정적이라 보고됨. 원래 NMT(neural machine translation)에서 처음 쓰여 표준화. 더 크면 underfit, 더 작으면 효과 미미.

### 함정

- knowledge distillation에서 teacher가 soft label을 만들면 label smoothing은 *역효과* — 이중 smoothing이라 정보 손실.
- model calibration이 중요하면 (예: 의료 진단) label smoothing이 confidence를 *부드럽게* 해 도움이 되지만, *정확한 확률*이 필요하면 temperature scaling 같은 사후 calibration이 더 깔끔.

---

## Triplet margin

**위치 — 본문 §2.4 분류 loss 변종**

Triplet loss는 메트릭 학습용. (anchor, positive, negative) 셋을 받아 anchor-positive 거리는 가깝게, anchor-negative 거리는 멀게:

$$L_{\text{triplet}} = \max(0, d(a, p) - d(a, n) + \text{margin})$$

### 표준값

- *Default*: margin = 0.2 (정규화된 임베딩 기준, FaceNet 권장). 임베딩이 단위 구 위에 있으면 거리가 [0, 2] 범위라 0.2가 충분히 의미 있는 분리.
- *작은 margin* (0.05–0.1): 학습 신호 약함, 거의 모든 triplet이 만족해 gradient 0. 학습 정체.
- *큰 margin* (0.5–1.0): 더 강한 분리 요구, 어려운 triplet을 강하게 학습하지만 수렴 어려움.
- 임베딩이 정규화 안 됐으면 데이터 스케일에 비례해 조정 — 잔차 분포의 1 standard deviation 정도가 시작점.

### 중요한 짝 — triplet mining

무작위 (a, p, n)은 대부분 *easy triplet*이라 학습 신호 없음. *Semi-hard mining* (anchor-positive보다 약간 멀지만 margin 안에 있는 negative만 선택)이 표준.

---

## Contrastive temperature τ

**위치 — 본문 §2.4 분류 loss 변종**

Contrastive Loss (SimCLR식 자기지도 학습)는 같은 sample의 두 augmentation은 가깝게, 다른 sample은 멀게. NT-Xent (normalized temperature-scaled cross entropy) loss가 표준 form. τ가 *어려운 negative*에 얼마나 집중할지 결정.

### 표준값

- *Default*: τ = 0.1 (SimCLR), τ = 0.07 (MoCo). 작을수록 어려운 negative에 집중.
- τ → 0: hard mining과 비슷한 효과. 어려운 sample에 강한 gradient. 단 너무 작으면 gradient 폭발.
- τ → ∞: 모든 negative가 비슷하게 취급, 학습 신호 약함.
- τ는 batch size와 함께 묶여 효과 — batch size↑이면 negative 수가 많아져 τ↑ 가능. 일반적으로 batch 256 이상에서 0.1, 4096 이상에서 0.2.

---

## 본문 연결

이 sub-page는 본문 §2의 다음 자리에서 호출된다.

- §2.2 회귀 loss → Huber δ
- §2.4 분류 loss 변종 → Focal γ, Label smoothing α, Triplet margin, Contrastive τ
