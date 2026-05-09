# Regularization hyperparameters — Weight decay λ, Dropout p

> 본문 §5.3 정규화의 보강. L2 weight decay와 dropout의 강도 결정. 본문에서는 한 줄 요약만 두고 여기서 깊게.

---

## Weight decay λ (L2 정규화)

**위치 — 본문 §5.3 정규화 5가지 전략**

L2 정규화는 loss에 $\lambda \|\theta\|^2$를 더해 가중치를 작게 유지. 큰 가중치가 작은 입력 변화에 큰 출력 변화를 일으키므로, 가중치를 작게 두면 함수가 부드러워진다.

수학적으로 SGD update에 추가 항이 들어간다:

$$\theta \leftarrow \theta - \eta(\nabla L + 2\lambda\theta) = (1 - 2\eta\lambda)\theta - \eta\nabla L$$

매 step 가중치가 살짝 0으로 끌려간다.

### 표준값

- *일반 신경망*: $\lambda = 10^{-4}$가 universal default. $\eta = 10^{-3}$, $\lambda = 10^{-4}$이면 매 step 가중치 norm이 약 0.0001 비율로 줄어든다.
- *Transformer/AdamW*: $\lambda = 10^{-2}$ 또는 $10^{-1}$로 훨씬 크다. AdamW가 weight decay를 정확히 처리하므로 큰 값을 넣어도 안전 (본문 §3.7).
- *작은 모델·작은 데이터*: λ 키움 (1e-3) — 정규화 강하게.
- *큰 데이터(>1M)·foundation model*: 오히려 큰 weight decay + 약한 dropout이 표준 패턴.

### 왜 그 정도 값인가

너무 강하면 underfit (가중치가 너무 빨리 0으로 끌려가 함수가 지나치게 부드러워짐), 너무 약하면 정규화 효과 없음. 1e-5 ~ 1e-1의 log scale로 sweep해 sweet spot을 찾는 게 일반적.

### 함정

- bias나 BN의 scale·shift 파라미터에는 weight decay를 *적용하지 말 것* — 이들이 0으로 끌려가면 표현력이 망가진다. PyTorch 기본은 모든 파라미터에 적용하므로, optimizer param group을 분리해 BN/bias에는 `weight_decay=0` 권장.
- 일반 Adam의 weight decay는 수학적으로 부정확하다 — AdamW 사용 (본문 §3.7).

---

## Dropout p

**위치 — 본문 §5.3 정규화 5가지 전략**

학습 중 일부 뉴런을 확률 $p$로 무작위 마스킹. 매 step 다른 sub-network. 평가 시 모든 뉴런 사용 (대신 활성값을 $1-p$로 스케일).

### 표준값

- *FC layer*: $p = 0.5$가 historical default (Hinton 2012). 절반 마스킹이 sub-network의 다양성을 가장 크게 만든다는 직관 + 큰 dropout이 큰 모델·작은 데이터 조합에서 효과적.
- *Conv layer*: $p = 0.1$ – $0.2$. 작게 잡는 이유는 conv가 이미 weight sharing으로 *암묵적 정규화*를 갖고 있어 추가 dropout이 적게 필요. 또 conv 출력은 spatial 구조라 한 채널을 통째로 끄는 *spatial dropout*이 더 자연스럽다는 분석.
- *Input layer*: $p = 0.1$ – $0.2$. 입력 augmentation 비슷한 효과지만 너무 강하면 정보 손실.

### 언제 키우고 줄이는가

- 데이터 적고 overfit 심함 → p 키움 (FC 0.5 → 0.7, conv 0.2 → 0.3).
- 데이터 매우 큼(>1M) → p 줄이거나 0. *Foundation model 시대(>10M)에선 $p = 0$이 흔하다* — dropout보다 weight decay + augmentation이 우세.

### 함정

- p가 너무 크면 underfit. train loss가 아예 떨어지지 않음.
- *Test 시 `model.eval()`을 잊으면 dropout이 켜진 채라 결과가 매번 들쭉날쭉* (본문 §9.1).
- LR·batch와 함께 dropout을 동시에 바꾸지 말 것 — ablation 신뢰도 떨어짐.

---

## 데이터 크기에 따른 정규화 강도 표 (참고)

본문 §5.4의 표 — 두 hyperparameter를 함께 결정할 때:

| 데이터 크기 | Dropout p | Weight decay λ | Augmentation |
|---|---|---|---|
| < 10K | 0.5 | 1e-3 | 강하게 |
| 10K – 100K | 0.2 – 0.3 | 1e-4 | 중간 |
| > 100K | 0.1 | 1e-5 | 약하게 |
| > 10M (foundation) | 0 | 1e-2 | label smoothing |

직관: 데이터 많으면 자연 정규화가 충분 (다양한 sample이 noise를 averaging). 데이터 적으면 외부 정규화로 보강.

---

## 본문 연결

- §5.3 정규화 깊이 → Weight decay λ, Dropout p
- §5.4 정규화 강도 → 위 표
- §3.7 AdamW → weight decay 정확한 처리
- §9.1 Train vs Eval → dropout test 시 `model.eval()`
