# 03. 분포 진단의 통계적 기초

> 이 장의 목표 — 그래프 너머의 정량적 진단 도구. 평균·분산이 부족하다는 건 2장에서 봤다. 그러면 *무엇을* 측정하면 분포를 안다고 할 수 있는가? 모멘트, 정규성 검정, Q-Q plot, 그리고 흔한 분포 family를 정리한다.

---

## 3.1 분포의 4가지 모멘트

분포를 *수치로* 묘사하는 표준은 4개의 모멘트(moment)다.

### 1차: 평균 (Mean) — 중심

$$\mu = E[X]$$

분포의 *중심 위치*. 단점: outlier에 매우 민감. 한 점이 100배 크면 평균이 끌려감.

대안 — **중앙값(median)**: outlier에 robust. skewed 분포에서 median ≠ mean이고, 보통 *median이 분포의 진짜 중심에 더 가깝다*.

### 2차: 분산 (Variance) — 퍼짐

$$\sigma^2 = E[(X - \mu)^2]$$

분포의 *퍼짐 정도*. 표준편차 $\sigma$가 같은 단위.

대안 — **IQR (interquartile range)**: Q3 - Q1, 가운데 50%의 폭. outlier robust.

### 3차: 왜도 (Skewness) — 비대칭성

$$\text{Skew}(X) = E\left[\left(\frac{X-\mu}{\sigma}\right)^3\right]$$

- Skew = 0 → 좌우 대칭 (정규분포 같은 것)
- Skew > 0 → 오른쪽으로 꼬리가 길다 (right-skewed). 평균 > 중앙값
- Skew < 0 → 왼쪽으로 꼬리가 길다 (left-skewed). 평균 < 중앙값

ML 데이터에서 skewness가 큰 경우(|skew| > 1) → **변환을 고려**.

- right-skewed → log, sqrt, box-cox
- left-skewed → exp, square (드묾)

### 4차: 첨도 (Kurtosis) — 꼬리 두께

$$\text{Kurt}(X) = E\left[\left(\frac{X-\mu}{\sigma}\right)^4\right]$$

- 정규분포의 kurtosis = 3 (또는 *excess kurtosis* = 0 표기)
- Kurtosis > 3 (heavy-tailed): 꼬리에 outlier가 자주. 가우시안 가정 위반
- Kurtosis < 3 (light-tailed): 꼬리가 얇음 (uniform 분포 등)

ML에서 kurtosis가 크면 → MSE 위험. MAE/Huber 또는 robust loss로 (본문 §2.2).

---

## 3.2 정규성 진단 — Q-Q plot

가장 강력한 도구는 **Q-Q plot** (quantile-quantile plot)이다.

### 원리

데이터의 분위수(quantile)를 정규분포의 분위수와 *직접 비교*. 데이터가 정규분포면 점들이 정확히 직선에 놓인다. 정규분포가 아니면 직선에서 벗어남.

### 패턴 읽기

- **직선**: 정규분포에 부합
- **S자 모양** (양 끝이 꺾임): heavy tail (kurtosis 큼)
- **반대 S자**: light tail
- **위쪽 꼬리만 위로**: right-skewed
- **아래쪽 꼬리만 아래로**: left-skewed
- **계단 모양**: 이산 데이터 또는 결측값

### 코드

```python
import scipy.stats as stats
import matplotlib.pyplot as plt

stats.probplot(data, dist="norm", plot=plt)
plt.show()
```

ML에서 Q-Q plot을 *반드시* 봐야 하는 곳:

1. **회귀 잔차** — 잔차가 정규분포면 MSE 정당, 아니면 Huber/MAE
2. **수치 feature** — 정규성이 깨졌으면 변환 검토
3. **target 변수** (회귀) — skewed면 log/box-cox

---

## 3.3 정규성 검정 (formal tests)

Q-Q plot이 직관적이지만 *수치 결정*이 필요하면 통계 검정.

### Shapiro-Wilk test

가장 강력한 정규성 검정. n < 5000일 때 표준.

```python
from scipy.stats import shapiro
stat, p = shapiro(data)
# p < 0.05면 정규분포가 아니라고 결론
```

### Kolmogorov-Smirnov test (KS test)

데이터의 누적분포(empirical CDF)와 정규분포 CDF의 *최대 차이*를 측정. Shapiro보다 약하지만 더 일반적 (정규 외 분포에도 적용 가능).

```python
from scipy.stats import kstest
stat, p = kstest(data, 'norm')
```

### Anderson-Darling test

KS의 강화 버전 — 꼬리 영역에 더 민감. heavy-tail 감지에 좋다.

### 주의 — 큰 n에서는 *모든* 검정이 reject

n = 10⁶이면 *아주 작은* 정규성 위반도 p < 0.05를 만든다. 큰 데이터에서 정규성 검정의 p값은 *거의 무의미*.

→ 큰 데이터에선 **Q-Q plot의 시각적 평가**가 더 신뢰. 또는 **effect size** (예: skew 절대값) 자체를 본다.

---

## 3.4 흔한 분포 family와 ML 처방

각 분포가 어떤 모양이고, ML에서 어떻게 처리하는가.

### (1) Normal (Gaussian)

$$X \sim \mathcal{N}(\mu, \sigma^2)$$

- 종 모양, 좌우 대칭, 두 모멘트로 완전히 묘사
- 자연·생물·측정 오차에 흔함 (**CLT**, central limit theorem — 여러 독립 random 변수의 합은 정규분포에 수렴. 그래서 "여러 요인의 합"인 noise는 자연히 정규에 가까워짐)
- **처방**: MSE loss, StandardScaler, 거의 모든 통계 기법 자연 적용

### (2) Log-normal

$$\log X \sim \mathcal{N}$$

- 양수만, right-skewed, heavy tail
- 소득, 인구, 입자 크기, 응답 시간
- **처방**: log 변환 후 정규로 — 그 후 MSE 가능. 또는 raw에 MAE/Huber.

### (3) Power-law / Pareto

$$P(X > x) \propto x^{-\alpha}$$

- 매우 heavy tail. 평균이 발산할 수도
- 도시 인구, 부의 분포, 단어 빈도 (**Zipf 법칙** — 자연어에서 k번째 흔한 단어의 빈도가 1/k에 비례), social network degree
- **처방**: log-log plot에서 직선이 보이면 power-law. 변환은 한계 — robust loss + capping(상하위 percentile 자르기) 또는 별도 모델

### (4) Exponential

$$X \sim \text{Exp}(\lambda)$$

- 양수, monotone 감소, **memoryless** (이미 t시간 대기했어도 추가 대기시간은 처음과 같은 분포)
- 대기 시간, decay
- **처방**: log 변환 또는 **Poisson regression** (count(횟수) 데이터에 자연 — y가 0,1,2,... 정수일 때 Poisson 분포 likelihood로 학습)

### (5) Bimodal (두 봉우리)

- 두 가우시안의 mixture (또는 더 복잡)
- 두 그룹이 섞인 경우 (남/여, 정상/이상)
- **처방**: cluster를 *별도 feature*로 추가하거나, 두 그룹을 각각 모델링 (**mixture model** — 여러 분포의 가중합으로 데이터를 설명, 각 sample이 어느 분포에서 왔는지 latent 변수로)

### (6) Zero-inflated

- 0이 매우 많고 (전체의 30%+), 0이 아닌 값은 다른 분포
- 강수량 (대부분 0), 지출액, 사기 거래
- **처방**: zero-inflated Poisson/negative binomial. 또는 **hurdle model** — *0인지 아닌지를 먼저 분류하고, 0이 아니면 그 값에 회귀를 따로 적용*하는 2단계 모델

### (7) Heavy-tailed + 비대칭 (Pareto, Cauchy, t-분포)

- outlier가 자주
- 금융 수익률, 보험 청구
- **처방**: Cauchy/t-분포 likelihood, robust loss (Huber, **Tukey biweight** — 큰 잔차를 *완전히 0으로* 처리하는 loss. Huber보다 더 robust). MSE 절대 안 됨

---

## 3.5 분포 변환 (transformation) 가이드

Skewed/heavy-tail 변수를 정규에 가깝게 만드는 표준 도구들.

### Log transformation

$$y = \log(x + 1)$$

- Right-skewed의 표준 처방
- 양수만 가능 (또는 +1로 0 안전)
- 직관: 곱셈성 노이즈를 덧셈성으로

### Square root

$$y = \sqrt{x}$$

- 약한 skew. count data
- log보다 약한 효과

### Box-Cox

$$y = \begin{cases} \frac{x^\lambda - 1}{\lambda} & \lambda \neq 0 \\ \log x & \lambda = 0 \end{cases}$$

- 양수만 가능
- $\lambda$ 자동 결정 (MLE)
- log·sqrt를 일반화. 가장 강력한 단변량 정규화 도구

### Yeo-Johnson

- Box-Cox의 일반화. 음수도 가능
- ML에서 음수 feature가 흔하므로 더 자주 사용

### Quantile transformation

- empirical CDF를 정규 분위수로 매핑
- 분포 모양과 무관하게 정규로 만듦
- 단점: 새 데이터의 quantile을 다시 계산해야 함

### Coding

```python
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

# Box-Cox or Yeo-Johnson
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X)

# Quantile to normal
qt = QuantileTransformer(output_distribution='normal')
X_qt = qt.fit_transform(X)
```

---

## 3.6 분포 비교 — 두 분포가 얼마나 다른가

train과 test의 분포가 같은지, 또는 시간에 따라 drift가 있는지 비교할 때.

### Two-sample KS test

```python
from scipy.stats import ks_2samp
stat, p = ks_2samp(train_data, test_data)
# p < 0.05면 분포가 다름
```

### Kullback-Leibler divergence

$$D_{KL}(P \,\|\, Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$$

- 비대칭 (P와 Q 순서 중요)
- 0이면 같은 분포
- 정보 이론적 distance

### Wasserstein (Earth Mover's) distance

- "한 분포를 다른 분포로 옮기는 데 필요한 일"
- 대칭, metric 성질 모두 만족 (KL과 달리 진정한 거리 함수)
- 최근 ML에서 인기 (**WGAN** — Wasserstein GAN, 분포 거리를 Wasserstein으로 두어 GAN 학습 안정화)

### PSI (Population Stability Index) — 산업 표준

$$\text{PSI} = \sum_i (a_i - b_i) \log\frac{a_i}{b_i}$$

- 금융·보험 업계의 drift monitoring 표준
- < 0.1: 분포 안정
- 0.1–0.25: 약한 drift, 주시
- \> 0.25: 강한 drift, 재학습 필요

---

## 3.7 ML 결정 흐름 — 분포 진단 → 처방

| EDA 결과 | 처방 |
|---|---|
| Skew |X| < 0.5, 정규 Q-Q OK | StandardScaler + MSE 그대로 |
| Skew > 1, right-skewed | log 또는 Yeo-Johnson 변환 후 MSE |
| Heavy tail (kurtosis > 5) | Huber/MAE loss, RobustScaler |
| Bimodal | cluster를 feature로, 또는 mixture model |
| Zero-inflated | hurdle 또는 zero-inflated 모델 |
| Power-law (log-log 직선) | 변환 한계 → robust loss + capping |
| train vs test PSI > 0.25 | drift 의심 — split 재검토 또는 domain adaptation |

---

## 3.8 흔한 함정

### (1) `describe()`의 평균만 보기

평균과 중앙값을 *같이* 봐야 한다. 차이 크면 skewed.

### (2) skew·kurtosis를 sample size 보정 없이 비교

작은 n에서는 skew·kurtosis 추정 자체가 noisy. n < 100이면 이 통계량을 신뢰하지 마라.

### (3) 정규성 검정 p값 맹신

큰 n에서는 *모두 reject*되고, 작은 n에서는 *모두 통과*된다. 시각적 Q-Q plot이 더 신뢰.

### (4) 변환을 train+test 합쳐서 적합

train만으로 변환 파라미터 ($\lambda$ 등)를 적합하고 test에 *적용*만. 합치면 leak (본문 §6.5).

### (5) target 변환 후 metric 해석 주의

회귀에서 target에 log를 적용했으면, RMSE는 *log scale*에서의 오차. raw scale로 보고하려면 역변환 후.

---

## 3.9 한 줄 요약

- 분포는 4개 모멘트(평균·분산·skew·kurtosis)와 *시각적 Q-Q plot*으로 진단.
- 정규에 가까우면 MSE·StandardScaler 자연. 멀면 log/box-cox/yeo-johnson 변환 또는 robust loss.
- 큰 n에서는 정규성 검정 p값 무의미 — 시각적 평가에 의존.
- train·test 분포 비교는 KS test, KL, Wasserstein, PSI 중 하나로.
- 변환은 *반드시 train으로 적합, test에 적용만* (leak 방지).

---

## 다음 장

→ [04. 결측치 — Rubin의 분류와 처리 전략](04_missing_data.md): MCAR/MAR/MNAR과 imputation 기법 8가지.

---

## 참고문헌

- Box, G. E. P. & Cox, D. R. (1964). *An analysis of transformations*. JRSS-B.
- Yeo, I. K. & Johnson, R. A. (2000). *A new family of power transformations*. Biometrika.
- Shapiro, S. S. & Wilk, M. B. (1965). *An analysis of variance test for normality*. Biometrika.
- scikit-learn: [`PowerTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html), [`QuantileTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)
