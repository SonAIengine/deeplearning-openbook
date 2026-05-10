# 01. 잔차의 기하학 — MSE를 정사각형의 면적으로 보기

> 이 장의 목표 — 회귀 loss의 *수식*이 아니라 *그림*에서 무엇을 의미하는가. MSE/MAE/Huber를 한 산점도 위에 그려 보고, 각각 *어떤 도형의 합*인지를 명확히 하면 outlier 민감성 같은 직관이 즉시 이해된다.

---

## 1.1 출발점 — 산점도와 잔차

회귀의 가장 단순한 그림은 다음이다.

- *데이터 점들* — 산점도에 흩어진 빨간 점들 $(x_i, y_i)$
- *모델 (회귀선)* — 데이터를 가로지르는 파란 직선 $\hat{y}(x) = ax + b$
- *잔차* — 각 데이터 점에서 회귀선까지의 *수직* 거리 $r_i = y_i - \hat{y}(x_i)$

<img src="../../assets/images/linear_least_squares_residuals.svg" alt="Scatter plot with linear least-squares fit and vertical green residual segments connecting data points to the fitted line" width="520">

> *빨간 점: 데이터. 파란 선: 최소제곱으로 fit한 회귀선. 초록 수직 선분: 각 점의 잔차 $r_i = y_i - \hat{y}(x_i)$. 어떤 점은 회귀선 위(잔차 양수), 어떤 점은 아래(잔차 음수). 회귀의 목표는 이 잔차들의 *어떤 통계량*을 최소화하는 직선을 찾는 것.*  
> *Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Linear_least_squares_example2.svg), Krishnavedala, CC BY-SA 3.0 / GFDL.*

이 그림에서 잔차는 *직선 길이*로 보인다. MSE/MAE/Huber는 이 *직선 길이를 어떻게 처리해서 합치느냐*의 차이다.

---

## 1.2 MSE = 정사각형의 면적 합

MSE의 정의는:

$$L_{\text{MSE}} = \frac{1}{N} \sum_i (y_i - \hat{y}_i)^2 = \frac{1}{N} \sum_i r_i^2$$

각 잔차 $r_i$를 *제곱*해서 더한다. 기하학적으로 이건 **잔차를 한 변으로 하는 정사각형의 면적**이다.

머릿속으로 위 산점도 그림에 다음을 추가하자.

- 각 잔차 선분(초록색 수직선)을 *한 변*으로 하는 *정사각형*을 그린다.
- 정사각형이 점 옆에 매달려 있는 모습. 짧은 잔차는 작은 정사각형, 긴 잔차는 큰 정사각형.

**MSE 최소화 = 모든 정사각형의 *총 면적*을 최소로 만드는 회귀선 찾기.**

이 시각이 중요한 이유:

### 왜 MSE는 outlier에 끌려가는가

잔차가 1인 점 100개와 잔차가 10인 점 1개가 있다고 하자.

- 잔차 1짜리 정사각형 면적: $1^2 = 1$. 100개의 합 = 100.
- 잔차 10짜리 정사각형 면적: $10^2 = 100$. 1개의 합 = 100.

**outlier 한 점이 정상 sample 100개와 같은 영향**을 준다. 회귀선이 outlier 쪽으로 *기울어진다* — 그 한 점의 큰 정사각형을 줄이기 위해서.

이게 "MSE는 outlier에 매우 민감"이라는 표현의 *기하학적* 의미. 제곱이 잔차의 영향력을 *비선형적으로* 키운다 — 잔차가 10배면 면적은 100배.

---

## 1.3 MAE = 직선 길이의 합

MAE의 정의는:

$$L_{\text{MAE}} = \frac{1}{N} \sum_i |y_i - \hat{y}_i| = \frac{1}{N} \sum_i |r_i|$$

각 잔차의 *절댓값*을 더한다. 기하학적으로는 **잔차 선분의 길이를 그대로 합한 것**.

위 산점도에서 *초록 수직선들의 총 길이*가 MAE에 비례.

### 왜 MAE는 robust하지만 부드럽지 못한가

같은 outlier 예를 다시 보자.

- 잔차 1짜리 직선 길이: 1. 100개의 합 = 100.
- 잔차 10짜리 직선 길이: 10. 1개의 합 = 10.

**outlier 한 점이 정상 sample 10개와 같은 영향**. MSE의 1대 100과 비교하면 영향이 *훨씬 작음*. 그래서 robust.

단점: 0에서 *꺾임*. 잔차가 정확히 0인 점에서 도함수가 정의되지 않아 학습 동역학이 부드럽지 못하다. 또 minimizer가 평균이 아니라 *중앙값*이라 분포 모양에 따라 평균과 다른 곳을 학습.

---

## 1.4 Huber = 두 도형의 절충

Huber의 정의는:

$$L_\delta(r) = \begin{cases} \frac{1}{2}r^2 & |r| \le \delta \\ \delta |r| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

기하학적 해석:

- 잔차 *작으면* (|r| ≤ δ): MSE처럼 *정사각형 면적* (단 1/2 곱셈으로 부드럽게)
- 잔차 *크면* (|r| > δ): MAE처럼 *직선 길이*에 비례 (단 정사각형과 매끄럽게 이어지도록 상수 항 보정)

위 산점도에서 그림으로 표현하면:

- δ보다 짧은 잔차: 정사각형 모양
- δ보다 긴 잔차: 직선 모양

그래서 Huber는 *데이터 대부분*에서는 MSE의 부드러움을 가지고(작은 잔차 영역), *outlier*에서는 MAE의 robustness를 가진다(큰 잔차 영역).

### Loss curve로 본 Huber

잔차 길이만 보지 말고 *loss 자체의 모양*도 보자.

<img src="../../assets/images/huber_loss.svg" alt="Huber loss (green) vs squared error loss (blue) plotted against the residual y - f(x)" width="520">

> *x축은 잔차 $r$, y축은 loss 값. 파란 곡선(MSE)은 잔차가 커질수록 *제곱*으로 폭발. 녹색 곡선(Huber, $\delta = 1$)은 |r| ≤ δ 구간에서 MSE와 거의 같은 부드러운 곡선이지만, |r| > δ에서는 *직선*으로 꺾인다. MAE는 이 그림에 없지만 모든 영역에서 직선 — Huber의 큰 잔차 영역과 같은 모양.*  
> *Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Huber_loss.svg), Qwertyus, CC BY-SA 4.0.*

이 그림과 위 산점도 그림을 함께 머리에 두면, *왜 Huber가 절충인지*가 두 시각으로 동시에 잡힌다.

---

## 1.5 한 표로 정리

| Loss | 잔차를 어떻게 처리 | 기하학적 도형 | outlier 영향 (잔차 10) | minimizer |
|---|---|---|---|---|
| **MSE** | 제곱 $r^2$ | 정사각형 (변 = $\|r\|$) | $10^2 = 100$ | 평균 |
| **MAE** | 절댓값 $\|r\|$ | 직선 (길이 = $\|r\|$) | $10$ | 중앙값 |
| **Huber (δ=1)** | δ 안: 제곱 / δ 밖: 직선 | 정사각형(작은 r) + 직선(큰 r) | $\delta\|r\| - \frac{1}{2}\delta^2 = 9.5$ | 평균에 가까움 |

이 한 표가 §2.2의 핵심. 본문은 이 정도로 압축하고 깊은 직관은 이 페이지에서.

---

## 1.6 한 줄 요약

- 회귀 loss는 *잔차를 어떤 도형으로 만들어 합치느냐*로 갈린다.
- MSE = 정사각형의 면적 합 → 큰 잔차의 영향이 *제곱*으로 폭발 → outlier 민감.
- MAE = 직선의 길이 합 → 잔차 영향이 *선형* → robust, 단 0에서 꺾임.
- Huber = δ 안에서는 정사각형, δ 밖에서는 직선 → 두 장점의 절충.

---

## 다음 장 (작성 예정)

- 02. Outlier sensitivity 시뮬레이션 — outlier 한 점을 움직일 때 MSE/MAE/Huber 회귀선이 얼마나 끌려가는지
- 03. Noise 모델 가정 — 가우시안 → MSE, 라플라스 → MAE의 통계적 정당성
- 04. 다변량·비선형 회귀에서 잔차 정의

---

## 본문 연결

- §2.2 MSE/MAE/Huber 정의 → 이 장이 그 시각적 직관
- §2.5 Loss 선택 사고 흐름 → §1.5 표가 빠른 참조

## 참고문헌

- Hastie, Tibshirani, Friedman. *The Elements of Statistical Learning* (2009). Chapter 3.
- Wikipedia: [Least squares](https://en.wikipedia.org/wiki/Least_squares), [Huber loss](https://en.wikipedia.org/wiki/Huber_loss)
