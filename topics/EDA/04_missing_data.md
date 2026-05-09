# 04. 결측치 — Rubin의 분류와 처리 전략

> 이 장의 목표 — 결측치를 *왜* 잃었는지에 따라 *어떻게* 처리해야 하는지가 갈린다. Donald Rubin의 1976년 분류(MCAR/MAR/MNAR)가 이 모든 결정의 토대다. 그 위에 8가지 imputation 기법과 ML 실무 결정 흐름.

---

## 4.1 결측치는 단순한 "공백"이 아니다

전형적 ML 책은 결측치를 다음 한 줄로 처리한다 — *"평균 또는 중앙값으로 채워라"*. 이게 위험한 이유는, **결측이 일어난 메커니즘이 채우는 방식과 결과 모두에 영향**을 주기 때문이다.

극단적 예시 — 의료 데이터에서 "혈압 측정" 열의 결측:

- *Case A*: 환자가 대기실에서 잠이 들어 측정 못함 → 결측이 *무작위*. 평균으로 채워도 무관.
- *Case B*: 의사가 *증상이 가벼워서 측정 안 했음* → 결측 자체가 "건강함"을 의미. 평균으로 채우면 noise 추가.
- *Case C*: 환자가 *너무 아파서 응급실로 직행해서 측정 못함* → 결측 자체가 "매우 심각함"을 의미. 평균으로 채우면 *반대 방향*의 정보를 주입.

세 경우 모두 "결측치"지만 의미가 정반대다. 이걸 정확히 구분하는 게 Rubin의 분류다.

---

## 4.2 Rubin의 분류 (1976)

Donald Rubin이 1976년 *Biometrika*에 정립한 결측 메커니즘 분류.

### MCAR — Missing Completely At Random

> *결측 여부가 데이터의 어떤 변수와도 무관하다.*

수식적으로:
$$P(\text{missing} \mid X, Y) = P(\text{missing})$$

즉, 관측된 값들과 결측된 값들 자체와 무관하게 결측이 발생.

- **예**: 설문 응답을 우편으로 받는데 일부 봉투가 *우체국에서 분실됨*. 분실 자체는 응답 내용과 무관.
- **처방**: 거의 어떤 imputation도 안전. 행 제거(listwise deletion)도 OK (단, n 손실).

현실적으로 *진짜 MCAR은 드물다*. 보통은 다음 둘이 더 흔하다.

### MAR — Missing At Random

> *결측 여부가 다른 관측 변수에는 의존하지만, 결측된 값 자체와는 무관하다.*

수식적으로:
$$P(\text{missing} \mid X, Y) = P(\text{missing} \mid X_{\text{observed}})$$

- **예**: 남성이 우울감 설문에 응답을 더 자주 빼먹는다 (그러나 빼먹는 정도는 우울 점수 자체와 무관). 성별(관측됨)이 결측 여부를 설명.
- **처방**: 다른 관측 변수를 *조건부로* imputation. 단순 평균은 편향. KNN, MICE, regression imputation이 적절.

### MNAR — Missing Not At Random

> *결측 여부가 결측된 값 자체에 의존한다.*

- **예 1**: 우울감이 매우 심한 사람들이 우울감 설문 자체를 빼먹는다. → "우울하다"는 사실이 결측의 원인.
- **예 2**: 매우 부유한 사람들이 자산 공개를 거부한다. → 자산 자체가 결측 원인.
- **처방**: 가장 어려운 경우. **결측 indicator 자체를 feature로 추가**하는 게 표준. 또는 도메인 모델링 (selection model, pattern mixture).

### 메커니즘 진단

세 종류를 정확히 구분하기는 *어렵다* (관측 못한 값과 비교해야 하니까). 그러나 다음 휴리스틱이 도움된다.

- **MCAR 검정 (Little's MCAR test)**: 결측 패턴이 다른 변수와 독립인지 통계 검정
- **결측 vs 관측의 다른 변수 비교**: 결측이 있는 행과 없는 행의 *다른* 변수 분포가 비슷? → MCAR/MAR. 다름? → MAR/MNAR.
- **도메인 지식**: "왜 이 값이 빠졌을까"를 도메인 전문가에게 물어보는 게 가장 강력.

---

## 4.3 Imputation 기법 — 8가지

### 1. Listwise Deletion (행 제거)

가장 단순. 결측 있는 행을 그냥 버린다.

- **장점**: 구현 0줄, 편향 없음 (MCAR이면)
- **단점**: n 손실, 결측 비율 크면 거의 모든 행이 사라짐
- **언제**: MCAR + 결측 비율 < 5%

### 2. Mean / Median / Mode Imputation

수치는 평균·중앙값, 카테고리는 최빈값으로 채움.

- **장점**: 빠름, 단순
- **단점**: *분산 축소* (모든 결측이 같은 값이라). feature 간 상관관계 왜곡. MAR·MNAR에 부적절
- **언제**: MCAR + 변수 자체가 중요하지 않음 + 결측 적음

### 3. Constant / Indicator Imputation

특정 값(예: -999, 0, "Unknown")으로 채우고 *결측 indicator를 별도 feature*로 추가.

- **장점**: 결측 패턴이 정보면 모델이 학습 가능. MNAR에 안전
- **단점**: feature 수 증가
- **언제**: MNAR 의심 + tree 기반 모델 (GBM은 이 패턴에 매우 강함)

```python
# pandas 예시
df['x_missing'] = df['x'].isna().astype(int)
df['x'] = df['x'].fillna(-999)
```

### 4. KNN Imputation

비슷한 K개 이웃의 값으로 채움.

- **장점**: 변수 간 관계 보존, 비선형 패턴 처리
- **단점**: 거리 정의 필요 (mixed type 어려움), 큰 데이터에서 느림
- **언제**: MAR + 중간 크기 데이터 + 변수 간 상관 강함

```python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
```

### 5. Regression Imputation

다른 변수로 결측 변수를 *예측*.

- **장점**: feature 간 관계 정확히 활용
- **단점**: 예측의 *불확실성을 무시* (single value로 채움). 분산 축소.
- **언제**: MAR + 단순한 baseline

### 6. Stochastic Regression / Multiple Imputation (MICE)

regression imputation + *예측 분포에서 sampling*. **MICE** (Multivariate Imputation by Chained Equations)는 여러 번 sampling해 m개의 완성된 데이터셋을 만들고, 분석 결과를 *평균*.

- **장점**: 불확실성 보존. MAR의 *표준* 처방
- **단점**: 복잡, 계산 비용 m배
- **언제**: MAR + 통계적 신뢰성 핵심 (의료, 사회과학)

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(max_iter=10, random_state=0)
X_imputed = imputer.fit_transform(X)
```

### 7. Time Series Imputation

시계열 결측에 특화.

- **Forward fill (`ffill`)**: 마지막 관측값을 다음 결측에 복사. 가격·온도 같은 *지속성 있는* 신호에 적합.
- **Backward fill (`bfill`)**: 다음 값을 이전 결측에 — 미래 leak 위험, 학습용 X
- **Linear interpolation**: 양옆 값 사이를 직선으로
- **Spline / seasonal interpolation**: 더 부드러운 보간

```python
# pandas
df['x'].fillna(method='ffill')
df['x'].interpolate(method='linear')
```

### 8. Deep Learning Imputation

VAE, GAN, denoising autoencoder로 결측 채움.

- **장점**: 복잡한 패턴, 큰 고차원 데이터
- **단점**: 모델·튜닝 부담, 작은 데이터에 overkill
- **언제**: 매우 큰 데이터 + 다른 방법 부족할 때

---

## 4.4 결정 흐름 — 무엇을 골라야 하나

```
결측 비율 < 5% + MCAR 의심
    └── listwise deletion 또는 median imputation

결측 비율 5–30% + MAR
    └── KNN, regression, MICE
    └── tree 기반 모델 쓸 거면 indicator 추가도 강력

결측 비율 > 30%
    └── 그 변수 사용 자체를 재검토
    └── 또는 결측 자체를 feature (MNAR 가능성)

MNAR 강한 의심
    └── 결측 indicator + constant fill (-999 같은)
    └── pattern mixture model (전문가 영역)

시계열 결측
    └── forward fill (학습용) — 미래 leak 안 됨
    └── interpolation (시각화·EDA용)

이미지 결측 픽셀
    └── inpainting 모델 또는 mask로 처리

텍스트 결측
    └── empty string + indicator
```

---

## 4.5 결측치를 *feature로* 활용하기

전통적 통계는 결측을 "메우려" 한다. ML은 종종 다르다 — *결측 자체를 정보로*.

### Tree 기반 모델 (GBM, Random Forest)

XGBoost, LightGBM, CatBoost 모두 *결측을 native하게 처리*. 결측을 별도 분기로 학습.

```python
# LightGBM 예시 — NaN 그대로 입력
model = lgb.LGBMClassifier()
model.fit(X_with_nan, y)  # NaN을 알아서 처리
```

이 모델들은 결측을 채우는 것보다 *그대로 두는 게 더 잘 동작*하는 경우가 많다 — 결측 패턴 자체를 학습.

### 신경망 — 명시적 indicator

신경망은 NaN을 직접 못 받음. 하지만 다음 패턴이 표준.

```python
# 각 numeric feature에 대해
x_filled = x.fillna(0)            # 결측을 0으로
x_missing = x.isna().astype(float)  # 결측 indicator

# concat해서 입력
input = torch.cat([x_filled, x_missing], dim=-1)
```

이러면 결측 패턴이 정보면 모델이 학습한다.

---

## 4.6 결측 패턴 시각화

EDA에서 결측을 단순히 "비율"로 보지 말고 *패턴*을 보자.

### Missing matrix

각 row를 가로, 각 column을 세로로 두고 결측을 검은 점으로. **`missingno`** 라이브러리가 표준.

```python
import missingno as msno
msno.matrix(df)
msno.bar(df)       # 변수별 결측 비율
msno.heatmap(df)   # 변수 간 결측 상관 (한 변수 결측이면 다른 변수도 결측?)
msno.dendrogram(df)  # 결측 패턴 클러스터링
```

### 시계열 결측

시간 축으로 결측 시점을 표시. 특정 기간에 결측이 몰렸으면 *시스템 장애·정책 변화* 가능성 — 도메인 조사 필요.

---

## 4.7 흔한 함정

### (1) train+test 합쳐서 imputation 적합

가장 흔한 leak. **train으로만 imputation 모델을 적합하고, test에는 *적용*만**.

```python
# 잘못
imputer.fit(np.concatenate([X_train, X_test]))
X_train_imp = imputer.transform(X_train)
X_test_imp = imputer.transform(X_test)

# 올바름
imputer.fit(X_train)
X_train_imp = imputer.transform(X_train)
X_test_imp = imputer.transform(X_test)
```

### (2) 평균 imputation 후 통계 검정

평균으로 채우면 분산이 인위적으로 줄어 들어 t-test, F-test의 p값이 *너무 작아짐*. 통계 추론에 쓸 거면 multiple imputation 필요.

### (3) "결측 = 0" 자동 처리

`x.fillna(0)`이 무지성 default가 되면 위험. 수치 변수에서 0이 *진짜 0*인지 *결측의 placeholder*인지 구분 못함.

### (4) MNAR을 MAR로 처리

MNAR을 평범한 imputation으로 채우면 모델이 *틀린 패턴*을 학습. 도메인 전문가의 진단이 중요.

### (5) 결측 indicator 추가하지 않음

GBM 같은 모델에 NaN을 그대로 줄 거면 OK. 하지만 imputation 후 indicator를 *추가하지 않으면* 결측 패턴 정보가 사라짐.

---

## 4.8 한 줄 요약

- 결측의 *왜*에 따라 처리가 갈린다. Rubin의 MCAR/MAR/MNAR 분류가 토대.
- MCAR이면 단순 imputation OK, MAR이면 KNN/MICE, MNAR이면 indicator + constant.
- Tree 기반 모델(GBM)은 NaN을 *그대로 받는 게* 종종 더 강력하다.
- imputation은 *반드시 train으로만 적합*하고 test에 적용만 — 합치면 leak.
- `missingno` 라이브러리로 결측 *패턴*을 시각적으로 진단.

---

## 다음 장

→ [05. 이상치 탐지](05_outliers.md): IQR, Z-score, Mahalanobis, Isolation Forest, autoencoder. 이상치를 *언제 제거하고 언제 살리는가*.

---

## 참고문헌

- Rubin, D. B. (1976). *Inference and missing data*. Biometrika, 63(3), 581–592.
- Little, R. J. A. & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.). Wiley.
- van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). [Free online](https://stefvanbuuren.name/fimd/).
- scikit-learn: [`IterativeImputer` (MICE)](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html), [`KNNImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)
- [`missingno` package](https://github.com/ResidentMario/missingno)
