# EDA — 분포에서 결정까지

> **한 페이지 hub**. EDA의 전체 그림을 처음부터 끝까지 한 호흡에 잡을 수 있도록 self-contained로 풀어 쓴다. 더 깊이 들어가고 싶은 키워드는 인라인 링크의 sub-page를 따라가면 된다.
>
> 본문 §10.2가 *"가장 자주 건너뛰는데 가장 중요"* 라고만 적은 EDA 단계의 본격 가이드.

---

## 왜 EDA가 가장 중요한가

ML 프로젝트에서 가장 자주 실패하는 단계는 모델 선택이 아니라 **데이터 단계**다. leak, drift, 라벨 noise, 분포 불일치 — 이런 사고의 90%는 *EDA로 분포를 미리 보면* 잡힌다. 모델·hyperparameter는 라이브러리가 거의 해결해 주지만, 데이터의 모양과 함정은 *사람이 직접 봐야 한다*.

EDA의 본질은 다음 세 질문에 답하는 것이다.

1. **이 데이터로 정말 ML이 가능한가?** (신호가 있는가, 양이 충분한가, 라벨 품질)
2. **어떤 모델·loss·전처리·metric이 이 분포에 맞는가?**
3. **무엇이 망가질 수 있는가?** (leak, drift, 결측 패턴, outlier)

이 셋에 답이 안 서면 모델 단계로 가지 마라.

→ 더 깊이: [01. EDA의 철학과 역사](01_philosophy.md) — Tukey 1977 비전, CDA vs EDA, ML 시대 재해석.

---

## 시각화는 옵션이 아니라 필수다 — Anscombe와 Datasaurus

Frank Anscombe(1973)는 **평균·분산·상관계수가 모두 같은데 분포가 4가지로 다른** 데이터셋을 보였다. 더 충격적으로, Matejka & Fitzmaurice(2017)는 simulated annealing으로 **통계량을 정확히 보존하면서 공룡 모양·별 모양 등 13가지 다른 분포**를 만들었다 (Datasaurus Dozen).

결론은 단순하다 — *통계량은 분포를 묘사하지 못한다*. `df.describe()`만 보면 위험. 반드시 그려야 한다.

**최소 시각화 11종** (프로젝트 첫 30분에 그릴 것)
- 수치 변수: 히스토그램·box plot·Q-Q plot
- 카테고리: value counts bar, target과 cross-tab
- feature 간: 상관 heatmap (Pearson + Spearman), pair plot
- target과 feature: 산점도 / box plot
- 시계열: 시간 vs target, ACF/PACF

→ 더 깊이: [02. Anscombe & Datasaurus](02_anscombe_datasaurus.md), [07. 시각화 기술의 무기고](07_visualization_techniques.md) (예정)

---

## 분포 진단 — 4 모멘트와 Q-Q plot

분포를 *수치로* 묘사하는 표준은 4개의 모멘트:

| 모멘트 | 의미 | robust 대안 |
|---|---|---|
| 1차 평균 | 중심 | 중앙값(median) |
| 2차 분산 | 퍼짐 | IQR |
| 3차 왜도(skew) | 비대칭성 | — (값 자체로 본다) |
| 4차 첨도(kurtosis) | 꼬리 두께 | — |

빠른 진단 흐름:

- |skew| < 0.5 + Q-Q plot 직선 → **정규에 가까움 → MSE·StandardScaler 그대로**
- skew > 1 right-skewed → **log / Yeo-Johnson 변환**
- kurtosis > 5 (heavy tail) → **MAE/Huber loss + RobustScaler**
- 봉우리 두 개 (bimodal) → **cluster를 feature로** 또는 mixture model
- 0이 매우 많음(>30%) → **zero-inflated 또는 hurdle 모델**

큰 데이터(n > 10⁶)에서는 정규성 검정의 p값이 *모두 reject*되므로 무의미. **시각적 Q-Q plot이 더 신뢰**.

→ 더 깊이: [03. 분포 진단의 통계적 기초](03_distributions.md) — Q-Q plot, 7가지 분포 family, 5가지 변환 도구, drift 비교(KS/KL/Wasserstein/PSI).

---

## 결측치 — Rubin의 분류로 갈린다

빈 값을 *어떻게* 채우느냐는 *왜* 빠졌느냐에 달렸다. Donald Rubin(1976)의 분류:

- **MCAR** (완전 무작위): 어떤 변수와도 무관하게 빠짐 → *평균/중앙값 imputation 또는 행 제거 OK*
- **MAR** (관측 변수 의존): 다른 관측 변수에는 의존하지만 결측값 자체와는 무관 → *KNN, MICE, regression imputation*
- **MNAR** (결측값 자체 의존): 결측 자체가 정보 (우울한 사람이 우울 설문 빼먹는 등) → *결측 indicator를 별도 feature로*

**Tree 기반 모델(XGBoost/LightGBM/CatBoost)은 NaN을 native로 처리** — imputation 없이 그대로 줘도 잘 동작한다. 오히려 imputation이 *결측 패턴이라는 정보*를 지워 모델 성능을 깎을 수 있다.

신경망은 NaN 직접 못 받음 → `x_filled + x_missing_indicator`를 concat해 입력.

→ 더 깊이: [04. 결측치 — Rubin 분류와 처리 전략](04_missing_data.md) — 8가지 imputation, missingno로 패턴 시각화, 함정 5가지.

---

## 이상치 — 제거할까, 살릴까

이상치는 두 종류로 나눈다.

- **진짜 이상치**: 측정 오류, 입력 실수 → *제거*
- **유효한 극단값**: 실제 그런 사례가 드물게 있음 → *유지하되 robust 처리*

탐지 도구:

| 방법 | 장점 | 단점 |
|---|---|---|
| **IQR 1.5×**: Q3 + 1.5·IQR 밖 | 단순, 직관적 | 정규 가정, 다변량 무시 |
| **Z-score |z| > 3** | 쉬움 | 이상치 자체가 평균·분산 끌어당김 |
| **Mahalanobis distance** | 다변량 + 상관 고려 | 차원의 저주 |
| **Isolation Forest** | 비모수, 고차원 OK | 해석 어려움 |
| **Autoencoder reconstruction error** | 이미지·고차원 | 학습 부담 |

도메인 지식 없이 IQR만으로 자르면 *진짜 신호*(사기 거래, 희귀 질환)를 잘라버린다. 도메인 검증 후 결정.

robust 처리: **MAE/Huber loss**, **RobustScaler**(median, IQR), **winsorize**(상하위 percentile에서 capping).

→ 더 깊이: [05. 이상치 탐지](05_outliers.md) (예정) — 5가지 방법 비교, 다변량 outlier, 시계열 이상치(change-point).

---

## 상관성과 의존성 — Pearson 너머

**Pearson 상관**은 *선형* 관계만 잡는다. y = x²처럼 완벽한 비선형 관계도 Pearson r ≈ 0.

대안 도구:

- **Spearman 상관**: 단조(monotonic) 관계 잡음. rank 기반이라 outlier robust
- **Kendall τ**: Spearman과 비슷, 작은 표본에 안정
- **Mutual Information (MI)**: 임의의 의존성. KSG estimator로 연속 변수 가능
- **Distance correlation**: 0이면 완전 독립 보장 (Pearson은 r=0이라도 종속 가능)
- **Copula**: 변수 간 *의존 구조*를 분리해서 모델링 (금융에서 표준)

다중공선성(|r| > 0.9 변수 쌍)은 선형 모델에서 계수 해석을 무의미하게 만든다. PCA 또는 feature selection. 신경망·GBM은 덜 민감.

target과의 상관이 *너무* 강하면 leak 의심 — target에서 파생된 feature가 섞이지 않았는지 확인 (본문 §6.5).

→ 더 깊이: [06. 상관성과 의존성](06_correlation.md) (예정) — Pearson 너머의 5가지 측도, 비선형 관계 진단, copula 입문.

---

## 데이터 type별 — 무엇을 보는가

### 표 (Tabular)

- 행 수 N, 열 수 D, dtype, 메모리
- 결측 비율과 패턴
- 수치 변수의 분포·outlier
- 카테고리 cardinality 분포
- target 분포와 feature-target 관계
- feature 간 상관

→ 더 깊이: [08. 표 데이터 EDA](08_tabular.md) (예정) — Pandas workflow + 코드 예시.

### 이미지

- 클래스별 sample 이미지를 *직접 눈으로* 보기 (가장 중요!)
- 이미지 크기 분포 (가변? 고정?)
- 채널별 평균·분산 (정규화 통계)
- 클래스별 sample 수 (불균형)
- 데이터 출처 — 같은 환경에서 수집됐나?

→ 더 깊이: [09. 이미지 데이터 EDA](09_image.md) (예정) — 라벨 noise 진단, 도메인 차이 검출.

### 텍스트

- 길이 분포 (토큰 수)
- 어휘 크기, OOV 비율
- 클래스별 단어 빈도 (TF-IDF)
- 라벨 일관성 (재라벨링 비교)
- 특수 패턴 — URL, 이모지, 코드, 다국어

→ 더 깊이: [10. 텍스트 데이터 EDA](10_text.md) (예정) — 어휘 분석, 라벨 일관성 측정.

### 시계열

- *반드시* 시간 vs target 그려보기
- Trend (추세), seasonality (주기성)
- Drift — 시점에 따른 분포 변화
- 결측 시점, 중복 시점
- 이벤트·휴일 표시

시계열은 split이 *반드시 시간 순*이어야 함. random split은 미래 leak (본문 §6.3).

→ 더 깊이: [11. 시계열 데이터 EDA](11_time_series.md) (예정) — STL decomposition, ACF/PACF, change-point detection.

---

## 분포에서 결정으로 — 핵심 매트릭스

이 책의 가장 중요한 표. EDA 결과 → 결정.

| EDA에서 본 것 | Loss | 전처리 | 모델 후보 | Metric |
|---|---|---|---|---|
| 잔차 종 모양·대칭 | MSE | StandardScaler | 일반 회귀 | RMSE |
| 잔차 long-tail / heavy-tail | MAE / Huber | log / Yeo-Johnson | robust regression | MAE |
| target right-skewed | MSE on log(target) | log 변환 | — | RMSE on log scale |
| 클래스 50/50 | CE | — | 무엇이든 | accuracy / AUC |
| 클래스 90/10 | weighted CE | — | weight 적용 | F1 / PR-AUC |
| 클래스 99/1 | focal loss (γ=2) | — | + threshold 조정 | PR-AUC, recall |
| 클래스 99.99/0.01 | — | anomaly detection 패러다임 | Isolation Forest, AE | precision@k |
| 결측 < 5% MCAR | (영향 없음) | median imputation | 무엇이든 | — |
| 결측 5–30% MAR | (영향 없음) | KNN / MICE | 무엇이든 | — |
| 결측 > 30% MNAR | (영향 없음) | indicator + constant | GBM 강함 | — |
| 다중공선성 강함 | — | PCA / feature selection | GBM, NN (덜 민감) | — |
| 카테고리 cardinality 큼 | — | embedding 또는 hashing | NN, CatBoost | — |
| Outlier 다수 | MAE / Huber | RobustScaler / winsorize | tree (덜 민감) | MAE |
| 시계열 trend·seasonality | — | lag, time feature | ARIMA / LSTM / GBM | MAPE |
| Drift 강함 | — | 시간 순 split | + 정기 재학습 | — |
| 이미지 도메인 차이 | — | 강한 augmentation | CNN + DA | — |
| 작은 데이터 (<1K) | — | — | GBM / linear, 강한 정규화 | CV 평균 |
| 큰 데이터 (>10M) | — | — | NN / Transformer | — |

이 매트릭스가 EDA를 "데이터 구경"에서 *결정 도구*로 바꾼다.

→ 더 깊이: [12. 분포에서 결정으로 — 매트릭스 확장판](12_distribution_decision.md) (예정).

---

## 흔한 함정 — 같은 사고가 반복된다

### 1. 평균·분산만 보고 분포 안 봄

`df.describe()`만 신뢰. Anscombe·Datasaurus가 보이는 위험.

### 2. train만 보고 test 안 봄

train과 test의 *분포가 다르면* (covariate shift) production에서 망한다. *동일한 EDA를 train·val·test 모두에 반복*.

### 3. 전처리 통계를 train+test 합쳐서 적합

가장 흔한 leak. 정규화·imputation·encoder 모두 *train으로만 fit*, test에는 transform만.

### 4. EDA에서 target 정보로 feature 검토

target에서 파생된 feature를 그대로 train에 넣으면 leak. *시간 순으로 보면 target보다 먼저 측정된 것만 feature로*.

### 5. Outlier를 무조건 제거

outlier가 *진짜 신호*인 경우 (사기, 희귀 질환) 제거하면 모델이 그 case를 못 학습. **도메인 지식으로 검증 후**.

### 6. EDA를 한 번만 함

데이터는 시간이 지나면 변한다. *프로젝트 시작 + 학습 후 잔차 EDA + production 데이터 EDA* 모두 필요.

→ 더 깊이: [13. 흔한 함정과 사례 연구](13_pitfalls.md) (예정) — 실제 production에서 망한 이야기들.

---

## 도구 — 자동화의 한계

### pandas-profiling / ydata-profiling

```python
from ydata_profiling import ProfileReport
ProfileReport(df).to_file("eda_report.html")
```

한 줄로 *수십 페이지* 리포트. 분포·상관·결측·중복·warning까지.

### sweetviz

train·test를 *나란히 비교*하는 데 특화. covariate shift 진단에 강력.

### autoviz, dataprep

다른 자동 EDA 라이브러리들. 도구별로 강점 다름.

### 한계

이런 도구들은 *분포를 보여주지만 결정을 해주지 않는다*. "이 변수는 long-tail이야"라고 알려주지만, "그래서 log 변환이 적절한가, robust loss가 적절한가, 아예 빼는 게 적절한가"는 *사람이 도메인 지식과 함께 결정*해야 한다.

자동화 도구는 EDA의 *입구*이지 출구가 아니다.

→ 더 깊이: [14. 도구와 자동화](14_tools.md) (예정) — 5가지 자동 EDA 도구 비교 + 한계 사례.

---

## 실전 workflow — 첫 30분에 무엇을 하나

```
[5분] 데이터 로드 + df.shape, df.dtypes, df.head()
[5분] 결측 비율 — df.isna().mean(), missingno.matrix(df)
[10분] 수치 변수 분포 — df.describe() + 히스토그램 + box plot
[5분] 카테고리 cardinality — value_counts()
[5분] target 분포 — 회귀: 히스토그램 / 분류: 클래스 비율
```

이 30분 후 다음 10가지가 답이 서야 한다.

1. 데이터 양 — 충분한가?
2. 타깃 분포 — 변환 필요? 클래스 처리?
3. 결측 처리 전략
4. 수치 정규화 — Standard / MinMax / Robust?
5. 카테고리 인코딩 — one-hot / embedding / hashing?
6. Loss · metric — 분포에 맞나?
7. 모델 후보
8. 분할 전략 — random / 시간 / 그룹?
9. Augmentation 전략
10. Drift · leak 위험

이 10개에 답이 서면 모델 단계로 가도 좋다. 하나라도 명확치 않으면 EDA를 더.

→ 더 깊이: [15. 실전 workflow와 체크리스트](15_workflow.md) (예정) — task type별 workflow 차이, 30분/2시간/1주 시나리오별 체크리스트.

---

## 한 줄 요약

- EDA는 *분포를 보고 결정을 내리는 단계*. 평균·분산이 아니라 *그림*을 본다.
- 거의 모든 모델·loss·전처리 결정이 분포에서 나온다 (위 매트릭스).
- train·val·test 모두에 EDA. 분포가 다르면 그게 첫 번째 위험.
- 자동화 도구는 입구. 결정은 사람이.
- EDA는 한 번이 아니라 프로젝트 내내 반복.

---

## sub-page 인덱스 (현재 작성된 것)

핵심 키워드 별 깊이 학습. 위 본문에서 인라인 링크로 호출.

- [00. Quick Reference](00_quick_reference.md) — 결정 매트릭스, 체크리스트
- [01. 철학과 역사](01_philosophy.md) — Tukey 1977 + ML 시대 재해석
- [02. Anscombe & Datasaurus](02_anscombe_datasaurus.md) — 시각화 정당성
- [03. 분포 진단](03_distributions.md) — 모멘트, Q-Q plot, 변환, drift 비교
- [04. 결측치 — Rubin 분류](04_missing_data.md) — MCAR/MAR/MNAR + 8 imputation

작성 예정: 05 outliers / 06 correlation / 07 visualization / 08 tabular / 09 image / 10 text / 11 time-series / 12 decision matrix 확장 / 13 pitfalls / 14 tools / 15 workflow.

---

## 본문과의 연결

이 hub는 본문 다음 자리에서 호출된다.

- **§1.2 학습 유형** → 시각화로 분포 검사 섹션
- **§2 Loss** → 분포 진단 + 결정 매트릭스
- **§5.4 정규화 강도** → 결정 매트릭스
- **§6 데이터 분할** → 시계열 / 함정 섹션
- **§7 평가 지표** → 결정 매트릭스
- **§10.2 ML 파이프라인의 EDA** → 이 hub 전체
- **§11 흔한 함정** → 함정 섹션
