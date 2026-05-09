# EDA 가이드 — 분포를 보고 모델·전처리를 결정하기

> **이 문서의 목표**
> EDA(Exploratory Data Analysis)가 *왜 가장 중요한 단계인가*, *무엇을 보는가*, *그래서 어떤 결정으로 이어지는가*를 한 페이지로 풀어 쓴다. 본문 §10.2는 "가장 자주 건너뛰는 단계인데 가장 중요"라고만 적혀 있는데, 이 문서가 그 빈자리를 메운다.

---

## 0. EDA가 왜 가장 중요한 단계인가

ML 프로젝트에서 가장 자주 실패하는 단계는 모델이 아니라 **데이터**다 (§11.1 데이터 누수 등). 그리고 데이터 단계의 사고 90%는 *EDA로 분포를 보면 미리 잡힌다*. 모델 architecture·hyperparameter는 라이브러리가 어느 정도 해결해 주지만, 데이터의 모양과 함정은 *사람이 직접 보지 않으면 모른다*.

EDA의 본질은 세 가지 질문에 답하는 것:

1. **이 데이터로 정말 ML이 가능한가?** (신호가 있는가, 양이 충분한가, 라벨 품질이 어떤가)
2. **어떤 모델·loss·전처리가 이 분포에 맞는가?** (가우시안? long-tail? 불균형?)
3. **무엇이 망가질 수 있는가?** (leak, drift, 결측 패턴, outlier)

이 셋에 답이 안 서면 모델을 짜기엔 이르다.

---

## 1. EDA 점검 항목 — 무엇을 보는가

### 1.1 데이터 형태 (shape & type)

가장 먼저 — *얼마나 있고, 어떻게 생겼나*.

- 행 수 N, 열 수 D (또는 이미지 크기 / 시퀀스 길이)
- 각 열의 dtype (수치 / 카테고리 / 텍스트 / 날짜 / 이미지 경로)
- 메모리 크기 — 한 번에 RAM에 올라가는가? GPU에 올라가는가?

**→ 결정으로**:
- N이 너무 작음 (<1000) → 딥러닝보다 GBM/선형 모델, 강한 정규화, cross-validation
- N이 매우 큼 (>10M) → mini-batch 필수, distributed 고려
- D가 매우 큼 (>1000) → 차원 축소(PCA, t-SNE) 고려, sparse 처리
- 메모리 안 들어감 → streaming 학습, lazy loading, IterableDataset

### 1.2 결측치 (missing values)

각 열의 결측 비율 + 결측 패턴을 본다.

- *MCAR* (Missing Completely At Random): 무작위 결측. imputation 안전.
- *MAR* (Missing At Random): 다른 변수에 의존한 결측. imputation 가능하지만 모델링 신중.
- *MNAR* (Missing Not At Random): 결측 자체가 정보. 결측 여부를 *별도 feature로* 만들기.

**→ 결정으로**:
- 결측 < 5% → 평균/중앙값 imputation 또는 결측 행 제거
- 결측 5–30% → KNN imputation, MICE (multiple imputation by chained equations), 또는 결측 indicator 추가
- 결측 > 30% → 그 열을 사용할지 자체를 재검토. 또는 *결측 자체가 신호*인지 확인 (병원 데이터의 "검사 안 함" = 의사가 필요 없다 판단)
- 결측 패턴이 시계열적이면 (특정 시점부터 갑자기 결측↑) → drift 의심, 별도 처리

### 1.3 분포 (distribution)

각 수치 변수의 히스토그램을 *반드시 직접 본다*. 평균·표준편차만 보면 함정.

- *대칭 종 모양 (bell-shaped)*: 가우시안 가정 적절
- *Long-tail / heavy-tailed*: 극단값이 자주 등장 — 평균이 분포 대표 못함
- *Skewed (한쪽 쏠림)*: log·box-cox 변환 검토
- *Bimodal (두 봉우리)*: 잠재적 두 그룹 — 별도 모델? 또는 cluster feature 추가?
- *Spike at zero*: 양수만 들어오는데 0이 많음 — zero-inflated 모델 또는 hurdle model

**→ 결정으로**:
- Long-tail → log 변환 (`log(1+x)`로 0 안전), 또는 robust loss (MAE/Huber, §2.2)
- Skewed → box-cox 또는 yeo-johnson 변환으로 정규성 회복 → MSE 가능
- Bimodal → cluster label을 feature로, 또는 mixture model
- 모델이 가우시안 가정 (linear regression의 conf interval 등) 사용한다면 변환 필수

### 1.4 분포의 scale

각 수치 변수의 *값 범위*를 본다.

- 한 변수는 0–1, 다른 변수는 0–1,000,000 → scale 차이 6자리
- 모델이 거리 기반(KNN, SVM, K-means)이면 큰 scale 변수가 모든 거리 지배
- gradient 기반 모델(신경망)은 큰 scale 입력에 weight가 작아짐 → 학습 느려짐 / 불안정

**→ 결정으로**:
- StandardScaler (mean 0, var 1) — 가우시안 가정
- MinMaxScaler (0–1) — 분포 가정 없음, 이미지 픽셀
- RobustScaler (median, IQR) — outlier 강한 데이터
- 신경망: 입력 정규화 + BN/LN으로 내부 정규화

### 1.5 카테고리 변수

각 카테고리의 *cardinality*(고유값 수)와 *분포*.

- Cardinality 작음 (<10) → one-hot 안전
- Cardinality 중간 (10–100) → target encoding 또는 embedding
- Cardinality 큼 (>1000) → embedding 거의 필수, hashing trick
- 분포가 매우 한쪽 쏠림 → rare category를 "Other"로 묶기

**→ 결정으로**:
- 표 데이터 + 신경망 → embedding 표
- 표 데이터 + GBM → label encoding (LightGBM은 카테고리 직접 지원, CatBoost는 더 깔끔)
- *unseen category 처리* — production에서 새 카테고리가 들어오면? "Other" bucket 또는 hashing

### 1.6 클래스 균형 (분류 task)

각 클래스의 빈도 비율.

- 50/50 — 균형. accuracy OK.
- 90/10 — 보통 불균형. 처리 필요.
- 99/1 — 심한 불균형. 신중한 처리 (의료, 사기 탐지, 이상 탐지).
- 99.99/0.01 — 극단 (희귀 질환, 자율주행 사고).

**→ 결정으로**:
- 균형 → accuracy/AUC OK
- 보통 불균형 → class weight, F1, PR-AUC
- 심한 불균형 → focal loss (§2.4), oversampling (SMOTE), threshold 조정 (§7.7)
- 극단 → anomaly detection 패러다임으로 전환 고려

### 1.7 상관관계 (correlation)

수치 변수 간 상관관계 (Pearson/Spearman).

- 강한 양의 상관 (|r| > 0.9) → 다중공선성. 한 변수만 남기거나 PCA.
- 약한 상관 → 독립적 정보, 모두 유지
- 비선형 관계 → Pearson은 못 잡음. mutual information 또는 산점도(scatter)로 확인

**→ 결정으로**:
- 다중공선성 → 선형 모델은 불안정 (계수 해석 무의미). 신경망/GBM은 덜 민감.
- 너무 많은 비슷한 feature → PCA 또는 feature selection
- target과 강한 상관 → 좋은 신호. *너무* 강하면 leak 의심 (target 자신을 포함했나?)

### 1.8 이상치 (outliers)

분포의 꼬리에 있는 점들.

- *진짜 이상치*: 측정 오류, 입력 실수 → 제거
- *유효한 극단값*: 실제로 그런 사례가 드물게 있음 → 유지, robust 처리
- *데이터 leak*: 너무 좋은 점이 train에 있음 → 의심

**→ 결정으로**:
- IQR 1.5배·3배 기준으로 표시
- 도메인 지식으로 진짜/가짜 판별 (예: 사람 키 250cm는 가짜)
- 진짜 이상치 → 제거 또는 winsorize (capping)
- 유효한 극단값 → 유지하고 robust loss (MAE/Huber) 또는 log 변환

### 1.9 시계열 패턴 (시계열 데이터)

시간 축으로 정렬한 plot을 *반드시* 본다.

- *Trend* (추세): 시간에 따라 평균이 움직임 → 차분(differencing) 또는 trend feature
- *Seasonality* (주기성): 일/주/년 단위 반복 → 시간 feature (sin·cos encoding), Fourier feature
- *Drift* (분포 변화): 시점에 따라 분포가 바뀜 → train/test split이 시간 순이어야 (§6.3)
- *Outlier 클러스터*: 특정 시점에 이상값 모음 → 이벤트(휴일, 장애 등) 확인

**→ 결정으로**:
- Trend·seasonality 강함 → ARIMA/Prophet baseline, 또는 lag feature + GBM
- Drift 강함 → 정기 재학습 + monitoring
- 이벤트 영향 강함 → 이벤트 indicator feature 추가

### 1.10 이미지 데이터

- 클래스별 sample 이미지를 *직접 눈으로 보기*. 라벨 noise·도메인 차이 즉시 보임.
- 이미지 크기 분포 (가변? 고정?)
- 색상 분포 (RGB 채널별 평균·분산)
- 클래스별 이미지 수
- 데이터 출처 — 같은 카메라? 다른 환경?

**→ 결정으로**:
- 가변 크기 → resize/crop 전략
- 색상 분포 차이 → 정규화 (ImageNet 평균/std로 시작)
- 클래스 불균형 → weighted sampling 또는 augmentation 비대칭
- 도메인 차이 → domain adaptation, 또는 균등 샘플링

### 1.11 텍스트 데이터

- 길이 분포 (단어 수, 토큰 수)
- 어휘 크기, OOV(out-of-vocabulary) 비율
- 클래스별 단어 빈도 차이
- 언어 종류 (multi-lingual?)
- 라벨 일관성 — 같은 sample을 다른 사람이 라벨링하면 일치하나?

**→ 결정으로**:
- 길이 매우 다양 → padding/truncation 전략, 또는 hierarchical 모델
- 어휘 크기 큼 → subword tokenization (BPE, WordPiece)
- 라벨 noise → label smoothing(§2.4), 또는 manual 정제

---

## 2. EDA → 결정 매트릭스 (빠른 참조)

| EDA에서 본 것 | 결정 |
|---|---|
| 잔차 분포가 종 모양·대칭 | MSE loss, gaussian likelihood 모델 |
| 잔차 long-tail / heavy-tail | MAE/Huber loss, log 변환 |
| target이 매우 skewed | log/box-cox 변환 후 학습 |
| 클래스 50/50 | accuracy·AUC OK, 일반 CE |
| 클래스 90/10 – 99/1 | class weight, focal loss, F1/PR-AUC |
| 클래스 99.99/0.01 | anomaly detection 패러다임 |
| 결측 < 5% | 단순 imputation (평균/중앙값) |
| 결측 > 30% | 결측 자체를 feature로, 또는 열 제거 |
| 다중공선성 강함 | PCA, feature selection, 또는 GBM |
| 카테고리 cardinality 큼 | embedding, hashing |
| Outlier 다수 | robust loss, winsorize, RobustScaler |
| 시계열 trend·seasonality | lag feature, time feature, ARIMA baseline |
| Drift | 시간 순 split, 정기 재학습, monitoring |
| 이미지 도메인 차이 | augmentation 강하게, domain adaptation |
| 작은 데이터 (<1000) | GBM/선형, 강한 정규화, CV |
| 큰 데이터 (>10M) | 딥러닝, mini-batch, distributed |

---

## 3. 데이터 type별 EDA 체크리스트

### 3.1 표 (Tabular)

- [ ] 행 수, 열 수, 각 열 dtype
- [ ] 결측 비율과 패턴 (MCAR/MAR/MNAR)
- [ ] 수치 변수 히스토그램 — 정규성·skewness·outlier
- [ ] 카테고리 cardinality + 분포
- [ ] target 분포 (회귀: 분포·outlier / 분류: 클래스 균형)
- [ ] feature 간 상관관계 + target과의 상관
- [ ] 시간 feature 있으면 시간 순 점검

### 3.2 이미지

- [ ] 클래스별 sample 이미지를 직접 눈으로 보기 (10–20장씩)
- [ ] 이미지 크기 분포
- [ ] 색상 분포 (채널별 평균/분산)
- [ ] 클래스별 sample 수
- [ ] 데이터 출처 — 같은 환경에서 수집됐나?
- [ ] 라벨 품질 — 의심 sample 직접 확인

### 3.3 텍스트

- [ ] 길이 분포 (토큰 수)
- [ ] 어휘 크기, OOV 비율
- [ ] 클래스별 단어 빈도 (TF-IDF로)
- [ ] 라벨 일관성 (가능하면 일부를 다시 라벨링)
- [ ] 특수 패턴 — URL, 이모지, 코드, 다국어 혼합

### 3.4 시계열

- [ ] 시간 순 plot을 *반드시* 그려보기
- [ ] Trend, seasonality (FFT 또는 자기상관)
- [ ] 이벤트·휴일·장애 표시
- [ ] 결측 시점, 중복 시점
- [ ] Drift 의심 — train·val·test 시점 비교

---

## 4. EDA의 흔한 함정

### 4.1 평균·분산만 보고 분포 안 봄

평균·분산이 같아도 분포가 완전히 다를 수 있다 (Anscombe's quartet, Datasaurus dozen). **반드시 히스토그램·산점도를 직접 그려본다.**

### 4.2 train만 보고 test 안 봄

분포가 train과 test에서 다르면 (covariate shift), 모델이 잘 학습돼도 production에서 망한다. *train·val·test 모두에서 같은 EDA를 반복*. 분포가 다르면 그게 첫 번째 위험.

### 4.3 EDA에서 target 정보로 feature 검토

target과 강한 상관을 가진 feature를 그대로 train에 넣으면 좋아 보임. 하지만 그 feature가 *target에서 파생*된 것이면 (예: 다른 target과 상관 강한 변수) leak. *시간 순으로 보면 target보다 먼저 측정된 것만 feature로*.

### 4.4 outlier를 무조건 제거

outlier가 *진짜 신호*인 경우 (사기 거래, 희귀 질환) 제거하면 모델이 그 case를 못 학습한다. **outlier 제거 전 도메인 지식으로 검증**.

### 4.5 EDA를 한 번만 함

데이터는 시간이 지나면 변한다. EDA는 *프로젝트 시작 시점에 한 번 + 정기 재실행*. drift 모니터링의 일부.

### 4.6 EDA 결과를 기록 안 함

"분포가 long-tail이라 log 변환 했다"를 기억으로만 두면 1주일 뒤 다시 본다. **EDA notebook + 분포 도식 + 결정 근거를 PR/문서로**.

---

## 5. EDA 후의 표준 체크리스트 (의사결정 정리)

EDA가 끝나면 다음 항목이 *명확한 답을 가진 상태*여야 한다.

1. **데이터 양** — 충분한가? 부족하면 augmentation/transfer learning 필요?
2. **타깃 분포** — 회귀: 변환 필요? 분류: 클래스 처리 필요?
3. **결측 처리 전략** — 어떤 방식? 결측 자체가 feature?
4. **수치 정규화** — Standard / MinMax / Robust 중 무엇?
5. **카테고리 인코딩** — one-hot / embedding / hashing?
6. **Loss·metric** — 분포에 맞는가? (가우시안→MSE, long-tail→Huber, 불균형→focal+PR-AUC)
7. **모델 후보** — GBM(표) / CNN(이미지) / RNN·Transformer(시퀀스) / 선형(작은 데이터)?
8. **분할 전략** — random / 시간 / 그룹? (§6 leak 점검)
9. **Augmentation 전략** — 어떤 변환? 얼마나 강하게? (§5.4 데이터 크기별 강도)
10. **Drift·leak 위험** — 어디서 깨질 수 있는가?

이 10개에 답이 서면 모델 단계로 가도 좋다. 하나라도 명확치 않으면 EDA를 더 한다.

---

## 6. 한 줄 요약

- EDA는 *분포를 보고 결정을 내리는 단계*. 평균·분산이 아니라 *그림*을 본다.
- 거의 모든 모델·loss·전처리 결정이 분포에서 나온다 — 가우시안이면 MSE, long-tail이면 Huber, 불균형이면 focal.
- train·val·test 모두에 EDA. 분포가 다르면 그게 첫 번째 위험.
- 결과를 기록한다. EDA notebook + 결정 근거.

---

## 관련 문서

- 본문 §10.2 — ML 파이프라인의 EDA 단계
- 본문 §2 — Loss 선택 (분포 가정에 맞는 loss)
- 본문 §6 — train/val/test 분할 (leak 방지)
- 본문 §7 — 평가 지표 (분포에 맞는 metric)
- 본문 §11 — 흔한 함정 (data leakage, distribution shift)
