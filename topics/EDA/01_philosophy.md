# 01. EDA의 철학과 역사

> 이 장의 목표 — Tukey가 1977년에 무엇을 주장했는지, 왜 그 주장이 ML 시대에 더 유효해졌는지, 그리고 "EDA가 뭘 봐야 하는가"라는 질문이 사실 *철학적 질문*임을 보인다.

---

## 1.1 1977년, John Tukey의 책

벨 연구소의 통계학자 John W. Tukey는 1977년 *Exploratory Data Analysis*라는 책을 출간했다. 그 시점의 통계학 주류는 **확증적 데이터 분석**(confirmatory data analysis, CDA)이었다 — 모델·가설을 *데이터를 보기 전에* 정해 놓고, 데이터로 검증하는 방식. 가설검정, 신뢰구간, 추정량의 일치성 같은 것들이 그 토대다.

Tukey는 이 흐름에 정면으로 반대한다. *통계학이 가설검정에 너무 많은 비중을 두고 있다*는 것. 그는 다음을 주장했다.

> "Far better an approximate answer to the right question, which is often vague, than an exact answer to the wrong question, which can always be made precise."
> — *대충 맞는 답이 정확히 틀린 답보다 낫다.*

즉, 모델이 옳은 질문을 다루는지부터 확인해야 한다. 그리고 그걸 확인하는 유일한 방법은 **데이터를 직접 보는 것**이다. 이게 EDA의 정신이다 — *모델·가설을 정하기 전에, 일단 데이터가 무엇을 말하는지 듣는다*.

Tukey는 이 책에서 **stem-and-leaf plot**, **box plot**, **median polish**, **resistant statistics** 같은 새 도구들을 발명·체계화했다. 모두 *극단값에 휘둘리지 않으면서 분포를 빠르게 보여주는* 도구들이다. 평균보다 중앙값을, 분산보다 IQR을, 회귀보다 잔차 plot을 강조한 것이 핵심.

---

## 1.2 통계학의 두 모드 — CDA vs EDA

|  | **확증적 데이터 분석 (CDA)** | **탐색적 데이터 분석 (EDA)** |
|---|---|---|
| 시점 | 데이터 보기 *전*에 가설·모델 결정 | 데이터를 *먼저* 보고 가설을 수립 |
| 도구 | 가설검정, 신뢰구간, p-value | 시각화, robust statistics, plot |
| 위험 | 잘못된 가설을 검증해도 정답이 나옴 | 가설이 데이터에 over-fit |
| 비유 | 법정 증명 (피고인을 정해 놓고 증거) | 형사 수사 (단서를 따라가기) |

이 둘은 적이 아니라 **사이클**이다. EDA로 가설을 만든다 → CDA로 검증한다 → 새 데이터 → EDA → CDA … 진짜 위험은 한쪽만 사용하는 것이다.

- *EDA만*: 보이는 패턴이 우연인지 진짜 신호인지 구별 못 함 (multiple testing)
- *CDA만*: 데이터가 가정을 만족하지 않으면 가설검정 자체가 무의미

Tukey의 메시지는 *EDA를 먼저, 충분히* 해서 가설이 데이터에 맞는지 확인한 뒤 CDA로 가라는 것이다.

---

## 1.3 ML 시대의 EDA — 더 강력해진 정당성

1977년의 통계학은 작은 데이터(n < 1000)에 깔끔한 모델(선형 회귀, ANOVA)을 적용하는 시대였다. 그런데 2010년 이후의 ML은:

- **데이터가 거대**(n > 10⁶, d > 10³)
- **모델이 매우 유연**(deep network는 거의 모든 함수를 표현 가능)
- **표현 학습**(feature를 사람이 만들지 않음)

이 조합에서 EDA는 *더 중요해졌다*. 왜?

### (1) 모델이 너무 유연해서 noise까지 학습한다

선형 회귀는 데이터가 가정을 위반해도 "어울리지 않는 직선"을 그릴 뿐이다 — 잔차 plot으로 즉시 보인다. 그러나 deep network는 **noise·outlier·leak까지 그대로 외워서** 매끄럽게 학습한다. 학습 곡선만 보면 "잘 됐다"고 착각하기 쉽다. EDA로 *데이터의 quirk를 모델보다 먼저 발견*하지 못하면 production에서 망한다.

### (2) feature engineering이 EDA로 결정된다

전통 통계: feature가 이미 정의됨. EDA는 신호 확인용.

ML: 어떤 feature를 만들지, 어떻게 변환할지가 *모델 성능의 절반*. 그 결정은 EDA에서 나온다 — long-tail은 log, skewed는 box-cox, 카테고리는 embedding 등.

### (3) 데이터 leak·drift가 사고의 90%

논문·실무 사례를 분석한 여러 보고에 따르면, ML 프로젝트 실패의 *주된 원인*은 모델 부족이 아니라 데이터 단계의 사고 — leak, drift, 라벨 noise, 불균형 등. 이 모두 EDA에서 일찍 발견 가능하다. 모델 단계까지 가서야 발견되면 비용이 100배 든다.

### (4) 분포 가정이 loss와 metric을 결정한다

- 잔차가 가우시안 → MSE
- 잔차가 long-tail → MAE/Huber
- 클래스가 99/1 → accuracy 무용 → PR-AUC

이 결정 흐름 (본문 §2·§7)이 *전부 EDA의 결과물*이다.

---

## 1.4 EDA의 본질 — 세 가지 질문

이 책 전체를 관통하는 EDA의 본질은 다음 세 질문에 답하는 것이다.

### 질문 1: 이 데이터로 정말 ML이 가능한가?

- 신호가 충분히 있는가? (target과 feature 사이에 의미 있는 패턴)
- 양이 충분한가? (curse of dimensionality에 안 빠질 정도)
- 라벨 품질은 어떤가? (noisy labels, 일관성)
- ML이 도덕적·법적으로 적절한가? (편향, 프라이버시)

답이 *No*면 모델 단계에 가지 마라. EDA로 데이터·문제 정의 단계로 돌아가야 한다.

### 질문 2: 어떤 모델·loss·전처리·metric이 이 분포에 맞는가?

- 분포가 가우시안이면 MSE+MLE 자연
- Long-tail이면 MAE/Huber 또는 log 변환 후 MSE
- 클래스 불균형이면 focal+PR-AUC
- 표 데이터+적은 N이면 GBM이 강자
- 이미지+많은 N이면 CNN/Transformer

이 매핑이 책의 12장에서 매트릭스로 정리된다.

### 질문 3: 무엇이 망가질 수 있는가?

- *Leak*: train과 test에 같은 그룹/시간/ID
- *Drift*: 시간에 따라 분포 변화
- *결측 패턴*: MNAR이면 imputation이 위험
- *Outlier*: 진짜 신호인가, 측정 오류인가
- *Selection bias*: 데이터 수집 자체가 편향

이 셋이 EDA의 도착지다. 답이 명확해지면 모델 단계로, 명확하지 않으면 EDA를 더.

---

## 1.5 EDA에 대한 현대의 두 가지 오해

### 오해 1: "EDA는 자동화 도구가 다 해줘"

`pandas-profiling`, `sweetviz`, `ydata-profiling` 같은 도구가 한 줄로 리포트를 생성한다. 매우 유용하지만 이걸로 EDA가 끝났다고 생각하면 위험.

이런 도구들은 *분포를 보여주지만 결정을 해주지 않는다*. "이 변수는 long-tail이야"라고 알려주지만, "그래서 log 변환이 적절한가, robust loss가 적절한가, 아예 빼는 게 적절한가"는 *사람이 도메인 지식과 함께 결정*해야 한다.

자동화 도구는 EDA의 *입구*이지 출구가 아니다. (자세히는 14장.)

### 오해 2: "EDA는 처음에 한 번만"

진짜 EDA는 프로젝트 *내내* 반복된다.

- *전*: 모델 결정 전 첫 EDA
- *중*: 학습 후 잔차·confusion matrix EDA (모델이 뭘 못 맞췄는가)
- *후*: production 데이터 EDA (drift 감지)

특히 *production EDA* 가 가장 자주 잊힌다. monitoring system이 그 자리를 일부 채우지만 (input/output drift detection), *사람이 직접 보는 것*을 대신할 순 없다.

---

## 1.6 한 줄 요약

- EDA는 **모델·가설·결정을 하기 전에 데이터를 직접 듣는 행위**다.
- Tukey가 1977년에 시작한 정신은 ML 시대에 *더* 강력해졌다 — 모델이 noise까지 외우고, feature engineering이 EDA에서 나오고, 사고의 90%가 데이터 단계에서 발생.
- 세 질문에 답하는 게 본질: *ML이 가능한가, 무엇이 맞는가, 무엇이 망가질 수 있는가*.
- 자동화 도구는 입구이지 출구가 아니다. EDA는 처음에 한 번이 아니라 프로젝트 내내 반복.

---

## 다음 장

→ [02. 시각화의 정당성 — Anscombe Q. & Datasaurus](02_anscombe_datasaurus.md): "왜 평균·분산만 보면 안 되는가"를 두 유명 사례로 보인다.

---

## 참고문헌

- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Wikipedia: [Exploratory data analysis](https://en.wikipedia.org/wiki/Exploratory_data_analysis)
- Tukey, J. W. (1962). *The Future of Data Analysis*. Annals of Mathematical Statistics, 33(1), 1-67.
