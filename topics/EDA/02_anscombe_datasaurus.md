# 02. 시각화의 정당성 — Anscombe Quartet & Datasaurus Dozen

> 이 장의 목표 — 왜 *반드시* 분포를 그려야 하는가. 평균·분산·상관계수만 봐서 EDA가 끝났다고 생각하면 안 되는 이유를 두 유명 사례로 못박는다.

---

## 2.1 1973년, Anscombe의 사중주

Frank Anscombe는 1973년 *American Statistician*에 짧은 논문 하나를 실었다. 제목은 *"Graphs in Statistical Analysis"*. 그 안에 다음 4개의 데이터셋이 있다.

| Dataset | 평균 X | 평균 Y | 분산 X | 분산 Y | 상관 (r) | 회귀선 |
|---|---|---|---|---|---|---|
| I | 9.0 | 7.5 | 11.0 | 4.12 | 0.816 | y = 3 + 0.5x |
| II | 9.0 | 7.5 | 11.0 | 4.13 | 0.816 | y = 3 + 0.5x |
| III | 9.0 | 7.5 | 11.0 | 4.12 | 0.816 | y = 3 + 0.5x |
| IV | 9.0 | 7.5 | 11.0 | 4.12 | 0.817 | y = 3 + 0.5x |

소수점 2자리까지 *모든 통계량이 같다*. 평균, 분산, 상관, 회귀선 기울기·절편 — 모두 동일.

그런데 산점도를 그려보면:

- **Dataset I**: 깨끗한 선형 관계 + 정규 잔차 → 회귀가 잘 어울림
- **Dataset II**: 명백한 *2차 곡선*. 선형 회귀가 부적절
- **Dataset III**: 거의 완벽한 선형이지만 *outlier 1개*가 회귀선을 끌어당김
- **Dataset IV**: x가 거의 한 값에 모여 있는데 *outlier 한 점*이 모든 신호를 만듦

네 데이터셋이 *완전히 다른* 함수 관계 / 다른 issue를 가지고 있는데, 통계량은 똑같다. 만약 평균·분산·상관계수만 보고 회귀를 적합시켰다면, *4개 데이터셋 모두에 같은 직선*을 그어 주고 "잘 됐다"고 했을 것이다. 실제로는 II에는 2차 모델이, III·IV에는 outlier 처리가 필요하다.

> Anscombe의 메시지: *"통계학자 사이에 '숫자 계산은 정확하지만 그래프는 거칠다'는 인상을 반박하기 위해"* 이 사중주를 만들었다고 그는 적었다.

---

## 2.2 2017년, Datasaurus Dozen — 공룡으로 못박기

Anscombe의 사례가 너무 약하다고 느낀 사람이 있었다. Justin Matejka와 George Fitzmaurice는 2017년 *"Same Stats, Different Graphs"*라는 논문에서, **simulated annealing**으로 다음을 보였다 — *통계량을 정확히 보존하면서 분포를 임의로 바꿀 수 있다*.

원래 데이터는 Alberto Cairo가 그린 **공룡 모양 산점도**(Datasaurus). 거기서 출발해, 평균·분산·상관을 *소수점 2자리까지 일치시키면서* 점들을 옮겼다. 결과는:

- 공룡 (원본)
- 별
- 가로 선
- 세로 선
- X자
- 원
- 다섯 점만 있는 점군
- 그 외 6가지 더

**13개 데이터셋의 모든 요약 통계가 정확히 같다.** 그런데 시각적으로 완전히 다르다.

이게 *Datasaurus Dozen*이다. 의미는 명확하다 — *통계량은 분포를 묘사할 수 없다. 분포를 보려면 그려야 한다.*

> "Same stats, different graphs" — 같은 통계량, 다른 그래프.

---

## 2.3 ML 실무에서의 함의

이 두 사례가 가르치는 바는 ML EDA에서 다음 형태로 매일 등장한다.

### 함의 1: 평균·표준편차·describe()로는 부족하다

```python
df.describe()
```

이걸 보고 끝났다고 생각하면 위험. 다음을 *반드시* 봐야 한다.

- **히스토그램**: 분포 모양 (대칭, skewed, bimodal, long-tail)
- **box plot**: outlier와 IQR
- **산점도**: feature와 target의 비선형 관계
- **Q-Q plot**: 정규성 여부 (3장에서 깊이)

### 함의 2: 상관계수(Pearson)는 *선형* 상관만 본다

Pearson r = 0인데 *완벽한 비선형 관계*가 있을 수 있다 (예: y = x²). Anscombe Dataset II가 이 경우.

따라서:

- **Spearman 또는 Kendall 상관**: 단조 관계 잡음
- **Mutual Information**: 임의의 의존성
- **산점도**: 그냥 눈으로 보기

(자세히는 6장 상관성)

### 함의 3: outlier 1개가 모든 통계량을 흔든다

Anscombe Dataset III·IV가 이 경우. 평균·표준편차·회귀선 모두 outlier에 끌려간다. EDA에서 outlier를 *반드시* 시각적으로 확인해야 하는 이유.

robust statistics (median, IQR, Mahalanobis distance)가 outlier에 덜 민감하지만, 그 전에 *outlier 자체를 인지*해야 한다.

### 함의 4: 자동 파이프라인의 위험

`fit → score → compare`만 하는 자동 ML 파이프라인은 Anscombe·Datasaurus형 데이터를 *완벽히 놓친다*. 학습·평가가 모두 동일 분포 가정 위에서 동작하기 때문에 분포가 잘못된 줄 알 길이 없다.

이게 EDA를 *수작업으로* 해야 하는 이유이고, AutoML이 만능이 아닌 이유.

---

## 2.4 그럼 무엇을 그려야 하는가 — 최소 시각화 세트

ML 프로젝트 시작 시 *반드시* 그려야 하는 최소 plot 세트.

### 모든 수치 변수에 대해

1. **히스토그램** (또는 KDE) — 분포 모양
2. **box plot** — outlier와 IQR
3. **Q-Q plot vs 정규분포** — 정규성 여부

### 모든 카테고리 변수에 대해

4. **bar chart of value counts** — 카테고리 빈도
5. **target과의 cross-tab** (분류) 또는 **box plot by category** (회귀)

### feature 간

6. **상관 heatmap** (Pearson + Spearman 둘 다)
7. **pair plot** (매우 작은 데이터·소수 변수일 때)

### Target과 feature

8. **각 feature vs target 산점도** (회귀) 또는 **box plot** (분류)
9. **feature importance** (단순 모델로 빠르게)

### 시계열이라면

10. **시간 vs target** plot (trend·seasonality)
11. **자기상관 (ACF/PACF)**

이 11종을 *프로젝트 시작 첫 30분에* 모두 그리면, Anscombe·Datasaurus형 함정의 99%를 피한다. 코드는 8장(표 데이터)·14장(도구)에서.

---

## 2.5 "그래프를 그려도 못 잡는" 경우

시각화도 만능은 아니다. 다음은 plot으로 잘 안 보이는 경우들 (다른 장에서 다룸).

- **고차원 의존성**: 3변수 이상의 상호작용은 2D plot으로 불가 → mutual information, decision tree feature importance (6장)
- **시간적 leak**: plot으로는 안 보임 → 도메인 지식, split 점검 (11·13장)
- **selection bias**: 데이터 수집 자체가 편향 → plot 자체가 이미 편향 (1장 §1.4 질문 1, 13장)
- **conceptual error**: 잘못된 target 정의 → 비즈니스 이해 (13장)

EDA의 시각화는 *광범위하지만 완전하지 않다*. 도메인 지식과 함께해야 한다.

---

## 2.6 한 줄 요약

- **Anscombe quartet**: 통계량이 같아도 분포가 4가지로 다를 수 있다.
- **Datasaurus Dozen**: 통계량을 정확히 보존하면서 분포를 13가지(공룡 포함)로 만들 수 있다.
- 결론: *평균·분산·상관계수만 보면 분포를 모른다. 반드시 그려야 한다.*
- 최소 시각화 세트 11종을 첫 30분에 그리면 Anscombe형 함정 대부분을 피한다.

---

## 다음 장

→ [03. 분포 진단의 통계적 기초](03_distributions.md): Q-Q plot, normality test, 모멘트, 분포 family — 그래프 너머의 정량적 진단 도구.

---

## 참고문헌

- Anscombe, F. J. (1973). *Graphs in Statistical Analysis*. The American Statistician, 27(1), 17–21.
- Matejka, J. & Fitzmaurice, G. (2017). *Same Stats, Different Graphs: Generating Datasets with Varied Appearance and Identical Statistics through Simulated Annealing*. CHI 2017.
- Wikipedia: [Anscombe's quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet), [Datasaurus dozen](https://en.wikipedia.org/wiki/Datasaurus_dozen)
- Autodesk Research: [Same Stats, Different Graphs](https://www.research.autodesk.com/publications/same-stats-different-graphs/)
