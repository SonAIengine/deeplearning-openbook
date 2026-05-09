# EDA — 분포에서 결정까지 (Book)

> ML 프로젝트에서 가장 자주 *건너뛰는* 단계인데 가장 자주 *실패하는* 곳이기도 한 EDA(Exploratory Data Analysis)를 책 한 권 분량으로 풀어 쓴다. 본문 §10.2가 한 줄로 압축한 자리를 챕터별로 깊게 채운다.

---

## 이 책의 목표

EDA를 단순한 "체크리스트 작업"이 아니라 **데이터의 분포를 보고 모델·loss·전처리·평가 metric까지 결정하는 사고 과정**으로 다룬다. 다음 세 종류의 독자에게 답한다.

- *시작하는 사람*: 무엇을 봐야 하는가, 어떻게 보는가, 어떤 코드로 보는가
- *경력 ML 엔지니어*: 무엇을 빠뜨리고 있는가, 흔한 함정은 무엇인가, 자동화 어디까지 가능한가
- *시험·면접 준비*: "EDA 단계에서 뭘 보나?" 류 질문에 1분 답할 수 있게

---

## 책의 구조 — 3부 15장

### 1부. 기초 (왜·무엇을·어떻게)

- **00. [Quick Reference](00_quick_reference.md)** — 결정 매트릭스, 체크리스트, 한 페이지 요약 (이전 EDA_guide의 핵심 내용)
- **01. [EDA의 철학과 역사](01_philosophy.md)** — Tukey의 1977년 비전과 ML 시대의 재해석
- **02. [시각화의 정당성 — Anscombe Q. & Datasaurus](02_anscombe_datasaurus.md)** — 평균·분산만 보면 왜 망하는가
- **03. [분포 진단의 통계적 기초](03_distributions.md)** — 히스토그램, Q-Q plot, normality test, 모멘트

### 2부. 데이터 type별 EDA

- **04. [결측치 — Rubin의 분류와 처리 전략](04_missing_data.md)** — MCAR/MAR/MNAR과 imputation
- **05. [이상치 탐지](05_outliers.md)** — IQR, Z-score, Mahalanobis, Isolation Forest, autoencoder
- **06. [상관성과 의존성](06_correlation.md)** — Pearson을 넘어: Spearman, mutual information, copula
- **07. [시각화 기술의 무기고](07_visualization_techniques.md)** — 기본 plot부터 dimension reduction까지
- **08. [표 데이터 EDA](08_tabular.md)** — 전형적 workflow + Pandas 코드 예시
- **09. [이미지 데이터 EDA](09_image.md)** — 채널 분포, 라벨 noise 진단, 도메인 차이
- **10. [텍스트 데이터 EDA](10_text.md)** — 길이 분포, 어휘, OOV, 라벨 일관성
- **11. [시계열 데이터 EDA](11_time_series.md)** — Trend, seasonality, drift, change-point

### 3부. 결정과 실전

- **12. [분포에서 결정으로 — 모델·전처리·loss·metric 매핑](12_distribution_decision.md)** — 책의 핵심 매트릭스
- **13. [흔한 함정과 사례 연구](13_pitfalls.md)** — 실제 production에서 망한 이야기들
- **14. [도구와 자동화](14_tools.md)** — pandas-profiling, sweetviz, ydata-profiling, 그리고 한계
- **15. [실전 workflow와 체크리스트](15_workflow.md)** — 첫 30분에 무엇을 하나, 마지막 점검

---

## 학습 흐름

| 시간이 30분 | Quick Reference (00) → Workflow (15) |
| 시간이 2–3시간 | 1부(00–03) + 12장(결정 매핑) + 본인 task type 챕터 (08–11 중) |
| 진지하게 1주 | 처음부터 끝까지 + 코드 직접 실행 |

---

## 본문과의 연결

이 책은 본문 13개 챕터의 다음 자리에서 호출된다.

- **§1.2 학습 유형의 변종** → 02장 (시각화로 분포 검사)
- **§2 Loss function** → 03·12장 (분포 → loss 결정)
- **§5.4 정규화 강도** → 12장 (데이터 크기 → 정규화)
- **§6 데이터 분할** → 11·13장 (시계열 leak, group leak)
- **§7 평가 지표** → 12장 (분포 → metric)
- **§10.2 ML 파이프라인의 EDA** → 이 책 전체 (가장 강한 연결)
- **§11 흔한 함정** → 13장

---

## Quick Reference로 바로 가기

지금 당장 결정이 필요하면 → [00. Quick Reference](00_quick_reference.md)의 §2 결정 매트릭스부터.

깊은 이해를 원하면 → [01. 철학](01_philosophy.md)부터 순서대로.
