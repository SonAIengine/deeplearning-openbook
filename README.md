# Deep Learning Open-Book Notes

서술·설명 위주 딥러닝 시험을 위한 오픈북 정리.
"정의 암기 X / 구조 이해 O / 왜·비교·설계·적용" 사고 패턴 중심.

## 구성

### 기초 (각 모델의 구조와 핵심 질문)
| # | 파일 | 핵심 |
|---|---|---|
| 01 | [ML Overview](01_ML_Overview.md) | 학습 유형, loss, GD, bias-variance, regularization, 평가 |
| 02 | [FNNs](02_FNNs.md) | 퍼셉트론 한계, MLP, activation, backprop, UAT |
| 03 | [DNNs](03_DNNs.md) | vanishing gradient, init, BN, dropout, optimizer, ResNet |
| 04 | [CNNs](04_CNNs.md) | conv·pooling·sharing 이유, 주요 architecture, transfer learning |
| 05 | [RNNs](05_RNNs.md) | recurrent 구조 이유, BPTT, arbitrary seq, seq2seq |
| 06 | [LSTM & GRU](06_LSTM_GRU.md) | gating 이유, vanishing 완화 메커니즘, LSTM vs GRU |

### 사고 확장 ("교수님처럼 사고하기")
| # | 파일 | 핵심 |
|---|---|---|
| 07 | [Cross-Architecture Design](07_Cross_Architecture_Design.md) | 이미지를 RNN으로? 텍스트를 CNN으로? — inductive bias 매칭 |
| 08 | [What-If / Ablation](08_WhatIf_Ablation.md) | 컴포넌트를 빼면 무엇이 어떻게 망가지는가 |
| 09 | [Architecture & Hyperparameter Decisions](09_Architecture_Hyperparameter_Decisions.md) | 깊이·너비·batch·LR·activation 결정 사고법 |
| 10 | [Training Diagnosis](10_Training_Diagnosis.md) | 학습 실패 시나리오별 원인 추적과 처방 |
| 11 | [Comparison & Decision](11_Comparison_Decision.md) | X냐 Y냐 — 옵션 간 본질적 차이와 결정 가이드 |
| 12 | [Big Picture & FAQ](12_BigPicture_FAQ.md) | 발전사 흐름 + 단골 Q&A 30선 + A4 1장 압축 요약 |

### 실전 적용
| # | 파일 | 핵심 |
|---|---|---|
| 13 | [Task Design Playbook](13_Task_Design_Playbook.md) | ML 엔지니어 관점 6개 task end-to-end 설계 (CV·NLP·시계열·이상탐지·추천) |

### 심화 주제 (`topics/`)

본문이 압축한 부분을 풀어 쓴 심화 가이드. 본문 해당 섹션에서 링크로 연결.

| 주제 | 본문 연결 | 핵심 |
|---|---|---|
| [EDA → 모델·전처리 결정](topics/EDA_guide.md) | §10.2 | 점검 항목 11종 + 결정 매트릭스 + 데이터 type별 체크리스트 + 흔한 함정 |

## 각 문서 공통 구조

1. 핵심 질문 (교수님이 던질 법한 형태)
2. 직관·구조 비교 (그림/표 위주)
3. 설계·사고 단계
4. 함정·실수 포인트
5. "생각해보라" 확장 질문
6. 한 줄 요약

## 사용법

- 시험·면접 직전 → `12_BigPicture_FAQ`의 Q&A 30선이 압축 요약.
- "X 어떻게 설계?" 질문 대비 → `13_Task_Design_Playbook` + `07_Cross_Architecture_Design`.
- "이거 왜 빼면 안 됨?" 질문 → `08_WhatIf_Ablation`.
- "X vs Y 무엇 선택?" → `11_Comparison_Decision`.

## 주의

- 표준 딥러닝 커리큘럼 기반으로 작성. 특정 강의 슬라이드의 고유 예시·숫자·도식은 빠질 수 있으니 시험 전 슬라이드와 대조 권장.
- 계산형 시험엔 별도 worked-example 자료가 필요 (현 자료는 서술·설명형 위주).

## 라이선스

개인 학습 정리용. 자유롭게 fork·수정·재사용.
