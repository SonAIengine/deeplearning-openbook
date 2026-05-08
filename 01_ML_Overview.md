# 01. Machine Learning Overview — 오픈북 정리

> **교수님 스타일 키워드**: 정의 암기 X / 구조 이해 O / "왜?" / 장단점 / 비교 / 적용

---

## 0. 큰 그림 — "ML이 왜 필요한가?"

전통 프로그래밍은 **규칙(rule) → 데이터 → 답**.
ML은 **데이터 + 답 → 규칙(모델)**.

**왜 ML?** 사람이 규칙을 못 적는 문제(이미지 인식, 음성, 번역, 의료 영상)에서, **함수 f를 데이터로부터 추정**한다.
즉 ML = "데이터로부터 함수를 근사하는 일".

생각해보라:
- 왜 모든 문제를 if-else로 못 푸는가? → 입력 공간이 너무 고차원/연속이고, 사람의 직관이 명시적 규칙으로 안 떨어짐.
- 그럼 ML이 못하는 건? → 데이터가 적거나, 분포가 바뀌거나, **인과**를 요구하는 문제.

---

## 1. 학습 유형 (왜 셋으로 나누나?)

| 유형 | 입력 | 정답 라벨 | 예시 | 핵심 질문 |
|---|---|---|---|---|
| **Supervised** | x | y 있음 | 분류, 회귀 | "y를 어떻게 예측?" |
| **Unsupervised** | x | y 없음 | 군집, 차원축소, 생성 | "데이터의 구조?" |
| **Reinforcement** | 상태 s | 보상 r (지연됨) | 게임, 로봇 | "장기 보상 최대?" |

**왜 이렇게 나누나?** → 풀려는 문제의 **신호 종류**가 다르기 때문. supervised는 정답이 강한 신호, unsupervised는 약한 신호(분포 자체), RL은 sparse·delayed 신호.

**비교**: semi-supervised(라벨 일부) / self-supervised(라벨을 입력에서 만들어냄, 예: BERT의 masked LM, SimCLR의 augmentation pair)는 위 셋의 혼합·확장.

---

## 2. Supervised: 분류 vs 회귀

| | 분류(Classification) | 회귀(Regression) |
|---|---|---|
| 출력 | 이산 클래스 | 연속 실수 |
| 마지막 층 | softmax / sigmoid | linear |
| Loss | Cross-Entropy | MSE / MAE |
| 평가 | Accuracy, F1, AUC | RMSE, MAE, R² |

**왜 분류에 MSE 안 쓰나?** → softmax + MSE는 gradient가 작아서 학습이 느려진다. CE는 softmax와 결합하면 gradient가 (예측 − 정답)으로 깔끔해진다 (수업에서 자주 묻는 포인트).

---

## 3. Loss Function — "왜 필요한가, 어떻게 고르나?"

Loss = "현재 예측이 얼마나 틀렸는지"의 정량화. 학습은 **loss를 줄이는 방향**으로 파라미터를 갱신.

| Loss | 형태 | 언제? | 주의 |
|---|---|---|---|
| **MSE** | (y−ŷ)² | 회귀, 노이즈가 가우시안 가정 | 이상치(outlier)에 민감 |
| **MAE** | \|y−ŷ\| | 회귀, 이상치에 강함 | 0에서 미분 불가 |
| **Huber** | MSE+MAE 절충 | 회귀, robust | 임계값 δ 튜닝 |
| **Binary CE** | −[y log ŷ + (1−y) log(1−ŷ)] | 이진 분류 | sigmoid와 짝 |
| **Categorical CE** | −Σ yᵢ log ŷᵢ | 다중 분류 | softmax와 짝 |
| **KL divergence** | Σ p log(p/q) | 분포 매칭 | 비대칭 |

**왜 CE가 분류에서 표준?** → 확률적 해석(MLE 관점). 정답 분포와 예측 분포의 KL을 최소화하는 것과 등가. 또 softmax의 saturation 영역에서도 CE는 gradient가 살아있다.

---

## 4. Gradient Descent — "왜 작동하는가?"

목표: $\theta^* = \arg\min_\theta L(\theta)$
업데이트: $\theta \leftarrow \theta - \eta \nabla_\theta L$

**왜 gradient의 반대 방향?** → 1차 Taylor 전개에서 함수가 가장 빨리 감소하는 방향이 −∇.

### 4.1 변종 비교

| 방식 | 한 step의 데이터 | 장점 | 단점 |
|---|---|---|---|
| **Batch GD** | 전체 데이터 | 안정, 정확한 gradient | 느림, 메모리 폭발 |
| **SGD** | 1개 샘플 | 빠름, **noise가 local minima 탈출에 도움** | 진동 큼 |
| **Mini-batch** | 32~512 | 둘의 절충, GPU 친화 | batch size 튜닝 필요 |

**핵심 통찰**: SGD의 noise는 단점이 아니라 **암묵적 정규화** 역할. (왜 mini-batch가 표준인가에 대한 답)

### 4.2 학습률 (η)

너무 크면 발산, 작으면 느림/local에 갇힘. → **scheduling** 필요(step decay, cosine, warmup).

생각해보라: warmup이 왜 필요? → 초기 random 파라미터에서 큰 LR은 발산 위험. 작게 시작해 안정화 후 키운다.

---

## 5. Bias–Variance Trade-off (★시험 단골)

테스트 에러 = **Bias² + Variance + Irreducible noise**

- **Bias 高**: 모델이 단순. Train도 Test도 못 맞춤 → **Underfitting**.
- **Variance 高**: 모델이 복잡, 데이터 noise까지 학습 → Train↓ Test↑ → **Overfitting**.

| 상황 | Bias | Variance | 해결 |
|---|---|---|---|
| Underfit | 高 | 低 | 모델 키우기, feature 추가, 학습 더 |
| Overfit | 低 | 高 | regularization, dropout, 데이터 늘리기, early stop, 모델 줄이기 |

**왜 이 trade-off가 발생?** → 같은 데이터로 더 잘 맞추려면 모델이 복잡해져야 하고, 복잡할수록 noise까지 외워버린다.

**핵심 관찰**: 딥러닝에서는 매우 큰 모델이 오히려 일반화 잘 되는 "double descent" 현상 — 고전적 trade-off만으로 설명 안 됨. 그래서 over-parameterization + 적절한 regularization이 현대 표준.

---

## 6. Generalization — "왜 안 본 데이터에서도 동작하나?"

가정: train과 test가 **같은 분포(i.i.d.)에서** 추출.

- 모델이 데이터의 **본질적 패턴**을 배우면 generalize.
- **noise**까지 배우면 overfit, generalize 실패.

### 6.1 Regularization (왜 필요?)

훈련 loss를 낮추는 것이 목적이 아니라 **테스트 loss**를 낮추는 게 목적이므로, 모델 용량을 인위적으로 제한.

| 방법 | 어떻게 | 효과 |
|---|---|---|
| **L1 (Lasso)** | + λΣ\|w\| | 일부 w를 정확히 0 → **sparse**, feature selection 효과 |
| **L2 (Ridge / weight decay)** | + λΣw² | w 전반을 작게 → smooth |
| **Dropout** | 학습 중 뉴런 랜덤 마스킹 | 암묵적 ensemble |
| **Early stopping** | val loss 증가 시 중단 | 시간을 통한 정규화 |
| **Data augmentation** | 입력에 변형 | 분포 확장 |
| **Batch Norm** | 층마다 정규화 | 부수 효과로 약한 정규화 |

**L1 vs L2**: L1은 corner에서 미분 불가지점이 있어 정확히 0을 만든다(sparse). L2는 부드럽게 줄인다. 둘 다 쓰면 Elastic Net.

생각해보라: 왜 dropout이 ensemble과 같은가? → 매 mini-batch마다 다른 sub-network를 학습하는 셈, test 시엔 평균.

---

## 7. 데이터 분할 — "왜 셋(train/val/test)으로?"

- **Train**: 파라미터 학습.
- **Validation**: 하이퍼파라미터 선택 + early stop. 모델 선택용.
- **Test**: 최종 평가. **딱 한 번** 본다.

**왜 val과 test를 분리?** → val로 모델 선택을 반복하면 val에 간접 overfit. test는 실세계 성능 추정의 마지막 보루.

### 7.1 Cross-validation
데이터가 적으면 **k-fold CV**: 데이터를 k 등분, 매번 1개를 val로. → 통계적 안정성 ↑, 계산량 ×k.

생각해보라: time-series 데이터에 k-fold 그대로 써도 되나? → No. 미래 데이터로 과거를 예측하는 leak 발생. **forward-chaining** 사용.

---

## 8. 평가 지표 — "언제 어떤 걸?"

### 분류 (혼동행렬 기반)
$$\text{Confusion Matrix} = \begin{bmatrix} TP & FN \\ FP & TN \end{bmatrix}$$

| 지표 | 정의 | 언제? |
|---|---|---|
| Accuracy | (TP+TN)/total | 클래스 균형일 때 |
| **Precision** | TP/(TP+FP) | FP 비용 큰 문제 (스팸 분류, 추천) |
| **Recall (Sensitivity)** | TP/(TP+FN) | FN 비용 큰 문제 (암 진단) |
| **F1** | 2·P·R/(P+R) | P와 R의 조화평균, 불균형 데이터 |
| **AUC-ROC** | ROC 아래 면적 | 임계값 무관 비교 |

**왜 accuracy만 보면 안 되나?** → 99% 음성인 암 데이터에서 "전부 음성" 모델은 99% accuracy지만 의미 없음. → recall, F1 봐야.

### 회귀
- **MAE**: 직관적, 이상치 robust.
- **RMSE**: 큰 오차에 민감.
- **R²**: 1에 가까울수록 좋음, 0이면 평균 예측 수준.

---

## 9. Curse of Dimensionality

차원이 커질수록:
- 같은 밀도를 유지하려면 데이터가 **지수적으로** 필요.
- 모든 점이 멀어져 거리 기반 알고리즘(KNN) 무력화.
- 표면이 부피의 대부분 차지.

**왜 딥러닝은 고차원에서 작동?** → 실제 데이터는 고차원 공간에 분포하더라도 **저차원 manifold**에 놓여있기 때문(manifold hypothesis). 모델이 이 manifold를 학습.

---

## 10. 학습 vs 추론 (Inference)

- **학습(train)**: forward + backward, 파라미터 갱신.
- **추론(inference)**: forward만, 파라미터 고정.
- BN, Dropout은 train/eval 모드가 다름 (★자주 묻힘).

---

## 11. ML 파이프라인 — "실제 문제에 어떻게 적용?"

1. **문제 정의**: 분류? 회귀? metric은? 비용 비대칭?
2. **데이터 수집·정제**: 결측치, 이상치, 라벨 noise.
3. **EDA**: 분포, 상관, 클래스 균형.
4. **전처리**: 정규화/표준화, encoding (one-hot, embedding), missing 처리.
5. **모델 선택**: baseline부터.
6. **학습 + val로 튜닝**.
7. **test로 최종 평가** (한 번만).
8. **배포 + 모니터링**: data drift 감지.

생각해보라: 왜 baseline부터? → 단순 모델이 충분하면 복잡한 모델 불필요. 또 복잡한 모델이 baseline을 못 이기면 뭔가 잘못된 것.

---

## 12. 실수하기 쉬운 포인트 (시험 함정)

1. **데이터 누수(leakage)**: test 정보가 train에 들어감. 정규화를 train+test 합쳐서 fit하면 leak.
2. **클래스 불균형**: accuracy 함정. → resampling, class weight, focal loss.
3. **분포 변화(distribution shift)**: train과 deploy의 분포가 다름.
4. **Train-test split이 i.i.d. 가정 위반** (시계열, 그룹).

---

## 13. 한 줄 요약 (시험 직전 복습)

- ML = 데이터로 함수 근사. **학습 = loss 최소화**.
- **GD가 작동하는 이유** = gradient의 반대 방향이 가장 빨리 감소.
- **CE를 쓰는 이유** = MLE와 등가, softmax와 만나면 gradient가 깔끔.
- **Bias-Variance**: 단순↔복잡의 trade-off. 현대 딥러닝은 over-param + regularization으로 깬다.
- **Regularization의 본질**: 모델 용량을 제한해 noise 학습 방지.
- **Val/Test 분리 이유**: val에 간접 overfit 방지.
- **Accuracy만 보면 안 되는 이유**: 클래스 불균형에서 무의미.
