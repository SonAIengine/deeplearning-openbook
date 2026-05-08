# 10. Training Diagnosis — "왜 학습이 안 되는가"

> 학습 실패는 모두 다른 얼굴을 하지만 **원인은 패턴이 있다**.
> 증상 → 원인 → 처방의 매트릭스로 빠르게 진단.

---

## 0. 진단 사고법

학습 문제는 항상 다음 4단계로:
1. **증상 관찰**: loss 곡선, gradient norm, 예측 분포.
2. **가설 좁히기**: 어디서 발생? (data / model / optimizer / loss / code).
3. **빠른 검증**: 작은 ablation으로 원인 격리.
4. **처방**: 원인에 맞는 fix.

---

## 1. Loss 곡선 패턴 진단표

| 패턴 | 진단 |
|---|---|
| 시작부터 NaN | LR 너무 큼 / log(0) / 잘못된 init / unstable softmax |
| 시작은 정상, 갑자기 NaN | exploding gradient / 데이터 이상치 / mixed precision overflow |
| 수십 step 후 plateau | LR 너무 작음 / 모델 부족 / 데이터 정규화 X |
| 처음부터 수평 | gradient 죽음 (vanishing, dead ReLU) / 학습 못 흐름 |
| Train ↓ Val ↑ | overfit |
| Train ↑ Val ↑ (둘 다 stuck) | underfit (모델·LR·데이터) |
| Train 진동 큼 | LR 너무 큼 / batch 작음 / noisy data |
| Val이 train보다 낮음 | val set 너무 쉽거나 데이터 누수 |

---

## 2. NaN / 발산 추적

### 2.1 흔한 원인
1. **LR 너무 큼** — 1/10로 줄여 재시도.
2. **log(0)** — softmax 후 직접 log → log_softmax 사용 / eps 추가.
3. **나눗셈 0** — Adam의 sqrt(v)에 eps 빠뜨림.
4. **Init 잘못** — 분산 폭발.
5. **Mixed precision** — fp16 overflow → loss scaling 점검.
6. **잘못된 입력** — NaN/inf가 데이터에 섞임.
7. **Gradient explosion** — RNN에서 자주.

### 2.2 디버깅 절차
```
1. 작은 batch로 forward만 → 출력 NaN인가?
2. backward 한 번 → gradient norm 출력
3. loss를 분해 (예: cross-entropy를 logits·log_softmax 단계로)
4. 데이터 한 sample씩 확인
5. LR 1/10
6. gradient clipping 추가 (norm 1)
```

### 2.3 대표 처방
- `torch.clamp`로 logits를 안전 범위에.
- BCE 대신 `BCEWithLogitsLoss` (수치 안정).
- `nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.

---

## 3. Vanishing Gradient 진단

### 3.1 신호
- 깊은 layer의 gradient norm이 매우 작음 (1e-7 이하).
- Train loss 감소 매우 느리거나 정체.
- 얕은 layer만 학습되고 깊은 layer는 random.

### 3.2 원인
- Sigmoid/Tanh 깊게.
- 잘못된 init.
- BN/LN 없음.
- Skip connection 없음.

### 3.3 처방
- Activation: ReLU 계열로.
- Init: He.
- BN/LN 추가.
- Skip connection (ResNet).
- LSTM/GRU의 gating.

### 3.4 진단 코드 (직관)
```python
for name, p in model.named_parameters():
    if p.grad is not None:
        print(name, p.grad.norm().item())
```
깊을수록 작아지면 vanishing.

---

## 4. Exploding Gradient 진단

### 4.1 신호
- gradient norm이 1e3, 1e6 이상.
- Loss 갑자기 폭증 후 NaN.
- 특히 RNN/LSTM에서 흔함.

### 4.2 원인
- LR 큼.
- $W_{hh}$의 고유값 > 1 (RNN).
- Init 분산 큼.

### 4.3 처방
- **Gradient clipping** (norm 1~5).
- LR 줄임.
- Init 점검.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 5. Overfitting 진단

### 5.1 신호
- Train loss 계속 감소.
- Val loss 일정 epoch 후 증가.
- 차이 큼 (gap).

### 5.2 원인
- 데이터 부족 / 다양성 부족.
- 모델 너무 큼.
- Regularization 부족.

### 5.3 처방 (강도 순)
1. **Early stopping** — 가장 단순.
2. **Dropout / Weight decay** 늘림.
3. **Data augmentation** 강화.
4. **Label smoothing**.
5. **Mixup / CutMix**.
6. **모델 크기 축소** (마지막 수단).
7. **외부 데이터 추가** / pretrained 활용.

### 5.4 신호로 처방 강도 결정
- 작은 gap → augmentation.
- 큰 gap → dropout, weight decay 강화.
- 매우 큰 gap → 모델 줄이거나 데이터 많이 늘림.

---

## 6. Underfitting 진단

### 6.1 신호
- Train loss도 안 떨어짐.
- Val도 마찬가지.

### 6.2 원인
- 모델 너무 작음.
- LR 부적절.
- 비선형 부족 / 활성화 잘못.
- 데이터 정규화 X.
- Loss 잘못 매칭.

### 6.3 처방
1. **모델 키움** (depth, width).
2. **LR 재탐색** (LR range test).
3. **Regularization 줄임**.
4. **데이터 정규화** 확인.
5. **Output-loss 매칭** 점검.

---

## 7. Train ≪ Val (val이 더 좋음)

### 7.1 흔한 원인
- BN/Dropout이 train mode 그대로 (val에서 평가 시 model.eval() 안 부름).
- Val set이 train보다 쉬움 (분포 편향).
- 데이터 누수 (val 일부가 train에 들어감).
- 또는 정상 (BN/Dropout 효과로 가능).

### 7.2 처방
- `model.eval()` 호출 확인.
- 데이터 split 점검.
- Augmentation을 train에만 적용했는지 확인.

---

## 8. Loss는 떨어지는데 Accuracy 안 오름

### 8.1 원인
- Loss 안 잘못된 metric.
- 클래스 불균형 (다수 클래스만 잘 맞춤).
- Threshold 부적절 (이진 분류).

### 8.2 처방
- Confusion matrix 출력.
- 각 클래스별 accuracy.
- Macro F1 / class-weighted CE / Focal loss.
- Threshold 조정 (precision-recall trade-off).

---

## 9. BN 관련 흔한 실수

### 9.1 train/eval mode 미스
- 추론 시 `model.eval()` 안 부름 → batch마다 결과 다름.

### 9.2 작은 batch에서 BN 통계 부정확
- batch=2~4면 BN 통계 매우 noisy.
- → GN/LN으로 교체.

### 9.3 BN 직전 conv에 bias 두기
- BN이 평균을 빼버리므로 bias가 의미 없음.
- conv → BN 시 conv의 bias=False로.

### 9.4 BN과 Dropout 같은 위치
- 분산이 흔들려 충돌 가능.
- 보통 conv → BN → ReLU → (Dropout) → 다음 layer.

---

## 10. 데이터 누수(Data Leakage)

### 10.1 증상
- val/test 성능이 비현실적으로 높음.
- 배포 후 성능 급락.

### 10.2 흔한 원인
- 정규화 통계를 train+test 합쳐 fit.
- ID 같은 sample이 train·val에 모두.
- 시계열을 random split (미래로 과거 예측 leak).
- Feature engineering이 future 정보 사용.

### 10.3 처방
- Train만으로 통계 계산, val/test에 적용.
- Group-aware split (같은 환자/사용자는 한쪽에만).
- 시계열은 forward-chaining.

---

## 11. 분포 변화 (Distribution Shift)

### 11.1 증상
- Val 좋은데 deploy에서 못함.
- 시간 지날수록 성능 하락.

### 11.2 원인
- Train과 deploy의 입력 분포 다름 (covariate shift).
- 라벨 분포 변경 (label shift).
- 컨셉 변경 (concept drift, 예: 사기 패턴 변화).

### 11.3 처방
- Domain adaptation.
- Online learning.
- 정기적 재학습.
- 모니터링 (deploy 데이터 분포 추적).

---

## 12. 클래스 불균형

### 12.1 신호
- 다수 클래스 accuracy 99%, 소수 0%.
- F1 / recall이 낮음.

### 12.2 처방
- **Class weight**: CE에서 가중.
- **Focal loss**: 어려운 sample에 가중.
- **Oversampling** (소수 클래스 복제).
- **Undersampling** (다수 클래스 줄임).
- **SMOTE** (소수 클래스 합성).
- 평가 지표를 macro F1 / recall로.

---

## 13. RNN 특화 진단

### 13.1 자주 발생
| 증상 | 처방 |
|---|---|
| Gradient NaN | Gradient clipping 1.0 |
| Long-term 정보 손실 | LSTM/GRU/Attention |
| Exposure bias (생성) | Scheduled sampling |
| Memory 폭발 | Truncated BPTT |
| 학습 매우 느림 | LR 재탐색, batch packing |

### 13.2 디버깅
- 짧은 시퀀스로 먼저 작동 확인.
- 점진적으로 길이 늘림.
- gradient norm을 시점별로 출력.

---

## 14. Transformer 특화 진단

### 14.1 자주 발생
- **Warmup 없으면 학습 발산**: 큰 모델 + Adam → 초반 발산.
- **Attention 폭발**: softmax 입력이 너무 큼 → scaled dot-product (1/√d_k).
- **Positional Encoding 미스**: 학습 PE가 일반화 안 됨 → relative PE 고려.
- **Memory O(T²) 폭발**: FlashAttention / sparse attention.

---

## 15. 디버깅 체크리스트 (학습 시작 전 / 안 될 때)

### 15.1 시작 전 점검
- [ ] 입력 정규화 (mean, std)
- [ ] Output-loss 매칭 (sigmoid+BCE, softmax+CE)
- [ ] BN/Dropout train/eval mode
- [ ] Init이 활성화에 맞음 (He for ReLU)
- [ ] Gradient clipping (RNN/Transformer)
- [ ] LR이 적절 (LR range test)
- [ ] Val loss를 모니터
- [ ] Tiny dataset(10 sample)으로 overfit 가능한지 확인

### 15.2 안 될 때
- [ ] 첫 batch의 입출력 직접 확인 (NaN, 분포)
- [ ] Loss를 분해 (loss components가 모두 의미 있나)
- [ ] Gradient norm을 layer별로 출력
- [ ] LR 1/10 / 10× 시도
- [ ] 작은 모델로 baseline부터
- [ ] 데이터 누수 점검

---

## 16. "이런 증상엔 이걸" 매트릭스

| 증상 | 첫 시도 | 두 번째 |
|---|---|---|
| NaN | LR 1/10 | Gradient clip |
| Vanishing | ReLU + He init | BN, ResNet |
| Exploding | Gradient clip | LR↓, Init 점검 |
| Overfit | Early stop | Dropout, aug, weight decay |
| Underfit | LR 점검 | 모델 키움 |
| BN train/eval | model.eval() | GN/LN 교체 |
| 클래스 불균형 | Class weight | Focal loss, resample |
| Plateau | LR decay | Scheduler 점검 |
| Train ≪ Val | model.eval() | 데이터 누수 점검 |

---

## 17. "생각해보라" 확장

1. **Tiny dataset(10 sample)에서도 overfit 못 하면?** → 모델·loss·학습 코드 자체에 문제. 항상 첫 디버깅 단계로 권장.
2. **Gradient는 살아있는데 loss 안 떨어지면?** → 데이터 정규화, output-loss 매칭, label 정확성 점검.
3. **왜 Mixed Precision에서 NaN 자주?** → fp16의 동적 범위가 좁음 → loss scaling 필요.
4. **Train loss는 정확히 0인데 val은 형편없으면?** → 완전 overfit 또는 라벨 leak. 즉 val에 대해 무관한 신호로 train을 외움.
5. **왜 LR을 너무 작게 잡아도 학습이 안 되나?** → 너무 느려서 의미 있는 epoch 안에 진전 없음. 또는 saddle/local에 갇힘.

---

## 18. 한 줄 요약

- 모든 학습 실패는 **증상 → 원인 → 처방** 패턴.
- 가장 흔한 원인: **LR, init, BN/Dropout mode, 데이터 정규화, output-loss 매칭**.
- **Tiny dataset overfit**으로 코드 정상성 먼저 확인.
- 진단의 핵심 도구 = loss 곡선 + gradient norm + 출력 분포.
