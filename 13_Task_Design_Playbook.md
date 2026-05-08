# 13. ML Engineer's Task Design Playbook — "이 task를 받으면 어떻게 설계하나"

> 시험·면접 단골:
> *"당신이 의료 영상 분류 시스템을 설계해야 한다면 어떻게 접근하겠는가?"*
>
> 정답은 정의 나열이 아니라 **8단계 사고 흐름**으로 답하는 것.

---

## 0. ML 엔지니어의 8단계 설계 흐름 (★ 모든 task에 공통)

| 단계 | 묻는 것 |
|---|---|
| **1. 문제 정의** | 정확히 무엇을 풀려는가? 비즈니스 metric은? 비용 비대칭? |
| **2. 데이터 분석** | 양·라벨·분포·결측·이상치·leakage 위험 |
| **3. 전처리 / feature** | 어떻게 모델 입력으로 변환? |
| **4. 모델 선택** | 데이터 구조와 매칭되는 inductive bias의 모델은? Baseline부터. |
| **5. Loss / metric** | 학습 loss와 평가 metric은 일치해야 하나? 다르면 왜? |
| **6. 학습 전략** | batch, optimizer, schedule, regularization |
| **7. 흔한 실패 / 함정** | overfit, leakage, distribution shift, 클래스 불균형 |
| **8. 배포 / 모니터링** | latency, drift, 재학습 주기 |

**핵심 사고**: 모델 선택은 *4단계*에 불과하다. 1~3단계와 7~8단계가 ML 엔지니어와 학생의 차이를 만든다.

---

## 1. Task: 이미지 분류 (예: 의료영상 — 폐 X-ray 폐렴 진단)

### 1.1 문제 정의
- **이진 분류**: 폐렴 vs 정상.
- **비용 비대칭**: FN(폐렴 놓침) 비용 매우 큼 → **Recall 중요**.
- 비즈니스 metric: 의사 보조 → 민감도(Recall) 95% 이상에서 specificity 최대화.

### 1.2 데이터 분석
- 라벨 분포: 보통 정상 80% / 폐렴 20% (클래스 불균형).
- 데이터 양: 수만~수십만 장.
- **Leakage 위험**: 같은 환자의 여러 X-ray가 train/val 양쪽에 들어가면 안 됨 → **patient-level split**.
- 도메인 변화: 병원·기기마다 영상 특성 다름.

### 1.3 전처리
- 크기 통일 (224×224).
- 정규화 (ImageNet mean/std 또는 데이터셋 자체 통계).
- Augmentation: 회전(작게), 좌우 flip(해부학적으로 OK인지 확인), brightness/contrast.
  - **주의**: 의료 영상은 강한 augmentation이 거짓 패턴 만들 수 있음.

### 1.4 모델 선택 (왜?)
- **CNN** — 이미지의 inductive bias(국소성·평행이동)가 의료 영상 패턴(병변 위치 가변)에 적합.
- **Pretrained ResNet50 / EfficientNet** + transfer learning.
  - 왜 ImageNet pretrain? 얕은 층의 edge·texture는 도메인 무관.
  - Fine-tuning: 마지막 FC만 → 깊은 층까지 점진적 unfreeze.
- 데이터 매우 많으면 ViT도 고려.

### 1.5 Loss / Metric
- **Loss**: Class-weighted CE 또는 **Focal Loss** (불균형 보정).
- **Metric (평가)**: AUC-ROC, **PR-AUC** (불균형에 더 적합), 임계값 고정 시 Sensitivity/Specificity.
- Accuracy는 의미 없음 (다수 클래스 외움).

### 1.6 학습 전략
- Batch 32~64.
- Optimizer: SGD+momentum (이미지 일반화 우위) 또는 AdamW.
- LR: 1e-3 (Adam) / 0.01 (SGD), step decay or cosine.
- Pretrained 가중치는 작은 LR (예: backbone 1e-4, head 1e-3).
- Early stopping (val PR-AUC).

### 1.7 흔한 실패 / 함정
- **Patient-level leak**: 같은 환자가 양쪽에 → val 성능 부풀림.
- **편향**: 폐렴 환자 X-ray에 의료기기 마커가 더 자주 → 모델이 마커를 학습.
- **분포 변화**: 다른 병원에서 성능 하락.
- **클래스 불균형 무시**: accuracy 99%지만 recall 0%.

### 1.8 배포 / 모니터링
- **임계값 보정**: 비즈니스 요구(recall 95%+)에 맞게 threshold 조정.
- 모니터링: 입력 영상 분포(밝기, 크기), 예측 분포 변화, 의사 피드백.
- 정기 재학습 (분기/반기).
- **Explainability**: Grad-CAM으로 주의 영역 시각화 → 의사 신뢰 확보.

### 1.9 한 문장 답안 골격
> "ResNet50 pretrained + fine-tuning, patient-level split, focal loss + PR-AUC, Grad-CAM 시각화. Recall이 비대칭 비용을 반영하므로 핵심 metric. 배포 후 분포 변화 모니터링과 정기 재학습."

---

## 2. Task: Object Detection / Semantic Segmentation (예: 자율주행 — 차량/보행자 검출)

### 2.1 문제 정의
- **Object Detection**: bounding box + 클래스.
- **Segmentation**: pixel별 클래스.
- 자율주행: 실시간 (≥30 FPS) + 높은 안전성.
- 비용 비대칭: **FN(보행자 놓침) 치명적**.

### 2.2 데이터 분석
- 데이터 양: 큰 라벨링 비용 (bbox/mask 그려야).
- 라벨 분포: 차 多, 보행자 中, 자전거 少 (long-tail).
- 다양한 환경: 낮/밤, 비/맑음, 도시/고속도로.

### 2.3 전처리
- 큰 해상도 유지 (작은 객체 검출에 중요).
- Augmentation: random crop, scale jitter, color jitter, mosaic (YOLO).
- Mask는 동일 transform 적용 (입력과 일치).

### 2.4 모델 선택
**Detection 옵션**:
- **YOLO** (single-stage): 빠름, 실시간 적합, 정확도 약간 ↓.
- **Faster R-CNN** (two-stage): 정확하지만 느림.
- **DETR**: Transformer 기반, NMS 불필요.

**Segmentation 옵션**:
- **U-Net**: encoder-decoder + skip connection. 의료·일반 표준.
- **DeepLab**: dilated conv로 RF 확장.
- **Mask R-CNN**: instance segmentation.

자율주행 → **YOLO 계열** (실시간) + **DeepLab/U-Net** (segmentation).

### 2.5 Loss / Metric
**Detection**:
- Loss: classification CE + box regression (Smooth L1 또는 GIoU/CIoU).
- Metric: **mAP** (mean Average Precision), latency.

**Segmentation**:
- Loss: pixel-wise CE + **Dice Loss** (작은 영역에 강함).
- Metric: **IoU (Intersection over Union)** / mIoU, pixel accuracy.

### 2.6 학습 전략
- Batch 작음 (큰 모델·해상도) → GroupNorm 또는 큰 batch에서 BN.
- Pretrained backbone 거의 필수.
- Multi-scale training.

### 2.7 함정
- **작은 객체 놓침**: anchor box 크기 분포 점검, FPN 사용.
- **클래스 불균형**: focal loss, hard negative mining.
- **NMS threshold**: 너무 낮으면 중복, 너무 높으면 누락.

### 2.8 배포 / 모니터링
- **양자화 / pruning**으로 latency 감소.
- TensorRT, ONNX 변환.
- Edge 디바이스 (NVIDIA Jetson 등) 최적화.
- 도메인 변화 (계절, 도시) 모니터링.

### 2.9 사고 패턴
- **분류 → segmentation 차이**: 분류는 공간 압축 OK, segmentation은 pixel 출력 필요 → encoder-decoder.
- **검출은 분류 + 회귀의 결합**: 두 loss 동시 학습.

---

## 3. Task: 텍스트 분류 / 감성 분석 (예: 영화 리뷰 긍·부정)

### 3.1 문제 정의
- **이진 분류**: 긍정/부정. 또는 다중 (1~5점).
- 비즈니스: 추천·필터링·리포트.

### 3.2 데이터 분석
- 라벨 분포 (긍·부 비율).
- 텍스트 길이 분포 → max_length 결정.
- 다국어? 도메인 특이 어휘?
- **Leakage**: 같은 사용자의 리뷰가 train/test 양쪽에 → user-level split.

### 3.3 전처리
- Tokenization (subword: BPE, WordPiece).
- Lower-casing, punctuation 처리 (도메인에 따라).
- **너무 짧은/긴 문장 처리**: padding/truncation.
- **긴 문서**: 첫 N토큰 또는 sliding window + 평균.

### 3.4 모델 선택 (3 단계)

**Baseline** (빠른 prototype):
- TF-IDF + Logistic Regression — 의외로 강력. 종종 충분.

**Mid (딥러닝 입문)**:
- **TextCNN** (Yoon Kim 2014): embedding + 1D conv + max pool.
  - 빠름, 짧은 텍스트에 강함.
- **BiLSTM + attention**: 순서 정보, 긴 문서.

**SOTA**:
- **BERT fine-tuning**: 사전학습 + [CLS] 토큰 분류.
- **DistilBERT**: BERT의 경량화.

### 3.5 Loss / Metric
- Loss: CE (이진은 BCE, 다중은 categorical CE).
- Metric: Accuracy (균형 시), **F1/Macro-F1** (불균형 시), AUC.

### 3.6 학습 전략
- AdamW + warmup + linear decay (Transformer 표준).
- LR 2e-5 ~ 5e-5 (BERT fine-tuning), 1e-3 (TextCNN from scratch).
- Batch 16~32 (BERT 기준).
- Epoch 2~5 (BERT는 epoch 적게).

### 3.7 함정
- **리뷰의 부정문 처리**: "not good"이 긍정으로 잘못 분류 → BERT 같은 contextual 모델이 잘함.
- **OOV (out-of-vocab)**: subword tokenization이 해결.
- **데이터 누수**: 같은 사용자/제품 split 주의.
- **편향**: 특정 영화 장르에 편향.

### 3.8 배포 / 모니터링
- 추론 latency: BERT는 GPU 필요. 서비스 규모 따라 distillation/quantization.
- Drift 모니터링: 신조어, 새 도메인.
- 라벨 noise 모니터링.

### 3.9 사고 패턴
- "왜 BERT가 이김?" → 사전학습 + 양방향 + 풍부한 문맥.
- "TextCNN이 충분한 task?" → 짧고 키워드성. 영화 한 줄평.

---

## 4. Task: 시계열 예측 (예: 전력 수요 24시간 후 예측)

### 4.1 문제 정의
- **회귀**: 다음 N 시점의 수요량.
- 비즈니스: 발전 계획, 가격 결정.
- Metric: MAPE (% 오차), RMSE.

### 4.2 데이터 분석
- 시계열 특성:
  - **Trend** (장기 추세).
  - **Seasonality** (일·주·연 주기).
  - **이상치** (휴일, 사고).
- 외생 변수: 기온, 요일, 휴일.
- **Leakage 위험**: random split은 절대 금지. **forward chaining**.

### 4.3 전처리
- 정상성 (stationarity) 점검 → diff/log transform.
- 결측 처리 (보간 / forward fill).
- **Lag features**: 1시간 전, 24시간 전, 168시간 전 값.
- **Rolling stats**: 최근 N의 평균·분산.
- **시간 feature**: hour-of-day, day-of-week (sin/cos encoding).

### 4.4 모델 선택

**Baseline (필수 비교)**:
- **Naive**: 어제 같은 시간 값.
- **MA, ARIMA, ETS**: 전통 통계.

**ML/DL**:
- **GBM (XGBoost, LightGBM)**: feature 잘 만들면 매우 강력. 표 데이터의 강자.
- **LSTM / GRU**: 자동 시퀀스 학습.
- **TCN (Temporal CNN)**: 1D conv, 병렬, 빠름.
- **Transformer (Informer, Autoformer)**: 장기 의존성.

→ **GBM이 baseline에서 자주 우위**. 딥러닝은 데이터 매우 많고 비선형 강할 때.

### 4.5 Loss / Metric
- Loss: MSE (큰 오차에 민감) 또는 MAE (이상치 robust) 또는 Huber.
- Metric: **MAPE** (사업적 해석), RMSE.

### 4.6 학습 전략
- **Forward chaining CV**: 시간 순서대로 [t1~t100 train, t101~t110 val] → [t1~t110 train, t111~t120 val] ...
- 미래 정보 누수 절대 금지.
- Hyperparameter: window 길이, lag 개수.

### 4.7 함정
- **Random split** = 미래로 과거 예측 → 비현실적 성능.
- **Future leakage**: feature engineering에 미래 값 사용 (예: rolling mean이 미래 포함).
- **이상치 무시**: 휴일 효과 없으면 큰 오차.
- **Concept drift**: 코로나 같은 큰 이벤트로 패턴 변경.

### 4.8 배포 / 모니터링
- 실시간 예측 → 매 시간 / 매 분 batch.
- Drift 감지: 최근 오차 분포 모니터링.
- 정기 재학습 (월/분기).

### 4.9 사고 패턴
- "왜 시계열에 RNN이 자연?" → 시간 sharing + 가변 길이.
- "왜 GBM이 자주 이김?" → feature engineering이 시계열에 매우 효과적, 트리는 비선형 자연.

---

## 5. Task: 이상 탐지 (예: 신용카드 사기 탐지)

### 5.1 문제 정의
- **이진 분류**: 정상 vs 사기.
- **극단적 불균형**: 사기 < 0.1%.
- 비용 비대칭: FN(사기 통과) >>> FP(정상을 사기로).
- 라벨 부족 / 라벨 늦게 도착 (사후 confirm).

### 5.2 데이터 분석
- 사기 비율: 0.01~1%.
- 사기 패턴 변화: 새 수법 등장 (concept drift).
- Feature: 거래액, 시간, 위치, 사용자 history.

### 5.3 전처리
- 표 데이터 → **scale·encoding** (StandardScaler, target encoding).
- Time-based split (random split 금지).
- Aggregated features: 사용자별 최근 거래 통계.

### 5.4 모델 선택 — 라벨 유무에 따라

**Supervised** (라벨 충분):
- **GBM** (XGBoost, LightGBM, CatBoost): 표 데이터 강자.
- **Logistic Regression**: baseline + 해석성.
- DNN은 잘 안 씀 (feature engineering 우위).

**Semi-supervised / Unsupervised** (라벨 적음):
- **Isolation Forest**: 트리 기반 이상치.
- **Autoencoder**: 정상만 학습 → 재구성 오차로 이상치 탐지.
- **One-class SVM**.

→ 보통 **GBM 또는 GBM + AE 앙상블**.

### 5.5 Loss / Metric
- Loss: weighted CE / **Focal loss** (불균형 강함).
- Metric: 절대 accuracy 아님. **PR-AUC** (precision-recall), F1, Recall@k%.
- 비즈니스: precision 30% 이상에서 recall 최대화 (FP 비용 감안).

### 5.6 학습 전략
- **Resampling**: SMOTE (소수 클래스 합성), undersampling 다수.
- **Class weighting** in loss.
- Threshold 조정 (default 0.5 ≠ 최적).

### 5.7 함정
- **사후 라벨**: 사기는 며칠 후 confirm → train 시점에 사기로 안 알려진 sample이 정상으로 잘못 라벨.
- **Concept drift**: 새 사기 수법 → 모델 무력화. 정기 재학습 + monitoring.
- **Feedback loop**: 모델이 사기로 막은 거래는 confirm 안 됨 → 라벨 편향.

### 5.8 배포 / 모니터링
- **실시간 추론** (밀리초 단위).
- 임계값을 비즈니스 비용으로 결정 (FN 비용 vs FP 고객 불편).
- Drift detection: 거래 분포, 모델 confidence 분포.
- Human-in-the-loop: 모델이 의심하면 수동 검토.

### 5.9 사고 패턴
- "왜 accuracy 안 됨?" → 99.9% 정상 데이터에서 "전부 정상" 모델이 99.9% accuracy.
- "왜 AE?" → 정상 패턴만 학습 → 이상이면 재구성 안 됨.

---

## 6. Task: 추천 시스템 (예: 영화 추천)

### 6.1 문제 정의
- 사용자에게 좋아할 만한 아이템 N개 제시.
- Metric: CTR, watch time, 만족도(별점).
- 평가: **offline metric**(NDCG, Recall@k) → **online A/B test**.

### 6.2 데이터 분석
- **암묵적 피드백** (시청, 클릭) >> 명시적 (별점, 주로 적음).
- 매우 sparse: 사용자 수 × 아이템 수.
- **Cold start**: 신규 사용자/아이템.
- **Power law**: 인기 아이템에 데이터 집중, long tail.

### 6.3 전처리
- 사용자/아이템 ID → embedding으로.
- 시청 history → 시퀀스.
- 사용자 demographic, 아이템 metadata 활용.

### 6.4 모델 선택 — 단계별

**Collaborative Filtering (전통)**:
- **Matrix Factorization**: 사용자 × 아이템 행렬을 두 행렬의 곱으로.

**Deep Learning**:
- **Neural CF**: MF의 비선형 확장.
- **Wide & Deep** (Google): linear + DNN 결합.
- **Two-Tower**: 사용자 tower + 아이템 tower → embedding 유사도.
- **Session-based RNN/Transformer** (SASRec, BERT4Rec): 시청 순서 활용.
- **Graph (PinSage, LightGCN)**: 사용자-아이템 그래프.

**산업 표준**:
- **Two-tower for retrieval** (수백만 후보 → 수백) + **Ranking model** (DeepFM, DIN) for ranking (수백 → 수십).

### 6.5 Loss / Metric
- Loss:
  - **Pointwise**: BCE (클릭 / 미클릭).
  - **Pairwise**: BPR (좋아한 것 > 안 좋아한 것).
  - **Listwise**: Softmax CE over candidates.
- Metric: **NDCG@k, Recall@k, MAP** (offline) → CTR (online).

### 6.6 학습 전략
- Negative sampling: 안 본 아이템을 negative로 (random or popularity-weighted).
- 큰 batch (수천) + Adam.
- Cold start 처리: feature-based fallback.

### 6.7 함정
- **선호 ≠ 클릭**: 클릭 많아도 만족 안 할 수 있음.
- **Popularity bias**: 인기 아이템만 추천 → 다양성 감소.
- **Filter bubble**: 사용자 선호에만 갇힘.
- **Feedback loop**: 추천한 것만 데이터로 들어옴 → 추천 안 된 아이템은 영원히 안 보임.
- **A/B test 필수**: offline metric ≠ online 성과.

### 6.8 배포 / 모니터링
- **2단계 architecture**: retrieval (빠름, 큰 후보) → ranking (정밀, 작은 후보).
- 추천 latency < 100ms 목표.
- **Embedding** 주기적 갱신.
- A/B test, multi-armed bandit으로 신모델 점진 배포.

### 6.9 사고 패턴
- **추천 = 검색 + 랭킹**. retrieval 모델은 효율, ranking은 정확.
- **Two-tower의 본질**: 사용자·아이템을 같은 embedding 공간에 → dot product로 유사도.
- "왜 sequence model?" → 사용자의 최근 행동이 다음 행동에 강한 영향.

---

## 7. 6개 Task 비교 — "공통점과 차이"

| Task | 데이터 구조 | 핵심 모델 | Loss | 핵심 metric | 가장 큰 함정 |
|---|---|---|---|---|---|
| 이미지 분류 | 격자 | CNN + transfer | CE / Focal | PR-AUC | Patient leak, augmentation |
| Detection / Seg | 격자 | YOLO/U-Net | CE+IoU/Dice | mAP/mIoU | 작은 객체, NMS |
| 텍스트 분류 | 시퀀스 | TextCNN/BERT | CE | F1 / Accuracy | 부정문, OOV, leak |
| 시계열 | 시퀀스 | GBM/LSTM/TCN | MSE/MAE | MAPE | Future leakage |
| 이상 탐지 | 표/시퀀스 | GBM/AE | weighted CE | PR-AUC | 라벨 noise, drift |
| 추천 | 그래프/시퀀스 | Two-tower/MF | BCE/BPR | NDCG/CTR | Filter bubble, A/B |

---

## 8. ML 엔지니어 사고 — 공통 원칙 정리

### 8.1 Baseline부터
**언제나** 단순한 baseline (linear, GBM, naive) 먼저. 복잡한 모델은 baseline을 못 이기면 잘못된 것.

### 8.2 데이터 split이 무기보다 중요
- 의료 → patient-level
- 텍스트 → user-level
- 시계열 → forward chaining
- 추천 → time-based

random split은 거의 항상 leakage.

### 8.3 Metric은 비즈니스 비용 반영
- Accuracy는 거의 항상 부적절.
- 비대칭 비용 (의료, 사기) → recall / precision 따로.
- 분포 무관 (불균형) → PR-AUC.
- 순위 (검색·추천) → NDCG.

### 8.4 Loss와 Metric이 다를 수 있다
- Loss는 미분 가능해야 함 → CE.
- Metric은 비즈니스 신호 → F1, NDCG, mAP.
- 일치하지 않을 수 있음. 일치 안 하면 "왜?"를 답할 수 있어야.

### 8.5 함정 5가지 (어디서나)
1. **Data leakage** (split 잘못)
2. **Distribution shift** (배포 후)
3. **Class imbalance** (다수만 학습)
4. **Concept drift** (시간 따라 변경)
5. **Feedback loop** (추천·이상탐지)

### 8.6 배포 후가 진짜 시작
- 모니터링 (입력 분포, 예측 분포, 성능).
- 정기 재학습.
- A/B test로 신모델 검증.
- Explainability (의료·금융에서 필수).

---

## 9. 시험·면접 답안 골격

> **"X task를 어떻게 설계?"**
> 1. 문제·metric 정의 (비대칭 비용? 평가 기준?)
> 2. 데이터 split (leakage 방지)
> 3. Baseline (단순한 것부터)
> 4. 모델 선택 + 그 이유 (inductive bias 매칭)
> 5. Loss·metric 차이 (있으면 이유)
> 6. 학습 전략 (불균형·long-tail 처리)
> 7. 함정 (그 task 특이)
> 8. 배포·모니터링·재학습

이 8단계를 외워두면 모든 "어떻게 설계?" 질문에 답 가능.

---

## 10. "생각해보라" 확장

1. **의료 영상 분류에서 why pretrained ImageNet?** → 얕은 층의 edge·texture는 도메인 무관. 데이터 적을 때 매우 효과적.
2. **추천 시스템에서 왜 explicit rating보다 implicit feedback?** → 사용자는 별점 잘 안 줌. 클릭/시청은 자연스럽게 발생, 양 多.
3. **시계열에 deep learning이 GBM보다 못한 경우 多 — 왜?** → 시계열은 feature engineering이 강력. GBM이 그걸 잘 활용. 딥러닝은 raw data + 큰 데이터에서 빛남.
4. **이상 탐지에 supervised vs unsupervised?** → 라벨 충분하고 알려진 패턴 → supervised. 새 패턴 / 라벨 없음 → unsupervised(AE, Isolation Forest).
5. **Object detection의 anchor box는 왜?** → 다양한 크기·비율의 객체에 대한 prior. anchor-free (CenterNet, FCOS) 식도 발전.
6. **번역을 RNN으로 풀면? Transformer로 풀면?** → RNN은 순차 → 느림 + long-range 약함. Transformer는 병렬 + long-range. 번역 길이 길수록 Transformer 우위.

---

## 11. 한 줄 요약

- ML 엔지니어 = **8단계 흐름** (정의 → 데이터 → 전처리 → 모델 → loss/metric → 학습 → 함정 → 배포).
- 모델 선택은 한 단계에 불과. **데이터 split·loss·metric·함정 인식**이 차이를 만든다.
- 모든 "이 task 어떻게?" 질문엔 **8단계 + 그 task 특이 함정** 으로 답.
- Baseline → 모델 비교 → 함정 점검 → 배포 → 모니터링의 사이클이 ML 엔지니어링.
