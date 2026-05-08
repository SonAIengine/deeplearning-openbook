# 13. ML Engineer's Task Design Playbook — "이 task를 받으면 어떻게 설계하나" 심층

> **이 문서의 목표**:
> 시험·면접 단골인 *"당신이 X task를 맡으면 어떻게 설계하겠는가"* 질문에 *완전한 답안*을 만든다.
>
> 답안의 골격 = **8단계 설계 흐름**. 모델 선택은 그중 한 단계에 불과 — *데이터 split·loss/metric·함정 인식·배포·모니터링*이 ML 엔지니어와 학생을 가르는 차이.
>
> 6개 보편 task를 깊이 다룬다 — 의료 이미지, 자율주행, 텍스트 분류, 시계열 예측, 이상 탐지, 추천 시스템.

---

## 0. ML 엔지니어의 8단계 설계 흐름

### 0.1 8단계

| 단계 | 묻는 것 |
|---|---|
| **1. 문제 정의** | 정확히 무엇을 풀려는가? 비즈니스 metric? 비용 비대칭? Latency 요구? |
| **2. 데이터 분석** | 양·라벨·분포·결측·이상치·leakage 위험·도메인 차이 |
| **3. 전처리 / Feature** | 어떻게 모델 입력으로 변환? 정규화·인코딩·augmentation |
| **4. 모델 선택** | 데이터 구조 매칭 inductive bias? Baseline부터. |
| **5. Loss / Metric** | 학습 loss와 평가 metric. 일치 여부, 다르면 왜? |
| **6. 학습 전략** | Batch, optimizer, schedule, regularization, 클래스 처리 |
| **7. 흔한 실패 / 함정** | 그 task 특이 함정. Leakage, drift, 불균형 |
| **8. 배포 / 모니터링** | Latency, drift detection, 정기 재학습 |

이 8단계가 *답안의 표준 구조*. 어떤 task든 이 골격으로 답.

### 0.2 핵심 사고

**모델 선택은 4단계에 불과**. 1~3단계와 7~8단계가 ML 엔지니어와 학생의 차이를 만든다. 학생은 "어떤 모델?"만 답하고, 엔지니어는 8단계 모두를 답한다.

### 0.3 본 챕터의 흐름

§1: 의료 이미지 분류 (CV 전형)
§2: 자율주행 Detection/Segmentation (CV 응용)
§3: 텍스트 분류·감성 (NLP)
§4: 시계열 예측 (시계열)
§5: 이상 탐지 (불균형·라벨 부족)
§6: 추천 시스템 (sparse feedback)
§7: Task별 비교
§8: ML 엔지니어 사고 정리
§9: 면접 Q&A
§10: 생각해보라

---

## 1. Task: 의료 영상 분류 — 폐 X-ray 폐렴 진단

### 1.1 문제 정의

**핵심 task**: 흉부 X-ray에서 폐렴 vs 정상 분류 (이진).

**비즈니스 metric**:
- 의사 보조용 — 의사가 *놓친* 폐렴을 잡는 것이 핵심.
- **FN (폐렴 놓침)이 매우 비싼 비용** — 환자 사망 위험.
- **FP (정상을 폐렴으로)는 추가 검사로 흡수** — 비용 적음.

**비용 비대칭 → Recall 중요**. 표적: Recall 95%+ 에서 specificity 최대화.

**Latency**: 의사 review 보조이므로 분 단위 OK.

### 1.2 데이터 분석

**양**: NIH 등 공개 데이터셋 수십만 장. Hospital-internal 데이터 추가 가능.

**라벨**:
- 보통 정상 80~85%, 폐렴 15~20%.
- 라벨 noise — 의사 간 disagreement (kappa 0.5~0.7 정도).

**분포 위험**:
- **병원·기기 간 차이**: 다른 X-ray 기기, 다른 환자 demographics.
- **시간 변화**: 코로나 같은 새 패턴.

**Leakage 위험 (★)**:
- **Patient-level**: 같은 환자의 여러 X-ray가 train·test 양쪽에. 모델이 "환자" 학습 → val 부풀림.
- **Image marker leakage**: 폐렴 환자 X-ray에 *의료기기 마커가 더 자주* (ICU 등) → 모델이 마커를 학습. 다른 환경에서 무력.

### 1.3 전처리

**(1) 크기 통일**: 224×224 또는 512×512.

**(2) 정규화**: ImageNet mean/std 또는 데이터셋 통계.

**(3) Augmentation** (의료 특이 주의):
- Rotation 작게 (±5도) — 큰 회전은 비의학적.
- 좌우 flip — 해부학적으로 OK인지 점검 (좌심·우심).
- Brightness/contrast 작은 jitter.
- Crop·zoom 가능.
- *피하기*: 강한 색 변환 (X-ray는 그레이), 큰 affine.

**(4) 라벨 정제**: 라벨 noise 줄임. Multi-rater consensus 또는 expert review.

### 1.4 모델 선택

**Baseline**: 작은 ResNet-18 + transfer learning. 빠른 prototype.

**SOTA 시도**:
- **ResNet-50/EfficientNet** + ImageNet pretrained + fine-tune. 표준.
- **DenseNet-121** — 의료 영상에서 자주 사용 (CheXNet).
- **데이터 매우 많으면 ViT** (단 의료엔 종종 CNN이 우위).

**왜 CNN?** 
- 이미지의 국소성·평행이동 inductive bias가 의료 영상에 자연.
- 작은 데이터에서 transfer learning 매우 효과적.

**왜 transfer learning?**
- 얕은 layer의 generic feature (edge, texture) 도메인 무관.
- 의료 데이터 적어도 ImageNet 가중치로 시작 → 빠른 수렴.

### 1.5 Loss / Metric

**Loss**:
- **Class-weighted CE** (불균형 보정).
- 또는 **Focal Loss** ($\gamma=2$) — 어려운 sample에 가중.

**Metric**:
- **PR-AUC** — 클래스 불균형에 강함.
- **Recall**: 비즈니스 표적 (95%+).
- **AUC-ROC** — 임계값 무관 비교.
- 임계값 고정 시 Sensitivity/Specificity.
- **Accuracy 무의미** — 80% 정상에서 "전부 정상"이 80% accuracy.

### 1.6 학습 전략

**Batch**: 32~64.

**Optimizer**: SGD+momentum (이미지 분류, 일반화 우위) 또는 AdamW.

**LR**:
- Pretrained backbone: 1e-4 (작게).
- 새 head: 1e-3 (크게).
- Discriminative LR.

**Schedule**: Step decay (30/60/90 epoch에 1/10) 또는 cosine.

**Regularization**:
- Dropout 0.3 (FC head).
- Weight decay 1e-4.
- Augmentation 위 명시.

**Early stopping**: Val PR-AUC 기준.

### 1.7 흔한 실패 / 함정

**(1) Patient-level Leak**:
- 증상: Val PR-AUC 매우 높음, deploy에서 급락.
- 처방: split 시 patient ID 단위.

**(2) Spurious Correlation**:
- 증상: 모델이 의료기기 마커를 학습. 새 환경 무력.
- 처방: Grad-CAM으로 모델이 보는 영역 점검. 마커 영역 mask 또는 데이터 다양화.

**(3) Distribution Shift**:
- 증상: 다른 병원에서 성능 하락.
- 처방: Multi-center 데이터, domain adaptation.

**(4) 라벨 Noise**:
- 증상: Train loss 0 도달 못함.
- 처방: Label smoothing, multi-rater consensus, noise-robust loss (Symmetric CE).

**(5) 클래스 불균형 무시**:
- 증상: Accuracy 95%지만 recall 30%.
- 처방: Class weight, focal loss, threshold 조정.

### 1.8 배포 / 모니터링

**임계값 보정**:
- 비즈니스 표적 (recall 95%)에 맞게.
- Precision-Recall curve 분석 후 결정.
- 단순 0.5는 거의 사용 안 함.

**모니터링**:
- 입력 분포: 영상 mean·std·percentile 추적.
- 예측 분포: 폐렴 비율 추적 (자연 변동 vs drift).
- 의사 피드백: 모델 예측이 의사 결정과 일치 비율.
- 정기 재학습: 분기 또는 반기.

**Explainability** (의료 필수):
- **Grad-CAM**: 주의 영역 시각화. 의사 신뢰 확보.
- **Confidence calibration**: 모델 출력이 진짜 확률에 근접 (Platt scaling, temperature).
- **Uncertainty quantification**: 모델이 *확신 없으면 그렇게 표시*.

### 1.9 답안 골격 (★ 시험·면접용)

> "ResNet50/EfficientNet pretrained + fine-tune. **Patient-level split** 필수. **Focal loss + PR-AUC**가 핵심 metric — recall이 비대칭 비용 (FN 치명적) 반영. **Grad-CAM 시각화**로 의사 신뢰 확보. **함정**: patient leak, spurious correlation (의료기기 마커), distribution shift (병원 간), 라벨 noise. **배포**: threshold 비즈니스 비용에 맞게 조정, 분포 모니터링, 정기 재학습. ML 엔지니어 일은 모델보다 *데이터 split·평가·explainability·모니터링*."

---

## 2. Task: 자율주행 — 차량/보행자 Detection·Segmentation

### 2.1 문제 정의

**핵심 task**: 카메라·LiDAR 입력에서 객체 검출 (bbox + class) 및/또는 segmentation (pixel mask).

**비즈니스 metric**:
- **실시간 (≥30 FPS)** — latency 매우 중요.
- **안전성** — FN (보행자 놓침) 치명적.
- 정확도 + 재현율 + latency의 trade-off.

**Latency**: 30 ms 이하. Edge device (NVIDIA Jetson, Tesla FSD chip) 동작.

### 2.2 데이터 분석

**양**: KITTI, Cityscapes 등 공개 + 회사 내부 수백만 frame.

**라벨링 비용**:
- Bounding box: 비교적 빠름.
- Pixel-level segmentation: 매우 비쌈 (한 frame에 시간~분).
- Polygon annotation 또는 active learning으로 효율화.

**라벨 분포 (long-tail)**:
- 차 多, 보행자 中, 자전거·motorcycle 少, 동물·debris 매우 少.
- 클래스 불균형 심함.

**다양한 환경**:
- 낮·밤, 비·맑음, 도시·고속도로.
- 카메라 종류, 시야각, 해상도.

### 2.3 전처리

**해상도**: 큰 해상도 유지 (작은 객체 검출에 중요). 1920×1080 등.

**Augmentation** (자율주행 특이):
- Random crop, scale jitter.
- Color jitter (낮·밤 일반화).
- Mosaic (YOLO식 — 4 frame을 하나로 합쳐 다양한 scale).
- Cutmix, MixUp.
- 단 *해부학적*으로 합리적이어야 — 좌우 flip은 OK 도로에선.

**Mask 처리**: 입력 transform과 동일한 mask transform (rotation, crop).

### 2.4 모델 선택

**Detection 옵션**:

| 모델 | 특징 |
|---|---|
| **YOLO (single-stage)** | 빠름, 실시간. 자율주행 표준 |
| **Faster R-CNN (two-stage)** | 정확하지만 느림. 학술 연구 |
| **DETR** | Transformer 기반, NMS 불필요. 최근 |
| **CenterNet, FCOS** | Anchor-free |

자율주행: YOLO 계열 (YOLOv5, YOLOv8) — 실시간 + 정확도 균형.

**Segmentation 옵션**:

| 모델 | 특징 |
|---|---|
| **U-Net** | 의료에서 시작. 일반 표준 |
| **DeepLab** | Dilated conv로 RF 확장 |
| **Mask R-CNN** | Instance segmentation |
| **PSPNet** | Pyramid pooling |

자율주행 segmentation: DeepLab 또는 U-Net 기반.

**Backbone**: ResNet, EfficientNet, 또는 MobileNet (edge 친화).

### 2.5 Loss / Metric

**Detection**:
- Loss: classification CE/focal + box regression (Smooth L1, GIoU/CIoU).
- Metric: **mAP** (mean Average Precision), latency (FPS), failure rate.

**Segmentation**:
- Loss: pixel-wise CE + **Dice loss** (작은 영역에 강함).
- Metric: **mIoU** (mean Intersection over Union), pixel accuracy.

### 2.6 학습 전략

**Batch**: 작음 (큰 모델·해상도 메모리). 8~32.

**Norm**: GroupNorm (작은 batch BN 부정확).

**Optimizer**: SGD+momentum 또는 AdamW.

**LR**: 1e-2 (SGD), 1e-4 (AdamW). Warmup + cosine.

**Pretrained**: ImageNet pretrained backbone 거의 필수.

**Multi-scale training**: 다양한 해상도로 학습 → robust.

### 2.7 흔한 실패 / 함정

**(1) 작은 객체 놓침**:
- 증상: 멀리 있는 보행자 검출 실패.
- 처방: Anchor box 크기 분포 점검, FPN (Feature Pyramid Network) — 다양한 RF 동시 사용.

**(2) 클래스 불균형**:
- 증상: 보행자 mAP 매우 낮음 (long-tail).
- 처방: Focal loss, hard negative mining, copy-paste augmentation.

**(3) NMS Threshold**:
- 너무 낮음: 중복 박스. 너무 높음: 누락.
- Soft NMS 또는 학습된 NMS.

**(4) Edge device 효율**:
- 증상: 학습 모델은 정확하지만 실시간 안 됨.
- 처방: Quantization (FP32 → INT8), pruning, distillation.

**(5) 데이터 누수**:
- 같은 동영상의 frame들이 train·test에 → 정보 누수.
- 처방: 비디오 단위 split.

### 2.8 배포 / 모니터링

**최적화**:
- **Quantization**: FP32 → INT8 (4배 빠름, 정확도 약간 손해).
- **Pruning**: 불필요 가중치 제거.
- **Distillation**: 큰 모델 → 작은 student.
- **TensorRT, ONNX 변환**: 추론 최적화.

**Edge device**:
- NVIDIA Jetson, Tesla FSD chip.
- 메모리·연산 제약 → 작은 모델 사용.

**Monitoring**:
- 환경 분포 (계절, 도시).
- 새 객체 종류 (e-scooter 등).
- Failure mode (실수 case 수집).

### 2.9 답안 골격

> "**YOLO (실시간) + DeepLab/U-Net (segmentation)** 조합. Backbone은 ImageNet pretrained ResNet 또는 MobileNet (edge). **Loss**: CE + GIoU (detection), CE + Dice (segmentation). **Metric**: mAP, mIoU, latency. **Train**: SGD/AdamW + cosine + multi-scale + heavy augmentation (mosaic). **함정**: 작은 객체 (FPN), 클래스 불균형 (focal), NMS, edge 효율 (quantization). **배포**: TensorRT 변환, INT8 quantization, 환경 분포 모니터링."

---

## 3. Task: 텍스트 분류·감성 분석

### 3.1 문제 정의

**핵심 task**: 영화 리뷰의 긍·부정 분류 (또는 1~5 별점).

**비즈니스 metric**:
- 추천·필터링·리포트 사용.
- 보통 균형 라벨 → accuracy/F1.
- 불균형이면 (예: 부정 리뷰 5%) Recall, F1.

**Latency**: 보통 sec 단위 OK (배치 처리). 실시간 챗봇이면 <100ms.

### 3.2 데이터 분석

**양**: IMDB (50K), Yelp (수백만), 회사 내부.

**라벨**:
- 명시적 별점: 신뢰 좋음.
- Implicit (좋아요): 신뢰 약간 낮음.
- 라벨 noise — 별점 1점이 "그저 그럼"일 수도.

**텍스트 길이 분포**:
- 트윗 (매우 짧음, ~140자).
- 리뷰 (중간, ~500단어).
- 긴 문서 (수천 단어).

→ Max length 결정.

**다국어? 도메인 특이?**: 영화 vs 식당 vs 전자제품.

**Leakage 위험**:
- **User-level**: 같은 사용자의 리뷰가 train·test에. 모델이 사용자 스타일 학습.
- **Time leakage**: 영화/제품의 인기 시기. Random split이 우연히 leak.

### 3.3 전처리

**Tokenization**:
- Subword: BPE (GPT), WordPiece (BERT). OOV 처리.
- 한국어: SentencePiece 또는 KoBERT 토크나이저.

**Cleaning**:
- Lower-casing (case-sensitive task가 아니면).
- 특수문자 처리 (도메인에 따라).
- Emoji 처리 (감성에 정보 있음).

**Length 처리**:
- Max length 256~512 (BERT 표준).
- 긴 문서: 첫 N토큰, 또는 sliding window + 평균.

**라벨 정제**: 명시적 별점이 있으면 그것 사용.

### 3.4 모델 선택

**3 단계 접근**:

**Baseline**:
- TF-IDF + Logistic Regression. 5분 학습.
- 의외로 강력. 종종 충분.

**Mid (딥러닝 입문)**:
- **TextCNN** (Yoon Kim 2014): embedding + 1D conv + max pool. 30분 학습. 짧은 텍스트 강함.
- **BiLSTM + attention**: 순서 정보, 긴 문서.

**SOTA**:
- **BERT/DistilBERT fine-tuning**: 1~3시간 학습.
- **DeBERTa, RoBERTa**: BERT 개선.
- 매우 큰 데이터 + 다양한 task: GPT-3.5/4 zero-shot.

**왜 BERT?**
- 사전학습으로 generic 언어 이해.
- 양방향 문맥 (decoder-only GPT는 단방향).
- Fine-tuning이 작은 데이터로도 효과적.

### 3.5 Loss / Metric

**Loss**:
- Binary CE (이진) 또는 Categorical CE (다중).
- Class weight (불균형).
- Label smoothing (큰 모델).

**Metric**:
- **Accuracy** (균형 시).
- **F1 / Macro-F1** (불균형 시).
- **AUC** (임계값 무관).
- 다중 분류: Macro-F1 vs Micro-F1 의식적 선택.

### 3.6 학습 전략

**Batch**: 16~32 (BERT 기준, 메모리 한계).

**Optimizer**: AdamW + linear decay + warmup.

**LR**: 2e-5 ~ 5e-5 (BERT fine-tuning), 1e-3 (TextCNN from scratch).

**Epoch**: 2~5 (BERT는 적게 — overfit 빠름).

**Regularization**:
- Dropout 0.1 (BERT default).
- Weight decay 0.01 (AdamW).

### 3.7 흔한 실패 / 함정

**(1) 부정문 처리**:
- "not good"이 긍정으로 잘못 분류. TextCNN의 max pool이 "good"만 잡음.
- 처방: BERT 같은 contextual 모델, 또는 negation 처리 feature.

**(2) OOV (Out-of-Vocabulary)**:
- 새 단어 (신조어, 고유명사).
- 처방: Subword tokenization (BPE, WordPiece) — OOV 거의 사라짐.

**(3) Leakage**:
- User-level leak: 같은 사용자가 train·test에. 처방: user-level split.
- Time leak: 영화 출시 시기 학습. 처방: time-based split.

**(4) 편향**:
- 특정 영화 장르·작가·국적에 편향.
- 처방: 데이터 다양화, fairness metric 점검.

**(5) Class Imbalance**:
- 부정 리뷰 5%만.
- 처방: Class weight, focal loss, oversampling.

### 3.8 배포 / 모니터링

**추론 Latency**:
- BERT는 GPU 필요. 서비스 규모 따라 distillation/quantization.
- DistilBERT (4배 빠름, 95% 정확도).

**Drift Monitoring**:
- 신조어, 새 도메인 (밈, 트렌드).
- 입력 토큰 분포 추적.
- 정기 재학습 (월·분기).

**Bias Audit**:
- 다양한 demographic의 리뷰 성능 확인.
- Fair ML practices.

### 3.9 답안 골격

> "**BERT/DistilBERT fine-tune** + AdamW + warmup + cosine. **Subword tokenization** (BPE/WordPiece) + max length 256~512. **User-level/time split** 필수. **Loss**: CE + class weight (불균형) + label smoothing. **Metric**: F1/Macro-F1 (불균형), AUC. **함정**: 부정문 처리 (contextual model), OOV (subword), leakage (split 신중), 편향 (audit), 불균형. **배포**: distillation/quantization for latency, drift monitoring (신조어). 짧은 텍스트엔 TextCNN baseline로 시작 — 의외로 강력."

---

## 4. Task: 시계열 예측 — 전력 수요 24시간 후

### 4.1 문제 정의

**핵심 task**: 다음 24시간의 전력 수요량 예측.

**비즈니스 metric**:
- 발전 계획·가격 결정.
- 큰 오차 비용 — 부족하면 정전, 과잉하면 비효율.
- **MAPE (% 오차)** — 비즈니스 해석 좋음.

**Latency**: 시간 단위 (매시간 또는 매분 batch).

### 4.2 데이터 분석

**시계열 특성** (★):

**(1) Trend**: 장기 추세 (인구·기술 변화).

**(2) Seasonality**:
- 일 주기 (낮·밤).
- 주 주기 (평일·주말).
- 연 주기 (여름·겨울).

**(3) 이상치**:
- 휴일 (설·추석, 크리스마스).
- 사고·재해.
- 코로나 같은 큰 이벤트.

**(4) 외생 변수**:
- 기온, 습도.
- 요일.
- 휴일.
- 경제 지표.

**Leakage 위험 (★)**:
- **Random split 절대 금지** — 미래로 과거 예측.
- **Forward chaining** 표준.

### 4.3 전처리

**(1) 정상성 점검**: ADF test, KPSS test. Trend 있으면 diff/log transform.

**(2) 결측 처리**: Interpolation, forward fill, 또는 별도 처리 (트리 모델).

**(3) Lag features**:
- 1시간 전, 24시간 전 (전일 같은 시간), 168시간 전 (전주 같은 시간).
- 가장 강한 신호 (자기상관).

**(4) Rolling stats**:
- 최근 N의 평균·분산·max·min.
- 단 *미래 정보 누수 점검* — "오늘 포함" 계산은 leak.

**(5) 시간 feature**:
- Hour-of-day, day-of-week.
- **Sin/cos encoding**: 주기성 유지 (12시와 0시가 가까움).

### 4.4 모델 선택

**Baseline (필수)**:
- **Naive**: 어제 같은 시간 값. 의외로 강력.
- **MA (Moving Average)**: 최근 N 평균.
- **ARIMA, ETS**: 전통 통계. 단순한 trend·seasonality에 강함.

**ML/DL**:

**(1) GBM (XGBoost, LightGBM)**:
- 표 데이터의 강자.
- Feature engineering 잘 되어 있으면 매우 효과적.
- 시계열 baseline에서 자주 우위.

**(2) LSTM/GRU**:
- 자동 시퀀스 학습.
- 단 GBM보다 느리고 종종 못함.

**(3) TCN (Temporal Convolutional Network)**:
- 1D conv, dilated.
- 병렬, 빠름.
- LSTM과 비슷한 성능, 더 안정.

**(4) Transformer (Informer, Autoformer)**:
- 장기 의존성에 강함.
- 단 데이터 매우 많아야.

**선택 가이드**:
```
시계열 예측?
├── Feature engineering 잘됨, baseline 충분 → GBM
├── 짧은 의존성, 데이터 적음 → LSTM/TCN
├── 장기 의존성, 데이터 많음 → Transformer
└── Production simplicity 원함 → GBM
```

### 4.5 Loss / Metric

**Loss**:
- **MSE**: 큰 오차 민감. 일반.
- **MAE**: 이상치 robust.
- **Huber**: 절충.
- **Quantile loss**: 분위수 예측 (예: 10%, 50%, 90% 분위 동시).

**Metric**:
- **MAPE** (% 오차) — 비즈니스 해석 좋음.
- **RMSE** — 큰 오차 강조.
- **MAE** — 직관적.
- **sMAPE** — symmetric, 안정.

### 4.6 학습 전략

**Forward Chaining CV**:
```
[t1~t100 train, t101~t110 val]
[t1~t110 train, t111~t120 val]
[t1~t120 train, t121~t130 val]
...
```

미래로 과거 예측 leakage 절대 방지.

**Hyperparameter**:
- Window 길이 (lag 개수).
- Forecast horizon (몇 step ahead).

**Regularization**:
- Trend·seasonality 강하면 정규화 약하게 (overfit 위험 적음).
- Noise 많으면 강하게.

### 4.7 흔한 실패 / 함정

**(1) Random Split**:
- 미래로 과거 예측 → 비현실적 성능.
- 처방: Forward chaining.

**(2) Future Leakage**:
- Rolling mean이 "오늘 포함"이면 leak.
- Feature에 미래 정보 (예: "다음 주 휴일") 포함 → leak.
- 처방: Feature engineering 점검.

**(3) 이상치 무시**:
- 휴일 효과, 코로나 같은 이벤트.
- 처방: 명시적 feature, robust loss.

**(4) Concept Drift**:
- 코로나, 산업 변화로 패턴 변경.
- 처방: 정기 재학습, 최근 데이터 가중.

**(5) Holiday Effect**:
- 설·추석은 평소와 매우 다름.
- 처방: Holiday feature, 별도 모델.

### 4.8 배포 / 모니터링

**추론**:
- 실시간 예측 → 매 시간 또는 매 분 batch.
- Latency 매우 빠름 (ms 단위).

**모니터링**:
- 최근 N일 오차 분포 추적.
- Drift detection — 갑작스런 변화 alert.
- 정기 재학습 (월).

### 4.9 답안 골격

> "Baseline부터: naive (어제 값) → MA → ARIMA/ETS → GBM (표 변환) → LSTM/TCN → Transformer. **Feature engineering이 핵심** — lag, rolling stats (미래 leak 점검), 시간 feature (sin/cos encoding), 외생 변수 (기온, 휴일). **Forward chaining CV** 필수. **Loss MSE/MAE/Huber, Metric MAPE/RMSE**. **함정**: random split 금지, future leakage, 이상치 (휴일·이벤트), concept drift. **GBM이 baseline에서 자주 우위** — 딥러닝은 매우 큰 데이터·강한 비선형에서. 정기 재학습 + 모니터링."

---

## 5. Task: 이상 탐지 — 신용카드 사기

### 5.1 문제 정의

**핵심 task**: 거래의 사기 vs 정상 분류 (이진).

**특이성**:
- **극단적 불균형**: 사기 < 0.1%.
- **비용 비대칭**: FN (사기 통과) >>> FP (정상을 사기로).
- **라벨 부족 / 라벨 늦게 도착** — 사기는 사후 confirm.
- **Concept drift 강함** — 새 사기 수법 등장.

**비즈니스 metric**:
- Precision 30%+ 에서 Recall 최대화 (FP 비용 — 고객 불편 — 감안).
- Real-time alert.

### 5.2 데이터 분석

**양**: 수백만~수억 거래.

**라벨**:
- 사기 비율 0.01~1%.
- 라벨 지연 — 사기 confirm은 며칠 후.

**Feature**:
- 거래액, 시간, 위치.
- 사용자 history (최근 거래량, 빈도).
- 카드 정보, 가맹점 정보.

**분포 변화**:
- **Concept drift**: 새 사기 패턴.
- 거래 패턴 자연 변화 (시즌, 명절).

### 5.3 전처리

**(1) Scale 처리**: StandardScaler 또는 RobustScaler (이상치).

**(2) Categorical Encoding**:
- One-hot (low cardinality).
- Target encoding (high cardinality, 단 leakage 점검).

**(3) Aggregated Features (시간 window)**:
- 사용자 최근 1시간/1일 거래 수, 총액.
- 가맹점 최근 사기율.
- 단 *미래 정보 누수* 점검.

**(4) Time-based Split**:
- 시간 순서대로 train/val/test.
- Random split 금지.

### 5.4 모델 선택

**라벨 충분 (Supervised)**:

**(1) GBM (XGBoost, LightGBM, CatBoost)**:
- 표 데이터 강자.
- 사기 탐지 표준.
- Feature engineering 효과 큼.

**(2) Logistic Regression**:
- Baseline + 해석성.
- Production에서 자주 사용 (단순, 빠름).

**(3) DNN**:
- GBM보다 종종 못함.
- 매우 큰 데이터 + 강한 카테고리 (수만 개)에서만 경쟁.

**라벨 부족 (Semi-supervised / Unsupervised)**:

**(1) Isolation Forest**:
- 트리 기반 이상치.
- 정상 거래 분포 학습 → 이상 거래는 멀리.

**(2) Autoencoder**:
- 정상 거래로만 학습.
- 재구성 오차로 이상치 탐지.

**(3) One-class SVM**:
- 정상 boundary 학습.

**실무**: 보통 **GBM + AE 앙상블** — supervised의 정확도 + unsupervised의 신규 패턴 탐지.

### 5.5 Loss / Metric

**Loss**:
- **Class-weighted CE** (불균형 보정).
- **Focal loss** ($\gamma=2$) — 어려운 sample 가중.

**Metric**:
- **PR-AUC** — 클래스 불균형 강함. 표준.
- **F1** — 보조.
- **Recall@k%** — top k% 거래에서 recall.
- **Accuracy 절대 무의미** — 99.9% 정상에서 "전부 정상"이 99.9%.

**비즈니스 metric**:
- Precision 30%+에서 Recall 최대화.
- 또는 reviewer's queue 처리 효율.

### 5.6 학습 전략

**Resampling**:
- **Oversampling**: 사기 sample 복제.
- **SMOTE**: 사기 sample 합성.
- **Undersampling**: 정상 sample 줄임 (정보 손실 위험).

**Class Weighting in loss**.

**Time-based Split** + **Walk-forward Validation**.

**Threshold 조정**: Default 0.5 ≠ 최적. PR curve 분석 후 결정.

### 5.7 흔한 실패 / 함정

**(1) 사후 라벨 (Label Delay)**:
- 사기는 며칠 후 confirm.
- Train 시점에 사기로 안 알려진 sample이 정상으로 잘못 라벨.
- 처방: Label correction, 시간 buffer.

**(2) Concept Drift**:
- 새 사기 수법 → 기존 모델 무력.
- 처방: 정기 재학습 (주·월), drift detector, online learning.

**(3) Feedback Loop**:
- 모델이 막은 거래는 confirm 안 됨 → 라벨 편향.
- 처방: A/B test, 일부 sample은 모델 무시.

**(4) Random Split**:
- 시간 순서 무시 → 미래 정보 누수.
- 처방: Time-based split.

**(5) Imbalance Mishandling**:
- Accuracy 99.9%지만 recall 0%.
- 처방: PR-AUC, focal loss, threshold.

### 5.8 배포 / 모니터링

**Real-time Inference**:
- 밀리초 단위 latency.
- GBM이 빠르고 가벼움 (모바일 가능).

**Threshold 조정**:
- 비즈니스 비용 (FN vs FP)에 따라 동적.
- Reviewer's queue 길이에 따라.

**Drift Detection**:
- 거래 분포 추적.
- 모델 confidence 분포.
- 새 패턴 발견 시 alert + 재학습 trigger.

**Human-in-the-loop**:
- 모델이 의심하면 reviewer queue.
- Reviewer 결정으로 라벨 보강.

### 5.9 답안 골격

> "**GBM (XGBoost) + Autoencoder 앙상블** — supervised 정확 + unsupervised 신규. **Feature engineering** 핵심 — 사용자/가맹점 aggregated stats. **Time-based split + walk-forward CV**. **Class-weighted loss + focal**, **PR-AUC** metric (accuracy 무의미). **Threshold** 비즈니스 비용에 맞게. **함정**: label delay, concept drift, feedback loop, random split 금지. **배포**: 실시간 latency (ms), drift monitoring, 정기 재학습 (주·월), human-in-the-loop. ML 엔지니어의 일은 사기 탐지에선 *데이터 split·feature·drift handling*이 모델보다 critical."

---

## 6. Task: 추천 시스템 — 영화 추천

### 6.1 문제 정의

**핵심 task**: 사용자에게 좋아할 영화 N개 추천.

**비즈니스 metric**:
- **CTR (Click-through rate)**, watch time, 사용자 만족도.
- **Offline metric** (NDCG, Recall@k) → **Online A/B test** (CTR, retention).

**Latency**: <100ms (실시간 추천).

### 6.2 데이터 분석

**Feedback 종류**:
- **Explicit** (별점, 좋아요): 신뢰 높지만 적음.
- **Implicit** (시청, 클릭): 신뢰 약간 낮지만 풍부.
- 일반적으로 **implicit이 dominant**.

**Sparsity**: 사용자 × 아이템 행렬이 매우 sparse (사용자가 시청한 영화 비율 1% 미만).

**Cold Start**:
- 신규 사용자 (history 없음).
- 신규 아이템 (interaction 없음).

**Power Law**:
- 인기 아이템에 데이터 집중.
- Long tail (대부분 아이템은 적은 interaction).

### 6.3 전처리

**(1) ID Embedding**:
- 사용자 ID, 아이템 ID를 dense vector로.
- 보통 32~256차원.

**(2) Sequence (시청 history)**:
- 최근 N개 시청 영화 시퀀스.
- 순서 의미 있음 (RNN/Transformer).

**(3) User/Item Features**:
- 사용자: 나이, 성별, 위치, 가입일.
- 아이템: 장르, 출시년도, 출연자, 평균 평점.

**(4) Negative Sampling**:
- Positive: 본 영화.
- Negative: 안 본 영화 중 random 또는 popularity-weighted.
- Implicit feedback에 필수.

### 6.4 모델 선택

**전통**:
- **Matrix Factorization (MF)**: 사용자×아이템 행렬을 두 행렬 곱으로 분해.
- **Item-based / User-based CF**: 코사인 유사도.

**Deep Learning**:

**(1) Neural CF**: MF의 비선형 확장.

**(2) Wide & Deep** (Google): linear (memorization) + DNN (generalization) 결합.

**(3) Two-Tower**: 사용자 tower + 아이템 tower → embedding → dot product. Retrieval에 표준.

**(4) Session-based RNN/Transformer** (SASRec, BERT4Rec): 시청 순서 활용.

**(5) Graph (PinSage, LightGCN)**: 사용자-아이템 그래프 + GNN.

**산업 표준**:
- **2단계 architecture**: 
  - **Retrieval** (수백만 후보 → 수백): Two-tower.
  - **Ranking** (수백 → 수십): DeepFM, DIN, DCN.

### 6.5 Loss / Metric

**Loss**:
- **Pointwise BCE**: 클릭/미클릭. 단순.
- **Pairwise BPR (Bayesian Personalized Ranking)**: 좋아한 것 > 안 좋아한 것.
- **Listwise (Softmax CE over candidates)**: 더 정교.
- **Contrastive (InfoNCE)**: 대조 학습.

**Metric**:

**Offline**:
- **NDCG@k**: 위치 가중 + 정규화. 표준.
- **Recall@k**: 상위 k에 진짜 관련 비율.
- **Precision@k**.
- **MAP, MRR**.

**Online**:
- **CTR**: A/B test.
- **Watch time, retention**: 진짜 satisfaction.

### 6.6 학습 전략

**Negative Sampling**:
- Random sampling.
- Popularity-weighted (인기 negative가 더 어려움).
- In-batch negatives (batch 내 다른 sample을 negative로).

**Batch**: 큰 batch 가능 (수천).

**Optimizer**: Adam.

**LR**: 1e-3 (embedding) + 1e-4 (DNN).

### 6.7 흔한 실패 / 함정

**(1) 선호 ≠ 클릭**:
- 클릭 많아도 만족 안 할 수 있음 (clickbait).
- 처방: Watch time, retention 같은 더 직접적 신호.

**(2) Popularity Bias**:
- 인기 아이템만 추천 → 다양성 감소, long tail 무시.
- 처방: Diversity loss, exploration (multi-armed bandit).

**(3) Filter Bubble**:
- 사용자 선호에만 갇힘 → 새 콘텐츠 발견 못함.
- 처방: Exploration, 의식적 다양화.

**(4) Feedback Loop**:
- 추천한 것만 데이터로 → 추천 안 된 아이템은 영원히 안 보임.
- 처방: Position bias 보정, off-policy evaluation.

**(5) A/B Test Mismatch**:
- Offline NDCG 좋은데 online CTR 안 좋음.
- Offline metric은 *historical 데이터*에 fit. Online은 *새 user에게 새 추천*.
- 처방: A/B test 필수.

**(6) Cold Start**:
- 신규 사용자 demographic feature 활용.
- 신규 아이템: content-based feature.

### 6.8 배포 / 모니터링

**2단계 Architecture**:
- **Retrieval**: ANN (Approximate Nearest Neighbor) — Faiss, ScaNN.
- **Ranking**: DNN inference.

**Latency**: <100ms.

**Embedding 갱신**:
- 매일 또는 실시간.
- 새 user/item embedding 추가.

**A/B Test**:
- 새 모델은 작은 traffic에 먼저.
- 통계적 유의성 확인 후 전체 rollout.

**Multi-Armed Bandit**:
- 새 추천 알고리즘 점진적 배포.
- Exploration vs exploitation balance.

### 6.9 답안 골격

> "**2단계 architecture**: retrieval (Two-tower) + ranking (DeepFM/DIN). **Negative sampling** 필수 (random/popularity-weighted/in-batch). **Loss**: BCE/BPR/InfoNCE. **Metric**: NDCG@k offline, CTR online. **함정**: 선호≠클릭, popularity bias, filter bubble, feedback loop, A/B mismatch, cold start. **배포**: ANN retrieval (<100ms), embedding 매일 갱신, A/B test로 rollout, multi-armed bandit. **Offline ≠ Online** — A/B test 절대 필수."

---

## 7. 6개 Task 비교

| Task | 데이터 구조 | 핵심 모델 | Loss | 핵심 Metric | 가장 큰 함정 |
|---|---|---|---|---|---|
| 의료 영상 분류 | 격자 (이미지) | CNN + transfer | CE / Focal | PR-AUC | Patient leak, marker spurious |
| Detection / Seg | 격자 | YOLO / U-Net | CE+IoU/Dice | mAP / mIoU | 작은 객체, NMS, latency |
| 텍스트 분류 | 시퀀스 | TextCNN / BERT | CE | F1 / Accuracy | 부정문, OOV, leak |
| 시계열 예측 | 시퀀스 | GBM / LSTM / TCN | MSE / MAE | MAPE | Future leakage, drift |
| 이상 탐지 | 표 / 시퀀스 | GBM / AE | weighted CE | PR-AUC | Label delay, drift |
| 추천 시스템 | 그래프 / 시퀀스 | Two-tower / MF | BCE / BPR | NDCG / CTR | Filter bubble, A/B |

---

## 8. ML 엔지니어 사고 — 공통 원칙

### 8.1 Baseline부터 항상

복잡한 모델 시도 전에 단순 baseline. 이유:
- (1) 단순 모델이 충분하면 복잡한 모델 불필요.
- (2) Baseline이 데이터·전처리·평가 파이프라인 sanity check.
- (3) 복잡 모델이 baseline 못 이기면 무언가 잘못.

### 8.2 데이터 Split이 모델보다 중요

- 의료 → patient-level
- 텍스트 → user-level
- 시계열 → forward chaining
- 추천 → time-based

Random split은 거의 항상 leakage. 신뢰할 수 없는 평가는 모델 선택을 오도.

### 8.3 Metric은 비즈니스 비용 반영

- Accuracy는 거의 항상 부적절.
- 비대칭 비용 (의료, 사기) → recall / precision 따로.
- 분포 무관 (불균형) → PR-AUC.
- 순위 (검색·추천) → NDCG.

### 8.4 Loss와 Metric이 다를 수 있다

- Loss는 미분 가능해야 → CE.
- Metric은 비즈니스 신호 → F1, NDCG, mAP.
- 일치 안 하면 *왜* 답할 수 있어야.

### 8.5 5대 함정 (어디서나)

1. **Data leakage** — split 잘못, feature 미래 정보.
2. **Distribution shift** — 배포 후.
3. **Class imbalance** — 다수만 학습.
4. **Concept drift** — 시간 따라 변화.
5. **Feedback loop** — 추천·이상탐지.

### 8.6 배포 후가 진짜 시작

- 모니터링 (입력·예측·outcome 분포).
- 정기 재학습.
- A/B test로 신모델 검증.
- Explainability (의료·금융 필수).

---

## 9. 시험·면접 답안 골격

> **"X task를 어떻게 설계?"**의 표준 답안:
>
> 1. **문제·metric 정의** (비대칭 비용? 평가 기준? Latency?)
> 2. **데이터 split** (leakage 방지)
> 3. **Baseline** (단순한 것부터)
> 4. **모델 선택 + 그 이유** (inductive bias 매칭)
> 5. **Loss·metric 차이** (있으면 이유)
> 6. **학습 전략** (불균형·long-tail 처리, optimizer)
> 7. **함정** (그 task 특이)
> 8. **배포·모니터링·재학습**

이 8단계를 외우면 모든 "어떻게 설계?" 질문에 답 가능. 짧게 답해도 8단계 골격이 보이면 좋은 답.

---

## 10. 면접 단골 Q&A

### Q1. 의료 영상 분류 어떻게 설계?
"ResNet50/EfficientNet pretrained + fine-tune. **Patient-level split** 필수. **Focal loss + PR-AUC** — recall이 비대칭 비용 반영. **Grad-CAM 시각화**로 의사 신뢰. 함정: patient leak, spurious correlation (의료기기 마커), distribution shift, 라벨 noise. 배포: threshold 비즈니스 비용에 맞게, 분포 모니터링, 정기 재학습."

### Q2. 자율주행 detection 어떻게?
"YOLO (실시간) + 가능하면 segmentation (U-Net/DeepLab). Backbone ImageNet pretrained. CE+GIoU/Dice loss. mAP/mIoU + latency metric. Multi-scale + heavy augmentation (mosaic). 함정: 작은 객체 (FPN), 클래스 불균형 (focal), NMS, edge 효율 (quantization). 배포: TensorRT + INT8."

### Q3. 텍스트 감성 분류 가장 빠른 prototype?
"단계별. (1) TF-IDF + Logistic — 5분. (2) TextCNN — 30분. (3) BERT/DistilBERT fine-tune — 1~3시간. 첫 두 baseline이 충분하면 거기서. SOTA면 BERT. Class weight, threshold 조정. F1/Macro-F1 metric (불균형 시)."

### Q4. 시계열 예측 안 되면 어떻게 디버깅?
"순서. (1) 데이터 정규화 (train만). (2) Lag feature (1일, 7일, 30일 전). (3) 외생 변수 (요일, 휴일, 기온). (4) **Forward chaining CV** — random split 금지. (5) Baseline (naive, MA, ARIMA) 비교 — 못 이기면 잘못. (6) GBM/LSTM/TCN 비교."

### Q5. 클래스 불균형 (양성 1%) 분류?
"5축 처방. (1) Class weight 또는 focal loss. (2) Oversampling/SMOTE 또는 undersampling. (3) Threshold 비즈니스 비용에. (4) Metric은 PR-AUC, F1, recall (accuracy 무의미). (5) 모델 GBM 강자. 의료는 recall, 사기는 PR-AUC."

### Q6. 사기 탐지 - supervised vs unsupervised?
"라벨 충분 + 알려진 패턴 → supervised (GBM + class weight + focal). 라벨 적음 + 새 패턴 → unsupervised (AE, Isolation Forest). 실무는 hybrid — 알려진 패턴은 supervised, 새 패턴은 AE. 함정: label delay, concept drift, feedback loop. PR-AUC, time-based split."

### Q7. 추천 시스템 cold start?
"세 처방. (1) Content-based — feature (장르, 태그). (2) Demographic — 나이, 성별. (3) Hybrid — content + collaborative. 신규 사용자엔 demographic + 인기 아이템. 신규 아이템엔 content-based. Active learning으로 점진. Foundation model (text/image embedding)도 cold start에 효과."

### Q8. 모델 deploy 후 성능 하락?
"Distribution shift 의심. 종류: covariate (입력), label, concept drift, selection bias. 처방: (1) 입력 분포 통계 추적, (2) 예측 confidence 분포, (3) 정기 재학습, (4) drift detector (ML 기반), (5) domain adaptation, (6) online learning. 의료·금융 critical."

### Q9. Object detection의 anchor box?
"다양한 크기·비율 객체에 대한 prior. 한 위치에서 작은 사람, 큰 차, 가로형 버스 등 동시 검출. 각 anchor가 'object 예/아니오 + box offset' 예측. Anchor-free (CenterNet, FCOS)도 발전 — 중심점 기반. Trade-off: anchor 多 = 정확도↑ 비용↑, anchor-free = 단순↑."

### Q10. 번역을 RNN으로 풀면? Transformer로 풀면?
"RNN: 순차 → 느림 + long-range 약함. 단 작은 데이터에 안정. Transformer: 병렬 + long-range. 큰 데이터에서 압도. 번역 길이 길수록 Transformer 우위. 현재 SOTA 번역 (Google, DeepL)은 모두 Transformer 기반. RNN/LSTM 시대는 거의 끝."

### Q11. ML 엔지니어 8단계 흐름?
"(1) 문제 정의 (비즈니스 metric, 비용). (2) 데이터 분석 (양·라벨·분포·leak). (3) 전처리 (정규화·encoding·aug). (4) 모델 선택 (inductive bias 매칭, baseline부터). (5) Loss/Metric (출력-loss 매칭, 비즈니스 metric). (6) 학습 전략 (batch/optim/schedule). (7) 함정 (task별 특이). (8) 배포·모니터링 (latency, drift, 재학습). 모델 선택은 한 단계 — 다른 7개가 ML 엔지니어와 학생 차이."

### Q12. 표 데이터에 GBM이 신경망보다 강한 이유?
"트리의 분할 기반 학습. (1) Feature value 임의 분할로 비선형. (2) 트리 분기로 feature 상호작용 자연. (3) Scale 무관. (4) 이상치·결측 robust. (5) Feature 수에 잘 적응. 신경망은 inductive bias가 표 데이터에 안 맞음. 매우 큰 데이터 + 강한 카테고리에서만 신경망 경쟁. 사기 탐지·시계열에서 GBM 자주 우위."

---

## 11. 생각해보라 — 단락 답안

**Q. 왜 의료에 *Explainability*가 그토록 중요한가?**

세 이유:

(1) **의사 신뢰**: 의사가 "AI가 폐렴이라고 했어요"로는 환자에게 설명 못함. *왜* 그런 결론인지 알아야 의료 의사결정에 통합. Grad-CAM 같은 시각화가 의사 신뢰의 도구.

(2) **법적·윤리적 책임**: 의료 결정에 책임. AI 결정이 틀렸을 때 "왜?"를 답할 수 없으면 책임 모호. 규제도 explainability 요구 (EU AI Act).

(3) **Spurious Correlation 발견**: 모델이 의료기기 마커 같은 비의학적 feature에 의존하는지 탐지. Explainability가 디버깅 도구.

이래서 의료 ML은 *정확도만큼이나 explainability*. 정확도 95% + 설명 불가 모델보다 정확도 90% + 명확한 설명이 자주 production 채택.

**Q. 자율주행의 *Edge Device 제약*이 어떻게 architecture를 형성?**

자율주행은 차량 내에서 추론 — 클라우드 의존 안 됨 (latency, 통신 끊김). Edge device 제약:
- 메모리: 수GB (서버의 1/100).
- 연산: GPU 약함 (NVIDIA Jetson 등).
- Power: 배터리 (소모 제약).

이 제약이 architecture 형성:
- **MobileNet, EfficientNet** 같은 효율적 backbone.
- **Quantization** (FP32 → INT8, 4배 빠름).
- **Pruning** — 불필요 가중치 제거.
- **Distillation** — 큰 teacher → 작은 student.
- **YOLO** 같은 single-stage detector (two-stage Faster R-CNN보다 빠름).

서버 친화 모델 (큰 ViT, 깊은 ResNet)은 자율주행 부적절. 같은 task가 *환경*에 따라 매우 다른 architecture 선택.

**Q. NLP의 *Subword Tokenization*이 왜 OOV 문제를 풀었나?**

전통 word-level tokenization:
- "hello", "world"가 별도 토큰.
- 새 단어 ("Twitter", "OpenAI") = OOV → <unk>.
- Vocabulary size 수십만 — 메모리 부담.

Subword (BPE, WordPiece, SentencePiece):
- "Twitter" = "Twit" + "ter".
- 서브워드는 *대부분 재사용*.
- 새 단어도 서브워드 조합으로 표현 가능.
- Vocabulary size 30K~50K — 효율.

또 한국어, 일본어 같은 *형태소* 풍부한 언어에 매우 적합. "먹었다" = "먹" + "었" + "다" 식으로 morphological 의미 반영.

이게 BERT, GPT 같은 modern NLP의 토대. OOV 문제가 사실상 사라짐.

**Q. 시계열의 *Concept Drift*가 다른 task와 어떻게 다른가?**

대부분 task의 distribution shift는 점진적 또는 환경 변경 (다른 병원, 새 도시).

시계열의 concept drift는 *시간이 본질*:
- 사용자 선호 시간 따라 변경.
- 경제 사이클.
- 기술 발전 (스마트폰 등장).
- 갑작스런 이벤트 (코로나, 전쟁).

**처방의 차이**:
- 다른 task: pretrained 모델 + 정기 재학습.
- 시계열: *최근 데이터에 더 큰 가중* (exponential weighting), online learning, 명시적 trend·seasonality 모델링.

또 시계열은 *ground truth가 즉시 나옴* (다음 시점 도착하면 답 알 수 있음). 빠른 feedback으로 drift 감지·재학습 가능. 이상 탐지 등 라벨 지연 task와 큰 차이.

**Q. 추천 시스템에서 *Feedback Loop*가 왜 다른 task보다 심각한가?**

다른 task의 feedback loop도 있지만 추천이 가장 심각:

(1) **모델 결정이 데이터 생성에 직접 영향**: 추천 안 한 영화는 *영원히 안 보임* → 다음 학습 데이터에 없음.

(2) **Bias 증폭**: 모델이 인기 영화 추천 → 더 많이 보임 → 더 많이 추천. 점점 다양성 감소.

(3) **A/B Test의 어려움**: Control 그룹도 *기존 모델*의 영향 받음. True counterfactual 어려움.

처방:
- **Off-policy evaluation**: 과거 정책의 데이터로 새 정책 평가. Importance sampling.
- **Position bias 보정**: 위치마다 click probability 다름. 모델링에 반영.
- **Exploration**: 일부 추천을 random 또는 unfamiliar.
- **Multi-armed Bandit**: Exploration vs exploitation balance.
- **Counterfactual ML**: 인과추론 도구 활용.

추천이 ML의 *가장 어려운 분야 중 하나*. 단순 supervised learning으로 풀 수 없음.

**Q. 6개 task의 *공통 사고 패턴*이 무엇?**

8단계 모두 같지만 *각 단계의 무게 중심*이 다름:

| Task | 가장 critical 단계 |
|---|---|
| 의료 영상 | Explainability (8) + patient split (2) |
| 자율주행 | Latency·edge (4, 8) + small object (7) |
| 텍스트 분류 | 빠른 baseline (3, 4) — 의외로 단순 모델로 충분 |
| 시계열 | Forward chaining + feature engineering (3) |
| 이상 탐지 | PR-AUC + drift handling (5, 8) |
| 추천 | Two-tower + A/B + bandits (4, 8) |

공통 통찰: *가장 critical 단계는 task의 본질 제약*에 따라 다름. 면접 답안은 해당 task의 critical 단계를 강조해야 좋은 답.

**Q. Foundation Model 시대에 *Task-specific 설계*가 여전 의미 있는가?**

일정 부분 yes:
- 매우 작은 데이터·niche 도메인 (특수 의료, 산업 검사) — task-specific.
- Edge device·real-time 제약 — task-specific.
- 표 데이터, 시계열 — GBM 등 traditional이 여전 강함.
- 도메인 특이 해석성 (의료) — task-specific.

Foundation model이 *대체*하는 영역:
- 일반 NLP (분류, 번역) — GPT-4, BERT 등.
- 일반 vision (분류) — CLIP, ViT.
- Multimodal — CLIP, GPT-4V.

미래: Foundation model을 *backbone*으로, task-specific은 *fine-tuning + 도메인 처리*. 이게 modern ML 엔지니어링의 표준 흐름이 됨. Task-specific 사고는 여전 valuable — 단 layer가 바뀜.

---

## 12. 한 줄 요약

- **8단계 흐름**: 문제 정의 → 데이터 → 전처리 → 모델 → loss/metric → 학습 → 함정 → 배포.
- **모델 선택은 한 단계** — 다른 7개가 ML 엔지니어 일.
- **Baseline부터** 항상.
- **Split이 모델보다 중요** — patient/user/time level.
- **Metric은 비즈니스 비용** 반영. Accuracy 거의 무용.
- **Loss와 metric** 다를 수 있음, 이유 답할 수 있어야.
- **5대 함정**: leakage, drift, imbalance, concept drift, feedback loop.
- **배포 후가 진짜 시작** — 모니터링·재학습·A/B test.
- 의료엔 **explainability**, 자율주행엔 **edge 효율**, 텍스트엔 **baseline 우선**, 시계열엔 **forward chaining**, 사기엔 **drift handling**, 추천엔 **A/B test + bandits**.
- ML 엔지니어 = *모델 + 데이터 + 평가 + 배포 + 모니터링*의 통합 사고.
