# 04. Convolutional Neural Networks (CNN) — 오픈북 정리

> 핵심: **"왜 이미지에 FNN이 안 좋은가? 왜 sharing/local인가?"** 이 질문을 답할 수 있으면 CNN의 절반 이상.

---

## 0. 동기 — "왜 FNN으로 이미지를 안 하나?"

224×224 RGB 이미지 = 입력 150,528개.
은닉층 1000개라면 첫 층 가중치 ≈ 1.5억개 → **파라미터 폭발**.

게다가 FNN은:
- **위치 독립성 무시**: (10,10)의 고양이 눈을 학습해도 (50,50)의 눈은 새로 배워야.
- **국소성 무시**: 멀리 떨어진 픽셀과 가까운 픽셀을 같은 가중치로 묶음.

→ 이미지의 **inductive bias**(국소성, 평행이동 불변성, 계층 구조)를 모델 구조에 인코딩하자 = **CNN**.

---

## 1. Convolution 연산 — "어떻게 작동?"

### 1.1 정의
입력 $X$, 커널 $K$ (예: 3×3), 출력 $Y$:
$$Y[i,j] = \sum_{u,v} K[u,v] \cdot X[i+u, j+v]$$
(엄밀히는 cross-correlation, 딥러닝에서 "convolution"이라 부름.)

각 출력 위치는 입력의 **국소 영역**에서 같은 커널을 적용.

### 1.2 핵심 두 가지 (★시험 단골)

| 성질 | 의미 | 왜 좋은가? |
|---|---|---|
| **Local connectivity** | 출력은 입력의 작은 영역만 본다 | 파라미터↓, 국소 패턴 학습 |
| **Weight sharing** | 같은 커널을 모든 위치에 | 파라미터↓, **translation equivariance** |

### 1.3 "왜 sharing 하겠냐?" (★교수 질문)
1. **파라미터 효율**: 위치마다 다른 가중치를 학습할 필요 없음. 같은 "edge 검출기"는 어디서나 같은 일.
2. **Translation equivariance**: 입력이 평행이동하면 출력도 같은 만큼 평행이동. 즉 위치가 달라도 같은 패턴은 같은 응답.
3. **데이터 효율**: 한 위치에서 학습한 패턴이 모든 위치에 적용 → 사실상 **데이터 augmentation의 효과**.

**equivariance vs invariance** (혼동 주의):
- Conv = **equivariance** (출력이 같이 움직임).
- Pooling/Global avg pool = **invariance** 부여 (출력이 같음).

---

## 2. 주요 하이퍼파라미터

| 파라미터 | 의미 | 효과 |
|---|---|---|
| **Kernel size (k)** | 필터 크기 | 클수록 receptive field↑, 파라미터↑ |
| **Stride (s)** | 이동 간격 | s=2면 출력 크기 절반 |
| **Padding (p)** | 입력 가장자리 채움 | 출력 크기 유지, 가장자리 정보 보존 |
| **Channel (C)** | 입출력 깊이 | 표현력↑ |

출력 크기: $\lfloor (W + 2p - k)/s \rfloor + 1$

**왜 padding?** → 없으면 매 conv마다 크기 줄어들고 가장자리 픽셀이 적게 사용됨. "same" padding으로 크기 유지.

---

## 3. Pooling — "왜 필요?"

### 3.1 종류
- **Max pooling**: 영역 내 최댓값. **두드러진 특징**.
- **Average pooling**: 평균. 부드러움.
- **Global Average Pooling (GAP)**: feature map 전체 평균. **공간을 1×1로**.

### 3.2 효과
1. **공간 차원 축소** → 계산량↓, receptive field↑.
2. **약한 translation invariance** (조금 움직여도 같은 결과).
3. **노이즈 강건성**.

### 3.3 현대 트렌드
- Stride conv가 pooling 대체하는 경향(파라미터화된 down-sampling).
- 분류 마지막엔 GAP + FC가 표준 (FC만 쓰면 파라미터 폭발, overfitting).

생각해보라: **왜 GAP가 FC보다 좋은가?** → 파라미터 0, overfitting 적음, spatial 정보 보존(채널별 평균이 클래스 활성으로 직결 — CAM).

---

## 4. Receptive Field

한 출력 뉴런이 보는 입력 영역의 크기.
- 깊을수록 RF 커짐.
- 3×3 conv를 두 번 쌓으면 RF = 5×5, 세 번이면 7×7.
- **VGG의 통찰**: 7×7 1개보다 3×3 3개가 같은 RF면서 파라미터 적고 비선형 多.

| | 7×7 1개 | 3×3 3개 |
|---|---|---|
| RF | 7×7 | 7×7 |
| 파라미터 | 49C² | 27C² |
| 비선형 횟수 | 1 | 3 |

→ "더 작은 커널을 더 깊이"가 일반 원칙.

---

## 5. 1×1 Convolution — "왜 필요해?" (★)

공간 차원에서는 변화 없고, **채널 차원만 섞음**.

용도:
1. **채널 수 조절** (bottleneck).
2. **계산량 줄이기** (Inception, ResNet bottleneck).
3. **채널 간 비선형 결합**.

생각해보라: 1×1 conv는 사실상 각 픽셀 위치에서의 **fully connected layer** (채널만). 공간 정보는 그대로.

---

## 6. CNN의 계층적 표현

| Layer 깊이 | 학습되는 표현 |
|---|---|
| 얕은 층 | edge, color blob, gradient |
| 중간 층 | texture, simple shape |
| 깊은 층 | object part (눈, 바퀴) |
| 최상위 | 전체 object, 의미 단위 |

→ "CNN은 자연스럽게 **계층적 visual hierarchy**를 학습한다." (실험으로 확인됨)

---

## 7. 주요 Architecture — "각자의 핵심 아이디어"

| 모델 | 연도 | 핵심 |
|---|---|---|
| **LeNet** | 1998 | 최초의 CNN. conv-pool-conv-pool-FC. 우편번호 |
| **AlexNet** | 2012 | ReLU, Dropout, GPU, 데이터 augmentation. ImageNet 혁명 |
| **VGG** | 2014 | "3×3 conv를 깊이 쌓자". 단순+깊이 |
| **GoogLeNet/Inception** | 2014 | **Inception module**: 여러 크기 conv 병렬 + 1×1로 차원 축소 |
| **ResNet** | 2015 | **Skip connection**으로 100+ 층. degradation 해결 |
| **DenseNet** | 2017 | 모든 이전 층과 연결. feature reuse |
| **MobileNet** | 2017 | **Depthwise separable conv**. 모바일용 경량 |
| **EfficientNet** | 2019 | depth/width/resolution 동시 scaling |
| **ConvNeXt** | 2022 | Transformer 디자인을 CNN으로 역수입 |

### 7.1 각 모델의 "왜?"

- **AlexNet**이 ImageNet에서 압도한 이유: ReLU(빠른 학습), Dropout(regularization), GPU(연산), 데이터 augmentation. **Deep + data + compute의 첫 증명**.
- **VGG의 통찰**: 작은 커널을 깊이. 단순함의 미.
- **Inception의 통찰**: 어떤 receptive field가 좋은지 모르면 **여러 크기를 동시**에. 1×1로 비용 절감.
- **ResNet의 통찰**: depth 자체가 어려움 → identity shortcut으로 학습 가능.
- **MobileNet의 통찰**: 표준 conv = depthwise (공간) × pointwise (채널). 이걸 **분리**하면 파라미터·연산 9배 감소.

---

## 8. Depthwise Separable Convolution

표준 conv (k×k, C_in → C_out): k²·C_in·C_out 곱셈 per pixel
Separable: k²·C_in (depthwise) + C_in·C_out (pointwise)
→ 비율 = 1/C_out + 1/k². k=3, C_out=256이면 약 1/9.

생각해보라: 정확도 거의 안 떨어지는데 9배 빠름. **왜 항상 안 쓰나?** → GPU에서 메모리 접근 패턴이 비효율적인 경우 있음. 모바일/embedded에서 진짜 빛남.

---

## 9. Transfer Learning — "왜 가능?"

ImageNet에서 학습한 CNN의 **얕은 층은 일반적인 visual feature** → 다른 task에도 유용.

### 9.1 방식
1. **Feature extractor**: 학습된 가중치 freeze, 마지막 FC만 새로 학습.
2. **Fine-tuning**: 전체를 작은 LR로 다시 학습.
3. 데이터 적으면 1, 많으면 2.

생각해보라: 왜 **얕은 층을 freeze**하나? → edge/texture는 도메인 무관, 깊은 층이 task-specific.

---

## 10. 다양한 task

| Task | 핵심 구조 |
|---|---|
| **Classification** | Conv backbone → GAP → FC → softmax |
| **Object Detection** | backbone + region proposal (R-CNN) / single-shot (YOLO, SSD) |
| **Semantic Segmentation** | Encoder-Decoder (U-Net), FCN |
| **Instance Segmentation** | Mask R-CNN |
| **Generation** | GAN, Diffusion (UNet 기반) |

**왜 segmentation은 encoder-decoder?** → 분류는 공간을 압축해도 되지만, segmentation은 **pixel-level 출력**이 필요 → upsampling 경로 필요.

---

## 11. CNN vs FNN vs RNN vs Transformer 비교 (★)

| | inductive bias | 데이터 | 장점 | 단점 |
|---|---|---|---|---|
| **FNN** | 거의 없음 | 일반 | 범용 | 파라미터 비효율 |
| **CNN** | 국소성, 평행이동 | 격자 데이터(이미지) | 효율적, 강한 prior | 장거리 의존 약함 |
| **RNN** | 시간 순서 | 시퀀스 | 가변 길이, 순서 | 병렬화 어려움, vanishing |
| **Transformer** | 거의 없음(self-attention) | 시퀀스/이미지/뭐든 | 장거리, 병렬 | 데이터 多, O(n²) |

생각해보라: 왜 ViT(Vision Transformer)가 충분한 데이터에서는 CNN을 이기나? → CNN의 inductive bias는 **편향(prior)이자 제약**. 데이터가 충분하면 그 제약 없는 Transformer가 더 풍부한 패턴 학습.

---

## 12. 흔한 함정

1. **입력 크기 불일치** — Conv는 size 무관하지만 FC는 고정 크기 필요. 그래서 GAP가 표준.
2. **Padding 미스** — output size 계산 잘못.
3. **BN을 eval에서 train 모드로** — 추론 결과 들쭉날쭉.
4. **데이터 normalization 불일치** — train/test에서 다른 mean/std 쓰면 망함.

---

## 13. 한 줄 요약

- **CNN의 근거**: 국소성 + weight sharing → 파라미터 효율 + translation equivariance.
- **Conv = equivariance, Pooling/GAP = invariance**.
- **3×3 깊게**가 큰 커널 1개보다 효율.
- **1×1 conv**: 채널 차원만 섞고 공간 그대로. bottleneck용.
- **AlexNet → VGG → Inception → ResNet**: 깊이를 어떻게 학습 가능하게 만드느냐의 역사.
- **Transfer learning**: 얕은 층은 도메인 무관 feature.
- CNN은 **이미지에 강하지만 장거리 의존엔 약함** → Transformer 등장 동기.
