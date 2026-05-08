# 04. Convolutional Neural Networks (CNN) — 심층 정리

> **이 문서의 목표**
> CNN을 이해하는 두 축: **(1) 왜 이미지에 FNN을 쓰면 안 되고 어떤 inductive bias가 필요한가**, **(2) 그 bias를 구현하는 conv·pooling·sharing이 정확히 어떤 일을 하고 어떻게 발전해왔는가**.
>
> 핵심 사고 패턴: **"왜 sharing 하겠냐"**, **"equivariance vs invariance"**, **"왜 더 깊게 갈수록 어려워지나"**, **"transfer learning이 왜 작동하나"**, **"ViT가 CNN을 이기는 데이터 조건은?"** 이 다섯 질문에 한 단락씩 답할 수 있게 만든다.

---

## 0. 큰 그림 — 이미지에는 왜 특별한 architecture가 필요한가

### 0.1 FNN으로 이미지 분류를 시도하면

ImageNet 224×224×3 이미지를 입력으로. 첫 fully-connected layer를 1000 뉴런으로 만들면 가중치 수:

$$224 \times 224 \times 3 \times 1000 = 150{,}528{,}000 \approx 1.5 \times 10^8$$

**1억 5천만 개**. 한 layer에. 메모리·연산 모두 폭발.

이걸 줄이려고 첫 layer를 100 뉴런으로 줄여도 1500만 개. 여전히 큰데, 이러면 표현력이 거의 없어 분류 정확도 매우 낮음. 양쪽 다 막혔다.

게다가 이게 *유일한* 문제도 아니다.

### 0.2 FNN의 또 다른 큰 문제 — 위치 정보 미활용

FNN은 입력의 모든 차원을 *대등하게* 본다. 픽셀 (10, 20)과 픽셀 (100, 50)을 다른 가중치로 처리하지만, 이 둘이 어떤 *공간적 관계*에 있는지 모른다.

그 결과:

(1) **Translation에 약함**. 고양이 사진에서 고양이가 왼쪽 위에 있을 때 학습된 모델은, 고양이가 오른쪽 아래로 옮겨지면 다시 처음부터 학습해야 한다. "왼쪽 위 코너의 눈"과 "오른쪽 아래의 눈"은 같은 패턴인데 FNN에선 별도.

(2) **국소성 활용 못 함**. 자연 이미지의 핵심 성질은 *가까운 픽셀이 강한 상관*을 가진다는 것. 멀리 떨어진 픽셀은 거의 독립. FNN은 이 사실을 활용 못 하고 모든 픽셀 쌍을 동등하게 다룸.

(3) **계층적 구조 활용 못 함**. 자연 이미지의 또 다른 성질: **edge → texture → part → object**의 계층. 작은 패턴들이 모여 큰 패턴을 만드는 구조. FNN은 한 layer에 모든 것을 담으려 함.

이 셋이 합쳐서 **이미지에는 다른 inductive bias가 필요**라는 결론. CNN이 그 답이다.

### 0.3 CNN의 세 가지 핵심 설계 결정

CNN은 세 가지를 도입한다 — 이게 CNN의 정체성:

**(1) Local connectivity**: 출력의 각 뉴런이 입력의 작은 영역(receptive field)만 본다. 멀리 떨어진 픽셀은 영향 없음.

**(2) Weight sharing**: 같은 가중치(커널)를 모든 위치에 적용. 한 위치에서 학습한 패턴이 모든 위치에 자동 적용.

**(3) Hierarchical composition**: 깊은 layer가 얕은 layer의 feature를 조합. 작은 패턴 → 큰 패턴.

이 셋의 직접적 효과:
- 파라미터 감소 (1억 → 수만)
- Translation equivariance (위치 이동에 출력도 같이 이동)
- 자연 이미지의 계층 구조 학습

이 셋의 간접적 효과:
- 데이터 효율 (한 위치 학습 = 모든 위치 학습)
- 일반화 향상 (강한 inductive bias = 강한 prior)
- 학습 가능성 (작은 모델이 빠르게 학습)

이 모든 것이 1989년 LeCun의 첫 CNN (LeNet)에서 이미 설계되어 있었다. 30년 후에도 본질은 같다.

### 0.4 본 챕터의 큰 흐름

이 챕터는 다음 순서로 간다:

1. Convolution 연산 자체 (§1~2)
2. CNN의 다른 부품들 — pooling, padding, stride (§3~5)
3. 받는 영역 — receptive field (§6)
4. 1×1 conv의 마법 (§7)
5. Architecture의 진화 — LeNet에서 ConvNeXt까지 (§8~9)
6. Transfer learning (§10)
7. Detection / Segmentation (§11)
8. CNN vs Vision Transformer (§12)
9. 면접 Q&A (§13)
10. 생각해보라 (§14)

---

## 1. Convolution 연산 — 정확히 무엇을 하는가

### 1.1 수학적 정의 (continuous)

신호 처리에서 두 함수 $f, g$의 convolution:

$$(f * g)(t) = \int f(\tau) g(t - \tau) d\tau$$

이건 $g$를 뒤집고($g(-\tau)$) 평행이동($g(t-\tau)$)한 다음 $f$와 곱해 적분. 신호 처리에서 filter를 적용하는 표준 연산.

### 1.2 Discrete 1D convolution

이산화하면:

$$(f * g)[n] = \sum_m f[m] g[n - m]$$

$g$가 커널(kernel) 또는 filter. 시간 신호 처리에서 전형적.

### 1.3 Discrete 2D convolution (이미지)

이미지 $X$, 커널 $K$:

$$(X * K)[i, j] = \sum_{u, v} K[u, v] \cdot X[i - u, j - v]$$

엄밀히는 이게 진짜 convolution. 그런데 딥러닝에서 "convolution"이라 부르는 건 사실 **cross-correlation**:

$$Y[i, j] = \sum_{u, v} K[u, v] \cdot X[i + u, j + v]$$

차이는 커널을 뒤집느냐 안 뒤집느냐. 학습된 커널은 어차피 임의이므로 실용적 차이 없음. 그래서 "convolution"이라는 이름을 유지하지만 실제는 cross-correlation.

### 1.4 직관 — 패턴 매칭

Convolution을 이해하는 가장 쉬운 방법: **커널은 "찾고 싶은 패턴", 출력은 "그 패턴이 각 위치에 얼마나 있는지의 점수"**.

3×3 커널이 edge detector라면:

```
K = [-1  -1  -1]
    [ 0   0   0]
    [ 1   1   1]
```

이건 horizontal edge (위가 어둡고 아래가 밝음) 검출기. 입력 이미지의 각 3×3 patch와 이 커널의 element-wise 곱 + 합 = "이 위치에 그런 edge가 얼마나 있나"의 점수.

CNN의 학습은 **이 커널을 데이터로부터 자동 학습**. 사람이 edge detector를 손으로 적지 않아도 데이터가 알려준다.

### 1.5 다채널 convolution

실제 이미지는 RGB 3채널. 커널도 깊이 차원이 있다:

$$Y[i, j] = \sum_{c} \sum_{u, v} K[c, u, v] \cdot X[c, i + u, j + v]$$

3채널 입력에 3×3×3 커널 → 단일 채널 출력. 여러 커널 ($K_1, K_2, ..., K_F$)을 적용하면 $F$채널 출력 — 이게 다음 layer의 입력.

전체 layer:
- 입력: $H \times W \times C_{in}$
- 커널: $C_{out}$개의 $k \times k \times C_{in}$
- 출력: $H' \times W' \times C_{out}$

총 가중치: $C_{out} \cdot k \cdot k \cdot C_{in}$. 일반적인 conv layer가 $3 \times 3 \times 256 \times 256 = 590{,}000$ 정도. FNN의 1억 5천만에 비하면 약 *250배 적음*.

### 1.6 행렬 곱으로의 환원 (im2col)

GPU에서 conv를 효율적으로 계산하는 방법: **input을 펴서 행렬 곱으로**.

각 출력 픽셀에 해당하는 input patch들을 행으로 쌓아 큰 행렬 만든다. 커널들도 펴서 행렬. 두 행렬의 곱이 conv 결과.

$O(N) \cdot O(K)$의 conv가 $O(N \cdot K)$ 행렬 곱이 됨. cuBLAS의 매우 빠른 GEMM 사용 가능. cuDNN이 이 변환을 자동.

---

## 2. Weight Sharing의 본질 — "왜 sharing 하겠냐"

### 2.1 Sharing의 정의

같은 가중치(커널)를 입력의 *모든 위치*에 적용. 위치 (1,1)에서의 conv와 위치 (50,50)에서의 conv가 *같은 커널 값*을 사용.

대안: **Locally Connected Network (LCN)**. 같은 local connectivity (작은 RF)이지만 위치마다 *다른* 가중치. CNN처럼 가중치 공유 안 함.

### 2.2 왜 sharing? 첫 번째 이유 — 파라미터 효율

LCN은 위치마다 다른 가중치 → 파라미터 수가 위치 수에 비례. 224×224 이미지에 3×3 커널 256채널이면 LCN의 가중치 수:

$$224 \times 224 \times 3 \times 3 \times 256 \approx 1.16 \times 10^8$$

CNN은 위치 무관하게:
$$3 \times 3 \times 3 \times 256 = 6{,}912$$

**약 1만 7천 배 적음**. 같은 표현력 영역에 있다고 가정하면 압도적 효율.

### 2.3 왜 sharing? 두 번째 이유 — Translation Equivariance

이게 더 중요. 수학적으로 정의:

**함수 $f$가 translation $T$에 equivariant**: $f(T(x)) = T(f(x))$

CNN의 conv 연산이 정확히 이걸 만족. 입력을 $(\Delta x, \Delta y)$만큼 평행이동하면 출력도 같은 만큼 평행이동.

**증명 (직관)**: Conv는 매 위치에서 *같은* 패턴 매칭. 입력의 패턴이 위치 $(i, j)$에서 위치 $(i + \Delta x, j + \Delta y)$로 옮겨지면, 출력의 그 패턴 점수도 같은 위치로 옮겨감.

이게 자연 이미지에 매우 적합한 prior. 고양이는 사진에서 어디에 있든 고양이고, 같은 visual feature (눈, 코, 귀)를 가진다.

LCN은 이 성질이 없다. 위치마다 다른 가중치라 같은 패턴이 다른 위치에 있으면 다른 응답.

### 2.4 왜 sharing? 세 번째 이유 — 데이터 효율

한 위치에서 학습한 패턴이 *모든* 위치에 자동 적용된다. 이게 사실상 **데이터 augmentation 효과**.

생각해보라: 만약 한 위치의 고양이만 학습 데이터에 있으면, FNN/LCN은 다른 위치의 고양이를 *별도로* 학습해야 한다. CNN은 한 번 학습하면 끝. 데이터 효율이 압도적.

### 2.5 Sharing의 한계

모든 위치에서 같은 패턴이 의미 있을 때만 sharing이 좋음. 어떤 task에서는 *위치별 다른 패턴*이 필요.

예: **얼굴 인식**. 얼굴 영상에서 눈은 항상 위쪽, 입은 항상 아래쪽. 위치에 따라 다른 패턴 학습이 효율적일 수 있음. 그래서 일부 face recognition 모델은 LCN 사용 (DeepFace, Facebook).

하지만 일반 자연 이미지에선 sharing이 압도적 우위. 그래서 표준은 항상 CNN.

### 2.6 Equivariance vs Invariance — 핵심 구분

| | 정의 | 예시 |
|---|---|---|
| **Equivariance** | $f(T(x)) = T(f(x))$ | Conv 연산 |
| **Invariance** | $f(T(x)) = f(x)$ | Pooling, GAP |

CNN의 conv 부분은 equivariant — 입력 이동에 출력도 같이 이동. Pooling, 특히 Global Average Pooling은 invariance를 부여 — 입력 위치 무관하게 같은 출력.

이 구분이 중요한 이유: **분류 task에는 invariance가 최종 목표** (어디 있든 고양이는 고양이). 하지만 segmentation이나 detection에서는 *위치 정보 보존*이 필요. CNN의 강력한 점 — equivariance 통해 위치 정보 유지하다가, 필요한 곳(분류 직전)에서만 invariance 부여.

면접 답안 골격: "Conv는 equivariance, pooling/GAP는 invariance를 부여. 이 둘의 조합이 CNN을 detection·segmentation·classification 모두에 사용 가능하게 만든다."

---

## 3. Convolution의 하이퍼파라미터

### 3.1 Kernel Size

커널 한 변의 픽셀 수. 일반적으로 1, 3, 5, 7. 거의 모두 홀수 (중앙 픽셀 명확).

| Kernel Size | 의미 |
|---|---|
| 1×1 | 채널만 변환, 공간 그대로 |
| 3×3 | 가장 흔함. RF 작음, 깊이로 보충 |
| 5×5 | 큰 패턴. VGG 이전 표준 |
| 7×7 | 매우 큰 RF. ResNet의 첫 conv |

**왜 3×3가 표준?** VGG의 통찰. 3×3 두 번 = 5×5 RF 한 번 (RF 같음). 3×3 세 번 = 7×7 RF 한 번. 파라미터:
- 7×7 한 번: $7 \times 7 = 49$
- 3×3 세 번: $3 \times 3 \times 3 = 27$ (45% 적음)

게다가 비선형 활성화가 세 번 들어감 → 표현력 ↑.

이래서 "큰 kernel 한 번"보다 "작은 kernel 여러 번"이 일반 원칙.

### 3.2 Stride

커널이 한 step에 이동하는 픽셀 수. Default 1.

Stride 1: 출력 크기 (거의) 입력과 같음.
Stride 2: 출력 크기 절반. **down-sampling 효과**.

큰 stride는 pooling 대신 사용 가능 — 학습 가능한 down-sampling. 현대 표준이 점점 stride conv가 pooling 대체.

### 3.3 Padding

입력 가장자리에 0을 채워 출력 크기 조절. 두 가지 모드:

**Valid (no padding)**: 출력이 줄어듦. $H_{out} = (H - k)/s + 1$.

**Same**: 출력이 입력과 같은 크기. Padding으로 채움.

```
H_out = floor((H + 2p - k)/s) + 1
```

stride=1, kernel=3이면 padding=1로 same.

**왜 padding?**:
- 출력 크기 유지로 architecture 설계 단순.
- 가장자리 픽셀이 적게 사용되는 문제 완화.
- 매우 깊은 망에서 크기가 너무 작아지는 거 방지.

### 3.4 Dilation (Atrous Convolution)

커널 사이에 빈 공간 둠. Effective receptive field를 키우면서 파라미터 그대로.

```
일반 3×3:        Dilation 2:
■ ■ ■            ■ . ■ . ■
■ ■ ■    →       . . . . .
■ ■ ■            ■ . ■ . ■
                 . . . . .
                 ■ . ■ . ■
```

Effective kernel: 5×5. 파라미터: 9 (3×3과 같음).

용도: Segmentation에서 RF 키우면서 해상도 유지. DeepLab 등.

### 3.5 출력 크기 계산

기본 식:
$$H_{out} = \left\lfloor \frac{H + 2p - d(k-1) - 1}{s} \right\rfloor + 1$$

(d=dilation. 일반 conv는 d=1)

흔한 case:
- 3×3, stride 1, padding 1 (same): $H_{out} = H$
- 3×3, stride 2, padding 1: $H_{out} = H/2$ (down-sample)
- 1×1, stride 1, padding 0: $H_{out} = H$

이 계산은 architecture 설계의 기본. 자주 잘못 계산하면 모델이 안 fit하거나 메모리 폭발.

---

## 4. Pooling — 공간 차원 줄이기

### 4.1 Max Pooling

각 영역의 최댓값:

```
Input 4×4:           After 2×2 max pool:
[1 3 2 4]            [3 4]
[5 6 1 2]      →     [7 8]
[7 2 8 0]
[3 1 0 4]
```

영역 (2,2): max(1,3,5,6) = 6 → 출력 (1,1) 위치.

**왜 max?**: 가장 강한 신호 보존. "이 영역에 패턴이 있나"의 신호. 두드러진 feature.

### 4.2 Average Pooling

각 영역의 평균. 부드러움. Max보다 정보 보존 多, 두드러진 신호 약함.

### 4.3 Global Average Pooling (GAP)

전체 feature map을 평균. $H \times W \times C$ → $1 \times 1 \times C$. 공간 정보를 완전히 압축.

**왜 GAP?**: 분류의 마지막. 공간 정보 → 채널 vector → FC → 클래스 분류. 파라미터 0, overfitting 적음.

전통적인 FC 분류 head:
```
14×14×512 → flatten → 100,352 → FC 4096 → FC 4096 → FC 1000
```
파라미터: 약 1억 2천만 (FC 부분만!).

GAP 대체:
```
14×14×512 → GAP → 512 → FC 1000
```
파라미터: 약 50만. **240배 적음**.

이래서 modern CNN은 GAP가 표준.

### 4.4 Pooling의 효과

**(1) 공간 차원 축소**: 메모리·연산 ↓. 깊은 layer에서 작아진 feature map은 큰 RF.

**(2) Translation invariance**: 영역 내에서는 어디 있든 같은 출력. 약한 invariance.

**(3) 노이즈 강건성**: max는 두드러진 신호만, average는 부드럽게.

### 4.5 Pooling vs Strided Conv — 현대 트렌드

| | Pooling | Strided Conv |
|---|---|---|
| 학습 | 비학습 | 학습 가능 |
| 표현력 | 단순 max/avg | 복잡 변환 |
| 메모리 | 적음 | 약간 더 |
| 현대 | GAP만 | down-sampling 표준 |

ResNet 같은 modern CNN은 stride conv로 down-sampling, max pool은 거의 안 씀 (첫 layer 후 한 번만). 마지막에 GAP.

**왜 학습된 down-sampling이 좋은가?**: Hand-crafted (max/avg)는 단순. 학습은 task-적응적. 단 max pool은 단순함이 장점이라 Edge device에서 여전히 사용.

---

## 5. CNN의 흐름 — 전형적 architecture

### 5.1 일반 패턴

```
Input
  ↓
[Conv → BN → ReLU] × N    (block 1, 작은 RF, 큰 spatial)
  ↓ Pool/Stride
[Conv → BN → ReLU] × N    (block 2, 중간 RF, 중간 spatial)
  ↓ Pool/Stride
[Conv → BN → ReLU] × N    (block 3, 큰 RF, 작은 spatial)
  ↓ GAP
FC → softmax
  ↓
Output
```

특징:
- 깊이 따라 spatial ↓, channel ↑
- 마지막 GAP + FC

### 5.2 Spatial vs Channel의 trade-off

전형적 변화:
- 224×224×3 → 112×112×64 → 56×56×128 → 28×28×256 → 14×14×512 → 7×7×1024

Spatial ÷2 할 때마다 channel ×2. 메모리는 거의 일정 (몫 4 ÷ 2 = 2 정도).

**왜 channel을 늘리나?**: 깊은 layer에서 abstract feature가 다양해짐. 한 위치에 여러 종류의 feature가 동시에 있을 수 있음. 그래서 채널 多.

---

## 6. Receptive Field — 한 뉴런이 보는 영역

### 6.1 정의

깊은 layer의 한 뉴런이 입력 이미지의 어느 영역에 영향을 받는가. 깊이가 깊어질수록 RF는 커진다.

### 6.2 계산

3×3 conv 한 번: RF = 3×3.
3×3 conv 두 번 (stride 1): RF = 5×5.
3×3 conv 세 번: RF = 7×7.

일반 식 (stride 1):
$$RF_{l+1} = RF_l + (k - 1)$$

stride > 1이면:
$$RF_{l+1} = RF_l + (k - 1) \cdot \prod_{i=1}^{l} s_i$$

(stride의 누적 곱이 RF 증가량을 키움)

### 6.3 Effective Receptive Field

이론상 RF와 실제로 영향력 있는 영역이 다르다 — 가운데 픽셀이 더 강한 영향. 가우시안 분포에 가까움 (Luo et al. 2017).

함의: 매우 깊은 망의 RF가 이론상 입력 전체를 덮어도, 실제로는 가운데만 강하게 본다. 이게 long-range 모델링에선 약점이고, attention이 등장한 이유 중 하나.

### 6.4 RF 키우는 방법

(1) **깊이 증가**: 자연스러운 방법. 단 매우 느림.
(2) **큰 커널**: 7×7 같은. 단 파라미터 多.
(3) **Stride**: down-sampling으로 RF 폭발.
(4) **Dilation**: 파라미터 그대로 RF 증가. Segmentation에 유용.
(5) **Pooling**: Stride와 비슷.

### 6.5 RF와 task의 관계

- **분류**: 큰 RF 필요 (전체 객체 봐야).
- **Detection**: 다양한 크기 RF 필요 (큰 객체와 작은 객체 모두). FPN.
- **Segmentation**: 큰 RF + 고해상도 (위치 보존). U-Net, DeepLab.

---

## 7. 1×1 Convolution — 작은 마법

### 7.1 정의

커널 크기 1×1. 공간 차원에는 변화 없음. 채널 차원만 변환.

수식:
$$Y[i, j, c'] = \sum_c K[c', c] \cdot X[i, j, c] + b[c']$$

각 픽셀 위치에서 채널 vector를 입력으로 받아 다른 채널 vector로. 사실상 **각 위치마다 적용된 fully connected layer**. 단 위치 간 weight sharing.

### 7.2 용도 1 — 채널 차원 조절

256 → 64 채널로 줄이기. 또는 64 → 256 채널로 늘리기. 비용 거의 없음.

### 7.3 용도 2 — Bottleneck

ResNet bottleneck block:
```
input 256 channel
  ↓ 1×1 conv → 64 channel  (압축)
  ↓ 3×3 conv → 64 channel  (작업, 적은 채널)
  ↓ 1×1 conv → 256 channel (복구)
  ↓ + skip
output 256 channel
```

3×3 conv를 256 채널 그대로 했다면 $3 \times 3 \times 256 \times 256 = 590{,}000$ 파라미터. Bottleneck:
- 1×1 256→64: $256 \times 64 = 16{,}000$
- 3×3 64→64: $3 \times 3 \times 64 \times 64 = 37{,}000$
- 1×1 64→256: $64 \times 256 = 16{,}000$
- 합: $69{,}000$ — **8.6배 적음**

같은 표현력, 적은 비용.

### 7.4 용도 3 — Inception에서의 채널 선택

GoogLeNet의 Inception 모듈은 1×1, 3×3, 5×5를 동시에 적용 후 합침. 1×1이 모든 path 앞뒤에 들어가 채널 차원 조절.

### 7.5 용도 4 — 채널 간 비선형 결합

1×1 conv는 사실 *각 픽셀의 채널 vector에 대한 비선형 변환*. 공간 정보는 그대로 두고 채널 정보만 처리. "Network in Network" 아이디어 (Lin et al.)에서 시작.

이 시각으로 보면 1×1은 매우 강력 — 어떤 채널 처리도 가능, 공간 처리는 후속 conv가.

### 7.6 1×1 conv의 의미 정리

1×1 conv = "공간 정보 보존 + 채널 차원 자유 조절 + 채널 간 비선형 결합". 작은 트릭이지만 modern CNN의 거의 모든 architecture에 핵심 부품.

---

## 8. CNN Architecture의 진화 — LeNet에서 EfficientNet까지

### 8.1 LeNet (1998, Yann LeCun)

CNN의 시작. 우편번호 인식.

```
Input 32×32×1
  ↓ Conv 5×5, 6 channels → 28×28×6
  ↓ Avg Pool 2×2 → 14×14×6
  ↓ Conv 5×5, 16 channels → 10×10×16
  ↓ Avg Pool 2×2 → 5×5×16
  ↓ FC 120 → FC 84 → FC 10
```

특징:
- 작은 모델 (60K 파라미터).
- Tanh activation (ReLU 이전).
- Avg pooling (max pool 이전).

당시는 SVM에 밀렸지만 CNN의 기본 형태를 정립. **Conv → Pool → FC**의 패턴이 30년 후에도 본질 그대로.

### 8.2 AlexNet (2012, Krizhevsky-Sutskever-Hinton)

ImageNet 우승 (top-5 에러 26% → 15%). 딥러닝의 부흥의 신호탄.

핵심 혁신:
- **ReLU**: 깊은 망 학습 가능.
- **Dropout**: 정규화.
- **GPU**: 두 GPU에 모델 나눠서 학습 (당시 GPU 메모리 부족).
- **Data augmentation**: 회전, crop, color jitter.
- **Local Response Normalization**: 후에 BN으로 대체.

```
Input 224×224×3
  ↓ Conv 11×11, stride 4 → 55×55×96
  ↓ Max Pool 3×3, stride 2 → 27×27×96
  ↓ Conv 5×5, padding 2 → 27×27×256
  ↓ Max Pool → 13×13×256
  ↓ Conv 3×3 → 13×13×384
  ↓ Conv 3×3 → 13×13×384
  ↓ Conv 3×3 → 13×13×256
  ↓ Max Pool → 6×6×256
  ↓ FC 4096 → FC 4096 → FC 1000
```

총 60M 파라미터. 그중 FC가 50M+ — 마지막 FC layer가 거대 (이후 GAP로 대체).

**왜 ImageNet에서 압도?**: ReLU + Dropout + GPU + 데이터 + 깊이 (8층). 모든 요소의 동시 발전. "Deep + Data + Compute"의 첫 증명.

### 8.3 VGG (2014, Oxford)

"3×3 conv를 깊이 쌓자."

VGG-16 (16층) / VGG-19 (19층). 모든 conv가 3×3, padding 1. Pool 2×2.

```
Block 1: Conv 3×3 64 → Conv 3×3 64 → MaxPool
Block 2: Conv 3×3 128 → Conv 3×3 128 → MaxPool
Block 3: Conv 3×3 256 → Conv 3×3 256 → Conv 3×3 256 → MaxPool
Block 4: Conv 3×3 512 → Conv 3×3 512 → Conv 3×3 512 → MaxPool
Block 5: Conv 3×3 512 → Conv 3×3 512 → Conv 3×3 512 → MaxPool
GAP → FC 4096 → FC 4096 → FC 1000
```

총 138M 파라미터.

**핵심 통찰**:
- 큰 커널 1개보다 작은 커널 여러 개 (RF 같으면서 파라미터 적고 비선형 多).
- 단순 + 깊이.

VGG는 **transfer learning의 표준 backbone**이 됐다 — 단순한 구조라 변형하기 쉬움. 단 파라미터 多, 메모리 부담.

### 8.4 GoogLeNet / Inception (2014, Google)

"어떤 RF가 좋은지 모르면 여러 크기 동시에."

**Inception module**:
```
input
  ├── 1×1 conv
  ├── 1×1 conv → 3×3 conv
  ├── 1×1 conv → 5×5 conv
  └── 3×3 maxpool → 1×1 conv
  → concat
output
```

다양한 RF의 정보를 동시에 추출. 1×1로 비용 절감.

GoogLeNet: 22층. ImageNet top-5 6.7%. AlexNet의 절반.

총 6.8M 파라미터 — VGG의 1/20. 1×1 bottleneck 덕에.

**핵심 통찰**:
- 다양한 RF 동시 사용.
- 1×1 conv로 비용 절감.
- Auxiliary classifier (중간에 보조 분류 head, vanishing 완화).

Inception v2/v3/v4로 발전, BN 추가, 5×5를 두 개의 3×3으로 분해 등 정교화.

### 8.5 ResNet (2015, He et al.) — 혁명

"깊이의 한계를 깨자."

**Skip connection**:
$$h_{l+1} = h_l + F(h_l)$$

이 단순한 변경으로:
- 100~150층이 표준 가능.
- ImageNet 성능 큰 향상 (top-5 3.57%, 사람 수준).
- 후속 모든 모델의 핵심 부품.

ResNet block:
```
input x
  ├── Conv 3×3 → BN → ReLU
  ├── Conv 3×3 → BN
  └── + (skip)
      └── ReLU
output
```

Bottleneck (50+층용):
```
input x (256 ch)
  ├── 1×1 conv → 64 ch → BN → ReLU
  ├── 3×3 conv → 64 ch → BN → ReLU
  ├── 1×1 conv → 256 ch → BN
  └── + (skip)
      └── ReLU
output
```

ResNet-50: 25M 파라미터, top-5 6.7%.
ResNet-152: 60M 파라미터, top-5 4.5%.

**왜 혁명**:
- Degradation problem 해결 — 깊이가 더해도 train accuracy 안 떨어짐.
- Vanishing gradient 우회 — 덧셈 경로.
- 후속 모든 모델의 표준 부품.

ResNet의 skip은 LSTM의 cell state와 본질적으로 같은 정신 — **곱셈 누적을 덧셈 경로로**.

### 8.6 DenseNet (2017)

"모든 이전 layer와 연결."

각 layer가 모든 *이전* layer의 출력을 입력으로:
$$h_l = F([h_0, h_1, ..., h_{l-1}])$$

[·]는 concatenation. ResNet의 sum 대신 concat.

장점:
- Feature reuse 강함.
- 적은 파라미터로 좋은 성능.
- Gradient flow 매우 좋음.

단점:
- 메모리 多 (모든 이전 출력 저장).
- ResNet만큼 보편화 안 됨.

### 8.7 MobileNet (2017, Google)

"모바일 디바이스를 위한 경량 CNN."

**Depthwise separable convolution**: 표준 conv를 두 단계로 분해.

표준 conv (3×3, $C_{in} \rightarrow C_{out}$): $3 \times 3 \times C_{in} \times C_{out}$ 파라미터.

Depthwise separable:
1. **Depthwise**: 각 입력 채널에 독립적인 3×3 conv ($C_{in}$채널). 파라미터: $3 \times 3 \times C_{in}$.
2. **Pointwise (1×1)**: 채널 mixing. 파라미터: $C_{in} \times C_{out}$.

총: $3 \times 3 \times C_{in} + C_{in} \times C_{out}$.

비율: 표준 / separable = $\frac{1}{C_{out}} + \frac{1}{9}$. $C_{out} = 256$이면 약 1/9 — **9배 적음**.

정확도는 거의 같으면서 파라미터·연산 9배. 모바일에 혁명.

MobileNet v2: + inverted residual + linear bottleneck.
MobileNet v3: + squeeze-excitation + h-swish.

### 8.8 EfficientNet (2019, Google)

"Depth, width, resolution을 균형 있게 scaling."

기존 모델 키울 때 보통 한 축만 키움 (깊이만, 너비만). EfficientNet은 셋을 동시에:

$$\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi$$

$\alpha\beta^2\gamma^2 \approx 2$ 제약. $\phi$가 compound coefficient.

EfficientNet-B0 ~ B7. 같은 정확도에 5~10배 적은 파라미터.

**핵심 통찰**: "Depth 혼자, width 혼자"는 sub-optimal. 셋의 균형이 중요.

### 8.9 ConvNeXt (2022)

"Transformer 디자인을 CNN으로 역수입."

ResNet에 modern Transformer 디자인 적용:
- Larger kernels (7×7).
- LayerNorm 대신 BatchNorm.
- GELU activation.
- 적은 activation·BN.
- Inverted bottleneck.

결과: ViT와 비슷하거나 우위. **CNN이 ViT를 따라잡음**.

이 발견의 의미: ViT의 우위는 attention 자체가 아니라 *modern training 기법* 때문일 수 있다. CNN도 같은 기법으로 학습하면 비슷.

---

## 9. Architecture 진화의 큰 흐름

### 9.1 무엇이 발전했나

| 측면 | 1998 (LeNet) | 2012 (AlexNet) | 2015 (ResNet) | 2019 (EfficientNet) |
|---|---|---|---|---|
| 깊이 | 5 | 8 | 152 | ~200 |
| 파라미터 | 60K | 60M | 25M | 7~66M |
| Top-5 (ImageNet) | - | 16.4% | 4.5% | 2.5% |
| 정규화 | - | Dropout | BN | BN + DropPath |
| 활성화 | Tanh | ReLU | ReLU | Swish/SiLU |
| Down-sample | AvgPool | MaxPool | StridedConv | StridedConv |

### 9.2 발전의 패턴

1. **깊이 ↑** (LeNet 5 → ResNet 152): vanishing 문제 해결과 함께.
2. **파라미터 효율 ↑**: 1×1 bottleneck, depthwise separable, compound scaling.
3. **정규화 정교화**: Dropout → BN → LN → mixup → label smoothing.
4. **Architecture 자동화**: Hand-design → NAS (Neural Architecture Search).
5. **Modern training 기법**: AdamW + cosine + warmup + augmentation.

### 9.3 핵심 통찰의 누적

각 architecture가 *직전 한계의 답*으로 등장:

- AlexNet: 깊이 + GPU의 첫 증명.
- VGG: 단순한 깊이가 좋다.
- GoogLeNet: 다양한 RF 동시 사용.
- ResNet: 깊이의 vanishing 문제 해결.
- DenseNet: feature reuse 극대화.
- MobileNet: 모바일 효율.
- EfficientNet: scaling의 균형.
- ConvNeXt: ViT와 경쟁 가능.

각 모델의 *핵심 아이디어*를 외우면 면접 답안이 만들어진다.

---

## 10. Transfer Learning — 왜 작동하고 어떻게 쓰나

### 10.1 동기

매 task마다 ImageNet 같은 큰 데이터로 from-scratch 학습은 비현실적. ImageNet 학습된 모델의 *일부를 재사용*하자.

### 10.2 왜 작동하는가

**얕은 layer는 도메인 무관**. Edge, texture, color blob 같은 low-level feature는 자연 이미지 어디서나 공통.

**깊은 layer는 task-specific**. 마지막 layer는 ImageNet 1000 클래스에 특화 — 다른 task엔 무용.

→ 얕은 layer는 **freeze**, 깊은 layer만 fine-tune.

이게 transfer learning의 본질.

### 10.3 두 가지 방식

**Feature Extractor**: 얕은 layer freeze, 마지막 FC만 새 task로 학습. 데이터 매우 적을 때.

```python
model = resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # 모두 freeze
model.fc = nn.Linear(2048, num_classes)  # 마지막만 재학습
```

**Fine-tuning**: 전체를 작은 LR로 다시 학습. 데이터 좀 있을 때.

```python
model = resnet50(pretrained=True)
model.fc = nn.Linear(2048, num_classes)
# 전체 학습. 단 LR을 처음 학습보다 작게 (예: 1e-4)
```

**Discriminative LR**: 얕은 layer는 더 작은 LR, 깊은 layer는 큰 LR. ULMFit 등에서.

### 10.4 데이터 크기에 따른 결정

| 데이터 | Strategy |
|---|---|
| < 1k | Feature extractor only |
| 1k ~ 10k | Fine-tune 마지막 몇 layer |
| 10k ~ 100k | Full fine-tune, 작은 LR |
| > 100k | From scratch도 가능, pretrain은 여전히 우위 종종 |

### 10.5 도메인 차이의 영향

**ImageNet과 비슷한 도메인** (자연 사진, 동물 등): Pretrained 모델이 매우 효과적.

**다른 도메인** (의료 영상, 만화, 위성 이미지): 효과 줄지만 여전히 random init보다 우위. 깊은 layer는 도메인 적응으로 다시 학습.

**완전 다른 modality** (음성, 텍스트): 직접 transfer 어려움. 단 self-supervised pretrain은 modality 무관 도움.

### 10.6 Transfer Learning의 함정

(1) **분포 불일치**: Source와 target 분포 너무 다르면 negative transfer (오히려 손해).

(2) **Output mismatch**: 분류 클래스 수 다름 → 마지막 layer 새로.

(3) **Input size mismatch**: ImageNet은 224×224. 다른 크기면 resize 또는 architecture 조정.

(4) **Pretrained 통계와 새 데이터 통계의 차이**: BN running 통계가 다름. Fine-tuning 시 BN unfreeze.

---

## 11. CNN의 다른 task — Detection, Segmentation

### 11.1 Object Detection

객체 위치 (bounding box) + 클래스 동시 예측.

**Two-stage** (Faster R-CNN):
1. Region Proposal Network (RPN): 후보 영역 생성.
2. Classification + Box Regression: 각 영역 분류 + 정확한 box.

장점: 정확. 단점: 느림.

**Single-stage** (YOLO, SSD):
- Region proposal 없이 한 번에 prediction.
- 격자(grid) 기반 — 이미지를 격자로 나눠 각 cell이 object 예측.

장점: 빠름 (실시간). 단점: 약간 정확도 손해.

**Anchor box**: 다양한 크기·비율의 prior box. 객체가 이 anchor 중 하나에 fit한다고 가정.

**Anchor-free** (FCOS, CenterNet): anchor 없이 중심점 기반.

**Loss**:
- Classification: focal loss (불균형 강함).
- Box regression: Smooth L1, GIoU, CIoU.

**Metric**: mAP (mean Average Precision). IoU threshold별 AP 평균.

### 11.2 Semantic Segmentation

각 픽셀에 클래스 라벨. 객체 instance 구분 안 함.

**Encoder-Decoder (U-Net)**:
```
Encoder: 점진적 down-sample (CNN backbone)
Decoder: 점진적 up-sample (transposed conv)
Skip connections: encoder의 같은 해상도 layer를 decoder에 concat
```

Skip connection 핵심: 깊은 layer는 high-level semantic, 얕은 layer는 정확한 위치 정보. 둘 다 필요.

**DeepLab (Google)**:
- Dilated conv로 RF 키우고 해상도 유지.
- Atrous Spatial Pyramid Pooling (ASPP): 다양한 dilation의 conv concat.

**Loss**: pixel-wise CE 또는 Dice loss.

**Metric**: mIoU (mean Intersection over Union).

### 11.3 Instance Segmentation

Semantic + 객체 instance 구분. "고양이 1, 고양이 2" 식.

**Mask R-CNN**: Faster R-CNN에 mask branch 추가. 각 object box 안에서 segmentation mask.

### 11.4 핵심 통찰 — 분류는 압축, 다른 task는 위치 보존

| Task | Output | 디자인 |
|---|---|---|
| 분류 | 클래스 (위치 무관) | 점진적 압축 + GAP |
| Detection | Box + class | 다양한 RF + anchor + FPN |
| Segmentation | 픽셀 라벨 | Encoder-Decoder + skip |
| Instance Seg | Box + mask | Detection + segmentation |

분류는 위치 정보를 *버려도* 됨. Detection·segmentation은 위치 정보를 *보존*해야. 이게 architecture 차이의 핵심.

---

## 12. CNN vs Vision Transformer

### 12.1 Vision Transformer (ViT, 2020)

이미지를 patch로 자르고 (16×16 patch), patch를 token으로. Transformer encoder를 그대로.

```
Input: 224×224×3 → 14×14 patches of 16×16
Each patch: 16×16×3 = 768 → linear projection → 768-d token
+ positional encoding
↓
Transformer Encoder (12층)
↓
Class token → MLP → 분류
```

### 12.2 CNN vs ViT 비교

| | CNN | ViT |
|---|---|---|
| Inductive bias | 국소성, 평행이동 | 거의 없음 (PE만) |
| 데이터 효율 | 좋음 | 많이 필요 |
| Long-range | 깊이 필요 | 한 layer에 전역 |
| 병렬 | 가능 | 매우 잘됨 |
| Scaling | 한계 보임 | 매우 잘 됨 |
| 일반 이미지 | 표준 | 데이터 충분 시 우위 |

### 12.3 ViT가 CNN을 이기는 조건

데이터 매우 많을 때. ImageNet만이면 CNN이 우위 또는 비슷. JFT-300M (Google 내부 3억 장 데이터셋) 같은 대량 사전학습 후 ImageNet fine-tune이면 ViT가 CNN을 이김.

**왜?**: CNN의 inductive bias (국소성)는 prior이자 *제약*. 데이터 적을 때 좋은 prior, 데이터 많을 때 제약. ViT는 prior 없어 데이터에서 자유롭게 학습. 데이터 무한이면 ViT 우위.

### 12.4 ConvNeXt — CNN의 반격

위에서 언급. ResNet에 modern Transformer 디자인 적용 → ViT와 비슷한 성능. **현 시점에서 CNN과 ViT는 거의 같은 트랙**.

### 12.5 Hybrid 모델

CoAtNet, MaxViT 등 — CNN block (얕은 부분) + Transformer block (깊은 부분). 양쪽 장점 결합.

### 12.6 결론

"CNN vs ViT" 식 이분법은 점점 흐려진다. 두 패러다임이 서로의 기법을 흡수. 미래는 *더 통합된 architecture*로 갈 가능성 큼.

---

## 13. 면접 단골 Q&A

### Q1. 왜 이미지에 FNN을 안 쓰나?
"세 가지입니다. (1) 파라미터 폭발 — 224×224×3 입력에 1000 hidden FC layer만 1.5억 파라미터. (2) 위치 정보 미활용 — 같은 패턴이 다른 위치에 있어도 별도 학습. (3) 자연 이미지의 계층 구조 활용 못함. CNN의 weight sharing + local connectivity + 계층적 깊이가 이 셋을 동시에 해결합니다."

### Q2. Weight sharing이 왜 좋은가?
"세 가지입니다. (1) 파라미터 효율 — 위치별 가중치 대신 한 커널을 모든 위치에. 1만 배 이상 적은 파라미터. (2) Translation equivariance — 같은 패턴이 어디 있든 같은 응답. (3) 데이터 효율 — 한 위치 학습이 모든 위치에 자동 적용. 사실상 데이터 augmentation 효과. 이 셋이 자연 이미지의 inductive bias와 정확히 맞아서 CNN이 효율적입니다."

### Q3. Equivariance와 Invariance 차이?
"Equivariance: $f(T(x)) = T(f(x))$. 입력 이동에 출력도 같은 만큼 이동. Conv 연산이 이걸 만족. Invariance: $f(T(x)) = f(x)$. 입력 변화에 출력 변화 없음. Pooling, GAP가 이걸 부여. CNN의 강력한 점은 conv로 equivariance 유지하다가 마지막에 GAP로 invariance 부여 — 그래서 분류·detection·segmentation 모두 가능합니다."

### Q4. 3×3 conv를 깊이 쌓는 이유?
"VGG의 통찰입니다. 7×7 conv 1번과 3×3 conv 3번이 같은 RF (7×7). 파라미터: 49 vs 27 — 45% 적음. 비선형 활성화 횟수: 1 vs 3 — 표현력 ↑. 같은 RF면서 더 적은 파라미터로 더 풍부한 표현. 그래서 modern CNN은 거의 모두 3×3."

### Q5. 1×1 conv는 무엇을 하나?
"공간 정보는 그대로 두고 채널 차원만 변환합니다. 사실상 각 픽셀 위치의 fully connected layer (위치 간 weight sharing). 용도: (1) 채널 차원 조절 (256 → 64), (2) Bottleneck (1×1로 압축 후 3×3 작업 후 1×1로 복구 — ResNet 50+층의 표준), (3) Inception에서 다양한 RF 합칠 때 비용 절감, (4) 채널 간 비선형 결합."

### Q6. ResNet의 skip connection이 왜 혁명?
"두 측면입니다. (1) Degradation problem 해결 — F가 0을 학습하면 자동 항등. 깊이 추가가 최소한 더 나빠지지 않음. (2) Vanishing gradient 우회 — $\partial h_{l+1}/\partial h_l = I + \partial F$. Identity 항으로 gradient가 덧셈으로 흐름. 곱셈 누적이 없어 vanishing 약함. 100~1000층 학습 가능. 후속 모든 architecture (DenseNet, Transformer 등)의 핵심 부품. LSTM의 cell state와 본질 같음 — 곱셈 누적을 덧셈 경로로."

### Q7. Pooling vs Strided Conv?
"Pooling은 비학습 down-sampling — max는 두드러진 신호, avg는 부드러움. Strided conv는 학습 가능 down-sampling — task-적응적. 현대 CNN은 stride conv가 표준 (ResNet, EfficientNet). 단 max pool은 첫 layer 후 한 번 사용 정도. 마지막엔 GAP. 이유는 학습된 down-sampling이 hand-crafted보다 유연."

### Q8. Transfer Learning이 왜 작동?
"얕은 layer는 도메인 무관한 일반 feature (edge, texture, color), 깊은 layer는 task-specific. ImageNet 학습 모델의 얕은 layer는 다른 task에도 유용 → 그 부분 freeze + 깊은 layer만 fine-tune. 데이터 적을 때 매우 효과적 — 1k 데이터로도 50%+ 정확도 가능. 단 도메인 차이 큰 경우 (의료 영상 등)는 깊은 layer도 다시 학습 필요."

### Q9. ViT가 CNN을 이기는 조건?
"데이터 매우 많을 때입니다. CNN의 국소성·평행이동 inductive bias는 자연 이미지의 prior로 매우 효과적, 단 *제약*이기도 합니다. 데이터 적을 때 (ImageNet 정도) 이 prior가 도움 → CNN 우위. 데이터 많을 때 (JFT-300M 같은 3억 장) prior가 제약이 됨 → ViT가 더 자유로운 학습으로 우위. 결국 CNN과 ViT의 trade-off는 inductive bias의 prior vs 자유도."

### Q10. Detection에서 anchor box는 왜?
"다양한 크기·비율의 객체에 대한 prior. 한 위치에서 작은 사람, 큰 차, 가로형 버스 등을 동시에 검출하려면 다양한 anchor 필요. 각 anchor가 'object 예/아니오 + box offset' 예측. Anchor-free 방법(CenterNet, FCOS)도 발전 중 — 중심점 기반. Trade-off: anchor 多 = 정확도↑ 비용↑, anchor-free = 단순↑."

### Q11. Segmentation에 encoder-decoder가 왜 필요?
"분류는 공간 정보를 *압축해도 됨* — '어디'보다 '무엇'이 중요. Segmentation은 *pixel-level 출력*이 필요 → up-sampling 경로 필수. Encoder가 점진 압축, decoder가 점진 복원. Skip connection이 핵심 — 깊은 layer의 semantic + 얕은 layer의 정확한 위치 정보 결합. U-Net이 의료 영상 segmentation의 표준이 된 이유."

### Q12. CNN의 receptive field가 충분히 크면 왜 long-range 문제가 있나?
"이론상 RF와 effective RF가 다릅니다. 매우 깊은 망에서 RF가 입력 전체를 덮어도, 가운데 픽셀의 영향이 압도적이고 가장자리는 약함 (가우시안 분포에 가까움). 그래서 RF 크기와 실제 long-range 모델링 능력이 다름. 이게 attention이 등장한 이유 중 하나 — 거리 무관 직접 연결. ViT가 한 layer에서 전역 정보 통합 가능."

### Q13. MobileNet의 depthwise separable conv가 왜 효율적?
"표준 conv를 두 단계로 분해. (1) Depthwise: 각 입력 채널에 독립적인 3×3 — 채널별 공간 처리. (2) Pointwise (1×1): 채널 mixing. 비율: $1/C_{out} + 1/9$. $C_{out}=256$이면 약 1/9 — 9배 적은 파라미터. 정확도는 거의 같음. 모바일·임베디드에서 빛남. ConvNeXt도 비슷한 정신 (큰 depthwise + pointwise)."

### Q14. EfficientNet의 compound scaling이란?
"기존 모델 키울 때 보통 한 축만 (깊이만, 너비만, 해상도만). EfficientNet은 셋을 *균형 있게* — depth $\alpha^\phi$, width $\beta^\phi$, resolution $\gamma^\phi$. $\alpha\beta^2\gamma^2 \approx 2$ 제약. $\phi$가 compound coefficient. 같은 정확도에 5~10배 적은 파라미터. 통찰: 한 축 혼자 키우면 sub-optimal, 셋의 균형이 중요."

### Q15. 왜 FC layer 대신 GAP를 쓰는가?
"VGG 식 분류 head는 14×14×512 → flatten → 100K → FC 4096 → FC 4096 → FC 1000. 파라미터 1.2억. GAP 식: 14×14×512 → GAP → 512 → FC 1000. 파라미터 50만. 240배 적음. 정확도 비슷하거나 더 좋음. Overfitting 적음 (파라미터↓), 공간 정보 보존 (CAM으로 시각화 가능). 그래서 modern CNN의 분류 head는 GAP가 표준."

---

## 14. 생각해보라 — 단락 답안

**Q. 왜 자연 이미지에 weight sharing이 자연스러운가?**

자연 이미지의 두 핵심 성질이 있다. (1) **Translation invariance of recognition**: 객체가 어디 있든 같은 객체. 고양이는 사진 어디 있든 고양이. (2) **Local correlation**: 가까운 픽셀이 강한 상관. 멀리 있는 픽셀은 거의 독립.

이 둘이 weight sharing의 정당화. 같은 커널을 모든 위치에 = (1)을 가정. 작은 RF = (2)를 가정. 자연 이미지가 이 두 가정을 만족하므로 sharing이 효율적 prior.

만약 이 가정이 깨지면? (1) 위치별 다른 패턴 — 얼굴 인식. 일부 LCN 쓰임. (2) 강한 long-range — 매우 깊은 객체 (사람의 머리부터 발). 큰 RF 또는 attention 필요.

**Q. CNN의 inductive bias가 *제약*도 된다는 것의 의미?**

Inductive bias는 모델이 가진 사전지식. 좋은 prior일 수 있지만 *제약*일 수도. CNN의 국소성·평행이동 가정은:
- 자연 이미지에 잘 맞음 → 좋은 prior, 학습 효율 ↑.
- 데이터가 매우 많고 prior가 안 맞을 수도 있는 경우 → 제약, 더 자유로운 모델보다 손해.

ViT는 prior 거의 없어 데이터에서 *모든 것*을 학습. 데이터 적으면 (ImageNet) 학습이 어려움 → CNN이 우위. 데이터 매우 많으면 (JFT-300M) prior 없이도 충분 → ViT 우위.

이래서 "강한 inductive bias가 항상 좋은가?"는 데이터에 따라 답이 다른 trade-off.

**Q. 매우 깊은 CNN(1000층)이 왜 잘 안 되는가?**

ResNet의 skip connection으로 100~152층은 표준. 1000층도 시도된 적 있다 (DenseNet의 변종, ResNet v2 등). 하지만 보편적으로는 안 가는 이유:

(1) **Receptive field는 충분히 큼** — 100층이면 이미 입력 전체 덮음. 더 깊다고 RF 더 안 커짐.

(2) **Diminishing returns**: 한 layer 더 추가의 표현력 향상이 점차 감소.

(3) **메모리·연산 비용**: 깊이에 선형 비례. 효율 ↓.

(4) **Effective Receptive Field의 한계**: 이론상 RF가 커도 effective는 가운데 집중 → long-range 한계.

(5) **Inductive bias의 한계**: CNN의 국소성 가정이 한계점에 도달.

이래서 *깊이 대신 너비, 해상도, 또는 attention*으로 가는 게 후속 발전. EfficientNet의 compound scaling, ViT의 attention 등.

**Q. 1×1 conv를 *fully connected*로 봐도 되는가?**

엄밀히 보면 1×1 conv는 *각 픽셀 위치에서의 fully connected layer + 위치 간 weight sharing*. 한 픽셀의 채널 vector를 받아 다른 채널 vector로 — 정확히 FC.

차이는: FC는 모든 픽셀의 모든 채널을 입력으로, 1×1 conv는 한 픽셀의 모든 채널만. 즉 *공간 정보는 보존* (위치 무관 처리), *채널 정보는 자유롭게 변환*.

함의:
- FC layer는 1×1 conv의 일반화 (공간 차원이 1×1일 때).
- 1×1 conv는 "공간을 건드리지 않는 채널 변환".
- 그래서 spatial 정보 보존이 필요한 모든 자리 (segmentation, detection)에 1×1 conv가 자주 등장.

**Q. ResNet의 skip이 학습된 항등 함수보다 더 강한 이유?**

만약 항등 함수가 좋은 시작점이라면, 일반 깊은 망도 항등을 학습할 수 있어야. 그런데 실제로는 안 됨 (degradation problem). 왜?

답: SGD의 implicit bias. 일반 깊은 망의 가중치 공간에서 항등 함수에 해당하는 영역은 매우 좁고 sharp. SGD가 그 좁은 영역에 도달하기 어려움. 가중치가 0에 가깝게 init된 후 살짝 update 하면 항등에서 멀어짐.

ResNet은 항등을 *구조적으로 default*로. F=0이면 자동 항등. SGD가 F를 0 근처에서 시작해 *잔차*만 학습. 항등 자체를 학습할 필요 없음. 이게 깊은 학습이 가능해진 핵심.

이래서 "skip connection은 단순한 트릭이 아니라 학습 동역학을 근본적으로 바꾸는 디자인"이라는 분석.

**Q. Detection의 anchor box가 *prior*로서 어떤 역할?**

Anchor box는 "이런 크기·비율의 object가 흔하다"의 prior. 학습 전에 anchor를 정의 — 예: $\{32, 64, 128, 256, 512\}$ 크기, $\{1:1, 1:2, 2:1\}$ 비율. 각 위치에서 9개 anchor 활성.

학습은 각 anchor에 대해 (1) object 있나 (binary), (2) class, (3) box 보정 (anchor 대비 offset)을 예측.

**왜 anchor가 필요?**: 단순히 "(x, y, w, h) 회귀"하면 학습이 어려움 — 매우 작은 좌표 변화가 scale 다른 object에서 큰 의미 차이. Anchor가 reference이라 학습이 더 안정.

**왜 anchor-free가 다시 나오나?**: Anchor 자체가 hyperparameter, 데이터에 따라 anchor 분포 다시 설계. CenterNet, FCOS 등은 객체의 중심점에 직접 회귀 — 단순. 정확도 비슷.

이래서 detection 분야는 "anchor 사용 vs 아님"의 사이를 진동. 본질적 trade-off는 prior의 강함.

**Q. CNN과 ViT가 합쳐지는 미래?**

이미 진행 중. ConvNeXt가 ResNet에 modern Transformer 디자인 — ViT와 비슷. CoAtNet, MaxViT 등 hybrid — 얕은 부분 CNN, 깊은 부분 Transformer.

미래 가능성:
- (1) **Inductive bias의 점진적 흡수**: ViT가 conv-like local attention 추가, CNN이 global pooling/attention 추가.
- (2) **Architecture 자동화**: NAS로 CNN+Transformer 최적 조합 탐색.
- (3) **Foundation 모델**: 매우 큰 데이터로 pretrain된 단일 backbone이 모든 vision task 처리. CLIP, DINO 등.

본질적으론 "**locality vs global**"의 trade-off를 어떻게 잘 결합하느냐의 문제. CNN은 locality의 극단, Transformer는 global의 극단. 둘의 sweet spot이 미래.

**Q. Transfer learning의 *negative transfer*?**

Source 도메인과 target 도메인이 너무 다르면 pretrained 가중치가 *해가 됨*. From scratch보다 못함. 예: ImageNet (자연 사진)에서 학습한 모델을 만화 또는 추상 미술에 적용하면 약함.

처방:
- 깊은 layer freeze 안 하고 fully fine-tune.
- Pre-training을 target 도메인 비슷한 데이터로.
- Self-supervised pre-training (모든 자연 이미지에 일반 적용).

함의: Transfer는 항상 좋은 게 아님. 도메인 차이 평가 후 결정. "ImageNet pretrain이 default"는 자연 이미지에선 맞지만 모든 비전 task에 일반화 안 됨.

---

## 15. 한 줄 요약 (시험 직전)

- **CNN의 근거** = 자연 이미지의 *국소성 + 평행이동* prior를 sharing + local connectivity로 구현.
- **Conv = equivariance, Pool/GAP = invariance**. 이 둘의 조합으로 분류·detection·segmentation 모두 가능.
- **Weight sharing의 세 효과** = 파라미터 효율 + translation equivariance + 데이터 효율.
- **3×3 conv 깊이 쌓기**가 큰 커널 1번보다 효율 (RF 같으면서 파라미터↓ 비선형↑).
- **1×1 conv** = 공간 보존 + 채널 변환. Bottleneck, 채널 조절, Inception의 핵심.
- **Pooling vs Stride conv** = hand-crafted vs 학습된 down-sampling. Modern은 stride.
- **GAP**가 FC head보다 좋음 — 파라미터↓, overfitting↓, CAM 가능.
- **ResNet skip의 효과** = identity default + gradient 직접 흐름 (덧셈 경로). LSTM cell state와 본질 같음.
- **Architecture 진화**: LeNet → AlexNet (deep+GPU) → VGG (단순 깊이) → GoogLeNet (다양 RF) → ResNet (skip) → DenseNet (reuse) → MobileNet (separable) → EfficientNet (균형 scale) → ConvNeXt (modern training).
- **Transfer learning** = 얕은 layer 도메인 무관, 깊은 layer task-specific. 데이터 적을 때 매우 효과적.
- **Detection**: anchor 또는 anchor-free, two-stage 또는 single. **Segmentation**: encoder-decoder + skip.
- **CNN vs ViT**: 데이터 적으면 CNN, 매우 많으면 ViT. ConvNeXt로 격차 줄어듦. 미래는 hybrid.
