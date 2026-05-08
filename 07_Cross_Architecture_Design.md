# 07. Cross-Architecture Design — "다른 모델로 풀어봐" 심층

> **이 문서의 핵심 사고**: 모든 데이터를 모든 모델로 *풀 수 있다*. 단 효율, 잃는 정보, 얻는 능력이 다르다. 그 trade-off를 *수식·사례·역사*로 풀어내는 게 목적.
>
> **교수님 직격타 질문**:
> "이미지를 CNN이 아니라 RNN으로 학습하려면 어떻게 설계해야 하고, 무엇이 다르냐?"
>
> 이 질문의 *완전한 답안*을 만든다 — 어떻게 설계, 무엇을 잃는가, 무엇을 얻는가, 언제 합리적인가, 언제 비합리적인가.

---

## 0. 큰 그림 — Inductive Bias 매칭의 사고

### 0.1 모델 = Inductive Bias의 묶음

머신러닝 모델은 *학습 가능한 함수의 공간*을 정의한다. 이 공간은 데이터에 따라 어떤 함수가 더 *그럴듯*한지에 대한 사전 가정 (inductive bias)에 의해 좁혀진다.

각 architecture의 inductive bias:
- **FNN**: 거의 없음. 모든 입력 차원이 동등.
- **CNN**: 국소성 (가까운 픽셀이 강한 상관) + 평행이동 등변성 (위치 독립 패턴) + 계층적 합성.
- **RNN**: 시간 순서 + 시간 축 weight sharing (시간 독립 패턴).
- **Transformer**: 거의 없음 (위치 인코딩으로 약하게 시간 정보).
- **GNN**: 노드 순서 무관 + 인접성으로 정보 전파.

### 0.2 데이터 = 본래 구조

각 데이터 종류는 본래의 구조를 갖는다:
- **이미지**: 2D 격자, 국소 상관, 평행이동에 의미 보존.
- **텍스트**: 1D 시퀀스, 시간 순서, 다양한 거리의 의존성.
- **시계열**: 1D 시퀀스, 시간 순서, trend·seasonality·noise.
- **그래프**: 노드 + 엣지, 임의 위상, 인접성 정보.
- **표(tabular)**: 컬럼 (순서 임의), feature간 의미적 관계.
- **동영상**: 시간 + 공간 (3D).
- **오디오**: 시간 (raw waveform) 또는 시간×주파수 (spectrogram).

### 0.3 Cross-Architecture 사고의 세 단계

데이터 D를 모델 M으로 풀려면:

**Step 1 — 데이터 변환**:
D의 본래 형태를 M의 입력 형식에 맞게 변환. 종종 정보 손실.

**Step 2 — 무엇을 잃나**:
D의 본래 inductive bias가 M의 inductive bias와 어긋나는 부분. 모델이 *이 prior를 활용 못함*.

**Step 3 — 무엇을 얻나**:
M이 가진 inductive bias가 D에 *추가로* 도움이 되는 부분. 또는 M의 다른 장점 (병렬화 등).

이 세 단계로 분석하면 어떤 cross-architecture 설계도 *체계적*으로 답할 수 있다.

### 0.4 이 챕터의 흐름

각 cross-architecture 사례를 다음 패턴으로:
1. 어떻게 설계? (변환 방법)
2. 무엇을 잃나? (잃는 inductive bias)
3. 무엇을 얻나? (얻는 능력)
4. 언제 합리적? (실용적 사용 case)
5. 답안 골격

다루는 사례:
- 이미지 → RNN (§1, ★ 단골)
- 텍스트 → CNN (§2)
- 시계열 → FNN (§3)
- 시퀀스 → Transformer vs RNN (§4)
- 그래프 → CNN (§5)
- 표 데이터 → 신경망 (§6)
- 동영상의 다양한 처리 (§7)
- 오디오의 시간/주파수 도메인 (§8)

---

## 1. 이미지를 RNN으로 — ★ 교수님 단골

### 1.1 어떻게 설계?

**기본 아이디어**: 이미지의 픽셀(또는 patch)을 1D 시퀀스로 펼쳐서 RNN에 입력.

펼치는 방법 (linearization):

**(1) Raster scan**: 좌→우, 위→아래. 가장 단순.
```
픽셀 순서: (0,0), (0,1), (0,2), ..., (0,W-1), (1,0), (1,1), ...
```

**(2) Hilbert curve**: 공간 국소성을 보존하는 fractal 곡선. 인접 픽셀이 시퀀스에서도 가까움.

**(3) Diagonal scan**: 대각선 단위.

**(4) Patch sequence**: 16×16 patch 단위로 펼침. ViT 식 (Transformer지만 같은 정신).

### 1.2 PixelRNN — 실제 사례 (DeepMind, 2016)

**목표**: 이미지의 자기회귀 생성 모델. $P(\text{image}) = \prod_{i,j} P(x_{i,j} | x_{<i,<j})$.

**구조**:
- 각 픽셀을 raster order로 시퀀스화.
- LSTM이 이전 픽셀들의 컨텍스트로 다음 픽셀 분포 예측.
- 각 픽셀의 RGB 값을 256-way categorical로 모델링 (각 채널 독립 또는 조건부).

**변종**:
- **Row LSTM**: 행 단위로 처리. 한 행의 픽셀들을 conv처럼 처리하면서 시간 축은 행.
- **Diagonal BiLSTM**: 대각선 단위. 더 효율적인 의존성 모델.

**성능**: 이미지 생성 task에서 그 시대 SOTA. 단 Diffusion model 등장 후 사라짐.

### 1.3 무엇을 잃나? (★)

(1) **2D 공간 inductive bias 상실**: 
CNN은 *모든 위치에서 같은 커널* (translation equivariance). RNN은 *시간 축에서 같은 cell* — 1D만. 2D 공간 sharing 없음.

같은 패턴 (예: 고양이 눈)이 이미지의 다른 위치에 있어도 RNN은 *별도로* 학습해야. 데이터 효율이 매우 떨어짐.

(2) **장거리 픽셀 의존성 약함**:
224×224 이미지를 raster scan하면 시퀀스 길이 50,176. 행 끝에서 다음 행 시작까지의 *공간 거리*는 1 픽셀이지만 *시퀀스 거리*는 224 step. RNN은 이 distance를 거의 인식 못함.

LSTM도 100~200 step 정도가 한계. 50,000 step은 비현실적.

(3) **인접성 왜곡**:
Raster scan에서 (0, 223)과 (1, 0)은 *공간적 인접*이지만 *시퀀스 거리* 1 step. 반대로 (0, 100)과 (1, 100)은 공간 거리 1, 시퀀스 거리 224. 매우 부정확한 prior.

Hilbert curve가 이걸 부분적 해결 — 인접 시퀀스가 인접 공간. 단 완전 해결 아님.

(4) **병렬화 불가**:
CNN은 모든 위치 conv를 GPU에서 동시 계산. RNN은 시점 t가 t-1 의존 → 순차. 학습·추론 시간 50,000배 (시퀀스 길이만큼) 차이.

### 1.4 무엇을 얻나?

(1) **명시적 자기회귀 분포**: $P(\text{image}) = \prod P(x_i | x_{<i})$. 각 픽셀의 *조건부 분포*가 명확. 이미지 생성·확률 추정에 자연스러움.

(2) **가변 해상도** (이론상): RNN은 시퀀스 길이 무관 → 다른 크기 이미지 처리 가능. 단 학습 데이터의 길이 분포에 의존하므로 실용성은 제한적.

(3) **Sequential decision making**: 각 픽셀을 이전 컨텍스트로 결정. 일부 task (super-resolution, inpainting)에 유용한 시각.

### 1.5 결론 — 언제 합리적?

**비합리적 case**:
- **분류·인식**: CNN이 압도적 효율. RNN 절대 안 씀. 같은 정확도 위해 100~1000배 학습 시간.

**합리적 case**:
- **자기회귀 이미지 생성** (PixelRNN 시대): 픽셀 분포의 명시적 모델링이 필요할 때.
- **현재는 거의 사라짐** — Diffusion이 더 좋은 생성 품질, 더 빠른 학습.

### 1.6 답안 골격 (★ 시험·면접용)

> "이미지를 RNN으로 학습하려면 픽셀(또는 patch)을 raster scan 또는 Hilbert curve로 1D 시퀀스화. 각 시점에 한 픽셀(또는 patch)을 LSTM/GRU에 입력. PixelRNN이 대표 사례 — 이미지의 자기회귀 생성에 사용.
>
> 무엇을 잃나: (1) CNN의 2D 공간 inductive bias (국소성 + translation equivariance) 상실. 같은 패턴이 다른 위치에 있어도 별도 학습. (2) 224×224 이미지가 50,000 step 시퀀스 — LSTM의 long-range 한계 초과. (3) 인접성 왜곡 — raster scan은 공간 인접 ≠ 시퀀스 인접. (4) 병렬화 불가 — 학습·추론 매우 느림.
>
> 무엇을 얻나: 명시적 자기회귀 분포 — 이미지 생성에 자연스러움. 가변 해상도 (이론상).
>
> 결론: 분류·인식엔 비현실적 (CNN의 100~1000배 비효율). 자기회귀 생성에는 의미 있었으나 현재는 Diffusion에 밀림."

---

## 2. 텍스트를 CNN으로 — TextCNN 사례

### 2.1 어떻게 설계?

**기본 아이디어**: 텍스트를 1D 시퀀스로 보고 1D conv 적용.

**구조** (Yoon Kim, TextCNN 2014):
1. 단어를 embedding (V → d 차원). 시퀀스 (T × d) 행렬.
2. 1D conv를 d 채널 위에 슬라이딩. 커널 폭 k = 2, 3, 4 (= n-gram).
3. **Max-over-time pooling**: 각 필터 출력에서 max. 길이 무관 고정 vector.
4. FC → softmax.

```
"This movie is great"
embedding → (4, d)

Conv (k=3): 슬라이딩 윈도우, n-gram 추출
Output: (2, num_filters)

Max pool: 가장 강한 신호만
Output: (num_filters,)

FC → 클래스
```

### 2.2 무엇을 얻나?

(1) **병렬화**: RNN은 시점 t가 t-1 의존, 순차. CNN은 모든 위치 동시. **학습 50~100배 빠름**.

(2) **국소 패턴 (n-gram) 학습**:
- 커널 k=2: bigram ("not good", "very nice")
- 커널 k=3: trigram ("I love you", "not very good")
- 데이터에서 자동 학습.

(3) **단순·강력한 baseline**: 감성 분석에서 LSTM과 거의 동급. 빠르고 단순.

(4) **위치 무관한 패턴 검출**: max pool이 "이 패턴이 어디 있어도 강한 신호" — 위치 독립.

### 2.3 무엇을 잃나?

(1) **장거리 의존성**: 
RF가 깊이로 제한. "I went to the *bank* of the *river*"에서 "bank"의 의미는 멀리 떨어진 "river" 의존. 깊이 5~6 layer로 약 10단어 RF — 긴 문서엔 부족.

처방: dilated conv, attention 추가. 단 RNN/Transformer가 더 자연.

(2) **순서 정보 일부 손실**:
Max pool이 위치를 흐림. "good not bad"와 "bad not good"이 같은 max 값이 될 수 있음.

처방: positional embedding 추가 (Transformer 식). 또는 max pool 대신 attention.

(3) **가변 길이 처리 부자연**:
Conv 자체는 가변 길이 OK지만 max pool 후 고정 vector → 분류엔 OK, 생성엔 부적절.

### 2.4 언제 합리적?

**합리적 case**:
- **짧은 텍스트 분류**: 트윗 감성, 리뷰 분류. 키워드성이 중요한 task.
- **빠른 prototype**: 1시간에 학습.
- **Baseline 모델**: SOTA 시도 전에 reference로.
- **Edge device**: 모바일에서 실시간 추론.

**비합리적 case**:
- **긴 문서 이해**: Transformer가 우위.
- **생성**: 자기회귀 모델이 자연. CNN으론 어려움.
- **Strong long-range 의존**: QA, 추론.

### 2.5 RNN/LSTM과 비교

| | TextCNN | LSTM | Transformer |
|---|---|---|---|
| 장거리 | 약함 | 보통 | 강함 |
| 병렬 | 가능 | 불가 | 매우 잘 |
| 데이터 효율 | 좋음 | 보통 | 많이 필요 |
| 짧은 분류 | **강력** | OK | overkill 가능 |
| 학습 속도 | 빠름 | 느림 | 빠름 |

짧은 텍스트 + 데이터 적음 → TextCNN이 의외로 강력. 큰 데이터·긴 문서 → Transformer.

### 2.6 답안 골격

> "단어 embedding 후 1D conv를 d 채널 위에 슬라이딩. 커널 폭 2~5로 n-gram 패턴 학습. Max-over-time pooling으로 길이 무관 고정 vector. FC + softmax로 분류.
>
> 얻는 것: 병렬화 (RNN의 50~100배), n-gram 자동 학습, 단순·강력한 baseline.
>
> 잃는 것: 장거리 의존성 (RF 한계), 순서 정보 일부 (max pool), 생성에 부자연.
>
> 짧은 텍스트 분류·감성 분석에 매우 강력. 긴 문서·생성엔 Transformer 우위."

---

## 3. 시계열을 FNN으로 — Sliding Window

### 3.1 어떻게 설계?

**기본 아이디어**: 최근 N 시점을 펼쳐서 FNN 입력.

```
시계열: x_1, x_2, x_3, ..., x_T
Window N = 5
입력: [x_{t-4}, x_{t-3}, x_{t-2}, x_{t-1}, x_t] → FNN → x_{t+1} 예측
```

학습: 각 시점에서 window 추출, target은 다음 값.

### 3.2 무엇을 잃나?

(1) **가변 길이 처리 불가**:
N 고정. 100 step 데이터와 1000 step 데이터를 같은 모델로 처리 못함.

(2) **시간 축 weight sharing 없음**:
시점 t의 패턴과 t-5의 같은 패턴을 *별도로* 학습. 데이터 효율 낮음.

예: "주말 효과"가 모든 주에서 같은 패턴이라도 FNN은 매주 따로 학습.

(3) **N 너머 정보 완전 차단**:
N=10이면 11 step 전 정보는 없음. 강한 trend가 한 달 전부터 누적된 것이라면 못 잡음.

(4) **N을 키우면 파라미터 폭발**:
N=100, hidden=100이면 입력 layer만 10,000 가중치. N=1000이면 100,000. 학습 어려움.

### 3.3 무엇을 얻나?

(1) **단순함**: 학습이 매우 안정. RNN의 vanishing 문제 없음.

(2) **단기 예측에 의외로 강함**: 단기 패턴이 dominant하면 baseline으로 강력.

(3) **빠른 학습**: 병렬, 메모리 적음.

(4) **표 데이터로 환원**: 시계열을 *table*로 변환 (각 lag을 별도 feature) → GBM 등 적용 가능. 실무에서 자주 사용.

### 3.4 언제 합리적?

**합리적 case**:
- **단기 예측** (1~10 step ahead): 단기 패턴이 dominant.
- **강한 외생 변수**: 기온, 요일 등이 dominant하면 시간 의존 약함.
- **Baseline**: 복잡 모델 전에 reference.
- **Edge device**: 매우 단순·빠름.

**비합리적 case**:
- **장기 예측**: 한 달 + 미래.
- **가변 길이 시퀀스**.
- **복잡한 trend·seasonality**.

### 3.5 1D CNN과의 비교

1D CNN (TCN)이 FNN보다 시계열에 자연:
- 시간 축 weight sharing (한 위치 학습 = 모든 위치).
- Dilated conv로 RF 키움.
- 가변 길이 처리.

TCN이 LSTM과 비슷한 성능에 학습 더 안정.

### 3.6 답안 골격

> "최근 N 시점을 펼쳐서 FNN 입력으로. 각 시점이 다른 feature 채널.
>
> 잃는 것: (1) 가변 길이 처리 불가, (2) 시간 축 weight sharing 없음 — 같은 패턴을 위치마다 따로 학습, (3) N 너머 정보 완전 차단, (4) N 키우면 파라미터 폭발.
>
> 얻는 것: 단순, 학습 안정, 단기 예측에 의외로 강함, 빠름.
>
> 단기 예측·외생 변수 dominant·baseline에 합리적. 장기·가변 길이엔 RNN/Transformer/TCN."

---

## 4. 시퀀스를 Transformer로 — RNN 대체

### 4.1 어떻게 설계?

**기본 구조**:
1. Tokenization → token sequence.
2. Embedding + Positional Encoding.
3. Multi-head self-attention (시퀀스 내 모든 토큰 간).
4. FFN (각 토큰별 MLP).
5. Skip connection + LayerNorm.
6. 여러 layer stack.

### 4.2 RNN 대비 다른 점

| | RNN | Transformer |
|---|---|---|
| 시간 축 처리 | 순차 ($h_t$는 $h_{t-1}$ 필요) | 동시 (모든 시점 attention) |
| Long-range | 거듭제곱 → 약함 | 거리 무관 직접 |
| 병렬 | ✗ | ✓ 매우 잘 |
| Inductive bias | 시간 순서 강함 | 거의 없음 (PE만) |
| 메모리 | $O(T \cdot d)$ | $O(T^2 \cdot d)$ ★ |
| 데이터 요구 | 적당 | 많음 |
| 위치 정보 | 자동 (sequential) | 명시적 PE 필요 |

### 4.3 무엇을 얻나?

(1) **장거리 의존성 직접**: 토큰 1과 토큰 1000을 한 layer에서 연결. RNN의 chain 우회.

(2) **병렬 학습**: GPU 활용 극대화 → 큰 모델 가능. GPT-3 (175B) 같은 모델은 RNN으론 거의 불가.

(3) **Scaling 잘 됨**: 데이터·파라미터·연산을 늘릴수록 성능 멱법칙으로 향상. RNN은 한계점 빨리 옴.

(4) **Self-attention의 표현력**: 토큰 간 관계를 동적으로 학습. 학습된 attention 패턴이 의미적으로 해석 가능.

### 4.4 무엇을 잃나?

(1) **순서 정보의 자연스러움**: PE에 의존. 잘못 설계하면 long-range generalization 약함.

(2) **메모리 $O(T^2)$**: 시퀀스 길이의 제곱. 1만 토큰 입력은 1억 attention 항. 매우 부담.

처방: Sparse attention, Linear attention, FlashAttention.

(3) **데이터·연산 多 필요**: 작은 데이터에선 RNN/CNN보다 못함. ViT가 ImageNet에서 처음 CNN에 밀린 이유.

(4) **inductive bias 부족**: RNN의 *시간 sharing*이 자연스러운 prior. Transformer는 PE에서만. 작은 데이터에서 prior 약점.

### 4.5 왜 NLP가 Transformer로?

세 압도적 이유:

(1) **병렬화**: RNN은 시퀀스 길이만큼 step 직렬. GPU 활용 못함. Transformer는 거의 100% GPU 활용.

(2) **Long-range**: 100+ 토큰의 의존성을 직접. 긴 문서 이해, 다중 문장 추론.

(3) **Scaling**: 모델·데이터 키울수록 성능 향상. Foundation model 시대 가능.

이 셋이 합쳐서 NLP 패러다임 전환. RNN은 niche로.

### 4.6 답안 골격

> "Token embedding + PE → Multi-head self-attention → FFN을 stack. RNN의 순차 처리 대신 모든 시점 동시 attention.
>
> 얻는 것: (1) 장거리 직접 — 거리 무관 한 layer 연결, (2) 병렬 — GPU 활용 극대, (3) Scaling — 큰 모델 가능, (4) 표현력.
>
> 잃는 것: (1) 순서 정보가 PE 의존, (2) 메모리 $O(T^2)$ — 긴 시퀀스 부담, (3) 데이터·연산 많이 필요, (4) inductive bias 약함.
>
> NLP 큰 모델은 Transformer 표준. 작은 데이터·짧은 시퀀스엔 RNN 여전히 의미."

---

## 5. 그래프를 CNN으로 — 격자가 아닌 데이터

### 5.1 그래프의 본질

그래프 $G = (V, E)$:
- 노드 $V$: 임의 개수, 임의 순서.
- 엣지 $E$: 노드 간 임의 연결.
- 노드별 차수 (이웃 수) 가변.

**격자가 아님** — CNN의 정해진 인접성 가정 깨짐.

### 5.2 강제로 CNN 적용?

**시도 1**: 노드를 격자에 매핑.
- 노드 순서를 정해 1D 또는 2D 격자에.
- 문제: *노드 순서가 임의* → 결과가 순서에 의존. Permutation invariance 깨짐.

**시도 2**: 인접 행렬을 이미지로.
- $V \times V$ 인접 행렬을 2D 이미지로 보고 CNN.
- 문제: 큰 그래프는 거대한 이미지. 또 인접 행렬의 *순서가 임의*.

**시도 3**: 그래프를 시퀀스로 (random walk).
- DeepWalk 식. 그래프 위 random walk으로 시퀀스 생성, 그 시퀀스에 RNN/CNN.
- 문제: 정보 손실. 한 random walk이 그래프의 일부만.

이 셋 모두 *부자연*. 그래프의 구조와 모델의 가정이 안 맞음.

### 5.3 그래프에 맞는 architecture — GNN

**Graph Convolutional Network (GCN)**: 노드의 이웃 표현을 평균/합:

$$h_v^{(l+1)} = \sigma\left( W \cdot \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} h_u^{(l)} \right)$$

핵심: *노드 순서 무관* + 차수 무관 + 인접성 정보 활용.

**Graph Attention Network (GAT)**: 이웃에 attention 가중. 더 정교.

**Message Passing**: 더 일반적인 framework. 각 노드가 이웃에 메시지 보내고 받음.

### 5.4 무엇을 잃나? (CNN으로 시도 시)

(1) **Permutation invariance**: 노드 순서 바꾸면 결과 변함. CNN은 격자 가정.
(2) **차수 정보**: CNN은 정해진 RF, 차수 가변에 안 맞음.
(3) **장거리 정보**: CNN의 RF 한계.

### 5.5 답안 골격

> "그래프는 격자가 아니라 임의 위상. CNN의 정해진 인접성과 안 맞음. 강제 적용 (격자 매핑, 인접 행렬, random walk) 모두 부자연 — permutation invariance 깨짐 또는 정보 손실.
>
> 자연스러운 답: GNN (GCN, GAT). 노드 표현을 이웃 표현의 (가중) 평균으로 update. 노드 순서·차수 무관.
>
> 교훈: 데이터 구조에 맞는 inductive bias가 따로 필요. CNN을 강제 적용하지 말 것."

---

## 6. 표(tabular) 데이터를 신경망으로

### 6.1 표 데이터의 특성

- **컬럼 순서 임의**: 카테고리/숫자 feature, 순서에 의미 없음.
- **컬럼 간 의미적 관계**: 강하지만 위치적 의존 없음.
- **Feature scale 이질적**: 나이 (0~100), 소득 (0~수억), 카테고리 (one-hot).
- **결측치, 이상치 빈번**.

### 6.2 RNN/CNN 강제 적용?

**RNN**: 컬럼을 시퀀스로 → "컬럼 순서"가 의미 없으니 학습 신호 부정확. 다른 순서면 다른 결과 — 부자연.

**CNN**: 격자 가정 위배. 인접 컬럼이 의미적으로 관련 없을 수 있음.

이 둘 모두 표 데이터에 *부자연*.

### 6.3 합리적 옵션

**(1) MLP (FNN)**: 가장 단순. 적절한 normalization·embedding.
- 카테고리 → embedding.
- 숫자 → standardization 또는 quantile normalization.
- BN 또는 LayerNorm.

**(2) GBM (XGBoost, LightGBM, CatBoost)**: 트리 기반.
- 표 데이터의 강자.
- Feature 간 비선형 상호작용 자연 학습.
- 결측치, 이상치 robust.
- Feature scale 무관.
- 종종 딥러닝 압도.

**(3) TabTransformer / FT-Transformer**: 
- Feature를 embedding 후 self-attention.
- 컬럼 간 임의 상호작용 학습.
- 큰 데이터에서 GBM 경쟁 가능.

### 6.4 왜 GBM이 표 데이터에 강한가?

(1) **분할 기반 학습**: 트리가 feature value의 *임의 분할*을 학습. 비선형, 비단조 자연.

(2) **Feature interaction**: 여러 feature의 조합을 트리 분기로 학습. 상호작용 자연.

(3) **Scale 무관**: 트리는 분할만, scale 영향 없음. 신경망은 standardization 필수.

(4) **이상치 robust**: 분할은 이상치에 덜 민감.

(5) **결측치 자동 처리**: 일부 라이브러리는 결측치를 별도 분기로.

이 다섯이 합쳐서 표 데이터에 *압도적*. 신경망은 매우 큰 데이터 + 강한 카테고리 (수만 개)에서만 경쟁 가능.

### 6.5 답안 골격

> "표 데이터는 컬럼 순서 임의 + 위치적 의존 없음. RNN/CNN의 inductive bias와 안 맞음. 강제 적용 시 부자연.
>
> 합리적 옵션: (1) MLP — 단순, baseline. (2) GBM — 표 데이터의 강자. (3) TabTransformer — 컬럼 간 attention.
>
> GBM이 강한 이유: 분할 기반 비선형, feature 상호작용 자연, scale 무관, 이상치·결측 robust. 신경망은 매우 큰 데이터 + 강한 카테고리 feature에서만 경쟁."

---

## 7. 동영상 — 시간 + 공간

### 7.1 동영상의 본질

동영상 = T × H × W × 3. 시간 + 공간 모두 inductive bias 필요.

### 7.2 옵션 비교

**옵션 1 — 2D CNN per frame + LSTM**:
```
프레임 1 → 2D CNN → feature_1
프레임 2 → 2D CNN → feature_2
...
프레임 T → 2D CNN → feature_T

LSTM(feature_1, ..., feature_T) → 출력
```

장점: 단순, 잘 작동. 사전학습 CNN 활용 가능.
단점: 시간·공간 분리 처리 → 동시 패턴 약함.

**옵션 2 — 3D CNN**:
```
3D conv: (k_t, k_h, k_w) 커널을 H × W × T 위에 슬라이딩
```

장점: 시공간 동시 처리.
단점: 파라미터·연산 매우 많음. ResNet 50 → 3D는 8배 비용.

**옵션 3 — (2+1)D CNN**:
3D conv를 *spatial 2D + temporal 1D*로 분해. Depthwise separable의 시공간 버전.

장점: 효율, 잘 작동.
단점: 약간 표현력 ↓.

**옵션 4 — Video Transformer (TimeSformer, Video Swin)**:
Patch × time을 attention.

장점: 강력, 장거리.
단점: 연산·데이터 많이 필요.

### 7.3 task별 선택

| Task | 권장 |
|---|---|
| 짧은 액션 인식 | (2+1)D CNN, 3D CNN |
| 긴 영상 분류 | 2D CNN + LSTM/Transformer |
| 영상 캡셔닝 | 2D CNN + LSTM (encoder) + LSTM (decoder) |
| 영상 생성 | Diffusion (3D 또는 conv-attention) |

### 7.4 사고 패턴

동영상의 두 inductive bias:
- 시간 (sequential, 짧은 의존성).
- 공간 (2D, 국소성).

어떻게 분배할지가 architecture 선택의 핵심:
- 둘 다 강한 모델 → 3D CNN, Video Transformer (비용 高).
- 시간 가벼운 모델 → 2D CNN + LSTM (효율).
- 시간·공간 분리 → (2+1)D (절충).

### 7.5 답안 골격

> "동영상 = 시간 + 공간. 두 inductive bias를 어떻게 처리하느냐가 핵심.
>
> 옵션: (1) 2D CNN per frame + LSTM — 단순, 효율, pretrained 활용. (2) 3D CNN — 시공간 동시. 비용 높음. (3) (2+1)D — 효율 절충. (4) Video Transformer — 강력하지만 비싼.
>
> 짧은 액션 인식엔 (2+1)D, 긴 영상엔 2D CNN + Transformer가 표준."

---

## 8. 오디오 — 시간 도메인 vs 주파수 도메인

### 8.1 입력 형식

**Raw waveform**: 1D 시퀀스 (16,000~48,000 sample/s). 매우 긴 시퀀스 (1초 = 16k samples).

**Spectrogram**: 2D (시간 × 주파수). Short-Time Fourier Transform으로 변환.

**Mel-spectrogram**: Mel scale (사람의 청각 반영). 80~128 mel bin × 시간.

### 8.2 모델 매칭

| 입력 | 모델 |
|---|---|
| Raw waveform | 1D CNN (WaveNet), Transformer |
| Spectrogram | **2D CNN** (음성 인식, 음악 분류) |
| Mel-spectrogram | 2D CNN (가장 흔한) |
| Long clip | Transformer (Whisper) |

### 8.3 핵심 통찰 — 변환의 가치

오디오를 *spectrogram으로 변환*하면 "오디오를 이미지처럼". CNN의 모든 기법 적용 가능.

이게 본질적 사고 패턴:
- 데이터 표현을 모델 inductive bias에 맞게 변환.
- 변환은 정보 손실 또는 강조.

Spectrogram 변환:
- 잃는 것: phase 정보 (그렇지만 분류엔 무관).
- 강조: 주파수 구조 (음성·음악의 본질).

### 8.4 WaveNet — Raw waveform CNN

DeepMind 2016. Raw waveform에 dilated 1D conv. RF 매우 큼 (수천 sample).

장점: phase 보존, 직접 신호 처리.
단점: 매우 긴 시퀀스, 학습 비용.

### 8.5 답안 골격

> "오디오의 두 표현: raw waveform (1D 매우 긴 시퀀스) vs spectrogram (2D 시간×주파수).
>
> 변환의 가치: spectrogram으로 만들면 'CNN의 모든 기법 적용 가능'. 음성·음악의 주파수 구조가 강조됨. Phase 손실하지만 분류엔 무관.
>
> 모델: raw → 1D CNN (WaveNet), spectrogram → 2D CNN (음성 인식 표준). 긴 clip → Transformer (Whisper)."

---

## 9. 일반 원칙 정리 — Cross-Architecture 사고법

### 9.1 사고의 5단계

1. **데이터의 본래 구조** 파악 (격자? 시퀀스? 집합? 그래프? 표?).
2. **모델의 inductive bias** 파악.
3. 두 bias 매칭 — *어긋나는 부분 식별*.
4. **변환** 또는 **모델 변경** 또는 **둘 다**.
5. 변환 시 *잃는 정보*와 *얻는 능력* 명시적 비교.

### 9.2 일반 원칙

- **데이터 구조와 모델 bias가 맞으면 효율** (CNN + 이미지, RNN + 시퀀스).
- **어긋나면 비효율 또는 정보 손실** (RNN + 이미지, CNN + 그래프).
- **변환으로 우회 가능**, 단 trade-off 명시.
- **데이터 매우 많으면** prior 약한 모델 (Transformer)이 우위 — bias가 제약이 됨.

### 9.3 표 — 데이터 ↔ 모델 매칭

| 데이터 | 자연 모델 | 강제 적용 시 잃는 것 |
|---|---|---|
| 이미지 | CNN | 2D 공간 bias (RNN), 위치 정보 (FNN) |
| 텍스트 | RNN/Transformer | 순서 정보 (FNN), 장거리 (CNN) |
| 시계열 | RNN/Transformer/TCN | 시간 sharing (FNN), 가변 길이 (FNN) |
| 그래프 | GNN | 비격자 인접성 (CNN), 순서 무관 (RNN) |
| 표 | MLP/GBM/TabTransformer | (적절한 자리) |
| 동영상 | 3D CNN / 2D+LSTM | 시간 또는 공간 |
| 오디오(raw) | 1D CNN / Transformer | 시간 sharing (FNN) |
| 오디오(spec) | 2D CNN | (이미지처럼) |

---

## 10. 면접 단골 Q&A

### Q1. 이미지를 RNN으로 학습하려면?
"픽셀(또는 patch)을 raster scan 또는 Hilbert curve로 1D 시퀀스화. 각 시점에 한 픽셀을 LSTM/GRU에 입력. PixelRNN이 대표 사례 — 이미지의 자기회귀 생성. 잃는 것: CNN의 2D 공간 inductive bias (국소성, translation equivariance), 50,000 step 시퀀스의 long-range 한계, 인접성 왜곡, 병렬화 불가. 분류엔 비현실적, 자기회귀 생성에 의미. 현재는 Diffusion에 밀림."

### Q2. 텍스트를 CNN으로 풀면?
"단어 embedding 후 1D conv를 슬라이딩. 커널 폭으로 n-gram 학습. Max-over-time pooling으로 길이 무관 vector. TextCNN (Yoon Kim 2014). 얻는 것: 병렬 (RNN의 50배 빠름), n-gram 자동, 강력한 baseline. 잃는 것: 장거리 의존성 (RF 한계), 순서 정보 일부 (max pool). 짧은 텍스트 분류·감성에 매우 강력, 긴 문서·생성엔 부적절."

### Q3. 왜 그래프 데이터에 CNN을 안 쓰나?
"그래프는 격자가 아닌 임의 위상. CNN의 정해진 인접성 가정과 안 맞음. 강제 적용 (격자 매핑, 인접 행렬, random walk)은 모두 부자연 — permutation invariance 깨짐 또는 정보 손실. GNN (GCN, GAT)이 자연. 노드 표현을 이웃의 가중 평균으로 update. 노드 순서·차수 무관. 데이터 구조에 맞는 inductive bias가 따로 필요한 좋은 예."

### Q4. 시계열을 FNN으로 풀면 무엇을 잃나?
"네 가지. (1) 가변 길이 처리 불가 — N 고정. (2) 시간 축 weight sharing 없음 — 같은 패턴을 위치마다 따로 학습. (3) N 너머 정보 완전 차단. (4) N 키우면 파라미터 폭발. 단기 예측·외생 변수 dominant·baseline엔 합리. 장기·복잡 시계열엔 RNN/Transformer/TCN."

### Q5. ViT가 CNN보다 작은 데이터에서 못한 이유?
"CNN의 국소성·평행이동 inductive bias가 자연 이미지의 강한 prior. 데이터 적을 때 이 prior가 큰 도움. ViT는 prior 거의 없어 (PE만), 데이터에서 모든 것을 학습해야. 작은 데이터에선 학습이 어려움. JFT-300M 같은 매우 큰 데이터에선 prior 없이도 충분 → ViT 우위. CNN의 inductive bias는 prior이자 *제약*. 데이터 많으면 제약이 됨."

### Q6. NLP가 RNN→Transformer로 간 이유?
"세 압도적. (1) 병렬화 — RNN은 시점 t가 t-1 의존, GPU 활용 못함. Transformer는 모든 시점 동시. (2) Long-range — 한 layer에서 거리 무관 연결. RNN은 chain 통한 vanishing. (3) Scaling — 모델·데이터 키울수록 잘 따라감. RNN은 한계점 빨리 옴. 이 셋이 GPT, BERT 같은 foundation model 시대를 가능하게."

### Q7. 표 데이터에서 GBM이 딥러닝을 자주 이기는 이유?
"트리의 분할 기반 학습이 표 데이터에 강함. (1) Feature value의 임의 분할로 비선형·비단조 자연. (2) 트리 분기로 feature 상호작용 자연. (3) Scale 무관 — 정규화 불필요. (4) 이상치·결측 robust. (5) Feature 수에 잘 적응. 신경망은 inductive bias가 표 데이터에 안 맞음 — 컬럼 순서 무관, 위치 의존 없음. 매우 큰 데이터 + 강한 카테고리에서만 신경망 경쟁."

### Q8. 오디오를 spectrogram으로 변환하는 이유?
"변환의 가치 — 오디오를 *이미지처럼*. CNN의 모든 기법 적용 가능. 음성·음악의 주파수 구조가 강조됨. Phase 정보 손실하지만 분류엔 무관. 더 일반적 사고 패턴: 데이터 표현을 모델 inductive bias에 맞게 변환. Spectrogram이 그런 변환의 좋은 예."

### Q9. 동영상에 어떤 architecture?
"두 inductive bias (시간 + 공간) 처리 방법. (1) 2D CNN per frame + LSTM/Transformer — 효율, pretrained 활용. (2) 3D CNN — 시공간 동시. 비용 高. (3) (2+1)D — 효율 절충. (4) Video Transformer — 강력하지만 비싸고 데이터 많이 필요. 짧은 액션 (2+1)D 또는 3D, 긴 영상 2D + Transformer."

### Q10. Cross-architecture 설계의 일반 원칙?
"5단계. (1) 데이터의 본래 구조 파악. (2) 모델의 inductive bias 파악. (3) 두 bias 매칭, 어긋나는 부분 식별. (4) 변환 또는 모델 변경. (5) 변환 시 *잃는 정보*와 *얻는 능력* 명시. 데이터 매우 많으면 prior 약한 모델이 우위 — bias가 제약. 일반: 데이터 구조와 모델 bias가 맞으면 효율, 어긋나면 비효율."

### Q11. 이미지를 자연어처럼 처리한다는 ViT의 정신?
"이미지를 16×16 patch로 나눠 각 patch를 token처럼. Patch embedding + PE → Transformer encoder. 본질적으로 'Transformer for sequence' 그대로. 시퀀스 길이 (224/16)² = 196. CNN의 국소성 inductive bias 버림. 데이터 매우 많을 때 (JFT-300M) CNN 이김. ImageNet만이면 CNN 우위. 큰 데이터의 시대에서 prior의 trade-off를 보여주는 사례."

### Q12. RNN을 이미지·표 데이터에 *왜 안 쓰는지* 정리?
"이미지: 50,000 step의 long-range 한계 + 2D bias 없음 + 병렬 안 됨 + 인접성 왜곡. CNN의 100~1000배 비효율. 표: 컬럼 순서 임의 → 시간 순서 가정 안 맞음. 결과가 컬럼 순서에 의존. 두 경우 모두 RNN의 *시간 inductive bias*가 데이터 구조와 안 맞음. Inductive bias mismatch가 비효율의 원천."

---

## 11. 생각해보라 — 단락 답안

**Q. CNN의 inductive bias가 *제약*이 되는 데이터 규모는?**

CNN의 국소성·평행이동 가정은 *자연 이미지의 일반 성질*에 기반. 단 모든 자연 이미지가 정확히 이 가정을 만족하진 않음. 예외:
- **얼굴 영상**: 눈은 항상 위쪽, 입은 아래쪽. *위치 의존* 패턴.
- **위성 영상**: 텍스처가 위치(지리)에 강하게 의존.
- **의료 영상**: 장기의 위치가 정해짐.

이런 경우 CNN의 *위치 무관* 가정이 부적절. 단 실용적으론 CNN이 여전히 잘 작동 — 다른 데이터 augmentation으로 보완.

진짜 prior가 제약이 되는 시점: **매우 큰 데이터**. JFT-300M (3억 장) 같은 데이터에서는 ViT가 prior 없이 학습한 패턴이 CNN의 prior를 넘어섬. ImageNet (140만 장)은 그 임계 미만이라 CNN 우위.

함의: **prior의 가치는 데이터 크기에 반비례**. 데이터 작으면 강한 prior가 좋음, 매우 많으면 prior가 학습을 방해.

**Q. *모든* 데이터에 잘 작동하는 universal architecture가 가능할까?**

이론적 답은 "예" — UAT가 충분한 표현력 보장. 실용적 답은 "거의 No" — 학습 동역학이 inductive bias에 의존.

현재 추세: **Transformer가 universal에 가까이 가는 중**. 이미지 (ViT), 텍스트 (GPT), 음성 (Whisper), 코드, 단백질 (AlphaFold) 등에서 SOTA. *최소 inductive bias + 큰 데이터*의 정신.

단 한계도 있음:
- 표 데이터에서는 GBM이 여전히 우위 (대부분 case).
- 작은 데이터에서는 적절한 prior가 있는 모델이 더 유리.
- 메모리 $O(T^2)$의 부담.

미래: 단일 universal architecture보다는 *foundation model*의 정신 — 대량 데이터로 사전학습 후 다양한 task에 fine-tune. 사실상 universal에 가까움.

**Q. 강한 inductive bias가 *학습 동역학*에 어떻게 영향을 미치나?**

이론: SGD가 도달 가능한 함수 공간을 inductive bias가 좁힌다. 좁아진 공간이 *데이터의 진짜 함수를 포함*하면 학습 효율, 안 포함하면 학습 불가.

실증:
- CNN의 weight sharing → 한 위치 학습이 모든 위치에 적용 → 데이터 효율 매우 높음.
- ResNet의 skip connection → 항등 함수가 default → 깊은 학습 가능.
- Transformer의 attention → 모든 토큰이 쉽게 연결 → long-range 학습.

각 architecture의 prior가 *학습 동역학을 부드럽게* 만든다. SGD가 좋은 minima를 찾기 쉬움.

반대로 prior가 어긋나면: 학습이 안 됨 (좋은 minima가 표현 가능 공간에 없음) 또는 매우 어려움 (있지만 SGD가 도달 못함).

**Q. *Hybrid architecture*가 미래의 방향인가?**

현재 추세를 보면 yes:
- **CoAtNet, MaxViT**: 얕은 부분 CNN (강한 prior) + 깊은 부분 Transformer (강한 표현력).
- **ConvNeXt**: ResNet에 Transformer 디자인 흡수.
- **State Space Models (Mamba)**: RNN의 효율 + Transformer의 표현력.

각 architecture의 강점을 결합하려는 시도. 이론적으론 prior는 데이터 적을 때 도움, 표현력은 모든 데이터 크기에서 도움 → 둘의 적절한 결합이 sweet spot.

미래 가능성: NAS로 자동 hybrid 탐색. 매우 큰 모델은 단일 architecture (Transformer), 작은 specialized 모델은 hybrid.

**Q. *데이터 변환*이 항상 좋은 답인가?**

좋은 변환: 정보를 *더 잘 노출*. 예:
- 오디오 → spectrogram: 주파수 구조 강조.
- 이미지 → conv feature: edge·texture 추출.
- 텍스트 → embedding: 의미 공간으로.

나쁜 변환: 정보 *왜곡* 또는 *손실*.
- 그래프 → 격자: permutation invariance 깨짐.
- 이미지 → raster scan: 인접성 왜곡.

규칙: 변환은 *invertible* 또는 *task-relevant 정보 보존*해야. 변환 자체가 prior — 잘못 선택하면 모델 성능 한계.

함의: 좋은 ML 엔지니어의 일은 *모델 선택* + *데이터 표현 선택*. 둘 다 중요.

**Q. *State Space Model (SSM)*이 RNN의 부활인가?**

Mamba (2024) 같은 SSM이 화제. 본질:
- Recurrent state (RNN처럼).
- Linear in length (Transformer의 $O(T^2)$ 회피).
- 병렬 학습 가능 (특수 알고리즘).

장점:
- 매우 긴 시퀀스에 효율 (수만 토큰).
- Transformer 비슷한 성능.
- 메모리 매우 적음.

이게 의미하는 것:
- "*RNN은 죽었다*"의 종말. RNN 식 정신이 새 형태로 부활.
- Transformer의 quadratic 한계가 진짜 문제이고 새 답이 필요했음.
- 미래는 Transformer만이 아닐 가능성.

단 SSM도 한계 있음 — *inductive bias가 약함* (RNN보다도). 작은 데이터에선 약함. 큰 모델에서만 빛남.

**Q. Cross-architecture 설계가 *연구의 영역*인가, *실무의 영역*인가?**

둘 다.

**연구**: 새로운 architecture는 cross-architecture 사고에서 종종 등장. ViT는 "Transformer for image", PixelRNN은 "RNN for image". 연구자가 새 매칭을 시도하면서 분야 발전.

**실무**: 주어진 task에 어떤 architecture가 가장 효율적인지 결정. 표 데이터엔 GBM (cross 아님), 이미지엔 CNN (자연), 시퀀스엔 RNN/Transformer (자연). 보통 자연 매칭이 답이지만, 특수 case에서 cross가 우위일 때 시도.

면접·시험: cross-architecture 사고를 통해 *각 모델의 inductive bias*를 깊이 이해하는지 평가. 답안의 깊이가 사고의 깊이를 보여줌.

---

## 12. 한 줄 요약 (시험 직전)

- **Cross-architecture 설계** = inductive bias 매칭 사고.
- **이미지를 RNN으로**: raster/Hilbert로 시퀀스화. 2D bias 잃음, 50K step long-range 한계, 병렬 불가. PixelRNN이 사례. 분류엔 비현실적.
- **텍스트를 CNN으로**: TextCNN, n-gram 자동 학습. 병렬 빠름. 짧은 분류 강력. 긴 문서·생성엔 약함.
- **시계열을 FNN으로**: sliding window. 가변 길이 불가, 시간 sharing 없음, N 너머 차단.
- **시퀀스를 Transformer로**: 병렬 + long-range + scaling. 단 $O(T^2)$, 데이터 多.
- **그래프를 CNN으로**: 부자연. GNN이 답.
- **표를 신경망으로**: GBM이 강자. TabTransformer가 신경망의 답.
- **오디오 → spectrogram → CNN**: 변환의 가치.
- **일반 원칙**: 데이터 구조와 모델 bias 매칭. 어긋나면 비효율 또는 변환.
- **Prior의 가치는 데이터 크기에 반비례**. 작으면 강한 prior 좋음, 매우 많으면 제약.
- **답안 5단계**: 변환 방법 → 잃는 것 → 얻는 것 → 언제 합리적 → 결론.
