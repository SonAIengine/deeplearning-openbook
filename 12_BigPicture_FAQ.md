# 12. Big Picture & FAQ — "전체 흐름 + 시험 직전" 심층

> **이 문서의 목표**:
> 모든 챕터를 *하나의 큰 그림*으로 묶고, 시험·면접의 단골 Q&A를 풍부한 답안으로 정리. 시험 시작 5분 전, A4 1장만 본다면 이거.
>
> **세 부분**:
> 1. 딥러닝 발전사 — 각 기술이 *직전 한계의 답*으로 등장한 흐름
> 2. 단골 Q&A — 사고의 흐름까지 답안에 담음
> 3. 시험 직전 압축 요약

---

## 0. 큰 그림의 핵심 사고

### 0.1 딥러닝 = 한계 극복의 누적

각 기술은 *직전의 한계의 답*으로 등장. 그 흐름을 외우면:
- 각 기술의 *동기*를 답할 수 있음.
- 각 기술의 *한계*도 자연스레 떠오름 (다음 기술의 동기).
- 면접의 "왜 이 기술?" 질문에 *진화적 답*.

### 0.2 핵심 질문 패턴

좋은 답안은 항상:
1. **무엇을 묻나?** (정의? 메커니즘? 비교? 설계?)
2. **구조적 답변** — 어떻게 작동? 어떤 식?
3. **왜 그렇게?** — 메커니즘적 이유.
4. **다른 옵션과 비교** — 이거 대신 저거면?
5. **실용적 함의** — 언제 쓰고 언제 안 쓰나.

이 5단계가 좋은 답의 골격. 정의만 답하지 말 것.

---

## 1. 딥러닝 발전사 — 한계 극복의 흐름

### 1.1 1957~1969: Perceptron

**기여**:
- Rosenblatt의 perceptron — 데이터로 학습하는 첫 model.
- "전자 두뇌"로 보도.

**한계**:
- 선형 분리 가능한 문제만.
- XOR 못 풀음 (Minsky-Papert 1969).

**다음 동기**: "비선형으로 어떻게?"

### 1.2 1986: Backpropagation + MLP

**기여**:
- Rumelhart-Hinton-Williams의 backprop.
- Chain rule로 hidden layer의 credit assignment 풀음.
- MLP가 XOR 등 비선형 문제 해결.
- Universal Approximation Theorem (1989, Cybenko) — 표현력 보장.

**한계**:
- 깊으면 vanishing gradient (sigmoid).
- 데이터·연산 부족 → 작은 모델만.
- 1990s~2000s, SVM과 random forest에 밀림.

**다음 동기**: "깊은 망을 학습 가능하게?"

### 1.3 1998: LeNet (CNN의 시작)

**기여**:
- LeCun의 CNN — 우편번호 인식.
- Conv + Pool + FC의 패턴 정립.
- Weight sharing의 첫 실용.

**한계**:
- 작은 task에 한정.
- 데이터·연산 부족.

**다음 동기**: "더 큰 task로 확장?"

### 1.4 2006: Pre-training (Hinton)

**기여**:
- Hinton의 layer-wise unsupervised pretraining.
- Deep Belief Network.
- "Deep learning"이라는 용어 부활.

**한계**:
- 여전히 큰 모델 학습 불안정.
- 단 deep의 *가능성* 증명.

### 1.5 2012: AlexNet — 딥러닝의 폭발

**기여**:
- Krizhevsky-Sutskever-Hinton.
- ImageNet top-5: 26% → 16%.
- ReLU + Dropout + GPU + Data augmentation.
- "Deep + Data + Compute"의 첫 압도적 증명.

**한계**:
- 깊이 8층, 더 깊으면 학습 어려움.

**다음 동기**: "더 깊게 가려면?"

### 1.6 2014: VGG, GoogLeNet — 깊이 탐험

**VGG** (Oxford):
- 3×3 conv 깊이 19층까지.
- 단순함의 미.

**GoogLeNet (Inception)**:
- 다양한 RF 동시 (1×1, 3×3, 5×5 병렬).
- 1×1 conv로 비용 절감.
- Auxiliary classifier (vanishing 완화).

**한계**:
- VGG: 메모리 부담.
- 둘 다 깊이 50+ 어려움 (degradation problem).

**다음 동기**: "100층, 1000층?"

### 1.7 2015: ResNet — 깊이의 혁명

**기여**:
- He et al.의 skip connection.
- $h_{l+1} = h_l + F(h_l)$.
- 152층 학습 가능. ImageNet top-5 4.5% (사람 수준).
- Vanishing gradient 우회 — 덧셈 경로.
- 후속 모든 architecture의 핵심 부품.

**의의**:
- 깊이의 *근본적 한계 깨뜨림*.
- 그 이후 CNN의 표준.

### 1.8 1997: LSTM (시간 차원의 vanishing 답)

**기여**:
- Hochreiter-Schmidhuber.
- Cell state + 3 gate.
- RNN의 vanishing 완화 — 덧셈 update.

**역사적 위치**:
- 1997 발표지만 2010년대 초까지 비주류.
- Speech recognition (2013), translation (2014)에서 부흥.

**한계**:
- 100~200 step이 한계.
- Sequential 처리 → 병렬 못함.

**다음 동기**: "Long-range + 병렬?"

### 1.9 2014: GRU (LSTM 단순화)

**기여**:
- Cho et al.의 GRU.
- 2 gate + hidden 단일.
- LSTM과 거의 동급, 더 단순.

### 1.10 2014: Attention (RNN의 정보 병목 답)

**기여**:
- Bahdanau-Cho-Bengio의 attention for translation.
- Encoder-decoder의 단일 context vector 한계 해결.
- 매 decoder 시점에 encoder의 모든 시점 동적 결합.

**의의**:
- Long-range를 chain 없이 직접.
- 해석 가능 (attention 가중치).

**다음**: "Attention만으로 모든 것?"

### 1.11 2017: Transformer — NLP의 패러다임 전환

**기여**:
- Vaswani et al. "Attention is All You Need".
- RNN 완전 제거. Self-attention만.
- 병렬 + long-range + scaling.

**의의**:
- NLP 표준이 RNN → Transformer로.
- 큰 모델 시대 가능.

### 1.12 2018~: BERT, GPT — Foundation Model

**BERT** (Google):
- Bidirectional Transformer encoder.
- Masked language modeling.
- Pretrain + fine-tune의 표준.

**GPT** (OpenAI):
- Decoder-only Transformer.
- Autoregressive language modeling.
- GPT-3 (175B), GPT-4 — 매우 큰 모델 시대.

**의의**:
- Foundation model 시대 — 매우 큰 사전학습 후 다양한 task fine-tune.
- Zero-shot, few-shot의 부상.
- ML이 "데이터·모델·연산의 scaling"으로.

### 1.13 2020~: Vision Transformer, Multimodal

**ViT** (Google):
- 이미지를 patch sequence로 → Transformer encoder.
- 매우 큰 데이터에서 CNN 추월.

**CLIP, DALL-E, GPT-4V**:
- 이미지 + 텍스트 multimodal.
- Foundation model의 일반화.

### 1.14 2022~: Diffusion, RLHF

**Diffusion**:
- 이미지 생성의 새 표준.
- Stable Diffusion, DALL-E 2.

**RLHF (Reinforcement Learning from Human Feedback)**:
- ChatGPT의 핵심 기법.
- 인간 선호로 모델 fine-tune.

### 1.15 한 표로 — 전체 발전사

| 시대 | 모델 | 해결 | 새 한계 |
|---|---|---|---|
| 1958 | Perceptron | 선형 분리 | XOR |
| 1986 | MLP + Backprop | XOR | Vanishing |
| 1998 | LeNet | 이미지 inductive bias | 작은 규모 |
| 2012 | AlexNet | Deep + Data + Compute | Depth degradation |
| 2014 | VGG | 작은 커널 깊게 | 더 깊으면 어려움 |
| 2014 | GoogLeNet | 다양한 RF + 1×1 | depth 한계 |
| 2015 | ResNet | Skip connection | Long-range (text) |
| 1997 | LSTM | RNN의 vanishing | 병렬 불가 |
| 2014 | GRU | LSTM 단순화 | 여전한 long-range |
| 2014 | Attention | RNN의 long-range | RNN과 결합 시 순차 |
| 2017 | Transformer | 완전 병렬 + long-range | 데이터·연산 폭증 |
| 2018~ | BERT, GPT | Pretrain + scaling | 자원, hallucination |
| 2020~ | ViT, CLIP, DALL-E | Multimodal foundation | 더 큰 scale |
| 2022~ | Diffusion, RLHF | 생성·alignment | 안전성, calibration |

---

## 2. 단골 Q&A — 풍부한 답안

### A. 기초

**Q1. 머신러닝과 딥러닝의 차이?**

"ML은 데이터로부터 함수를 학습하는 일반 개념. 딥러닝은 다층 신경망 기반 ML로, 핵심 차별점은 **representation learning** — feature를 사람이 만들지 않고 데이터에서 자동 학습. 전통 ML은 feature engineering이 주요 일이었으나, 딥러닝은 raw input에서 feature까지 end-to-end. 또 딥러닝은 데이터·연산이 충분할 때 전통 ML을 압도, 부족할 때는 GBM 등이 더 유리."

**Q2. Bias-Variance Trade-off의 의미와 딥러닝에서의 재해석?**

"고전: 단순 모델은 bias 高/variance 低 (underfit), 복잡 모델은 bias 低/variance 高 (overfit). 둘의 합 최소화가 sweet spot. 딥러닝 시대 *double descent* 현상 — 모델 크기를 데이터 크기 너머로 키우면 다시 좋아짐. SGD의 implicit bias가 부드러운 함수 선호 → 일반화. 그래서 over-parameterized + 정규화가 현대 표준."

**Q3. Validation과 Test의 차이?**

"Val은 hyperparameter 선택용 (early stopping, 모델 선택). Test는 *최종 평가용 단 한 번*. 만약 test로 모델 반복 선택하면 그 test가 사실상 val이 되어 실세계 성능 추정 부풀림. 그래서 test는 *마지막 보루*로 격리. 데이터 매우 적으면 cross-validation, nested CV로 보완."

**Q4. ML과 통계학의 관계?**

"통계학은 모집단의 성질 추정 (가설 검정, 신뢰구간), ML은 새 입력에 대한 예측 성능. 도구·토대 겹치지만 묻는 질문 다름. 통계학자는 추정량 일치성·신뢰구간 따짐, ML 엔지니어는 hold-out test 성능과 production 동작. ML은 통계학의 도구를 빌려와 *예측을 잘 하는 함수*를 만드는 데 초점."

### B. FNN

**Q5. 비선형 활성화가 왜 본질적인가?**

"전체 망이 단일 선형 변환으로 환원 안 되도록 만드는 핵심. 비선형 없으면 $W_2(W_1 x + b_1) + b_2 = W'x + b'$ — 모든 layer가 합쳐져 logistic regression 수준. 100층 1000층 쌓아도 표현력 같음. XOR도 못 풀음. UAT의 본질도 비선형에 있음. 비선형이 신경망의 표현력의 원천."

**Q6. ReLU가 표준이 된 이유?**

"세 측면. (1) Vanishing — 활성된 영역에서 미분 1, sigmoid의 0.25 제약 없음. (2) 계산 — max만, exp 없음 → 빠름. (3) Sparse — 절반이 0 출력 → sparse representation. 단점은 dying ReLU지만 LeakyReLU나 He init으로 완화. 그래서 hidden layer는 ReLU가 거의 default. Transformer에선 GELU (부드러운 ReLU) 표준."

**Q7. Backprop의 본질?**

"Chain rule을 출력에서 입력으로 적용해 gradient를 재사용. Forward 1번 + backward 1번에 모든 가중치 gradient. Numerical은 가중치마다 forward 1번 → 1M 가중치면 1M배 느림. 자동미분의 reverse mode 특수 case로, 출력이 scalar(loss)인 신경망에 매우 적합. 4개 핵심 식으로 구성 (출력 delta, 역전파, bias gradient, weight gradient)."

**Q8. Universal Approximation Theorem이 약속하는 것과 안 하는 것?**

"표현력 보장 정리. 1 hidden layer + 충분한 뉴런 + 비선형이면 임의 연속함수를 임의 정확도로 근사 가능. 단 약속 안 하는 게 많음. (1) 학습 가능성 — 그런 가중치가 *존재*함과 SGD로 *찾는 것*은 별개. (2) 효율성 — 1 layer로 표현하려면 뉴런이 지수적일 수 있음. (3) 일반화 — train fit과 test 성능은 다름. 그래서 실용적으론 deep이 압도적 우위."

### C. DNN

**Q9. Vanishing Gradient의 본질?**

"Backprop의 chain rule이 매 layer마다 $W \cdot \sigma'$의 곱셈을 누적해서 발생. 곱이 < 1이면 지수적 감쇠. Sigmoid는 $\sigma' \le 0.25$ → 10층에서 $10^{-6}$. ReLU는 활성 영역에서 미분 1 → vanishing 약함. 해결의 다섯 축: 활성화(ReLU) + Init(He) + 정규화(BN) + 구조(Skip) + Optimizer(Adam). 현대 deep learning은 이 다섯 모두 기본 stack."

**Q10. BN이 왜 효과 있는가?**

"세 해석. (1) 원래 동기 — internal covariate shift 완화, 매 layer 입력 분포 안정. (2) 후속 분석 — loss landscape 평활화, 큰 LR 사용 가능. (3) 부수 효과 — mini-batch 통계의 noise가 약한 정규화. 또 init에 덜 민감해짐. 단점은 작은 batch에서 부정확, 시퀀스에 불편 → GroupNorm/LayerNorm으로 보완."

**Q11. Dropout의 ensemble 해석?**

"매 step 다른 sub-network를 학습. n개 뉴런이면 가중치 공유된 $2^n$개 sub-network 동시 학습. 평가 시 모든 뉴런 사용 + 활성값 스케일 = 사실상 평균 — exponentially many networks의 ensemble 근사. 또 co-adaptation 방지 효과. Bayesian 관점으론 variational inference."

**Q12. ResNet의 skip connection이 왜 혁명?**

"두 측면. (1) Identity가 default — F가 0을 학습하면 자동 항등. 깊이 추가가 최소한 더 나빠지지 않음 (degradation 해결). (2) Gradient 직접 흐름 — $\partial h_{l+1}/\partial h_l = I + \partial F$. Identity 항으로 gradient가 덧셈 경로. 곱셈 누적 없음 → vanishing 약함. 100~1000층 학습 가능. 후속 모든 architecture의 핵심 부품. LSTM의 cell state와 본질 같음."

**Q13. Adam이 SGD보다 항상 좋은가?**

"아님. Adam은 빠른 수렴 + LR 튜닝 부담 적음 → 일반 default. ImageNet 같은 task에서 SGD+momentum이 약 1% 더 일반화 (Adam이 sharp minima 경향). AdamW가 weight decay 정확 처리해서 Transformer 시대 표준. 결론: Transformer면 AdamW + warmup + cosine, ImageNet이면 SGD+momentum, 작은 데이터면 Adam (안전), 큰 batch (4096+)는 LARS/LAMB."

**Q14. Warmup이 필요한 이유?**

"학습 초기 통계 noise. (1) BN의 running 평균·분산이 부정확. (2) Adam의 second moment 추정이 noisy (bias correction 있어도 완벽 안 됨). (3) 큰 LR로 시작 시 이 부정확한 통계 때문에 발산 위험. 작게 시작 → 통계 안정 시간 제공 → 점진 증가. 큰 모델 + Adam 조합에 사실상 필수."

### D. CNN

**Q15. 이미지에 FNN을 안 쓰는 이유?**

"세 가지. (1) 파라미터 폭발 — 224×224×3 입력에 1000 hidden 첫 FC layer만 1.5억 파라미터. 메모리·연산 폭발. (2) 위치 정보 미활용 — 같은 패턴이 다른 위치에 있어도 별도 학습. (3) 자연 이미지의 계층 구조 활용 못함. CNN의 weight sharing + local connectivity + 계층적 깊이가 이 셋을 동시에 해결."

**Q16. Weight sharing이 왜 좋은가?**

"세 가지. (1) 파라미터 효율 — 위치별 가중치 대신 한 커널을 모든 위치에. 1만 배 이상 적은 파라미터. (2) Translation equivariance — 같은 패턴이 어디 있든 같은 응답. (3) 데이터 효율 — 한 위치 학습이 모든 위치 자동 적용. 사실상 데이터 augmentation 효과. 자연 이미지의 inductive bias와 정확히 매칭."

**Q17. Equivariance와 Invariance의 구분?**

"Equivariance: $f(T(x)) = T(f(x))$. 입력 이동에 출력도 같이 이동. Conv 연산이 만족. Invariance: $f(T(x)) = f(x)$. 입력 변화에 출력 불변. Pooling, GAP가 부여. CNN의 강력한 점: conv로 equivariance 유지하다가 마지막에 GAP로 invariance 부여 — 분류·detection·segmentation 모두 가능."

**Q18. 1×1 conv는 무엇을 하나?**

"공간 정보 그대로 두고 채널 차원만 변환. 사실상 각 픽셀 위치의 fully connected layer (위치 간 weight sharing). 용도: (1) 채널 차원 조절, (2) Bottleneck (ResNet 50+층 표준 — 1×1로 압축, 3×3 작업, 1×1로 복구), (3) Inception에서 다양한 RF 합칠 때 비용 절감, (4) 채널 간 비선형 결합."

**Q19. 3×3 conv를 깊이 쌓는 이유?**

"VGG의 통찰. 7×7 conv 1번과 3×3 conv 3번이 같은 RF (7×7). 파라미터: 49 vs 27 — 45% 적음. 비선형 활성화 횟수: 1 vs 3 — 표현력 ↑. 같은 RF면서 더 적은 파라미터로 더 풍부한 표현. 그래서 modern CNN은 거의 모두 3×3."

**Q20. Transfer Learning이 왜 작동?**

"얕은 layer는 도메인 무관한 일반 feature (edge, texture, color), 깊은 layer는 task-specific. ImageNet 학습 모델의 얕은 layer는 다른 task에도 유용 → 그 부분 freeze + 깊은 layer만 fine-tune. 데이터 적을 때 매우 효과적. 단 도메인 차이 큰 경우 깊은 layer도 다시 학습."

### E. RNN/LSTM

**Q21. RNN이 arbitrary length를 어떻게 처리?**

"시간 축 weight sharing 덕분. 같은 cell을 시간에 따라 반복 적용. 시퀀스 길이가 5이든 5000이든 같은 cell을 그만큼 반복. Hidden state가 과거 정보의 고정 크기 요약 — 가변 길이 정보를 고정 차원에 압축. 모델 가중치는 길이에 독립이고 forward pass의 반복 횟수만 길이에 의존."

**Q22. RNN에서 vanishing이 더 심한 이유?**

"같은 가중치 행렬 $W_{hh}$가 시간 축에 거듭제곱되기 때문. FNN은 매 layer마다 다른 $W$의 곱이라 우연히 균형 가능. RNN은 같은 행렬의 거듭제곱 — 고유값 < 1이면 즉시 vanishing, > 1이면 즉시 exploding. 한쪽 방향 폭주 또는 소멸이 거의 보장. 처방: gradient clipping (exploding), LSTM/GRU의 gating (vanishing), attention (장거리 직접)."

**Q23. LSTM이 vanishing을 어떻게 완화?**

"Cell state의 덧셈 update가 핵심. $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. 미분 $\partial c_t/\partial c_{t-1} = f_t$ — 행렬 곱이 아닌 element-wise scalar 곱. RNN의 행렬 거듭제곱 대신 forget gate만 곱해지므로, gate가 1 근처를 학습하면 gradient가 시간 축에 거의 그대로 흐름. ResNet skip connection과 본질 같은 정신 — 곱셈 누적을 덧셈 경로로 우회."

**Q24. LSTM의 각 gate 역할?**

"Forget: 과거 cell 중 무엇을 버릴지. Input + candidate: 새 정보 중 무엇을 추가할지. Output: cell의 어느 부분을 외부에 노출할지. Forget이 가장 중요 — 없으면 정보 포화. Input은 선택적 기록. Output은 task-relevant 정보만 노출. 세 gate가 분리되어 *상황별 다른 동작* 가능."

**Q25. LSTM vs GRU 어느 쪽?**

"거의 동급. LSTM은 3 gate + cell/hidden 분리, GRU는 2 gate + hidden 단일. 파라미터 GRU가 ~25% 적음, 학습 약간 빠름. Task별 차이 — 데이터 적고 빠라야 하면 GRU, 매우 긴 의존성 + 큰 데이터면 LSTM 약간 우위. 실무에선 둘 다 시도해 val로 결정."

**Q26. 이미지를 RNN으로 학습하려면? (★)**

"픽셀(또는 patch)을 raster scan 또는 Hilbert curve로 1D 시퀀스화. 각 시점에 한 픽셀(또는 patch)을 LSTM/GRU에 입력. PixelRNN이 대표 사례. 잃는 것: CNN의 2D 공간 inductive bias (국소성, translation equivariance), 50,000 step 시퀀스의 long-range 한계 (LSTM도 100~200 step 정도가 한계), 인접성 왜곡 (raster scan은 공간 인접 ≠ 시퀀스 인접), 병렬화 불가. 분류엔 비현실적 — CNN의 100~1000배 비효율. 자기회귀 생성에 의미 있었으나 현재는 Diffusion 우위."

### F. Transformer

**Q27. Transformer가 RNN을 대체한 이유?**

"세 압도적. (1) 완전 병렬화 — RNN은 시점 t가 t-1 의존, GPU 활용 못함. Transformer는 모든 시점 동시. (2) Long-range — 한 layer에서 거리 무관 직접 연결. RNN의 chain 통한 vanishing 우회. (3) Scaling — 모델·데이터 키울수록 성능 멱법칙으로 향상. RNN은 한계점 빨리 옴. 이 셋이 GPT-3, BERT 같은 foundation model 시대를 가능하게."

**Q28. Self-attention의 핵심?**

"모든 토큰이 모든 토큰을 동시에 본다. Query·Key·Value 셋의 dot product로 attention weight, softmax 정규화. 그 weight로 Value의 가중합. 거리 무관 정보 전달 + 병렬. Multi-head로 다양한 관계 동시 학습. Transformer block = MHA + FFN + skip + LN."

**Q29. Positional Encoding이 왜 필요?**

"Self-attention은 permutation invariant — 토큰 순서 무관. 시퀀스 task에는 치명. PE가 위치 정보를 명시적 주입. 종류: Sinusoidal (원조), Learned (BERT), Relative (T5), RoPE (LLaMA), ALiBi (extrapolation 좋음). PE 없으면 Transformer는 bag-of-tokens — 시퀀스 task 거의 모두 망함."

### G. 일반 사고

**Q30. "이거 빼면 어떻게 되나"의 위력?**

"모든 컴포넌트의 가치는 그것을 *제거했을 때* 드러남. 비선형 빼면 선형 환원. BN 빼면 큰 LR 못 씀. Skip 빼면 30+층 학습 불가. Forget gate 빼면 cell 폭발. 시간 sharing 빼면 RNN 본질 상실. 이 사고가 컴포넌트의 본질을 명확히. 면접·시험에서 매우 강력."

**Q31. ML 파이프라인의 핵심 단계?**

"문제 정의 → 데이터 수집·정제 → EDA → 전처리 → 데이터 분할 → Baseline → 모델 개발 → Hyperparameter 튜닝 → 최종 평가 → 배포 → 모니터링 → 재학습. ML 엔지니어의 일은 모델 선택보다 *데이터 split·평가 metric·함정 인식·배포·모니터링*의 의식적 관리. 함정: leakage, drift, 불균형, BN/Dropout mode, hyperparameter overfit."

**Q32. ML 모델이 production에서 망하는 5가지?**

"(1) Data leakage — 비현실적 val 성능. (2) Distribution shift — train·deploy 분포 다름. (3) Class imbalance — 다수 클래스만 학습. (4) Concept drift — 시간 따라 패턴 변화 (사기 새 수법 등). (5) Feedback loop — 모델 결정이 데이터 생성에 영향. 의식적으로 모니터링·점검·재학습."

---

## 3. 보너스 12선 — 응용·면접

**B1. ImageNet pretrained ResNet을 의료 영상에?**

"Transfer learning. 마지막 FC 교체 (클래스 수 변경) + 작은 LR로 fine-tune. 데이터 매우 적으면 (< 1k) 마지막만, 적당하면 깊은 layer까지. 도메인 차이 크면 self-supervised pretrain (의료 이미지로) + supervised fine-tune. Patient-level split 필수. Augmentation은 의료에 적절한 것만 (회전 작게, 좌우 flip은 해부학 점검)."

**B2. 텍스트 감성 분류 가장 빠른 prototype?**

"단계별. (1) Baseline: TF-IDF + Logistic Regression — 5분에 학습. (2) 약간 깊게: TextCNN — 30분. (3) SOTA: BERT/DistilBERT fine-tune — 1~3시간. 첫 두 baseline이 충분하면 거기서 멈춤. SOTA 추구하면 BERT. Class weight 또는 focal loss로 불균형 처리."

**B3. 시계열 단기 예측 안 되면?**

"디버깅 순서. (1) 데이터 정규화 점검 — train만으로 통계 fit. (2) Lag feature 추가 — 1일, 7일, 30일 전. (3) 외생 변수 (요일, 휴일, 기온) 포함. (4) Baseline: naive (어제 값), MA, ARIMA 비교 — 복잡 모델이 baseline 못 이기면 잘못. (5) 모델 선택: LSTM/TCN/GBM 비교. (6) Forward chaining CV로 평가."

**B4. 클래스 불균형 (양성 1%) 분류?**

"5축 처방. (1) Class weight 또는 focal loss. (2) Oversampling (SMOTE) 또는 undersampling. (3) Threshold 비즈니스 비용에 맞게. (4) Metric은 PR-AUC, F1, recall (accuracy 무의미). (5) 모델 자체는 GBM이 강자 (트리는 불균형에 robust). 의료·사기 등 분야별 정답 다름."

**B5. GAN 학습이 자주 collapse?**

"Mode collapse — generator가 같은 종류만 생성. 원인: G와 D의 균형 무너짐. 처방: (1) Spectral normalization (D의 Lipschitz 제약). (2) Gradient penalty (WGAN-GP). (3) Two time-scale update rule. (4) 다양성 loss 추가. (5) Diffusion으로 전환 — 더 안정적 생성. 최근엔 Diffusion이 GAN 대체."

**B6. 추천 시스템 cold start?**

"신규 사용자/아이템에 데이터 없음. (1) Content-based — feature (장르, 태그)로 시작. (2) Demographic — 나이, 성별 등. (3) Hybrid — content + collaborative. (4) Meta-learning — 적은 데이터로 빠른 적응. (5) Active learning — 신규 사용자에게 좋은 것 추천하며 학습. (6) Foundation model — 텍스트 representation 활용."

**B7. 이상 탐지에 supervised vs unsupervised?**

"라벨 충분 + 알려진 패턴 → supervised (focal loss + class weight). 라벨 적음 또는 새 패턴 → unsupervised (Autoencoder reconstruction error, Isolation Forest). 실무에선 hybrid — 알려진 패턴은 supervised, 새 패턴은 unsupervised로 detect 후 라벨링. 사기 탐지의 표준 패턴."

**B8. 모델이 deploy 후 성능 하락?**

"Distribution shift 의심. 종류: covariate (입력), label (라벨 분포), concept (P(Y|X) 변경), selection bias. 처방: (1) 입력 분포 통계 추적 (KL divergence). (2) 예측 confidence 분포. (3) 정기 재학습 (월/분기). (4) Drift detector (ML 기반). (5) Domain adaptation. (6) Online learning."

**B9. Knowledge distillation의 핵심?**

"큰 teacher 모델의 soft label로 작은 student 학습. Soft label에는 'dark knowledge' — 클래스 간 미묘한 관계. 학습: teacher logit / T → softmax (temperature T로 부드럽게) → student의 KL divergence. Hard label CE도 함께. Effects: 모델 크기 1/10, 정확도 거의 같음. Production deployment 표준 기법."

**B10. Self-supervised learning이 왜 중요?**

"라벨 없는 대량 데이터로 표현 학습. Pretext task (BERT의 masked LM, SimCLR의 contrastive 등)로 의미 있는 representation. 그 후 supervised fine-tune이 매우 적은 라벨로도 성능. Foundation model 시대의 토대. 이미지 (MAE, DINO), 텍스트 (BERT, GPT), 음성 (Wav2Vec)."

**B11. Attention의 O(n²)을 줄이는 방법?**

"여러 종류. (1) Sparse attention — 각 토큰이 일부 토큰에만 (Longformer, BigBird). (2) Linear attention — softmax 근사로 O(n). (3) FlashAttention — IO-aware 최적화, 메모리 ↓ 같은 정확도. (4) Sliding window attention — 국소만. (5) State Space Model (Mamba) — RNN 식 recurrent state. 매우 긴 시퀀스에 critical."

**B12. Mixed precision training?**

"fp16/bf16 활용으로 속도 2~8배. 단 fp16의 dynamic range 좁음 → underflow/overflow 위험. 처방: (1) Loss scaling — loss 큰 수로 곱한 후 backward. (2) Master weights는 fp32. (3) 일부 연산 (LayerNorm, softmax)은 fp32. PyTorch의 `torch.cuda.amp.autocast`가 자동. 큰 모델 학습에 사실상 필수."

---

## 4. 한 화면 빅 픽처 (압축)

```
                   ┌── Vanishing → ReLU + He + BN + Skip
       FNN → DNN ──┤
                   └── Overfit → Dropout + WD + Aug

                ┌── 위치 정보 → CNN의 sharing
       FNN ─────┤
                └── 시퀀스 → RNN의 시간 sharing

                ┌── Vanishing → LSTM/GRU의 gating
       RNN ─────┤
                └── Long-range + 병렬 → Attention → Transformer

                ┌── PE → 위치 정보
   Transformer ─┤── Skip + LN → 깊은 학습
                └── Scaling → Foundation Model
```

각 화살표 = "왜 이게 등장?" = 면접·시험 단골 답.

---

## 5. A4 1장 — 시험 직전 5분

### 핵심 공식
- **GD update**: $\theta \leftarrow \theta - \eta \nabla L$
- **CE+softmax gradient**: $\hat{y} - y$ (★깔끔)
- **Conv output size**: $\lfloor (W + 2p - k)/s \rfloor + 1$
- **LSTM cell update**: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$
- **Self-attention**: $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k}) V$

### 짝짓기 (외워두면 답 자동)
- **ReLU + He init**
- **Sigmoid + BCE**
- **Softmax + CE**
- **BN** (CNN, big batch), **LN** (Transformer/RNN), **GN** (small batch)
- **AdamW + warmup + cosine** (Transformer)
- **SGD + momentum** (ImageNet CNN, 일반화 우위)

### 단골 답안 골격
- "**왜 X 빼면 안 되나**?" = ablation으로 답 — 어떤 현상이 발생, 왜.
- "**왜 sharing**?" = 파라미터 + equivariance + 데이터 효율.
- "**Arbitrary length 어떻게**?" = 시간 축 sharing.
- "**LSTM 왜 vanishing 완화**?" = forget gate scalar 곱, 덧셈 경로.
- "**왜 Transformer가 RNN 대체**?" = 병렬 + long-range + scaling.
- "**이미지를 RNN으로**?" = inductive bias 매칭 + 잃는 것 + 얻는 것.

### 실수 방지
- BN/Dropout `model.eval()` 항상.
- Output-Loss 매칭 점검.
- 데이터 정규화 train만으로 fit.
- Group/time leakage 점검.
- Gradient clipping (RNN, Transformer).
- LR > Batch > Optimizer > Init 순으로 민감.
- **Tiny dataset overfit**으로 sanity check.

### 5단계 답변 흐름
1. 무엇을 묻나? (정의? 메커니즘? 비교? 설계?)
2. 구조적 답변 (어떻게? 어떤 식?)
3. 왜 그렇게? (메커니즘적 이유)
4. 다른 옵션 비교 (이거 대신 저거면?)
5. 실용 함의 (언제 쓰고 언제 안 쓰나)

---

## 6. 마지막 사고 — "교수님처럼 답하기"

질문이 오면 항상:

1. **무엇을 묻나** 식별. 정의를 묻는지 메커니즘을 묻는지 비교를 묻는지.
2. **구조 답변**으로 시작. "X는 Y라는 식으로 작동합니다."
3. **왜** 한 번 더. "이유는 Z 때문입니다."
4. **비교** 한 번 더. "A와 비교하면 B 상황에서 우위, C 상황에선 D입니다."
5. **실용** 마무리. "그래서 [내 task에선 X/Y]를 선택합니다."

**정의를 묻는 게 아니라 사고의 흐름**을 보고 싶어한다.

좋은 답: "X는 Y의 한계를 풀기 위해 Z 메커니즘을 사용하는데, A와 비교하면 B 상황에서 우위, C 상황에선 D"

나쁜 답: "X는 Y입니다."

---

## 7. 한 줄 요약 (최종)

- 딥러닝 = **각 시대의 한계를 극복하는 기술의 누적**.
- 모든 답은 "**왜 이 구조? + 빼면 어떻게? + 다른 옵션과 비교?**"로.
- 정의 외우기보다 **사고의 흐름**.
- "이미지를 RNN으로?"처럼 cross-architecture 질문엔 **inductive bias 매칭** + 잃는 것 + 얻는 것.
- **5단계 답변**: 무엇을 묻나 → 구조 → 왜 → 비교 → 실용.
- **실수 방지**: model.eval(), output-loss 매칭, leakage 점검, gradient clipping, tiny overfit sanity check.
- 시험 직전엔 본 챕터의 빅 픽처 표 + Q&A 30개 + A4 1장만으로 충분.
