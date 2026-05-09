# LR schedule — Step / Cosine / Warmup / OneCycle / Cyclic

> 본문 §3.8 LR schedule의 보강. 학습 진행에 따라 LR을 어떻게 바꾸는가. 본문에서는 한 줄 요약만 두고 여기서 깊게.

---

## 왜 LR을 바꾸는가

고정 LR은 종종 sub-optimal. 학습 초반엔 큰 LR로 빨리 움직이고, 후반엔 작은 LR로 정밀하게 — 이게 LR schedule의 정신.

---

## Step decay

- 표준 설정: 전체 epoch의 *50%, 75% 지점*에서 각각 LR을 1/10로. 또는 ImageNet 관행으로 epoch 30/60/90.
- *factor (감쇠 비율)*: 0.1이 흔함. 0.5이면 완만한 감쇠.
- 단순하고 실증적으로 잘 동작. 큰 모델·복잡한 schedule이 필요한 곳이 아니면 default로 충분.

---

## Cosine annealing

- 표준 설정: 전체 epoch을 한 cycle로, $\eta$가 max에서 0으로 코사인 곡선을 따라 부드럽게 감소.
- min LR을 0이 아니라 max의 1% 정도(eta_min=1e-5)로 두는 게 안전 — 완전 0이 되면 가중치가 멈춤.
- Transformer 표준. *Warmup과 짝지어 사용*하는 게 사실상 default (warmup → cosine).

---

## Warmup

- 표준 설정: 처음 *전체 step의 1–10%*를 warmup. 큰 모델은 길게(10%), 작은 모델은 짧게(1%) 또는 생략.
- LR이 0에서 본 LR로 *linear* 증가가 가장 흔함.
- *왜 필요한가*: 학습 초기엔 BN running 통계가 부정확하고 Adam의 2차 moment 추정이 noisy. 큰 LR로 시작하면 이 부정확한 통계 때문에 발산. (본문 §3.6의 β2=0.999 effective horizon 1000 step과 같은 맥락.)

---

## OneCycle (Smith 2018)

- 표준 설정: 전체 학습의 *45%까지 LR을 max까지 증가*, 다음 45%에서 max에서 min까지 감소, 마지막 10%에서 더 작은 값으로 추가 감소.
- max LR은 LR-finder로 측정한 값의 1/4 정도.
- 매우 빠른 학습이 가능 ("super-convergence"). 단 hyperparameter 민감.

---

## Cyclic LR

- 표준 설정: 주기 step 수 (예: 2000 step), min/max LR 사이를 삼각파(triangular)로 진동.
- 모델이 sharp minima를 빠져나오게 도움. 그러나 OneCycle보다 덜 사용됨.

---

## 무엇을 고를까

- *디폴트*: warmup + cosine annealing. Transformer·large model에 표준.
- *단순함이 필요*: step decay (epoch 50%, 75%에서 LR/10).
- *빠른 prototyping*: OneCycle.
- *fine-tuning*: 매우 작은 LR + cosine annealing (eta_max=1e-4 정도).

---

## 왜 warmup이 큰 모델·큰 LR에서 필수인가

학습 초기에 BN의 running 통계가 부정확하고, Adam의 second moment 추정이 noisy하다. 큰 LR로 시작하면 이 부정확한 통계 때문에 발산할 위험. 작게 시작해서 점진 증가.

---

## 면접 질문

**Q: Cosine vs Step decay?**

둘 다 잘 동작하나, cosine이 부드러워 일반화가 약간 좋다는 보고. Transformer/큰 모델은 cosine + warmup이 거의 표준.

---

## 본문 연결

- §3.8 LR schedule → 이 페이지 전체
- §3.6 Adam의 β₂ effective horizon → warmup 필요성과 연결
