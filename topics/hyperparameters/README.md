# Hyperparameters — 표준값과 결정 가이드 (hub)

> 본문 §2·§3·§5에서 등장하는 hyperparameter들의 *표준값·왜 그 값인가·언제 키우고 줄이나·함정*을 한 곳에 모은 hub. 본문은 한 줄 요약만 두고 깊은 가이드는 이 폴더의 sub-page로 위임한다.

---

## 빠른 참조 — 카테고리별 default

| 분류 | 파라미터 | Default | 변동 범위 | 어디서 |
|---|---|---|---|---|
| **Loss** | Huber δ | 1.0 | 0.3–5 | [loss.md](loss.md) |
| **Loss** | Focal γ | 2 | 0–5 | [loss.md](loss.md) |
| **Loss** | Label smoothing α | 0.1 | 0–0.2 | [loss.md](loss.md) |
| **Loss** | Triplet margin | 0.2 | 0.05–1.0 | [loss.md](loss.md) |
| **Loss** | Contrastive τ | 0.1 (SimCLR) | 0.05–0.5 | [loss.md](loss.md) |
| **Optimizer** | Momentum β | 0.9 | 0–0.99 | [optimizer.md](optimizer.md) |
| **Optimizer** | Adam β₁ | 0.9 | 거의 고정 | [optimizer.md](optimizer.md) |
| **Optimizer** | Adam β₂ | 0.999 | 거의 고정 | [optimizer.md](optimizer.md) |
| **Optimizer** | Adam ε | 1e-8 | 1e-8 ~ 1e-4 | [optimizer.md](optimizer.md) |
| **Optimizer** | Adam LR | 1e-3 | 1e-5 ~ 1e-3 | [optimizer.md](optimizer.md) |
| **Regularization** | Weight decay λ | 1e-4 (일반) / 1e-2 (AdamW) | 1e-5 ~ 1e-1 | [regularization.md](regularization.md) |
| **Regularization** | Dropout p | 0.5 (FC) / 0.1–0.2 (conv) | 0–0.7 | [regularization.md](regularization.md) |
| **LR schedule** | Step decay | epoch 50%·75%에서 ÷10 | factor 0.1–0.5 | [lr_schedule.md](lr_schedule.md) |
| **LR schedule** | Cosine + warmup | warmup 1–10% → cosine | (Transformer 표준) | [lr_schedule.md](lr_schedule.md) |
| **LR schedule** | OneCycle | 45/45/10 분할 | (super-convergence) | [lr_schedule.md](lr_schedule.md) |

---

## 튜닝 우선순위 (내림차순)

거의 모든 신경망 학습에서 *영향이 큰 순*:

1. **Learning rate** — 가장 민감. 다른 무엇보다 먼저 sweep.
2. **Batch size** — 메모리·속도·일반화 모두에 영향. LR과 묶임.
3. **Weight decay λ** — 큰 모델일수록 중요.
4. **Dropout p** (있으면)
5. **LR schedule** (warmup + cosine 같은 default가 거의 충분)
6. **Loss-specific** (focal γ, Huber δ 등) — 데이터 분포에 맞게
7. **Momentum β / Adam β₁β₂ε** — 거의 default 그대로

LR을 튜닝하지 않고 다른 걸 만지면 거의 다 노이즈에 묻힌다. LR이 가장 먼저.

---

## sub-page

- [**loss.md**](loss.md) — Huber δ, Focal γ, Label smoothing α, Triplet margin, Contrastive τ
- [**optimizer.md**](optimizer.md) — Momentum β, Adam (β₁ / β₂ / ε / LR)
- [**regularization.md**](regularization.md) — Weight decay λ, Dropout p
- [**lr_schedule.md**](lr_schedule.md) — Step / Cosine / Warmup / OneCycle / Cyclic

---

## 본문 cross-reference

각 sub-page는 본문 다음 자리에서 호출된다.

- §2.2 Huber → [loss.md#huber-δ](loss.md#huber-δ)
- §2.4 Focal·Label smoothing·Triplet·Contrastive → [loss.md](loss.md)
- §3.5 Momentum → [optimizer.md#momentum-β](optimizer.md#momentum-β)
- §3.6 Adam → [optimizer.md#adam](optimizer.md)
- §3.8 LR schedule → [lr_schedule.md](lr_schedule.md)
- §5.3 L2/Dropout → [regularization.md](regularization.md)
