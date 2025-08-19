# DVI: Draft → Verify → Improve

## Overview

DVI extends self-speculation by turning the verifier’s internal accept/reject signal into a training objective—resulting in a fully self-supervised, online reinforcement learning system.

---

## Background

### Speculative Decoding (SD)

Speculative decoding is an inference-time optimization designed to increase throughput for autoregressive generation. Instead of generating one token at a time, it:

1. Uses a **draft model** to propose multiple tokens in parallel.
2. Uses a **verifier model** to check those proposals.
3. Accepts matching tokens; regenerates the rest.

This yields large speedups while maintaining output quality, since the verifier ensures semantic and statistical correctness.

### Kangaroo: Self-Speculative Decoding

The **Kangaroo** method improves upon SD by using a single model, split into two parts:

- **Shallow layers** (layers $0 \to k$) act as the **draft** module.
- **Deep layers** (layers $k \to L$) act as the **verifier**.

The draft generates candidate logits, and the verifier reprocesses them. If both agree on a token, it is "accepted." This removes the need for an external draft model.

However, Kangaroo is only an inference-time trick: the accept/reject signal is discarded.

---

## DVI

DVI converts speculative decoding into an **online training signal**. It reuses the verifier's decision as **reinforcement feedback** to improve the draft module, forming a training-time loop of:

> **Draft → Verify → Improve**

By doing this, DVI enables continual, self-supervised training from natural interaction (e.g., chatbot transcripts or streaming dialogue).

### Bootstrap Challenge & Unified Loss (NEW)

Starting from an un-trained draft head yields **near-zero acceptance** → no gradient.
We therefore **bootstrap** with a *decaying* KL term:

```math
{\cal L}_{\text{total}}
  = (1-\lambda_t)\,{\cal L}_{\text{RL}}
  + \lambda_t\,D_{\text{KL}}\bigl(p_{\text{verify}}\;\|\;p_{\text{draft}}\bigr),
\qquad
\lambda_t = \exp(-t/\tau).
```

* Early: behaves like knowledge-distillation ⇒ dense signal.  
* Late: pure REINFORCE ⇒ unbiased optimisation.

This **unifies** distillation + RL into one objective and removes stage-wise complexity.

---

## Kangaroo vs. DVI

| Kangaroo (Inference-only)                        | DVI (Training-enabled)                                             |
| ------------------------------------------------ | ------------------------------------------------------------------ |
| Uses draft/verify split to accelerate generation | Uses same split, but converts accept/reject into a training signal |
| Accept/reject signal is discarded                | Accept = reward 1, Reject = reward 0 → REINFORCE (+ KL warm-up λ_t)|
| No learning from experience                      | Performs continual RL using on-device buffer                       |
| Requires static, fixed model                     | Online learning; model adapts to new data                          |
| Only improves runtime throughput                 | Improves both throughput **and** model quality over time           |
| No replay or learning memory                     | Maintains experience buffer on verifier device                     |

---

## Fast and Slow Updates (not always RL)

DVI has two distinct optimizations — **fast updates** and **slow updates** — to manage learning in the **draft** and the **verifier** modules respectively.

### Fast Updates (for Draft / Policy Improvement)

The fast update operates on every training step, focusing solely on improving the **draft** network's ability to generate tokens that are likely to be accepted by the verifier.

Formally, the draft adapter defines a stochastic policy $\pi_\theta(t \mid h_k)$, where $h_k$ is the intermediate hidden state at the split layer. At each decoding step, the model observes a binary reward $r_t \in \{0, 1\}$ based on whether the verifier accepted the token. The fast update applies the **REINFORCE** algorithm:

```math
\nabla_\theta \mathbb{E}[r_t] = \mathbb{E}[(r_t - b) \nabla_\theta \log \pi_\theta(t \mid h_k)]
````

where \$b\$ is a baseline used for variance reduction.

This provides an **unbiased policy gradient** signal encouraging the draft model to improve its match with the verifier. Crucially:

* The fast update **does not** modify the verifier.
* The signal is **local** (per token) and **immediate**.
* It assumes the verifier remains a stable source of feedback.

This assumption — that the verifier is reliable and well-calibrated — is the weak point. Over time, the **draft network may shift its distribution**, causing the verifier’s judgments to become **out-of-domain** or inconsistent. When this happens, the reward signal may no longer reflect actual token quality, and learning degrades.

---

### Slow Updates (Verifier Maintenance)

To address this drift, DVI introduces **slow updates** to the verifier component. These updates are performed **periodically** and are **supervised**, not policy-based.

The verifier serves as both:

1. An **executor** of speculative decoding (deciding accept vs reject), and
2. A **teacher** that provides reward signals to train the draft.

If the verifier becomes outdated — e.g., due to domain shift in streaming input or evolving draft behavior — then its accept/reject signals become meaningless. Worse, the system may reinforce poor behaviors.

To maintain verifier reliability, DVI applies a **slow update** schedule using a replay buffer of accepted and rejected tokens. The verifier is fine-tuned using a cross-entropy objective and regularized with a KL-divergence constraint:

```math
\mathcal{L}_{\text{verifier}} = \mathcal{L}_{\text{CE}} + \beta_{\text{KL}} \cdot D_{\text{KL}}(\pi_\phi \| \pi_{\phi_{\text{old}}})
```

Here:

* \$\mathcal{L}\_{\text{CE}}\$ trains the verifier on next-token prediction using its own historical outputs as soft targets.
* \$D\_{\text{KL}}\$ penalizes deviation from the previous verifier, preventing instability.
* \$\phi\$ are the verifier parameters (typically a LoRA adapter), and \$\phi\_{\text{old}}\$ is a frozen copy.

This slow, conservative tuning ensures that:

* The verifier continues to track the **domain distribution**.
* The accept/reject decisions remain a valid training signal.
* The overall learning loop remains **stable and aligned**.

---

## Mathematical Formulation

### Kangaroo: Inference-time Speculative Decoding

Let \$h\_k\$ be the hidden state at layer \$k\$ of an LLM:

1. **Draft** logits:

```math
z^{(D)} = W_{\text{out}}^{(S)} h_k + b^{(S)}
```

where \$W\_{\text{out}}^{(S)}\$ are projection weights from shallow layers.

2. Sample token \$t \sim \text{softmax}(z^{(D)})\$

3. **Verifier** re-computes hidden state:

```math
h_L = f_{k \to L}(h_k)
```

Then computes:

```math
z^{(V)} = W_{\text{out}}^{(D)} h_L + b^{(D)}
```

4. If \$\arg\max(z^{(D)}) = \arg\max(z^{(V)})\$, token is accepted.
   Otherwise, it is rejected and replaced.

This is purely deterministic logic. No training occurs. The model generates faster, but does not improve.

---

### DVI: Draft → Verify → Improve (Online RL)

#### Notation:

* \$\pi\_\theta(t \mid h\_k)\$: Draft policy (parameterized by LoRA adapter \$\theta\$).
* \$r\_t \in {0, 1}\$: Verifier accept signal (1 = accepted, 0 = rejected).
* \$b\$: EMA baseline for variance reduction.

#### Objective: REINFORCE

The draft adapter is trained via:

```math
\mathcal{L}_{\text{draft}} = - (r_t - b) \log \pi_\theta(t \mid h_k)
```

* Tokens accepted by the verifier are treated as positive reinforcement.
* This encourages the draft to mimic tokens likely to be accepted.

#### Slow Update: Verifier Training

We also train the verifier’s adapter (LoRA \$\phi\$) using supervised losses:

```math
\mathcal{L}_{\text{verifier}} = \underbrace{\mathcal{L}_{\text{CE}}(t, \pi_\phi)}_{\text{Supervised on GT token}} + \beta_{\text{KL}} \underbrace{D_{\text{KL}}(\pi_\phi \| \pi_{\phi_{\text{old}}})}_{\text{Conservative update}}
```

* \$\mathcal{L}\_{\text{CE}}\$: Cross-entropy loss using real token.
* \$D\_{\text{KL}}\$: Ensures the verifier doesn’t change too rapidly.



Here’s a clean, paste-ready section you can drop into the README.

---

## Notes on Design & Implementation Choices


### 1) Exit-Head Stabilization: Pre-Norm + Learnable Logit Scale

**What we do.** Before the draft head, we apply a normalization layer (clone of the model’s final RMSNorm when available, LayerNorm fallback) and multiply logits by a learnable global scalar:

* `ĥ = Norm(h_k)`
* `z_s = α · W_exit · ĥ` with `α` learnable

**Why it helps.**

* **Distribution matching.** Hidden magnitudes at layer `k` drift as LoRA updates accumulate. Pre-norm keeps the exit features in the same range the original LM head expects, making KL/CE well-behaved.
* **Single-knob calibration.** A global scale `α` prevents the exit head from becoming over-confident (or too flat) during early training, avoiding vanishing or exploding gradients without hand-tuned temperature hacks.

**Alternatives.**

* No pre-norm (simpler, but more likely to destabilize early training).
* Per-class temperature / vector scale (more expressive, higher variance).
* WeightNorm on `W_exit` (good control but slightly more plumbing).

We default to **Pre-Norm + scalar scale** for minimal complexity and strong stability.

---

### 2) “RL” Term Uses **Log-Probability** of the Verifier’s Top-1

**What we do.** The policy term optimizes `L_pg = - E[ log π_s(v_top1) ]` (draft log-prob on the verifier’s top-1 token).

**Why it helps.**

* **Stronger gradients when wrong.** `-π(v)` has tiny gradient when π is small; `-log π(v)` gives large corrective updates, accelerating bootstrap.
* **Low variance.** Using the verifier top-1 as a pseudo-target is a low-variance surrogate compared to binary accept bits.
* **Compatible with KL/CE.** Plays nicely with distillation and avoids double-counting issues better than raw probability objectives.

**Alternatives.**

* True REINFORCE on accept bits: `-(r - b) · log π_s(t)` (unbiased but noisy; needs a baseline `b` and/or accepted-only sampling).
* Raw probability `-E[π_s(v)]` (simpler but weak gradients off-mode).
* Margin losses between `v_top1` and runner-up (can help late-stage sharpening).

We picked **log-prob** as the best trade-off for fast, stable bootstrap. (Extending to `(r−b)·log π` with an EMA baseline is a natural next step.)

---

### 3) **Sharper Teacher** (Temperature **T = 1.0** by default)

**What we do.** KL uses the verifier distribution at **T=1.0** (no softening), plus a CE anchor on `v_top1`.

**Why it helps.**

* **Crisp signal early on.** When acceptance is near zero, a sharp teacher avoids “blurred” supervision where the mode is unclear.
* **Better alignment with acceptance.** In DVI, acceptance is a top-1 event. A sharper teacher makes the KL/CE signals reflect the same decision boundary.

**Alternatives.**

* Softer teacher (T=1.1–1.5): improves mode coverage and can help if the verifier is noisy or under-confident.
* Annealed temperature: start soft → hard as acceptance rises.

We default to **T=1.0**; if you see instability, try **T≈1.2** for a few thousand steps.

---

### 4) **Stronger CE Anchor** During Warm-Up

**What we do.** We add a CE term on the verifier’s top-1 (default weight `0.20`). Together with KL it provides dense gradients from step 1.

**Why it helps.**

* **Avoids cold-start collapse.** When acceptance is \~0, pure RL has no signal; CE guarantees a usable gradient.
* **Complementary to KL.** KL shapes the full distribution; CE pins the argmax.

**Alternatives.**

* Decay CE weight over time (common pattern: strong early, taper late).
* Replace CE with margin or contrastive terms (works but less plug-and-play).

If your model becomes over-confident too soon, lower CE (e.g., `0.05–0.10`) or add a tiny entropy bonus.

---

### 5) KL Direction & Scheduling

**What we do.** We use **forward KL** `D_KL(p_verify || p_draft)` and a **KL→RL** schedule (warm-up, then ramp up the policy term, keep a small KL residual).

**Why it helps.**

* **Mode-covering early.** Forward KL encourages the draft to match the teacher’s mass, helpful when the draft is under-trained.
* **Residual KL** stabilizes late training and prevents drift from the verifier.

**Alternatives.**

* Reverse KL (mode-seeking; sharper but brittle early).
* No residual KL in late RL (more exploration, more risk).

Keep a small **KL floor** (e.g., 0.05) unless you specifically want to push exploration.

---

### 6) Later Split by Default (Deeper Draft)

**What we do.** Default `early_layer=24` (vs very shallow layers).

**Why it helps.**

* **Higher acceptance ceiling.** A deeper draft is closer (representation-wise) to the verifier, raising baseline agreement and making the RL/KL objectives easier.
* **Better gradient alignment.** Hidden features at later layers are more “LM-head ready.”

**Alternatives.**

* Shallower split (more speedup, lower acceptance).
* Adaptive split (dynamic k): promising but more engineering.

Start with **24** (7B scale). Tune for your speed/quality target.

---

### 7) No Norm Clamping by Default (But Available)

**What we do.** We ship a Frobenius-norm clamp for the exit head **disabled** by default.

**Why.**

* The **learnable logit scale** already provides amplitude control. Hard clamping can silently fight the scale and slow learning.

**Alternatives.**

* Relative clamp (e.g., `‖W_exit‖ ≤ 1.5×‖W_init‖`) if you see norm blow-ups.
* Lower weight decay on the exit head if under-fitting.

Use clamps only as a **seatbelt** when you observe instability, not preemptively.

---

### 8) Accepted-Only Sampling (Optional)

**What we do.** You can enable **accepted-only** batch sampling (with a backoff if the pool is too small). The loss still trains on the verifier target; the filter changes which states are sampled.

**Why it helps.**

* **Variance reduction.** Focuses updates on states where the verifier and draft already interact positively, which often stabilizes mid-training.

**Trade-offs.**

* **Bias.** Skews experience toward “good” states; combine with periodic full-buffer sampling to avoid over-fitting.
* **Cold-start.** Keep it **off** during pure KL warm-up; consider turning it on once acceptance > \~10–20%.

---

### 9) Diagnostics Worth Watching

* **`std_s` vs `std_t`** (student vs teacher logit std): large gaps → mis-calibration; if `std_s` ≫ `std_t`, reduce CE or add entropy; if `std_s` ≪ `std_t`, increase scale/CE slightly.
* **`π_s(v)`** and **acceptance rate**: should trend upward as KL decays.
* **Exit-head norm** and **logit scale α**: rapid growth = over-confidence; flatlined α with rising loss = under-fitting.

---

### 10) What We Didn’t Do (and Why)

* **Direct binary-reward REINFORCE** in the default path: the accept bit is extremely sparse at bootstrap → high-variance updates. The current surrogate (`-log π_s(v_top1)`) gives most of the win with far better stability.
  *If you want unbiased RL*, add `(r−b)` as a multiplier to the log-prob term and maintain an EMA baseline `b`.
* **Hard temperature schedules or label smoothing**: the learnable scale plus CE were sufficient in practice; consider temperature annealing only if you see persistent over-confidence.

---

### TL;DR: Defaults that Make DVI Train

* **Pre-Norm + logit scale** before the exit head → calibrated logits.
* **Log-prob policy term** on verifier top-1 → strong, low-variance updates.
* **Sharp teacher (T=1.0) + CE anchor** → dense bootstrap signal.
* **Forward KL with residual weight** → stability as RL ramps.
* **Later split (k≈24)** → higher acceptance ceiling.
* **Clamps off by default**, enable only if needed.
* **Accepted-only sampling** optional; turn on after acceptance improves.

These choices aim to **maximize early learning signal** and **minimize variance**, so the draft can quickly align with the verifier and lift acceptance without babysitting.
