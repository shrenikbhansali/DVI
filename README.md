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


