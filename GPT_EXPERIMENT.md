# GPT-scale experiment guide (GPU machine)

This repo's MLP results suggest the most promising transformer integration is
**HybridSOAP** (Round 3): SOAP machinery where the *input-side* preconditioner
is the engine's exact `A = E[x xᵀ]` instead of the gradient covariance —
noise-free, residual-free, full-rank from step one. This doc is the playbook
to test it at transformer scale.

## 0. Setup

```bash
git clone https://github.com/Forbu/native-curvature-engine.git
cd native-curvature-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -U "jax[cuda12]"       # GPU jaxlib
python tests.py                    # must print: 14/14 checks passed
```

## 1. Engine extensions needed (not yet implemented)

The engine currently has rules for `dense / relu / tanh / residual-add` and
losses `MSE / BCE / CE`. A transformer additionally needs:

| layer | gradient rule | proposed curvature rule (diagonal-propagation) |
|---|---|---|
| embedding lookup | standard | per-row curvature = row usage frequency (or just let Adam handle embeddings) |
| RMSNorm `y = γ·x/rms(x)` | standard | `c_x = (γ/rms)² ⊙ c_y` (drops mean/covariance corrections — verify vs diag-path reference) |
| attention `Y = softmax(QKᵀ/√d)V` | standard | per-head, row-wise over the softmax: `c_s = (p⊙(1−p))² ⊙ c_A` (keep only Jacobian diagonal); value path: `c_V ≈ Aᵀ² c_Y`, `c_x` via `(W_q², W_k², W_v²)` chains |
| causal masking | standard | curvature only over unmasked positions |

**Verification methodology** (same as `tests.py`, do this first):
1. gradient slots ≡ `jax.grad` on a plain transformer forward (machine epsilon)
2. curvature ≡ the "diag-path" reference
   `(1/B)Σ_n Σ_d Σ_k (∂z_d/∂u_k)²(∂u_k/∂θ)²` computed by per-sample
   `jacrev` on a tiny model — this pins the approximation semantics
3. positivity + no `[B, params]` intermediates in the jaxpr

## 2. Experiment matrix

- **Model**: decoder-only GPT, e.g. 4–8 layers, d=256–512, 4 heads (nanoGPT /
  char-shakespeare for a first pass; TinyStories slice for the real run).
- **Optimizers**:
  1. AdamW (baseline, tuned lr + wd)
  2. Muon (public reference implementation, orthogonalized momentum — the
     strongest cheap competitor)
  3. SOAP-2s (both factors from gradients — the ablation control)
  4. **HybridSOAP** (this repo: `optimizers.HybridSOAP`, GGN input factor +
     gradient output factor)
- **Metrics**: val loss vs tokens seen, vs wall-clock (preconditioners cost
  ~2–3× per step; the claim must survive wall-clock), plus val loss at fixed
  compute budget. 2–3 seeds.
- **LR grids**: AdamW {3e-4, 1e-3, 3e-3}; Muon {0.02, 0.05} (its own scale);
  SOAP variants {3e-3, 1e-2, 3e-2}. T=10 for eigendecomps; Sophia-style
  trust clip (clip_ratio=0.02) on anything dividing by curvature.

## 3. What to expect (from the MLP rounds)

- The GGN input factor should help most on the **embedding table's first
  linear consumers** and the **LM head** (fixed, exact geometry), and least
  where output-side correlations dominate (attention value paths).
- Loss curvature for CE uses the Hutchinson seed `(Hv)⊙v` — keep the EMA
  (β2) and floors; the `soap_compare.py` digits results show what noise does
  without them.
- If HybridSOAP ≥ SOAP-2s on val-loss-per-wallclock at 2–3 model scales,
  that's a real result.

## 4. Known pitfalls (already solved in this repo — keep them)

- XLA fast-math NaNs on `log(sigmoid)`/saturated losses → use
  `softplus(z) − t·z` (see `engine._bce_fwd`)
- curvature collapse (confident predictions, dead ReLU) → relative floor +
  absolute floor (`optimizers.NatCurv`)
- EMA bias-correction missing on curvature state → weights jump ~16× in one
  step (the GNSoap bug, fixed — do not reintroduce)
- Sophia-style trust clip `RMS(Δθ) ≤ 0.02·RMS(θ)` is what makes Newton-like
  scaling stable at all
