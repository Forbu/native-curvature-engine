# GPT-scale results (NVIDIA L4, Lightning AI)

**Setup**: decoder-only GPT, 4L · d=256 · 4 heads · ctx 256 (3.2M params),
char-level **TinyStories** (2.2GB train, last 0.5% val), batch **128**,
**6000 steps = 196M tokens ≈ 9% of one epoch** — a fixed-compute
optimization comparison with no memorization confound (train≈val throughout).
All: grad-clip 1.0, 100-step warmup, wd 0.01, single seed, no per-optimizer
tuning beyond the LR noted.

The transformer curvature engine (`gpt_gpu.py`, dual path) propagates
gradient + diagonal GGN through attention, RMSNorm and GELU with
`custom_vjp` (`test_rules.py`): gradients exact vs autodiff (≤4e-8
per-rule; ~1e-4 full-model, f32 chain noise), CE seed is the **exact**
diagonal `p(1−p)`. One bug found and fixed during verification: the softmax
backward initially used the diagonal-only Jacobian — the gradient must be
`A(gA − ⟨gA,A⟩)`, only the *curvature* rule stays diagonal.

## Final results (val loss, char-level nats/token)

| optimizer | @1000 | @3000 | @6000 | ms/step | wall-clock |
|---|---|---|---|---|---|
| **AdamW** lr 3e-3 | 0.919 | 0.747 | **0.676** | 109 | **717 s** |
| HybridSOAP lr 1e-2 | 1.109 | 0.855 | 0.783 | 153 | 977 s |
| SOAP-2s lr 1e-2 | 0.957 | 0.795 | 0.812 | 133 | 867 s |
| NatCurv (newton) lr 1e-2 | 1.808 | 1.186 | 1.044 | 180 | 1080 s |
| NatCurv (newton) lr 3e-2 | 1.791 | 2.197 | diverged-ish | 180 | — |

Earlier 3000-step reference (same data/model, batch 128): AdamW@3e-3 0.7401.

## Reading the results (honest)

1. **AdamW wins decisively at this scale.** 0.676 vs 0.78–0.81 for the
   preconditioned family and 1.04 for Newton curvature. It is also the
   cheapest per step. At 3.2M params / 33k-token batches, per-coordinate
   adaptivity is already nearly optimal.
2. **The GGN input factor still helps SOAP** (0.783 vs 0.812 at step 6000;
   the gap opened after ~4k steps), consistent with the MLP findings — but
   it does not close the gap to AdamW here.
3. **NatCurv-newton is not competitive at LM scale out-of-the-box.** The
   dual backward itself is cheap (1.65× AdamW per step — the whole
   curvature signal for +65% step cost), but the Newton direction is
   constantly restrained by the floor + trust clip: CE curvature collapses
   (`p(1−p)→0`) as the model becomes confident, so late training is
   clip-dominated. This matches the prediction in `GPT_EXPERIMENT.md`:
   LayerNorm removes the input-anisotropy edge that made Newton scaling win
   on ill-conditioned regression MLPs.
4. **Scale caveat (important)**: SOAP/Shampoo's published wins are at
   ≥360M params and 2–4M-token batches — far from this 3.2M/33k setup.
   Before declaring curvature useless for LM training, the fair next step
   is a scaled run (d≥768, ≥12L) where preconditioning has room to matter.

## What was validated at GPT scale

- The dual-slot `custom_vjp` architecture ports to attention/RMSNorm/GELU
  with exact gradients and PSD, finite curvature at ~1.6× step cost.
- New exact rules: `c_S = (p⊙(1−p))² ⊙ ⟨v_j², c_t⟩` through softmax-attention,
  `c_x = (γ/rms)² ⊙ c_y` through RMSNorm, `p(1−p)` CE seed (no MC needed).
- Muon (fixed Newton–Schulz scaling) and the fused-JIT training loop
  (20–180 ms/step on L4).

## Repro

```bash
ssh <lightning-studio>
cd native-curvature-engine
~/nce-venv/bin/python test_rules.py        # rule verification
STEPS=6000 ~/nce-venv/bin/python gpt_gpu.py # full comparison
```

Logs: `logs/gpt_main.log`, `logs/gpt_nc.log`.

---

# Scale-up: 12 layers · d=768 · 12 heads (~85M params)

Same data/protocol (TinyStories, TF32 matmuls, grad-clip 1.0, wd 0.01,
rematerialized blocks, batch 64 — batch 128 OOMs the 24GB L4 without remat).
**5000 steps = 82M tokens**. Note: the originally requested LRs (AdamW 3e-3,
Hyb 1e-2) were miscalibrated at this scale — AdamW@3e-3 plateaued at val
≈2.33 (near-unigram; see `logs/gpt768_hotlr.log`) — so both optimizers got a
small fair grid instead.

| optimizer | lr | val @5000 | ms/step | wall-clock |
|---|---|---|---|---|
| AdamW | 3e-4 | 0.5817 | 830 | 69 min |
| AdamW | 1e-3 | 0.6453 | 825 | 69 min |
| **HybridSOAP** | **1e-3** | **0.5467** | 1474 | 123 min |
| HybridSOAP | 3e-3 | 0.6572 @2080 (studio suspended mid-run) | 1464 | — |

## Headline

**At 85M params the result flips: HybridSOAP beats AdamW by 6% val loss
(0.5467 vs 0.5817)** — even paying 1.8× wall-clock per step. At 3.2M params
AdamW won (0.676 vs 0.783); preconditioning needed the scale, exactly as the
SOAP literature suggests. The GGN input factor `A = E[x xᵀ]` (the engine's
contribution) is doing this inside a two-sided SOAP skeleton — no curvature
backprop needed, one `XᵀX` GEMM per linear per step.

Caveats: single seed, single fair-ish grid (3e-4 may not be AdamW's optimum —
no 1e-4 point), Hyb@3e-3 incomplete, val loss at fixed steps (not fixed
wall-clock: on wall-clock AdamW reaches 0.58 at 69 min while Hyb needs ~110+
min for 0.55).
