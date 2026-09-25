# Native Curvature Engine (JAX)

A from-scratch implementation of **"Native Curvature Backpropagation"** — a
backprop engine that propagates a **curvature signal alongside the gradient in
a single backward sweep**, computing the diagonal of a Gauss-Newton
approximation at (almost) gradient cost, with **no `[batch, params]`
intermediates** (the memory wall that hits BackPACK-style libraries).

Built with `jax.custom_vjp`: each parameter is wrapped as
`{'val': θ, 'curv': zeros}`, each layer's backward rule emits **two cotangents**
in those slots, so one `jax.value_and_grad` call returns gradients AND
curvatures.

## Files

| file | contents |
|---|---|
| `engine.py` | custom_vjp rules: dense (X² trick), relu/tanh, residual add, MSE/BCE/CE losses with curvature seeding |
| `models.py` | MLP + ResNet-MLP (engine + plain-autodiff twins, incl. activation-trace variant) |
| `tests.py` | 14 correctness checks vs independent autodiff references |
| `optimizers.py` | SGD-m, Adam, NatCurv (sqrt/newton), AdamCurv, AdamHybrid, SOAP (1s), SOAP2S, **GN-SOAP**, **HybridSOAP** (GGN input factor + gradient output factor) |
| `train.py` | benchmark harness: 3 datasets, LR sweeps, dual/plain steps, plots |
| `soap_compare.py` | Round 2: curvature as `v_t` + GN-SOAP vs SOAP |
| `hybrid_compare.py` | Round 3: HybridSOAP vs two-sided SOAP vs Adam |
| `GPT_EXPERIMENT.md` | playbook for the GPU/transformer-scale follow-up |
| `convergence.png` | Round 1 convergence curves |

Run: `.venv/bin/python tests.py` then `.venv/bin/python train.py`
(output: `convergence.png`).

## The math

Dense layer `Y = X@Wᵀ + b`, gradient signal `Gy`, curvature signal `Cy`:

```
gW = Gyᵀ X        gb = Σ Gy      Gx = Gy W          (standard backprop)
cW = Cyᵀ(X⊙X)/B   cb = mean Cy   Cx = Cy (W⊙W)     ("X² trick")
```

`cW[i,j] = mean_n[Cy[n,i] X[n,j]²]` — the batch dimension collapses **inside
the GEMM**, exactly like standard backprop, so per-sample gradients are never
materialized. Curvature seeds: MSE → `1` (exact), sigmoid-BCE → `p(1−p)`
(exact), softmax-CE → Hutchinson MC `(Hv)⊙v`.

This is the **diagonal-propagated Gauss-Newton** (Becker & LeCun 1988;
"diagonal-path HBP"). It is *not* the exact Hessian diagonal — cross-terms
between output units are dropped by design (a mathematical wall, not an
engineering one).

## Verification (tests.py — 14/14 pass, float64)

1. gradients ≡ `jax.grad` on the plain model (max |Δ| ≈ 1e-16)
2. single dense + MSE: curvature ≡ **exact Hessian diagonal** (`jax.hessian`, |Δ| ≈ 1e-16)
3. dense→tanh nets: curvature ≡ **exact diag GGN** `(1/B)ΣΣ(∂z/∂θ)²`
4. 2-dense net: top layer ≡ exact diag GGN; hidden layer ≡ the exact
   **diag-path reference** `mean_n Σ_k (∂z/∂u_k)² (∂u_k/∂θ)²` (independent
   per-sample jacrev computation) — pins down the approximation semantics
5. curvature ≥ 0; BCE seed exact; scaling input feature j by s scales curvature
   row j by exactly s²
6. CE MC seed unbiased for diag-H GGN (0.9% rel err over 400 draws)
7. **memory**: largest jaxpr intermediate 32.8 KB (engine) vs 2097 KB
   (vmap-per-sample reference) — **64× smaller**, no `[B, params]` tensors

## Convergence benchmark (ResNet-MLP, width 64, 2 blocks, batch 128, CPU)

Best LR per optimizer (sweep), mean over 3 seeds. NatCurv modes:
`sqrt` = Adam-style `m̂/√d̂`; `newton` = `m̂/d̂` with Sophia-style trust clips.

| dataset | optimizer | lr | ms/step | final loss | acc / R² |
|---|---|---|---|---|---|
| moons ill-cond (×10/×0.1, BCE) | **Adam** | 3e-3 | 3.2 | **0.047** | **0.984** |
| | NatCurv-newton | 0.3 | 9.7 | 0.273 | 0.849 |
| | SGD-m | 0.03 | 1.1 | diverged | — |
| digits (10-class CE) | NatCurv-newton | 0.3 | 9.6 | **0.0001** | 1.000 (test loss **0.053** vs Adam 0.096) |
| | Adam | 3e-3 | 3.3 | 0.0001 | 1.000 |
| regression ill-cond (cond ~1e4, MSE) | **NatCurv-newton** | 0.3 | 9.6 | **0.014** | 1.000 |
| | NatCurv-sqrt | 0.03 | 9.3 | 0.030 | 1.000 |
| | Adam | 3e-3 | 3.2 | 0.093 | 0.9999 |
| | SGD-m | — | 1.1 | diverged | — |

Epochs-to-loss ≤ 0.1 on the regression task: **NatCurv-newton 11 vs Adam 29**
(2.6× fewer); wall-clock 4.9 s vs 2.8 s per 30 epochs, but Adam never reaches
0.1 in that budget.

## Honest conclusions

* **Where curvature wins**: ill-conditioned MSE regression — GGN diag = E[x²]
  is exactly the right preconditioner and is not corrupted by residual noise
  (Adam's E[g²] is). NatCurv-newton reaches the noise floor ~6× lower loss /
  2.6× fewer epochs than Adam's best.
* **Where Adam wins**: the ill-conditioned binary classification (moons).
  Early logit saturation zeroes the BCE curvature `p(1−p)` and NatCurv never
  fully recovers; Adam's E[g²] proxy doesn't have this failure mode.
* **Real data (digits)**: a tie in accuracy; NatCurv's test loss ~2× lower.
* **Cost**: dual backward ≈ 3× a plain step on CPU here (estimate before
  running was 1.3–1.5×; the extra factor is the third GEMM per layer plus
  pytree wrap/unwrap overhead in Python-land per step). Memory stays at
  gradient level — the main structural win.

This mirrors the literature (AdaHessian, Sophia, K-FAC): second-order
preconditioning helps on quadratic-ish/ill-conditioned losses, needs
sophisticated damping elsewhere, and pays ~2–3× per step.

## Relationship to existing work

No mainstream framework ships a fused curvature-backward kernel; the concept
dates to Becker & LeCun (1988) / LeCun's *Efficient BackProp* (1998), with
modern library approximations: BackPACK (hooks, [B,P] memory wall), kfac-jax
(custom JVP/VJP registrations, Kronecker blocks), AdaHessian/Sophia
(Hutchinson estimates). This repo is the "JAX custom_vjp" version of that
idea: ~90% of a native engine for ~1% of the engineering.

## Round 2 — "curvature as better v_t" and GN-SOAP (`soap_compare.py`)

**Q1: is the engine's curvature a better `v_t` than `E[g²]` (AdamW)?** No on these tasks:

| variant | moons | reg-ill | digits |
|---|---|---|---|
| Adam (`v = E[g²]`) | **0.035** | 0.140 | **0.0000** |
| AdamCurv (`v = E[curv]`, pure swap) | 0.422 | nan@lr | 0.0001 |
| AdamHybrid (`v = max(g², curv)`) | 0.062 | 0.162 | 0.0013 |

Pure swap inherits curvature collapse (saturated BCE); hybrid is neutral-at-best.
Reason: in sqrt form, `sqrt(curv)` and `|g|` carry nearly the same scale
information (they differ mainly by residual magnitude), and `g²` never
collapses. The real second-order win in Round 1 came from the **Newton**
update `g/curv`, not from better `sqrt(v)`.

**Q2: can native curvature improve SOAP?** Implemented **GN-SOAP**: SOAP
machinery (Adam in the preconditioner's eigenbasis, rotate back) with the
preconditioner built from engine factors instead of gradient covariance:
input side `A = E[x xᵀ]` (exact GGN factor, one `XᵀX` GEMM in the same
step), output side exact diagonal `E[c_i]` row scaling. Comparison
(one-sided SOAP, T=10, Sophia-style trust clip on GN-SOAP):

| optimizer | moons | reg-ill | digits | ms/step |
|---|---|---|---|---|
| Adam | **0.035** | 0.140 | **0.0000** | 3.1 |
| SOAP (grad-cov) | 0.039 | 0.133 | 0.0000 | 7.0 |
| GN-SOAP (sqrt) | 0.160 | **0.111** | 0.063 | 11.0 |
| GN-SOAP (newton) | 2.05 | 0.337 | 0.018 | 11.0 |

GN-SOAP wins where curvature is exact (ill-conditioned regression), is
mid-pack on moons (BCE curvature fragility again), loses on digits (Hutchinson
noise + trust-clip slowdown). SOAP's own early convergence on moons was the
fastest of all (loss 0.098 vs Adam 0.382 at epoch 10) — full-matrix
preconditioning is doing real work that diagonals cannot.

Takeaway: gradient covariance and native GGN factors carry **complementary**
information — SOAP gets cheap full matrices from gradients (both sides, but
empirical-Fisher-corrupted); the engine gives exact input covariance + exact
diagonal output curvature (but no off-diagonal output structure). A hybrid
that uses GGN input factors *and* gradient output covariance is the obvious
next experiment.

## Round 3 — GGN input factor + gradient output factor (`hybrid_compare.py`)

HybridSOAP: two-sided SOAP rotation, `Q_R` from the engine's exact
`A = E[x xᵀ]`, `Q_L` from Shampoo's `G_L = EMA[g gᵀ]`. Control: SOAP-2s
(both factors from gradients).

| optimizer | moons | reg-ill | digits (test loss) |
|---|---|---|---|
| Adam | 0.035 | 0.140 | **0.043** |
| SOAP-1s (grad R) | 0.039 | 0.133 | 0.064 |
| SOAP-2s (grad L+R) | **0.028** | 0.255 | 0.063 |
| **HybridSOAP (GGN R + grad L)** | 0.035 | **0.130** | 0.066 |

* **reg-ill**: the GGN input factor fixes two-sided SOAP's degradation
  (0.255 → 0.130, best of all) and dominates the early epochs. Reason:
  `A = E[x xᵀ]` is exact, residual-free, and **full-rank from step one**,
  while `gᵀg` accrues rank-1 per step and is residual-corrupted.
* **moons**: the gradient right factor wins instead (input side is only
  2-D there; the output-side correlations matter more).
* **digits**: everything ties on train; Adam keeps the best test loss.

Conclusion: the engine's unique contribution to a SOAP-style optimizer is a
*noise-free, immediately-full-rank* input-side preconditioner. It is not
uniformly better than the gradient factor — the two are complementary, and
which side benefits depends on where the problem's conditioning lives.
