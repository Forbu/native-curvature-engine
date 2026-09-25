"""Q1: is native curvature a better v_t than E[g^2] (AdamW)?
Q2: does native GGN-factor preconditioning improve SOAP?

Optimizers compared (all fair LR sweeps, same model/data pipeline):
  Adam        : v_t = EMA[g^2]                       (baseline)
  AdamCurv    : v_t = EMA[curv]                      (pure swap)
  AdamHybrid  : v_t = EMA[max(g^2, curv)]            (safe swap)
  SOAP        : eigenbasis-Adam, preconditioner from g^T g (gradient cov.)
  GN-SOAP     : same machinery, preconditioner from ENGINE factors:
               A = E[x x^T] (exact GGN input) + row scale E[c_i] (exact
               GGN output diagonal)
"""
import numpy as np
from train import (load_moons_illcond, load_illcond_regression, load_digits,
                   train)
import optimizers as O

DATASETS = [
    ("moons",   load_moons_illcond,        1,  "bce", 40),
    ("reg-ill", load_illcond_regression,   1,  "reg", 30),
    ("digits",  load_digits,               10, "ce",  40),
]

CFGS = [
    ("Adam",         lambda lr: O.Adam(lr),                    [1e-3, 3e-3, 1e-2], False),
    ("AdamCurv",     lambda lr: O.AdamCurv(lr),                [3e-3, 1e-2, 3e-2], True),
    ("AdamHybrid",   lambda lr: O.AdamHybrid(lr),              [1e-3, 3e-3, 1e-2], True),
    ("SOAP",         lambda lr: O.SOAP(lr),                    [1e-3, 3e-3, 1e-2], False),
    ("GN-SOAP",      lambda lr: O.GNSoap(lr),                  [3e-3, 1e-2, 3e-2], True),
    ("GN-SOAP-Nwt",  lambda lr: O.GNSoap(lr, mode="newton"),   [3e-2, 1e-1, 3e-1], True),
]

for dname, load, n_out, task, epochs in DATASETS:
    ds = load()
    print(f"\n=== {dname} ({task}, {epochs} epochs) ===")
    best = {}
    for name, mk, lrs, use_curv in CFGS:
        rows = []
        for lr in lrs:
            h = train(mk(lr), ds, n_out, epochs=epochs, seed=0,
                      use_curv=use_curv, task=task)
            rows.append((h["loss"][-1], lr, h))
        score, lr, h = min(rows, key=lambda r: (not np.isfinite(r[0]), r[0]))
        print(f"  {name:13s} lr={lr:<7g} final loss {score:.4f}  "
              f"acc {h['acc'][-1]:.4f}  test loss {h['tloss'][-1]:.4f}  "
              f"({h['ms'][-1]:.1f} ms/step)")
        best[name] = h
    # early-convergence detail for the two most interesting comparisons
    eps = [0, 2, 4, 9, epochs - 1]
    print("  loss by epoch:  " + " | ".join(f"{e+1:>3d}" for e in eps))
    for name in ["Adam", "AdamHybrid", "SOAP", "GN-SOAP", "GN-SOAP-Nwt"]:
        print(f"    {name:13s}  " + " | ".join(
            f"{best[name]['loss'][e]:5.3f}" for e in eps))
