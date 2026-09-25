"""Round 3: GGN input factor + gradient output factor (HybridSOAP)
vs two-sided SOAP (control) vs Adam. Focus: does the EXACT GGN input
covariance (noise-free, residual-free) beat the gradient right factor?"""
import numpy as np
from train import (load_moons_illcond, load_illcond_regression, load_digits,
                   train)
import optimizers as O

DATASETS = [
    ("moons",   load_moons_illcond,      1,  "bce", 40),
    ("reg-ill", load_illcond_regression, 1,  "reg", 30),
    ("digits",  load_digits,             10, "ce",  40),
]

CFGS = [
    ("Adam",       lambda lr: O.Adam(lr),        [3e-3, 1e-2],          False),
    ("SOAP-1s",    lambda lr: O.SOAP(lr),        [3e-3, 1e-2],          False),
    ("SOAP-2s",    lambda lr: O.SOAP2S(lr),      [1e-3, 3e-3, 1e-2],    False),
    ("HybridSOAP", lambda lr: O.HybridSOAP(lr),  [1e-3, 3e-3, 1e-2],    True),
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
        print(f"  {name:12s} lr={lr:<7g} final loss {score:.4f}  "
              f"acc {h['acc'][-1]:.4f}  test {h['tloss'][-1]:.4f}  "
              f"({h['ms'][-1]:.1f} ms/step)")
        best[name] = h
    eps = [0, 2, 4, 9, epochs - 1]
    print("  loss by epoch:  " + " | ".join(f"{e+1:>3d}" for e in eps))
    for name in ["Adam", "SOAP-2s", "HybridSOAP"]:
        print(f"    {name:12s}  " + " | ".join(
            f"{best[name]['loss'][e]:6.3f}" for e in eps))
