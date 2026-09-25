"""Convergence benchmark: Adam vs NatCurv (native curvature engine).

Datasets
  * moons-illcond : sklearn two-moons, features scaled by (100, 0.01) —
                    a pathological input geometry where first-layer weights
                    live at wildly different scales. Binary, BCE loss
                    (exact, deterministic curvature seed).
  * digits        : sklearn 8x8 digits, 10 classes, standardized. Softmax CE
                    (1-sample MC curvature seed).

Model: ResNet MLP  X -> relu(dense) -> 2 x [h += relu(dense(relu(dense(h))))]
       -> dense logits.  Width 64.

Fairness: every optimizer gets a small LR sweep; curves are reported for
each optimizer's best LR (by final training loss). Adam/SGD run on a plain
jax.value_and_grad pipeline (no curvature computed); only NatCurv pays the
dual-backward cost — and we report per-step wall time of both pipelines.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import engine as eng
import models as M
import optimizers as O

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ data

def load_moons_illcond(n=2048):
    from sklearn.datasets import make_moons
    X, t = make_moons(n_samples=n, noise=0.15, random_state=0)
    scales = np.array([10.0, 0.1])         # 100x anisotropy: brutal for SGD
    X = X * scales[None, :]
    return (X.astype(np.float32), t.astype(np.int32),
            X.astype(np.float32), t.astype(np.int32))


def load_illcond_regression(n=2048, d=20, noise=0.1, seed=0):
    """y = <w*, x> + eps with anisotropic features: std(x_j) = 10^(u_j),
    u spread over [-2, 2] -> input-condition number ~1e4. MSE loss.
    This is the textbook case for diagonal Gauss-Newton: curvature is
    exactly E[x^2] and residual noise never corrupts it."""
    rng = np.random.default_rng(seed)
    scales = np.logspace(-2, 2, d).astype(np.float32)
    X = rng.standard_normal((n, d)).astype(np.float32) * scales
    w = (rng.standard_normal(d) / np.sqrt(d)).astype(np.float32)
    y = X @ w + noise * rng.standard_normal(n).astype(np.float32)
    idx = rng.permutation(n); n_tr = int(0.8 * n)
    tr, te = idx[:n_tr], idx[n_tr:]
    return X[tr], y[tr][:, None], X[te], y[te][:, None]


def load_digits():
    from sklearn.datasets import load_digits
    d = load_digits()
    X = d.data.astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    t = d.target.astype(np.int32)
    idx = RNG.permutation(len(X))
    n_tr = int(0.8 * len(X))
    tr, te = idx[:n_tr], idx[n_tr:]
    return X[tr], t[tr], X[te], t[te]


# ------------------------------------------------------------ model / step

def build(n_in, n_out, width=64, blocks=2, seed=0):
    sizes = [n_in, width] + [width] * (2 * blocks) + [n_out]
    return M.init_params(sizes, jax.random.PRNGKey(seed))


def make_steps(n_out, blocks, task="bce"):
    """Returns (step_dual, step_plain, eval_fn).

    step_dual   : engine pipeline -> loss, grads, curvs (used by NatCurv)
    step_plain  : plain pipeline  -> loss, grads       (used by Adam / SGD)
    """
    if task == "reg":
        loss_w = lambda o, t, aux: eng.mse_loss(o, t)
        fwd_plain = lambda p, X: M.resnet_plain(p, X, act="relu",
                                                n_blocks=blocks)
        loss_plain = lambda z, t, aux: jnp.mean(0.5 * jnp.sum((z - t) ** 2,
                                                              axis=1))
    elif n_out == 1:
        loss_w = lambda o, t, aux: eng.bce_loss(o, t)
        fwd_plain = lambda p, X: M.resnet_plain(p, X, act="relu",
                                                n_blocks=blocks).squeeze(-1)
        loss_plain = lambda z, t, aux: -jnp.mean(
            t * jax.nn.log_sigmoid(z) + (1 - t) * jax.nn.log_sigmoid(-z))
    else:
        loss_w = lambda o, t, v: eng.ce_softmax_loss(o, t, v)
        fwd_plain = lambda p, X: M.resnet_plain(p, X, act="relu",
                                                n_blocks=blocks)
        loss_plain = lambda z, t, aux: -jnp.mean(
            jax.nn.log_softmax(z)[jnp.arange(z.shape[0]), t])

    def step_dual(params_w, X, t, aux):
        out = M.resnet_engine(params_w, X, act="relu", n_blocks=blocks)
        L, g = jax.value_and_grad(lambda p: loss_w(
            M.resnet_engine(p, X, act="relu", n_blocks=blocks), t, aux))(params_w)
        grads, curvs = eng.split_dual_grad(g)
        return L, grads, curvs

    def step_dual_trace(params_w, X, t, aux):
        """Dual step that also returns per-layer factors for GN-SOAP:
        A = xᵀx/B (exact GGN input covariance) and rowc (bias curvature slot
        = exact E[c_i], the GGN output-side diagonal)."""
        def fwd(p):
            out, inputs = M.resnet_engine_trace(p, X, act="relu",
                                                n_blocks=blocks)
            B = float(X.shape[0])
            As = tuple((x.T @ x) / B for x in inputs)
            return loss_w(out, t, aux), As
        (L, As), g = jax.value_and_grad(fwd, has_aux=True)(params_w)
        grads, curvs = eng.split_dual_grad(g)
        factors = [{"A": A, "rowc": curvs[i]["b"]}
                   for i, A in enumerate(As)]
        return L, grads, curvs, factors

    def step_plain(params, X, t, aux):
        return jax.value_and_grad(
            lambda p: loss_plain(fwd_plain(p, X), t, aux))(params)

    def evaluate(params, X, t):
        z = fwd_plain(params, X)
        L = loss_plain(z, t, None)
        if task == "reg":
            acc = 1.0 - jnp.mean((z - t) ** 2) / (jnp.var(t) + 1e-12)  # R^2
        elif n_out == 1:
            acc = jnp.mean((z > 0).astype(t.dtype) == t)
        else:
            acc = jnp.mean(jnp.argmax(z, -1) == t)
        return L, acc

    return step_dual, step_plain, step_dual_trace, evaluate


# ------------------------------------------------------------------ train

def train(opt, dataset, n_out, blocks=2, epochs=30, bs=128, seed=0,
          use_curv=False, verbose=False, task="bce"):
    Xtr, ttr, Xte, tte = dataset
    n = len(Xtr)
    Xtr_j = jnp.asarray(Xtr); ttr_j = jnp.asarray(ttr)
    Xte_j = jnp.asarray(Xte); tte_j = jnp.asarray(tte)

    step_dual, step_plain, step_dual_trace, evaluate = make_steps(
        n_out, blocks, task)
    params = build(Xtr.shape[1], n_out, seed=seed)
    state = opt.init(params)

    needs_factors = getattr(opt, "needs_factors", False)
    step_dual_j = jax.jit(step_dual_trace if needs_factors else step_dual)
    step_plain_j = jax.jit(step_plain)
    eval_j = jax.jit(evaluate)

    hist = {"loss": [], "acc": [], "tloss": [], "tacc": [], "ms": []}
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        perm = rng.permutation(n)
        ep_t, k = 0.0, 0
        for i in range(0, n, bs):
            xb = Xtr_j[jnp.asarray(perm[i:i + bs])]
            tb = ttr_j[jnp.asarray(perm[i:i + bs])]
            aux = (jnp.asarray(
                rng.integers(0, 2, (len(xb), n_out)).astype(np.float32) * 2 - 1)
                if (n_out > 1 and task == "ce") else None)
            t0 = time.perf_counter()
            if use_curv:
                res = step_dual_j(eng.wrap_tree(params), xb, tb, aux)
                if needs_factors:
                    L, grads, curvs, factors = res
                    params, state = opt.apply(params, grads, state, curvs,
                                              factors)
                else:
                    L, grads, curvs = res
                    params, state = opt.apply(params, grads, state, curvs)
            else:
                L, grads = step_plain_j(params, xb, tb, aux)
                params, state = opt.apply(params, grads, state)
            ep_t += time.perf_counter() - t0
            k += 1
        hist["ms"].append(1000 * ep_t / k)
        Ltr, atr = eval_j(params, Xtr_j, ttr_j)
        Lte, ate = eval_j(params, Xte_j, tte_j)
        hist["loss"].append(float(Ltr)); hist["acc"].append(float(atr))
        hist["tloss"].append(float(Lte)); hist["tacc"].append(float(ate))
        if verbose:
            print(f"  ep {ep+1:3d}  loss {float(Ltr):.4f}  acc {float(atr):.4f}")
    hist["wall"] = time.perf_counter() - t_start
    return hist


# ------------------------------------------------------------------ main

def sweep(name, dataset, n_out, epochs, configs, seeds=(0, 1, 2),
          task="bce"):
    print(f"\n=== {name} ===")
    results = {}
    for cname, make_opt, use_curv in configs:
        per_lr = []
        lrs = make_opt(None)  # returns lr grid
        for lr in lrs:
            h = train(make_opt(lr), dataset, n_out, epochs=epochs, seed=0,
                      use_curv=use_curv, task=task)
            per_lr.append((h["loss"][-1] + h["tloss"][-1], lr, h))
            print(f"  {cname:14s} lr={lr:<8g} final train loss "
                  f"{h['loss'][-1]:.4f}  acc {h['acc'][-1]:.4f} "
                  f"({h['ms'][-1]:.1f} ms/step)")
        best = min(per_lr, key=lambda x: (not np.isfinite(x[0]), x[0]))
        curves = [train(make_opt(best[1]), dataset, n_out, epochs=epochs,
                        seed=s, use_curv=use_curv, task=task) for s in seeds]
        results[cname] = {"lr": best[1], "curves": curves,
                          "ms": float(np.median(curves[0]["ms"])),
                          "wall": float(np.mean([c["wall"] for c in curves]))}
    return results


def plot(results_a, results_b, results_c, fname):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for col, (res, tag) in enumerate([
            (results_a, "moons: ill-conditioned classification"),
            (results_b, "digits (real data, CE)"),
            (results_c, "ill-conditioned regression (MSE)")]):
        for name, r in res.items():
            ls = "-" if "Adam" in name or "SGD" in name else "--"
            for i, c in enumerate(r["curves"]):
                alpha = 0.25 if i else 1.0
                axes[0, col].plot(c["loss"], ls, color=None, alpha=alpha,
                                  label=f"{name} lr={r['lr']:g}" if i == 0
                                  else None)
                axes[1, col].plot(c["acc"], ls, alpha=alpha)
        axes[0, col].set_yscale("log")
        axes[0, col].set_title(f"train loss — {tag}")
        axes[0, col].set_xlabel("epoch")
        axes[1, col].set_title(f"train accuracy — {tag}")
        axes[1, col].set_xlabel("epoch")
        axes[1, col].set_ylim(0.45, 1.02)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"saved {fname}")


if __name__ == "__main__":
    moons = load_moons_illcond()
    digits = load_digits()
    reg = load_illcond_regression()

    configs = [
        ("SGD-m",      lambda lr: ([3e-2, 1e-1, 3e-1] if lr is None
                                   else O.SGDM(lr)),                 False),
        ("Adam",       lambda lr: ([3e-4, 1e-3, 3e-3] if lr is None
                                   else O.Adam(lr)),                 False),
        ("NatCurv-sqrt", lambda lr: ([1e-3, 3e-3, 1e-2, 3e-2] if lr is None
                                     else O.NatCurv(lr, mode="sqrt")), True),
        ("NatCurv-newton", lambda lr: ([1e-2, 3e-2, 1e-1, 3e-1] if lr is None
                                       else O.NatCurv(lr, mode="newton")), True),
    ]

    reg_configs = [
        ("SGD-m",      lambda lr: ([1e-2, 3e-2, 1e-1] if lr is None
                                   else O.SGDM(lr)),                 False),
        ("Adam",       lambda lr: ([3e-3, 1e-2, 3e-2] if lr is None
                                   else O.Adam(lr)),                 False),
        ("NatCurv-sqrt", lambda lr: ([1e-2, 3e-2, 1e-1] if lr is None
                                     else O.NatCurv(lr, mode="sqrt")), True),
        ("NatCurv-newton", lambda lr: ([3e-2, 1e-1, 3e-1] if lr is None
                                       else O.NatCurv(lr, mode="newton")), True),
    ]

    res_moons = sweep("moons (ill-conditioned classification)", moons, n_out=1,
                      epochs=40, configs=configs)
    res_digits = sweep("digits (10 classes, B=1437)", digits, n_out=10,
                       epochs=60, configs=configs, task="ce")
    res_reg = sweep("ill-conditioned regression (MSE)", reg, n_out=1,
                    epochs=40, configs=reg_configs, task="reg")

    print("\n=== summary (best LR per optimizer, 3 seeds) ===")
    print(f"{'dataset':22s} {'optimizer':15s} {'lr':9s} {'ms/step':8s} "
          f"{'final loss':>11s} {'final acc':>10s}")
    for tag, res in [("moons-illcond", res_moons), ("digits", res_digits),
                     ("reg-illcond", res_reg)]:
        for name, r in res.items():
            fl = np.mean([c["loss"][-1] for c in r["curves"]])
            fa = np.mean([c["acc"][-1] for c in r["curves"]])
            print(f"{tag:22s} {name:15s} {r['lr']:<9g} {r['ms']:<8.1f} "
                  f"{fl:>11.4f} {fa:>10.4f}")

    plot(res_moons, res_digits, res_reg, "convergence.png")
