"""Correctness tests for the Native Curvature Engine.

Every engine quantity is checked against an INDEPENDENT reference built from
plain autodiff (jax.grad / jax.jacrev on unwrapped models):

 1. gradients          == jax.grad on the plain model        (exact)
 2. curvature, 1 dense == exact Hessian diagonal (jax.hessian; exact because
                          the MSE loss is quadratic in W)
 3. curvature, dense->tanh == exact diag GGN  (1/B)ΣΣ(∂z/∂θ)²  (exact: all
                          Jacobians factor through a diagonal map)
 4. curvature, 2 dense: top layer == exact diag GGN; first layer == the
                          "diagonal-path" reference (cross terms dropped BY
                          DESIGN — pins down the approximation's semantics)
 5. curvature >= 0;  input-feature scaling: scaling input j by s scales row j
    of the first-layer curvature by s²
 6. BCE: exact diag-GGN formula; CE: MC seed unbiased for diag-H GGN
 7. memory: engine jaxpr has no [B, params]-sized intermediates, while the
    vmap-per-sample reference does
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # tight tolerances for exact tests
import jax.numpy as jnp

import engine as eng
import models as M

PASS = []


def check(name, ok, detail=""):
    PASS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")


# ---------------------------------------------------------------- test 1
def test_gradients_match_autodiff():
    print("test 1: gradients == plain jax.grad (ResNet MLP, MSE & CE)")
    key = jax.random.PRNGKey(0)
    p = M.init_params([6, 8, 8, 8, 3], key)  # head + block(2 layers) + out
    pw = eng.wrap_tree(p)
    X = jax.random.normal(jax.random.PRNGKey(1), (5, 6))
    t_mse = jax.random.normal(jax.random.PRNGKey(2), (5, 3))
    t_cls = jnp.array([0, 1, 2, 0, 1])
    v = jax.random.bernoulli(jax.random.PRNGKey(3), 0.5, (5, 3)) * 2 - 1

    fwd_w = lambda p: M.resnet_engine(p, X, act="tanh", n_blocks=1)
    fwd_p = lambda p: M.resnet_plain(p, X, act="tanh", n_blocks=1)

    for name, loss_w, loss_p in [
        ("MSE", lambda o: eng.mse_loss(o, t_mse),
         lambda z: jnp.mean(0.5 * jnp.sum((z - t_mse) ** 2, axis=1))),
        ("CE", lambda o: eng.ce_softmax_loss(o, t_cls, v),
         lambda z: -jnp.mean(jax.nn.log_softmax(z)[jnp.arange(5), t_cls])),
    ]:
        g_eng = eng.split_dual_grad(
            jax.grad(lambda p: loss_w(fwd_w(p)))(pw))[0]
        g_ref = jax.grad(lambda p: loss_p(fwd_p(p)))(p)
        diffs = jax.tree_util.tree_map(lambda a, b: float(jnp.abs(a - b).max()),
                                       g_eng, g_ref)
        worst = max(jax.tree_util.tree_leaves(diffs))
        check(f"gradient ({name})", worst < 1e-10, f"max|Δ|={worst:.2e}")


# ---------------------------------------------------------------- test 2
def test_single_dense_exact_hessian():
    print("test 2: single dense + MSE -> curvature == exact Hessian diagonal")
    key = jax.random.PRNGKey(0)
    p = M.init_params([4, 3], key)
    pw = eng.wrap_tree(p)
    X = jax.random.normal(jax.random.PRNGKey(1), (6, 4))

    fwd = lambda p_: eng.dense(eng.wrap_input(X), p_[0]["W"], p_[0]["b"])
    L, g = jax.value_and_grad(
        lambda p_: eng.mse_loss(fwd(p_), jnp.zeros((6, 3))))(pw)
    g, c = eng.split_dual_grad(g)

    # reference 1: closed form  cW[i,j] = mean_n X[n,j]²,  cb = 1
    cW_ref = jnp.tile((X * X).mean(0), (p[0]["W"].shape[0], 1))

    # reference 2: exact Hessian diagonal via jax.hessian (quadratic loss)
    def plain_loss(pp):
        Y = X @ pp[0]["W"].T + pp[0]["b"]
        return jnp.mean(0.5 * jnp.sum(Y ** 2, axis=1))
    H = jax.hessian(plain_loss)(p)[0]["W"][0]["W"]   # [m,n,m,n]
    m_, n_ = p[0]["W"].shape
    Hdiag = jnp.stack([jnp.stack([H[i, j, i, j] for j in range(n_)], 0)
                       for i in range(m_)], 0)
    d1 = float(jnp.abs(c[0]["W"] - cW_ref).max())
    d2 = float(jnp.abs(c[0]["W"] - Hdiag).max())
    check("closed form  cW = mean(X²)", d1 < 1e-12, f"max|Δ|={d1:.2e}")
    check("jax.hessian diagonal", d2 < 1e-12, f"max|Δ|={d2:.2e}")
    check("bias curvature == 1", float(jnp.abs(c[0]["b"] - 1).max()) < 1e-12)


# ---------------------------------------------------------------- test 3
def test_dense_tanh_exact_ggn():
    print("test 3: tanh MLP + MSE -> curvature == exact diag GGN")
    key = jax.random.PRNGKey(0)
    p = M.init_params([4, 5, 2], key)
    pw = eng.wrap_tree(p)
    X = jax.random.normal(jax.random.PRNGKey(1), (6, 4))

    fwd_w = lambda p_: M.mlp_engine(p_, X, act="tanh")
    L, g = jax.value_and_grad(lambda p_: eng.mse_loss(fwd_w(p_), jnp.zeros((6, 2))))(pw)
    g, c = eng.split_dual_grad(g)

    # exact diag GGN: (1/B) Σ_n Σ_d (∂z_nd/∂θ)²   (MSE seed c=1)
    def plain(pp, x):
        h = jnp.tanh(x @ pp[0]["W"].T + pp[0]["b"])
        return jnp.tanh(h @ pp[1]["W"].T + pp[1]["b"])
    J = jax.vmap(jax.jacrev(lambda pp, x: plain(pp, x)), in_axes=(None, 0))(p, X)
    ref = jax.tree_util.tree_map(lambda Jl: (Jl ** 2).sum(1).mean(0), J)

    for i in range(2):
        dw = float(jnp.abs(c[i]["W"] - ref[i]["W"]).max())
        db = float(jnp.abs(c[i]["b"] - ref[i]["b"]).max())
        check(f"layer {i}", max(dw, db) < 1e-10, f"W:{dw:.2e} b:{db:.2e}")


# ---------------------------------------------------------------- test 4
def test_two_dense_diagpath():
    print("test 4: 2 dense layers — layer2 exact GGN, layer1 = diag-path")
    key = jax.random.PRNGKey(0)
    p = M.init_params([4, 5, 2], key)   # u = dense1(X), z = dense2(u)
    pw = eng.wrap_tree(p)
    X = jax.random.normal(jax.random.PRNGKey(1), (6, 4))

    fwd = lambda p_: eng.dense(
        eng.dense(eng.wrap_input(X), p_[0]["W"], p_[0]["b"]),
        p_[1]["W"], p_[1]["b"])
    L, g = jax.value_and_grad(lambda p_: eng.mse_loss(fwd(p_), jnp.zeros((6, 2))))(pw)
    g, c = eng.split_dual_grad(g)

    U = X @ p[0]["W"].T + p[0]["b"]                       # [B, m]
    W2, b2 = p[1]["W"], p[1]["b"]

    # per-sample jacobians via the closure pattern (one arg, vmap over data)
    A = jax.vmap(jax.jacrev(lambda u: u @ W2.T + b2))(U)     # [B,d,m] dz/du
    B = jax.vmap(lambda x_: jax.jacrev(
        lambda p_: x_ @ p_["W"].T + p_["b"])(p[0]))(X)        # {'W','b'} du/dθ1
    J2W = jax.vmap(lambda u_: jax.jacrev(
        lambda W_: u_ @ W_.T + b2)(W2))(U)                    # [B,d,2,5]
    J2b = jax.vmap(lambda u_: jax.jacrev(
        lambda b_: u_ @ W2.T + b_)(b2))(U)                    # [B,d,2]
    A2 = (A ** 2).sum(1)                                      # [B, m]

    # diag-path reference: mean_n Σ_k A2[n,k] · (∂u_k/∂θ)²
    ref_W1 = jnp.mean(
        jnp.sum(A2[:, :, None, None] * B["W"] ** 2, axis=1), axis=0)
    ref_b1 = jnp.mean(A2 * jnp.diagonal(B["b"], axis1=1, axis2=2) ** 2, axis=0)

    # top layer: exact diag GGN = (1/B) Σ_n Σ_d (∂z/∂θ2)²
    ref_W2 = (J2W ** 2).sum(1).mean(0)             # [2, 5]
    ref_b2 = (J2b ** 2).sum(1).mean(0)             # [2]

    d1w = float(jnp.abs(c[0]["W"] - ref_W1).max())
    d1b = float(jnp.abs(c[0]["b"] - ref_b1).max())
    d2w = float(jnp.abs(c[1]["W"] - ref_W2).max())
    d2b = float(jnp.abs(c[1]["b"] - ref_b2).max())
    check("layer 1 == diag-path reference", max(d1w, d1b) < 1e-10,
          f"W:{d1w:.2e} b:{d1b:.2e}")
    check("layer 2 == exact diag GGN", max(d2w, d2b) < 1e-10,
          f"W:{d2w:.2e} b:{d2b:.2e}")


# ---------------------------------------------------------------- test 5
def test_positivity_and_scale():
    print("test 5: positivity, BCE exactness, input-scale property")
    key = jax.random.PRNGKey(0)
    p = M.init_params([5, 7, 7, 7, 2], key)
    pw = eng.wrap_tree(p)
    X = jax.random.normal(jax.random.PRNGKey(1), (8, 5))

    fwd = lambda p_: M.resnet_engine(p_, X, act="tanh", n_blocks=1)
    g, c = eng.split_dual_grad(
        jax.grad(lambda p_: eng.mse_loss(fwd(p_), jnp.zeros((8, 2))))(pw))
    ok = all(float(leaf.min()) >= -1e-12
             for leaf in jax.tree_util.tree_leaves(c))
    check("curvature >= 0 (MSE, tanh ResNet)", ok)

    # BCE: exact diag GGN for a single dense layer
    t = jax.random.bernoulli(jax.random.PRNGKey(2), 0.5, (8,)).astype(float)
    p1 = M.init_params([5, 1], key)
    pw1 = eng.wrap_tree(p1)
    fwd_b = lambda p_: eng.dense(eng.wrap_input(X), p_[0]["W"], p_[0]["b"])
    g1, c1 = eng.split_dual_grad(
        jax.grad(lambda p_: eng.bce_loss(fwd_b(p_), t))(pw1))
    z = (X @ p1[0]["W"].T + p1[0]["b"]).squeeze(-1)
    s = jax.nn.sigmoid(z)
    refW = ((s * (1 - s))[:, None] * X ** 2).mean(0)[None, :]
    check("BCE exact diag-GGN", float(jnp.abs(c1[0]["W"] - refW).max()) < 1e-10)

    # scaling input feature j by s scales row j of layer-1 curvature by s²
    # (needs a LINEAR first layer: with an activation right after, the
    #  propagated curvature C_u itself would change with the scaling)
    pl = M.init_params([5, 6, 2], key)
    pwl = eng.wrap_tree(pl)
    fwd_l = lambda p_, X_: eng.dense(
        eng.dense(eng.wrap_input(X_), p_[0]["W"], p_[0]["b"]),
        p_[1]["W"], p_[1]["b"])
    _, cl = eng.split_dual_grad(
        jax.grad(lambda p_: eng.mse_loss(fwd_l(p_, X), jnp.zeros((8, 2))))(pwl))
    s_scale = 3.0
    Xs = X.at[:, 2].set(X[:, 2] * s_scale)
    _, cs = eng.split_dual_grad(
        jax.grad(lambda p_: eng.mse_loss(fwd_l(p_, Xs), jnp.zeros((8, 2))))(pwl))
    ratio = cs[0]["W"][:, 2] / jnp.maximum(cl[0]["W"][:, 2], 1e-30)
    check("feature scaling -> curvature scales s²",
          float(jnp.abs(ratio - s_scale ** 2).max()) < 1e-8,
          f"ratios in [{float(ratio.min()):.3f}, {float(ratio.max()):.3f}]")


# ---------------------------------------------------------------- test 6
def test_ce_mc_seed():
    print("test 6: CE curvature seed unbiased for diag-H GGN (MC)")
    p = [{"W": jax.random.normal(jax.random.PRNGKey(1), (3, 5)) * 0.5,
          "b": jnp.zeros(3)}]
    pw = eng.wrap_tree(p)
    X = jax.random.normal(jax.random.PRNGKey(2), (8, 5))
    t = jnp.array([0, 1, 2, 0, 1, 2, 0, 1])

    fwd = lambda p_, v_: eng.ce_softmax_loss(
        eng.dense(eng.wrap_input(X), p_[0]["W"], p_[0]["b"]), t, v_)

    n_mc, acc = 400, None
    for i in range(n_mc):
        v = jax.random.bernoulli(jax.random.PRNGKey(100 + i), 0.5, (8, 3)) * 2 - 1
        _, c = eng.split_dual_grad(jax.grad(lambda p_: fwd(p_, v))(pw))
        acc = c[0]["W"] if acc is None else acc + c[0]["W"]
    c_mc = acc / n_mc

    pr = jax.nn.softmax(X @ p[0]["W"].T, axis=1)
    Jz = jax.vmap(lambda x_: jax.jacrev(
        lambda W_: x_ @ W_.T)(p[0]["W"]))(X)       # [B,d,n] dz/dW
    ref = jnp.mean(
        jnp.einsum("bd,bdmn->bmn", pr * (1 - pr), Jz ** 2), axis=0)
    rel = float(jnp.abs(c_mc - ref).max() / jnp.abs(ref).max())
    check("CE MC seed unbiased", rel < 0.05, f"rel err={rel:.1%}")


# ---------------------------------------------------------------- test 7
def test_no_batch_by_params_intermediates():
    print("test 7: jaxpr memory — no [B, params] intermediates in the engine")
    B, sizes = 64, [20, 64, 64, 10]
    p = M.init_params(sizes, jax.random.PRNGKey(0))
    pw = eng.wrap_tree(p)
    X = jax.random.normal(jax.random.PRNGKey(1), (B, 20))
    t = jax.random.randint(jax.random.PRNGKey(2), (B,), 0, 10)
    v = jax.random.bernoulli(jax.random.PRNGKey(3), 0.5, (B, 10)) * 2 - 1

    def engine_step(p_):
        return jax.value_and_grad(
            lambda pp: eng.ce_softmax_loss(
                M.mlp_engine(pp, X, act="relu"), t, v))(p_)

    def per_sample_sqgrad(p_):
        def one(pp, x, ti):
            g = jax.grad(
                lambda q: -jax.nn.log_softmax(
                    M.mlp_plain(q, x, act="relu"))[ti])(pp)
            return jax.tree_util.tree_map(lambda a: a ** 2, g)
        return jax.vmap(one, in_axes=(None, 0, 0))(p_, X, t)

    def max_intermediate(jx):
        tot = 0
        for eqn in jx.eqns:
            for ov in eqn.outvars:
                av = getattr(ov, "aval", None)
                if av is None:
                    continue
                tot = max(tot, int(np.prod(av.shape))
                          * np.dtype(av.dtype).itemsize)
        return tot

    jx_e = jax.make_jaxpr(engine_step)(pw)
    jx_r = jax.make_jaxpr(per_sample_sqgrad)(p)
    e, r = max_intermediate(jx_e), max_intermediate(jx_r)
    check("engine max intermediate << vmap per-sample",
          r // e > 8,
          f"engine={e/1e3:.1f}KB vs vmap={r/1e3:.1f}KB (ratio {r//e}x)")


if __name__ == "__main__":
    test_gradients_match_autodiff()
    test_single_dense_exact_hessian()
    test_dense_tanh_exact_ggn()
    test_two_dense_diagpath()
    test_positivity_and_scale()
    test_ce_mc_seed()
    test_no_batch_by_params_intermediates()
    n_ok = sum(ok for _, ok in PASS)
    print(f"\n{n_ok}/{len(PASS)} checks passed")
    raise SystemExit(0 if n_ok == len(PASS) else 1)
