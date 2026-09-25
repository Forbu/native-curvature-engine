"""
Native Curvature Engine (JAX, custom_vjp rules)
================================================

Backpropagation that carries TWO signals in a single backward sweep:

  * the usual gradient  g = dL/dθ          -> 'val' slot of the cotangent
  * a curvature signal  c ≈ diag(GGN)      -> 'curv' slot of the cotangent

The trick that makes this work with jax.grad:
  * every parameter is wrapped as   {'val': θ, 'curv': zeros_like(θ)}
  * every activation is wrapped as  {'y': a,   'aux': zeros_like(a)}
  * each layer is a jax.custom_vjp primitive whose backward rule receives a
    cotangent with two slots and returns two-slot cotangents for its inputs.

Because the cotangent pytree must mirror the primal pytree, the curvature
signal "rides along" for free: one call of jax.value_and_grad produces a
gradient tree where tree['...']['curv'] is the layer's diagonal curvature.

Math per layer (batch X [B,n], out Y [B,m], W [m,n]):
  gradient :  gW = Gyᵀ X / B,   gb = mean(Gy),  Gx = Gy W
  curvature:  cW = Cyᵀ (X⊙X) / B   ("X² trick"),  cb = mean(Cy),  Cx = Cy (W⊙W)

  cW[i,j] = E_n[ Cy[n,i] · X[n,j]² ]  = diagonal of the Gauss-Newton block of
  W under the *diagonal-propagation approximation* (off-diagonal cross terms
  between output units are dropped — the "no cross-talk" approximation).

Curvature seeding at the loss (per-sample loss-Hessian diagonal; the batch
average is applied inside the layer rules):
  * MSE           : c = 1                       (exact, deterministic)
  * sigmoid+BCE   : c = σ(z)(1-σ(z))            (exact, deterministic)
  * softmax+CE    : c = (Hv)⊙v, v ~ ±1          (Hutchinson MC, 1 sample)

What this is NOT: the exact Hessian diagonal. It is the exact diagonal of a
diagonal-propagated Gauss-Newton approximation (a.k.a. diagonal curvature
backprop, Becker & LeCun 1988; "HBP", Botev et al. 2019). PSD by
construction for the MSE/BCE seeds — it can never divide you uphill.

Limitation: layers are custom_vjp primitives, so forward-mode autodiff and
vmap through them are not supported. Reverse-mode + jit (what training uses)
work fine, and the batch dimension is an explicit leading axis, so no vmap
is ever needed — that is precisely how the [B, params] memory wall is
avoided.
"""

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# parameter / activation wrapping
# ---------------------------------------------------------------------------

def wrap_tree(tree):
    """θ -> {'val': θ, 'curv': zeros_like(θ)} (recursively)."""
    return jax.tree_util.tree_map(
        lambda a: {"val": a, "curv": jnp.zeros_like(a)}, tree
    )


_is_dual = lambda x: isinstance(x, dict) and "val" in x and "curv" in x


def unwrap_val(tree):
    return jax.tree_util.tree_map(lambda d: d["val"], tree,
                                  is_leaf=_is_dual)


def unwrap_curv(tree):
    return jax.tree_util.tree_map(lambda d: d["curv"], tree,
                                  is_leaf=_is_dual)


def wrap_input(X):
    """Plain data array -> wrapped activation (the aux slot is a carrier)."""
    return {"y": X, "aux": jnp.zeros_like(X)}


def split_dual_grad(grad_tree):
    """jax.grad output on wrapped params -> (grads, curvatures)."""
    return unwrap_val(grad_tree), unwrap_curv(grad_tree)


# ---------------------------------------------------------------------------
# Dense layer:  Y = X @ W.T + b     (the core custom rule)
# ---------------------------------------------------------------------------

def _dense_fwd(x, W, b):
    Y = x["y"] @ W["val"].T + b["val"]
    out = {"y": Y, "aux": Y}               # aux content is irrelevant
    return out, (x["y"], W["val"])         # residuals: X and W


def _dense_bwd(res, ct):
    X, W = res
    Gy, Cy = ct["y"], ct["aux"]            # gradient & curvature w.r.t. Y
    B = X.shape[0]

    # ---- gradient (identical to textbook backprop). NOTE: Gy already
    #      contains the 1/B of the batch-mean loss (put there by the loss
    #      seed), so no extra division happens here.
    gW = Gy.T @ X
    gb = jnp.sum(Gy, axis=0)
    Gx = Gy @ W

    # ---- curvature ("X² trick"): the batch dim dies inside the GEMM.
    #      Cy carries NO 1/B (it is the per-sample loss-Hessian diagonal),
    #      so the batch average is taken HERE.
    cW = (Cy.T @ (X * X)) / B              # cW[i,j] = mean_n[ Cy[n,i] X[n,j]² ]
    cb = jnp.mean(Cy, axis=0)
    Cx = Cy @ (W * W)                      # diag of  Wᵀ diag(Cy) W

    return (
        {"y": Gx, "aux": Cx},              # cotangent for the input activation
        {"val": gW, "curv": cW},           # cotangent for W
        {"val": gb, "curv": cb},           # cotangent for b
    )


@jax.custom_vjp
def dense(x, W, b):
    return _dense_fwd(x, W, b)[0]


dense.defvjp(_dense_fwd, _dense_bwd)


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def relu(x):
    """Plain ReLU: gradient uses 1[x>0] and (f')² = 1[x>0] coincide, so the
    standard autodiff rule is already the correct curvature rule."""
    return jax.tree_util.tree_map(jax.nn.relu, x)


def _tanh_fwd(x):
    Y = jnp.tanh(x["y"])
    return {"y": Y, "aux": Y}, x["y"]


def _tanh_bwd(res, ct):
    g, c = ct["y"], ct["aux"]
    d = 1.0 - jnp.tanh(res) ** 2
    return ({"y": g * d, "aux": c * d * d},)    # curvature scales with (f')²


@jax.custom_vjp
def tanh_act(x):
    return _tanh_fwd(x)[0]


tanh_act.defvjp(_tanh_fwd, _tanh_bwd)


def residual(x, y):
    """x + y: cotangents of '+' add in BOTH slots == branch-sum of curvatures
    (diagonal approx: cross-covariance between branches is dropped)."""
    return jax.tree_util.tree_map(lambda a, b: a + b, x, y)


# ---------------------------------------------------------------------------
# Losses (custom rules that SEED the two signals)
# ---------------------------------------------------------------------------

def _mse_fwd(out, t):
    r = out["y"] - t
    L = jnp.mean(0.5 * jnp.sum(r * r, axis=1))
    return L, (out["y"], t)


def _mse_bwd(res, ct):
    Y, t = res
    r = Y - t
    g = r / Y.shape[0]                 # dL/dy of the batch-mean loss
    c = jnp.ones_like(Y)               # diag Hessian of the *per-sample* loss
    return ({"y": g, "aux": c}, None)   # (no 1/B: rules average over batch)


@jax.custom_vjp
def mse_loss(out, t):
    return _mse_fwd(out, t)[0]


mse_loss.defvjp(_mse_fwd, _mse_bwd)


def _bce_fwd(out, t):
    z = out["y"].squeeze(-1)
    # numerically stable:  ℓ = softplus(z) − t·z   (never touches log(0);
    # the naive  -[t·log σ + (1-t)·log(1-σ)]  NaNs under XLA fusion once
    # logits saturate, e.g. |z| > ~80 from badly scaled inputs)
    L = jnp.mean(jax.nn.softplus(z) - t * z)
    return L, (z, t)


def _bce_bwd(res, ct):
    z, t = res
    p = jax.nn.sigmoid(z)
    g = ((p - t) / z.shape[0])[:, None]      # [B,1]
    c = (p * (1.0 - p))[:, None]             # exact diag of the loss Hessian
    return ({"y": g, "aux": c}, None)


@jax.custom_vjp
def bce_loss(out, t):
    return _bce_fwd(out, t)[0]


bce_loss.defvjp(_bce_fwd, _bce_bwd)


def _ce_fwd(out, t, v):
    Y, v = out["y"], v
    L = -jnp.mean(jnp.take_along_axis(
        jax.nn.log_softmax(Y), t[:, None].astype(int), axis=1))
    return L, (Y, t, v)


def _ce_bwd(res, ct):
    Y, t, v = res
    onehot = jax.nn.one_hot(t, Y.shape[-1])
    p = jax.nn.softmax(Y, axis=1)
    g = (p - onehot) / Y.shape[0]
    # Hutchinson MC of diag(H): c_d = (Hv)_d · v_d with H = diag(p) - p pᵀ
    Hv = p * v - p * jnp.sum(p * v, axis=1, keepdims=True)
    c = Hv * v
    return ({"y": g, "aux": c}, None, None)  # t, v are probes: zero cotangent


@jax.custom_vjp
def ce_softmax_loss(out, t, v):
    """v: Rademacher (+/-1) matrix [B, d] — one-sample MC curvature seed."""
    return _ce_fwd(out, t, v)[0]


ce_softmax_loss.defvjp(_ce_fwd, _ce_bwd)
