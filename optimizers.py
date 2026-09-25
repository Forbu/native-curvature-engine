"""Hand-rolled optimizers (pytree-functional, identical for fairness).

NatCurv consumes the engine's native curvature: like Adam, but the v_t
EMA of squared gradients is replaced by an EMA of the exact diagonal
Gauss-Newton curvature propagated by the engine.

  Adam    : v_t = EMA[g²]            (proxy curvature)
  NatCurv : d_t = EMA[diag GGN]      (true-ish curvature, PSD)

Both use bias correction; NatCurv floors d_t per tensor (dead ReLU units
give exactly-zero curvature) and supports two step rules:

  'sqrt'   :  step = m̂ / sqrt(d̂)    (Adam-like units; robust default)
  'newton' :  step = m̂ / d̂          (Newton-like units, scale-invariant)
"""

import jax
import jax.numpy as jnp


class SGDM:
    def __init__(self, lr, mu=0.9):
        self.lr, self.mu = lr, mu

    def init(self, params):
        return {"v": jax.tree_util.tree_map(jnp.zeros_like, params)}

    def apply(self, params, grads, state, curvs=None):
        v = jax.tree_util.tree_map(
            lambda v_, g: self.mu * v_ + g, state["v"], grads)
        out = jax.tree_util.tree_map(
            lambda p_, v_: p_ - self.lr * v_, params, v)
        return out, {"v": v}


class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def init(self, params):
        return {"t": 0,
                "m": jax.tree_util.tree_map(jnp.zeros_like, params),
                "v": jax.tree_util.tree_map(jnp.zeros_like, params)}

    def apply(self, params, grads, state, curvs=None):
        t = state["t"] + 1
        m = jax.tree_util.tree_map(
            lambda m_, g: self.b1 * m_ + (1 - self.b1) * g, state["m"], grads)
        v = jax.tree_util.tree_map(
            lambda v_, g: self.b2 * v_ + (1 - self.b2) * g * g, state["v"],
            grads)
        mh = jax.tree_util.tree_map(lambda x: x / (1 - self.b1 ** t), m)
        vh = jax.tree_util.tree_map(lambda x: x / (1 - self.b2 ** t), v)
        out = jax.tree_util.tree_map(
            lambda p_, m_, v_: p_ - self.lr * m_ / (jnp.sqrt(v_) + self.eps),
            params, mh, vh)
        return out, {"t": t, "m": m, "v": v}


class NatCurv:
    def __init__(self, lr, b1=0.9, b2=0.999, mode="sqrt",
                 floor_frac=0.05, min_curv=1e-8, clip_rms=5.0,
                 clip_ratio=0.02, eps=1e-12):
        self.lr, self.b1, self.b2 = lr, b1, b2
        self.mode, self.floor_frac = mode, floor_frac
        self.min_curv, self.clip_rms = min_curv, clip_rms
        self.clip_ratio, self.eps = clip_ratio, eps

    def init(self, params):
        return {"t": 0,
                "m": jax.tree_util.tree_map(jnp.zeros_like, params),
                "d": jax.tree_util.tree_map(jnp.zeros_like, params)}

    def apply(self, params, grads, state, curvs=None):
        assert curvs is not None, "NatCurv needs the engine's curvature"
        t = state["t"] + 1
        m = jax.tree_util.tree_map(
            lambda m_, g: self.b1 * m_ + (1 - self.b1) * g, state["m"], grads)
        d = jax.tree_util.tree_map(
            lambda d_, c: self.b2 * d_ + (1 - self.b2) * c, state["d"], curvs)
        mh = jax.tree_util.tree_map(lambda x: x / (1 - self.b1 ** t), m)
        dh = jax.tree_util.tree_map(lambda x: x / (1 - self.b2 ** t), d)

        def step(dh_leaf, mh_leaf, p_leaf):
            # relative floor (dead units) + absolute floor (saturated loss
            # curvature -> 0, e.g. BCE with extreme logits)
            floor = self.floor_frac * jnp.mean(dh_leaf) + self.min_curv
            d_safe = jnp.maximum(dh_leaf, floor) + self.eps
            u = (mh_leaf / d_safe if self.mode == "newton"
                 else mh_leaf / jnp.sqrt(d_safe))
            # trust clip 1: cap update RMS at clip_rms x gradient RMS
            rms_u = jnp.sqrt(jnp.mean(u * u)) + 1e-30
            cap = self.clip_rms * jnp.sqrt(jnp.mean(mh_leaf * mh_leaf)) + 1e-30
            u = u * jnp.minimum(1.0, cap / rms_u)
            # trust clip 2 (Sophia-style): RMS(Δθ) <= clip_ratio * RMS(θ)
            # protects against curvature collapse death-spirals
            rms_p = jnp.sqrt(jnp.mean(p_leaf * p_leaf)) + 1e-30
            max_u = self.clip_ratio * rms_p / self.lr
            return u * jnp.minimum(1.0, max_u / (jnp.sqrt(jnp.mean(u * u)) + 1e-30))

        upd = jax.tree_util.tree_map(step, dh, mh, params)
        out = jax.tree_util.tree_map(lambda p_, u: p_ - self.lr * u,
                                     params, upd)
        return out, {"t": t, "m": m, "d": d}


# ---------------------------------------------------------------------------
# "Better v_t inside AdamW": replace the second moment E[g²] by the engine's
# native curvature (diag GGN).  Pure swap, and a hybrid that never collapses.
# ---------------------------------------------------------------------------

class AdamCurv:
    """Adam with v_t = EMA[curvature] instead of EMA[g²]."""
    needs_curv = True

    def __init__(self, lr, b1=0.9, b2=0.999, floor_frac=0.05, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.floor_frac = floor_frac

    def init(self, params):
        return {"t": 0,
                "m": jax.tree_util.tree_map(jnp.zeros_like, params),
                "v": jax.tree_util.tree_map(jnp.zeros_like, params)}

    def apply(self, params, grads, state, curvs=None, factors=None):
        t = state["t"] + 1
        m = jax.tree_util.tree_map(
            lambda m_, g: self.b1 * m_ + (1 - self.b1) * g, state["m"], grads)
        v = jax.tree_util.tree_map(
            lambda v_, c: self.b2 * v_ + (1 - self.b2) * c, state["v"], curvs)
        mh = jax.tree_util.tree_map(lambda x: x / (1 - self.b1 ** t), m)
        vh = jax.tree_util.tree_map(lambda x: x / (1 - self.b2 ** t), v)

        def upd(p_, mh_, vh_):
            floor = self.floor_frac * jnp.mean(vh_)
            v_safe = jnp.maximum(vh_, floor) + self.eps
            return p_ - self.lr * mh_ / jnp.sqrt(v_safe)

        out = jax.tree_util.tree_map(upd, params, mh, vh)
        return out, {"t": t, "m": m, "v": v}


class AdamHybrid:
    """v_t = EMA[ max(g², curvature) ] — Adam's stability, curvature where it
    is informative. Never collapses (g² > 0 while any sample is wrong)."""
    needs_curv = True

    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def init(self, params):
        return {"t": 0,
                "m": jax.tree_util.tree_map(jnp.zeros_like, params),
                "v": jax.tree_util.tree_map(jnp.zeros_like, params)}

    def apply(self, params, grads, state, curvs=None, factors=None):
        t = state["t"] + 1
        m = jax.tree_util.tree_map(
            lambda m_, g: self.b1 * m_ + (1 - self.b1) * g, state["m"], grads)
        v = jax.tree_util.tree_map(
            lambda v_, g, c: self.b2 * v_ + (1 - self.b2) * jnp.maximum(g * g, c),
            state["v"], grads, curvs)
        mh = jax.tree_util.tree_map(lambda x: x / (1 - self.b1 ** t), m)
        vh = jax.tree_util.tree_map(lambda x: x / (1 - self.b2 ** t), v)
        out = jax.tree_util.tree_map(
            lambda p_, m_, v_: p_ - self.lr * m_ / (jnp.sqrt(v_) + self.eps),
            params, mh, vh)
        return out, {"t": t, "m": m, "v": v}


# ---------------------------------------------------------------------------
# SOAP-style optimizers: Adam in the eigenbasis of a per-matrix preconditioner
# ---------------------------------------------------------------------------

def _adam_core(g, m, v, t, b1, b2, eps):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    return mh / (jnp.sqrt(vh) + eps), m, v


class SOAP:
    """One-sided SOAP (input side): rotate gradients into the eigenbasis of
    the gradient-covariance preconditioner G_R = EMA[gᵀg], run Adam there,
    rotate back. Biases: plain Adam. (Simplified but faithful variant of
    Ros & Liu 2024, arXiv:2409.11321.)"""
    needs_curv = False
    needs_factors = False

    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, T=10):
        self.lr, self.b1, self.b2, self.eps, self.T = lr, b1, b2, eps, T

    def init(self, params):
        st = []
        for l in params:
            n_in = l["W"].shape[1]
            st.append({"W": {"m": jnp.zeros_like(l["W"]),
                             "v": jnp.zeros_like(l["W"]),
                             "G": jnp.zeros((n_in, n_in)) + 1e-4 * jnp.eye(n_in),
                             "Q": None},
                       "b": {"m": jnp.zeros_like(l["b"]),
                             "v": jnp.zeros_like(l["b"])}})
        return {"t": 0, "layers": st}

    def apply(self, params, grads, state, curvs=None, factors=None):
        t = state["t"] + 1
        new_layers, out_params = [], []
        for l, gl, sl in zip(params, grads, state["layers"]):
            W, gW = l["W"], gl["W"]
            s = sl["W"]
            G = self.b2 * s["G"] + (1 - self.b2) * (gW.T @ gW)
            Q = s["Q"] if s["Q"] is not None else jnp.eye(W.shape[1])
            if (t - 1) % self.T == 0:
                _, Q = jnp.linalg.eigh(G)
            gr = gW @ Q                       # rotate into eigenbasis
            du, m, v = _adam_core(gr, s["m"], s["v"], t, self.b1, self.b2,
                                  self.eps)
            dW = du @ Q.T                     # rotate back
            db, mb, vb = _adam_core(gl["b"], sl["b"]["m"], sl["b"]["v"], t,
                                    self.b1, self.b2, self.eps)
            out_params.append({"W": W - self.lr * dW, "b": l["b"] - self.lr * db})
            new_layers.append({"W": {"m": m, "v": v, "G": G, "Q": Q},
                               "b": {"m": mb, "v": vb}})
        return out_params, {"t": t, "layers": new_layers}


class GNSoap:
    """SOAP with NATIVE second-order factors from the curvature engine:

      * input-side preconditioner: A = E[x xᵀ]  (exact GGN/K-FAC input
        factor — comes from the same forward pass, NOT from gradients)
      * output-side: exact diagonal curvature E[c_i] per row (the bias
        curvature slot of the engine), used as a row-wise Newton scale.

    Same eigenbasis-Adam machinery as SOAP; only the information source
    changes: gradient covariance  ->  native Gauss-Newton factors."""
    needs_curv = True
    needs_factors = True

    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, T=10, mode="sqrt",
                 floor_frac=0.05, clip_ratio=0.02):
        self.lr, self.b1, self.b2, self.eps, self.T = lr, b1, b2, eps, T
        self.mode, self.floor_frac = mode, floor_frac
        self.clip_ratio = clip_ratio

    def init(self, params):
        st = []
        for l in params:
            n_in = l["W"].shape[1]
            st.append({"W": {"m": jnp.zeros_like(l["W"]),
                             "v": jnp.zeros_like(l["W"]),
                             "A": jnp.zeros((n_in, n_in)) + 1e-4 * jnp.eye(n_in),
                             "d": jnp.zeros(l["W"].shape[0]),
                             "Q": None},
                       "b": {"m": jnp.zeros_like(l["b"]),
                             "v": jnp.zeros_like(l["b"])}})
        return {"t": 0, "layers": st}

    def apply(self, params, grads, state, curvs=None, factors=None):
        t = state["t"] + 1
        new_layers, out_params = [], []
        for l, gl, sl, fa in zip(params, grads, state["layers"], factors):
            W, gW = l["W"], gl["W"]
            s = sl["W"]
            A = self.b2 * s["A"] + (1 - self.b2) * fa["A"]
            d = self.b2 * s["d"] + (1 - self.b2) * fa["rowc"]  # EMA of E[c_i]
            d_hat = d / (1 - self.b2 ** t)                      # bias corr!
            Q = s["Q"] if s["Q"] is not None else jnp.eye(W.shape[1])
            if (t - 1) % self.T == 0:
                ridge = 1e-6 * jnp.mean(jnp.diagonal(A))
                _, Q = jnp.linalg.eigh(A + ridge * jnp.eye(A.shape[0]))
            gr = gW @ Q
            du, m, v = _adam_core(gr, s["m"], s["v"], t, self.b1, self.b2,
                                  self.eps)
            dW = du @ Q.T
            # exact diagonal output-side curvature: row scaling
            floor = self.floor_frac * jnp.mean(d_hat)
            d_safe = jnp.maximum(d_hat, floor) + 1e-12
            row_scale = (1.0 / d_safe if self.mode == "newton"
                         else 1.0 / jnp.sqrt(d_safe))
            dW = dW * row_scale[:, None]
            # Sophia-style trust clip: RMS(ΔW) <= clip_ratio * RMS(W)
            rms_dw = jnp.sqrt(jnp.mean(dW * dW)) + 1e-30
            max_dw = self.clip_ratio * (jnp.sqrt(jnp.mean(W * W)) + 1e-30) \
                / self.lr
            dW = dW * jnp.minimum(1.0, max_dw / rms_dw)
            db, mb, vb = _adam_core(gl["b"], sl["b"]["m"], sl["b"]["v"], t,
                                    self.b1, self.b2, self.eps)
            out_params.append({"W": W - self.lr * dW, "b": l["b"] - self.lr * db})
            new_layers.append({"W": {"m": m, "v": v, "A": A, "d": d, "Q": Q},
                               "b": {"m": mb, "v": vb}})
        return out_params, {"t": t, "layers": new_layers}


class SOAP2S:
    """Faithful two-sided SOAP (control): both preconditioner factors from
    gradient covariance  G_L = EMA[g gᵀ],  G_R = EMA[gᵀ g]."""
    needs_curv = False
    needs_factors = False

    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, T=10):
        self.lr, self.b1, self.b2, self.eps, self.T = lr, b1, b2, eps, T

    def init(self, params):
        st = []
        for l in params:
            m_, n_ = l["W"].shape
            st.append({"W": {"m": jnp.zeros_like(l["W"]),
                             "v": jnp.zeros_like(l["W"]),
                             "GL": 1e-4 * jnp.eye(m_),
                             "GR": 1e-4 * jnp.eye(n_),
                             "QL": None, "QR": None},
                       "b": {"m": jnp.zeros_like(l["b"]),
                             "v": jnp.zeros_like(l["b"])}})
        return {"t": 0, "layers": st}

    def _rotate(self, gW, s, t):
        ridgeL = 1e-6 * jnp.mean(jnp.diagonal(s["GL"])) + 1e-12
        ridgeR = 1e-6 * jnp.mean(jnp.diagonal(s["GR"])) + 1e-12
        QL = s["QL"] if s["QL"] is not None else jnp.eye(s["GL"].shape[0])
        QR = s["QR"] if s["QR"] is not None else jnp.eye(s["GR"].shape[0])
        if (t - 1) % self.T == 0:
            _, QL = jnp.linalg.eigh(s["GL"] + ridgeL * jnp.eye(s["GL"].shape[0]))
            _, QR = jnp.linalg.eigh(s["GR"] + ridgeR * jnp.eye(s["GR"].shape[0]))
        return QL.T @ gW @ QR, QL, QR


class HybridSOAP(SOAP2S):
    """GGN input factor + gradient output factor (the "best of both" idea):

      Q_R  <-  A = EMA[E[x xᵀ]]   (EXACT GGN input covariance, from the
                                    engine's forward — noise-free, not
                                    corrupted by residuals)
      Q_L  <-  G_L = EMA[g gᵀ]    (Shampoo/SOAP gradient output factor —
                                    captures output-side correlations that
                                    diagonal curvature propagation drops)
    Adam runs in the rotated basis, updates rotate back. Inherits everything
    from SOAP2S; only the right-factor update changes."""
    needs_curv = True
    needs_factors = True

    def apply(self, params, grads, state, curvs=None, factors=None):
        t = state["t"] + 1
        new_layers, out_params = [], []
        for l, gl, sl, fa in zip(params, grads, state["layers"], factors):
            W, gW = l["W"], gl["W"]
            s = sl["W"]
            s = dict(s)
            # right factor: exact GGN input covariance replaces gᵀg
            s["GR"] = self.b2 * s["GR"] + (1 - self.b2) * fa["A"]
            s["GL"] = self.b2 * s["GL"] + (1 - self.b2) * (gW @ gW.T)
            g_rot, QL, QR = self._rotate(gW, s, t)
            du, m, v = _adam_core(g_rot, s["m"], s["v"], t, self.b1,
                                  self.b2, self.eps)
            dW = QL @ du @ QR.T
            db, mb, vb = _adam_core(gl["b"], sl["b"]["m"], sl["b"]["v"], t,
                                    self.b1, self.b2, self.eps)
            out_params.append({"W": W - self.lr * dW,
                               "b": l["b"] - self.lr * db})
            new_layers.append({"W": {"m": m, "v": v, "GL": s["GL"],
                                     "GR": s["GR"], "QL": QL, "QR": QR},
                               "b": {"m": mb, "v": vb}})
        return out_params, {"t": t, "layers": new_layers}


# give SOAP2S its own apply (two-sided, both factors from gradients)
def _soap2s_apply(self, params, grads, state, curvs=None, factors=None):
    t = state["t"] + 1
    new_layers, out_params = [], []
    for l, gl, sl in zip(params, grads, state["layers"]):
        W, gW = l["W"], gl["W"]
        s = sl["W"]
        s = dict(s)
        s["GL"] = self.b2 * s["GL"] + (1 - self.b2) * (gW @ gW.T)
        s["GR"] = self.b2 * s["GR"] + (1 - self.b2) * (gW.T @ gW)
        g_rot, QL, QR = self._rotate(gW, s, t)
        du, m, v = _adam_core(g_rot, s["m"], s["v"], t, self.b1,
                              self.b2, self.eps)
        dW = QL @ du @ QR.T
        db, mb, vb = _adam_core(gl["b"], sl["b"]["m"], sl["b"]["v"], t,
                                self.b1, self.b2, self.eps)
        out_params.append({"W": W - self.lr * dW,
                           "b": l["b"] - self.lr * db})
        new_layers.append({"W": {"m": m, "v": v, "GL": s["GL"],
                                 "GR": s["GR"], "QL": QL, "QR": QR},
                           "b": {"m": mb, "v": vb}})
    return out_params, {"t": t, "layers": new_layers}


SOAP2S.apply = _soap2s_apply
