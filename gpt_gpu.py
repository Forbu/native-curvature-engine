"""GPT-scale test (Lightning AI, NVIDIA L4): does the engine's exact GGN
input factor  A = E[x x^T]  inside SOAP beat the gradient right-factor g^T g?

Model: decoder-only GPT, 4 layers, d=256, 4 heads, ctx=256, char-level
tiny-shakespeare. NOTE: no curvature backprop needed here — the GGN input
factor is pure forward information (one X^T X GEMM per linear, collected
from the same forward pass that computes the loss).

Optimizers (all: grad-clip 1.0, 100-step LR warmup, wd=0.01 on matrices):
  AdamW       : baseline
  Muon        : momentum + Newton-Schulz orthogonalization on matrices
                (embedding/head/norms via AdamW — standard usage)
  SOAP-2s     : two-sided SOAP, Q_L and Q_R from gradient covariances
  HybridSOAP  : Q_L from gradient covariance, Q_R from exact E[x x^T]
"""
import os, time, urllib.request
import numpy as np
import jax
import jax.numpy as jnp

N_LAYER = int(os.environ.get("N_LAYER", "4"))
D = int(os.environ.get("D", "256"))
N_HEAD = D // 64
CTX = 256
BATCH = int(os.environ.get("BATCH", "128"))
jax.config.update("jax_default_matmul_precision", "tensorfloat32")
STEPS = int(os.environ.get("STEPS", "3000"))
LOG_EVERY, WARMUP = 250, 100

# ------------------------------- data ------------------------------------
URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
       "data/tinyshakespeare/input.txt")

def get_data():
    """Big corpus (TinyStories) if available, else tiny-shakespeare."""
    path = os.environ.get("NCE_CORPUS", "/tmp/tinystories_train.txt")
    if os.path.exists(path) and os.path.getsize(path) > 50_000_000:
        raw = open(path, "rb").read().decode("ascii", "replace")
        raw = raw.replace("\ufffd", "?")
        chars = sorted(set(raw))
        lut = np.zeros(256, np.uint8)
        for b, ch in enumerate("".join(chars)):
            lut[ord(ch)] = b
        ids = lut[np.frombuffer(raw.encode("ascii", "replace"), np.uint8)]
        n_val = len(ids) // 200                     # last 0.5% = val
        return ids[:-n_val], ids[-n_val:], len(chars)
    path = "/tmp/shakespeare.txt"
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int32)
    return ids[:-100_000], ids[-100_000:], len(chars)

class Batches:
    def __init__(self, ids, bs, ctx, seed):
        self.ids, self.bs, self.ctx = ids, bs, ctx
        self.rng = np.random.default_rng(seed)
    def next(self):
        ix = self.rng.integers(0, len(self.ids) - self.ctx - 1, self.bs)
        x = np.stack([self.ids[i:i + self.ctx] for i in ix]).astype(np.int32)
        y = np.stack([self.ids[i + 1:i + 1 + self.ctx] for i in ix]).astype(np.int32)
        return jnp.asarray(x), jnp.asarray(y)

# ------------------------------- model ------------------------------------
def init_params(key, V):
    def n(*s, std=0.02):
        return jax.random.normal(key, s) * std
    lin = [{"W": n(V, D), "b": None}]                        # 0 embedding
    for _ in range(N_LAYER):
        sc = 1.0 / np.sqrt(2 * N_LAYER)
        lin += [{"W": n(3 * D, D), "b": jnp.zeros(3 * D)},   # qkv
                {"W": n(D, D),   "b": jnp.zeros(D)},         # attn out
                {"W": n(4 * D, D), "b": jnp.zeros(4 * D)},   # mlp fc
                {"W": n(D, 4 * D, std=0.02 * sc), "b": jnp.zeros(D)}]
    lin += [{"W": n(V, D), "b": None}]                       # head
    return {"lin": lin, "norms": jnp.ones((2 * N_LAYER + 1, D))}

def rms(h, g):
    return g * h / jnp.sqrt(jnp.mean(h * h, -1, keepdims=True) + 1e-5)

def forward(P, x):
    lin, norms = P["lin"], P["norms"]
    B, T = x.shape
    As = []                                     # GGN input factors, lin order
    h = lin[0]["W"][x]
    freq = jnp.bincount(x.reshape(-1), length=lin[0]["W"].shape[0]) / (B * T)
    As.append(jnp.diag(freq))                   # E[onehot onehot^T] = diag(freq)

    def dense(i, u):
        u2 = u.reshape(-1, u.shape[-1])
        As.append((u2.T @ u2) / u2.shape[0])    # A = E[x x^T], exact
        l = lin[i]
        return u @ l["W"].T + (0.0 if l["b"] is None else l["b"])

    idx = 1
    shp = (D // N_HEAD)
    for li in range(N_LAYER):
        a = dense(idx, rms(h, norms[2 * li])); idx += 1
        q, k_, v = jnp.split(a, 3, -1)
        sh = lambda t: t.reshape(B, T, N_HEAD, -1).transpose(0, 2, 1, 3)
        q, k_, v = sh(q), sh(k_), sh(v)
        att = q @ k_.transpose(0, 1, 3, 2) / np.sqrt(shp)
        att = jnp.where(jnp.tril(jnp.ones((T, T), bool))[None, None], att, -1e30)
        att = jax.nn.softmax(att, -1)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        h = h + dense(idx, o); idx += 1
        m = jax.nn.gelu(dense(idx, rms(h, norms[2 * li + 1])), approximate=True)
        idx += 1
        h = h + dense(idx, m); idx += 1
    logits = dense(idx, rms(h, norms[2 * N_LAYER]))
    return logits, As

def loss_fn(P, x, y):
    logits, As = forward(P, x)
    logp = jax.nn.log_softmax(logits.reshape(-1, logits.shape[-1]))
    nll = -jnp.mean(jnp.take_along_axis(logp, y.reshape(-1)[:, None], 1))
    return nll, As

# ---------------------------- optimizers ----------------------------------
def adam_upd(g, m, v, t, b1, b2, eps):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    upd = (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
    return upd, m, v

def _misc_state(P):
    return {"m": jnp.zeros_like(P["norms"]), "v": jnp.zeros_like(P["norms"])}

def _upd_misc(g, s, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    upd, m, v = adam_upd(g, s["m"], s["v"], t, b1, b2, eps)
    return upd, {"m": m, "v": v}

def _lr_t(base_lr, t):
    return base_lr * jnp.minimum(1.0, t / WARMUP)

class AdamWOpt:
    use_as = False
    def __init__(self, lr, wd=0.01):
        self.lr, self.wd = lr, wd
    def init(self, P):
        st = [{"W": {"m": jnp.zeros_like(l["W"]), "v": jnp.zeros_like(l["W"])},
               "b": None if l["b"] is None else
                    {"m": jnp.zeros_like(l["b"]), "v": jnp.zeros_like(l["b"])}}
              for l in P["lin"]]
        return {"t": jnp.int32(0), "lin": st, "misc": _misc_state(P)}
    def apply(self, P, G, st, As=None, Cv=None):
        t = st["t"] + 1
        lr = _lr_t(self.lr, t)
        lin, nl = [], []
        for l, gl, sl in zip(P["lin"], G["lin"], st["lin"]):
            W = l["W"] * (1 - lr * self.wd)
            uW, mW_, vW_ = adam_upd(gl["W"], sl["W"]["m"], sl["W"]["v"], t,
                                    0.9, 0.999, 1e-8)
            sW = {"m": mW_, "v": vW_}
            W = W - lr * uW
            b = l["b"]
            if b is not None:
                ub, mb_, vb_ = adam_upd(gl["b"], sl["b"]["m"], sl["b"]["v"], t,
                                        0.9, 0.999, 1e-8)
                b = b - lr * ub
                sb = {"m": mb_, "v": vb_}
            lin.append({"W": W, "b": b})
            nl.append({"W": sW, "b": None if sl["b"] is None else sb})
        um, sm = _upd_misc(G["norms"], st["misc"], t, lr)
        norms = P["norms"] - lr * um
        return {"lin": lin, "norms": norms}, {"t": t, "lin": nl, "misc": sm}

class SOAPG:
    """Two-sided SOAP. hybrid=False: both factors from gradients (control).
    hybrid=True: right factor from the exact GGN input covariance A."""
    use_as = True
    def __init__(self, lr, wd=0.01, T=10, b1=0.9, b2=0.999, eps=1e-8,
                 hybrid=False):
        self.lr, self.wd, self.T = lr, wd, T
        self.b1, self.b2, self.eps = b1, b2, eps
        self.hybrid = hybrid
    def init(self, P):
        st = []
        # embedding (i=0) is a transposed dense: input axis is axis 0
        self.tr_flags = [i == 0 for i in range(len(P["lin"]))]
        for i, l in enumerate(P["lin"]):
            tr = self.tr_flags[i]
            W = l["W"].T if tr else l["W"]
            m_, n_ = W.shape
            st.append({"m": jnp.zeros_like(W), "v": jnp.zeros_like(W),
                       "GL": 1e-4 * jnp.eye(m_), "GR": 1e-4 * jnp.eye(n_),
                       "QL": jnp.eye(m_), "QR": jnp.eye(n_),
                       "b": None if l["b"] is None else
                            {"m": jnp.zeros_like(l["b"]),
                             "v": jnp.zeros_like(l["b"])}})
        return {"t": jnp.int32(0), "lin": st, "misc": _misc_state(P)}
    def apply(self, P, G, st, As=None, Cv=None):
        t = st["t"] + 1
        lr = _lr_t(self.lr, t)
        lin, nst = [], []
        for i, (l, gl, sl) in enumerate(zip(P["lin"], G["lin"], st["lin"])):
            W, gW = l["W"], gl["W"]
            if self.tr_flags[i]:
                gW = gW.T                       # embedding: input axis -> cols
            GL = self.b2 * sl["GL"] + (1 - self.b2) * (gW @ gW.T)
            if self.hybrid:
                GR = self.b2 * sl["GR"] + (1 - self.b2) * As[i]
            else:
                GR = self.b2 * sl["GR"] + (1 - self.b2) * (gW.T @ gW)
            def _eig():
                rL = 1e-6 * jnp.mean(jnp.diagonal(GL)) + 1e-12
                rR = 1e-6 * jnp.mean(jnp.diagonal(GR)) + 1e-12
                _, qL = jnp.linalg.eigh(GL + rL * jnp.eye(GL.shape[0]))
                _, qR = jnp.linalg.eigh(GR + rR * jnp.eye(GR.shape[0]))
                return qL, qR
            QL, QR = jax.lax.cond((t - 1) % self.T == 0, _eig,
                                  lambda: (sl["QL"], sl["QR"]))
            g_rot = QL.T @ gW @ QR
            uW, m, v = adam_upd(g_rot, sl["m"], sl["v"], t,
                                self.b1, self.b2, self.eps)
            dW = QL @ uW @ QR.T
            if self.tr_flags[i]:
                dW = dW.T
            W = W * (1 - lr * self.wd) - lr * dW
            b = l["b"]
            nb = sl["b"]
            if b is not None:
                ub, mb_, vb_ = adam_upd(gl["b"], sl["b"]["m"], sl["b"]["v"], t,
                                        self.b1, self.b2, self.eps)
                b = b - lr * ub
                nb = {"m": mb_, "v": vb_}
            lin.append({"W": W, "b": b})
            nst.append({"m": m, "v": v, "GL": GL, "GR": GR,
                        "QL": QL, "QR": QR, "b": nb})
        um, sm = _upd_misc(G["norms"], st["misc"], t, lr)
        norms = P["norms"] - lr * um
        return {"lin": lin, "norms": norms}, \
               {"t": t, "lin": nst, "misc": sm}

def ns5(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (jnp.linalg.norm(G) + 1e-7)
    tr = X.shape[0] > X.shape[1]
    if tr:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tr:
        X = X.T
    return X

class MuonOpt:
    use_as = False
    def __init__(self, lr, wd=0.01, mom=0.95, adam_lr=0.003):
        self.lr, self.wd, self.mom, self.adam_lr = lr, wd, mom, adam_lr
    def init(self, P):
        st = []
        for i, l in enumerate(P["lin"]):
            entry = {"W": None if i in (0, len(P["lin"]) - 1)
                     else jnp.zeros_like(l["W"]),
                     "b": None if l["b"] is None else
                          {"m": jnp.zeros_like(l["b"]),
                           "v": jnp.zeros_like(l["b"])},
                     "aW": ({"m": jnp.zeros_like(l["W"]), "v": jnp.zeros_like(l["W"])}
                            if i in (0, len(P["lin"]) - 1) else None)}
            st.append(entry)
        return {"t": jnp.int32(0), "lin": st, "misc": _misc_state(P)}
    def apply(self, P, G, st, As=None):
        t = st["t"] + 1
        lr = _lr_t(self.lr, t)
        alr = _lr_t(self.adam_lr, t)
        lin, nst = [], []
        for i, (l, gl, sl) in enumerate(zip(P["lin"], G["lin"], st["lin"])):
            W = l["W"]
            if sl["aW"] is not None:                    # emb & head: AdamW
                uW, mW_, vW_ = adam_upd(gl["W"], sl["aW"]["m"], sl["aW"]["v"], t,
                                        0.9, 0.999, 1e-8)
                W = W * (1 - alr * self.wd) - alr * uW
                nW = None
                aW = {"m": mW_, "v": vW_}
            else:                                       # hidden: Muon
                buf = self.mom * sl["W"] + gl["W"]
                g_eff = gl["W"] + self.mom * buf        # nesterov
                scale = np.sqrt(max(1.0, W.shape[0] / W.shape[1]))
                W = W * (1 - lr * self.wd) - lr * scale * ns5(g_eff)
                nW, aW = buf, None
            b, nb = l["b"], sl["b"]
            if b is not None:
                ub, mb_, vb_ = adam_upd(gl["b"], sl["b"]["m"], sl["b"]["v"], t,
                                        0.9, 0.999, 1e-8)
                b = b - alr * ub
                nb = {"m": mb_, "v": vb_}
            lin.append({"W": W, "b": b})
            nst.append({"W": nW, "b": nb, "aW": aW})
        um, sm = _upd_misc(G["norms"], st["misc"], t, alr)
        norms = P["norms"] - alr * um
        return {"lin": lin, "norms": norms}, \
               {"t": t, "lin": nst, "misc": sm}

def clip_grads(G, C=1.0):
    tm = jax.tree_util.tree_map
    sq = jax.tree_util.tree_reduce(
        lambda a, b: a + jnp.sum(b ** 2), G, initializer=jnp.zeros(()))
    s = jnp.minimum(1.0, C / (jnp.sqrt(sq) + 1e-6))
    return tm(lambda a: a * s, G)

# ------------------------------- run ---------------------------------------
def evaluate(P, val_batches):
    tot, n = 0.0, 0
    for x, y in val_batches:
        L = jax.jit(lambda P, x, y: loss_fn(P, x, y)[0])(P, x, y)
        tot += float(L); n += 1
    return tot / n

def train_run(name, mk_opt, lr, train_ids, val_ids, V, seed=0):
    params = init_params(jax.random.PRNGKey(seed), V)
    opt = mk_opt(lr)
    state = opt.init(params)
    train = Batches(train_ids, BATCH, CTX, seed)
    rng = np.random.default_rng(1234)
    val_batches = []
    for _ in range(8):
        ix = rng.integers(0, len(val_ids) - CTX - 1, 32)
        val_batches.append((jnp.asarray(np.stack([val_ids[i:i+CTX] for i in ix])),
                            jnp.asarray(np.stack([val_ids[i+1:i+1+CTX] for i in ix]))))
    eval_fn = jax.jit(lambda P, x, y: loss_fn(P, x, y)[0])

    def _full(P, S, x, y):
        if getattr(opt, "use_curv", False):
            (L, As), Gw = jax.value_and_grad(loss_fn_dual, has_aux=True)(
                eng.wrap_tree(P), x, y)
            Gc, Cv = eng.split_dual_grad(Gw)
            Gc = clip_grads(Gc)
            P, S = opt.apply(P, Gc, S, As, Cv)
        else:
            (L, As), G = jax.value_and_grad(loss_fn, has_aux=True)(P, x, y)
            G = clip_grads(G)
            P, S = opt.apply(P, G, S, As)
        return P, S, L
    full_j = jax.jit(_full)
    hist = {"step": [], "val": [], "sec": []}
    t_start = time.time()
    t_mark, step_mark = t_start, 50
    ms_per_step = 0.0
    train_loss_ema = None
    for step in range(1, STEPS + 1):
        x, y = train.next()
        params, state, L = full_j(params, state, x, y)
        train_loss_ema = float(L) if train_loss_ema is None \
            else 0.98 * train_loss_ema + 0.02 * float(L)
        if step == step_mark:
            jax.block_until_ready(L)
            n_in_block = step - (step_mark - 50)
            ms_per_step = (time.time() - t_mark) * 1000 / n_in_block
            t_mark, step_mark = time.time(), step + 50
        if step % LOG_EVERY == 0:
            tot, nb = 0.0, 0
            for vx, vy in val_batches:
                tot += float(eval_fn(params, vx, vy)); nb += 1
            val = tot / nb
            hist["step"].append(step); hist["val"].append(val)
            hist["sec"].append(time.time() - t_start)
            print(f"    [{name} lr={lr:g}] step {step:5d}  train {train_loss_ema:.4f}"
                  f"  val {val:.4f}  ({ms_per_step:.0f} ms/step)", flush=True)
    return hist, ms_per_step

def main():
    train_ids, val_ids, V = get_data()
    print(f"data: {len(train_ids):,} train / {len(val_ids):,} val chars, "
          f"vocab {V}; run = {STEPS * BATCH * CTX / 1e6:.0f}M tokens "
          f"({STEPS * BATCH * CTX / len(train_ids):.3f} epochs)", flush=True)
    P0 = init_params(jax.random.PRNGKey(0), V)
    n_par = sum(int(l["W"].size) for l in P0["lin"]) + \
        sum((0 if l["b"] is None else int(l["b"].size)) for l in P0["lin"]) \
        + int(P0["norms"].size)
    print(f"model: {N_LAYER}L d={D} h={N_HEAD} ctx={CTX}  params {n_par:,}",
          flush=True)
    CFGS = [
        ("AdamW",      lambda lr: AdamWOpt(lr),              [3e-3]),
        ("SOAP-2s",    lambda lr: SOAPG(lr, hybrid=False),   [1e-2]),
        ("HybridSOAP", lambda lr: SOAPG(lr, hybrid=True),    [1e-2]),
        ("NatCurv",    lambda lr: NatCurvG(lr),              [1e-2, 3e-2]),
    ]
    all_hist = {}
    for name, mk, lrs in CFGS:
        for lr in lrs:
            key = f"{name}@{lr:g}"
            print(f"  === {key} ===", flush=True)
            hist, ms = train_run(name, mk, lr, train_ids, val_ids, V)
            all_hist[key] = hist
            print(f"  => {key}: final val {hist['val'][-1]:.4f}, "
                  f"{ms:.0f} ms/step, {hist['sec'][-1]:.0f}s total", flush=True)
    print("\n================ SUMMARY (best val per optimizer) ================")
    best = {}
    for key, h in all_hist.items():
        name = key.split("@")[0]
        if name not in best or h["val"][-1] < best[name][1]["val"][-1]:
            best[name] = (key, h)
    for name, (key, h) in sorted(best.items(), key=lambda kv: kv[1][1]["val"][-1]):
        v1000 = h["val"][min(range(len(h["step"])),
                             key=lambda i: abs(h["step"][i] - 1000))]
        print(f"  {name:12s} (lr {key.split('@')[1]:<7s})  final val {h['val'][-1]:.4f}"
              f"   val@1000 {v1000:.4f}   wall {h['sec'][-1]:.0f}s")
    np.savez("/tmp/gpt_results.npz",
             **{f"{k}|{f}": np.array(v) for k, h in all_hist.items()
                for f, v in h.items()})
    print("saved /tmp/gpt_results.npz")

# ===========================================================================
# DUAL PATH: gradient + native curvature (diag GGN) in ONE backward sweep.
# Reuses engine.py conventions: params {'val','curv'}, activations
# {'y','aux'}; custom_vjp rules receive cotangents carrying both slots.
# New transformer rules: attention, RMSNorm, GELU, exact CE seed p(1-p).
# ===========================================================================
import engine as eng

# ----------------------------- RMSNorm --------------------------------------
@jax.custom_vjp
def rmsn(x, g):
    return _rmsn_fwd(x, g)[0]

def _rmsn_fwd(x, g):
    h = x["y"]
    inv = jax.lax.rsqrt(jnp.mean(h * h, -1, keepdims=True) + 1e-5)
    xh = h * inv
    y = g * xh
    return {"y": y, "aux": y}, (h, inv, g, xh)

def _rmsn_bwd(res, ct):
    h, inv, g, xh = res
    Gy, Cy = ct["y"], ct["aux"]
    u = Gy * g                                   # dL/du, u = pre-scale normed
    Gx = inv * (u - xh * jnp.mean(u * xh, -1, keepdims=True))
    gg = jnp.sum(Gy * xh, 0)                     # grad wrt scale (plain arg)
    Cx = (g * inv) ** 2 * Cy                     # curvature: (f')^2 = (g/rms)^2
    return ({"y": Gx, "aux": Cx}, gg)

rmsn.defvjp(_rmsn_fwd, _rmsn_bwd)

# ----------------------------- GELU (tanh) ----------------------------------
@jax.custom_vjp
def gelu_act(x):
    return _gelu_fwd(x)[0]

def _gelu_fwd(x):
    z = x["y"]
    y = jax.nn.gelu(z, approximate=True)
    return {"y": y, "aux": y}, z

def _gelu_bwd(res, ct):
    z = res
    c0 = 0.7978845608028654
    u = c0 * (z + 0.044715 * z ** 3)
    t = jnp.tanh(u)
    dphi = c0 * (1 + 3 * 0.044715 * z ** 2) * (1 - t * t)
    d = 0.5 * (1 + t) + z * 0.5 * dphi           # gelu'(z)
    return ({"y": ct["y"] * d, "aux": ct["aux"] * d * d},)

gelu_act.defvjp(_gelu_fwd, _gelu_bwd)

# ----------------------------- attention ------------------------------------
def mhsa(qkv, T, H):
    return _mhsa_fwd(qkv, T, H)[0]


mhsa = jax.custom_vjp(mhsa, nondiff_argnums=(1, 2))

def _mhsa_fwd(qkv, T, H):
    Y3 = qkv["y"]                                # [N, 3d]
    N = Y3.shape[0]
    B = N // T
    dh = Y3.shape[1] // (3 * H)

    def heads(a):
        return a.reshape(B, T, H, a.shape[-1] // H).transpose(0, 2, 1, 3)

    q, k_, v = (heads(Y3[:, :dh * H]), heads(Y3[:, dh * H:2 * dh * H]),
                heads(Y3[:, 2 * dh * H:]))
    att = q @ k_.transpose(0, 1, 3, 2) / jnp.sqrt(dh)
    att = jnp.where(jnp.tril(jnp.ones((T, T), bool))[None, None], att, -1e30)
    A = jax.nn.softmax(att, -1)
    y = (A @ v).transpose(0, 2, 1, 3).reshape(N, H * dh)
    return {"y": y, "aux": y}, (A, q, k_, v)

def _mhsa_bwd(T, H, res, ct):
    A, q, k_, v = res
    B, _, _, dh = q.shape
    rt = jnp.sqrt(dh)
    Gyh = ct["y"].reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    Cyh = ct["aux"].reshape(B, T, H, dh).transpose(0, 2, 1, 3)

    # ---- gradient (standard)
    gA = jnp.einsum("bhti,bhji->bhtj", Gyh, v)
    gv = jnp.einsum("bhtj,bhti->bhji", A, Gyh)
    # full softmax backward: dL/dS = A * (gA - <gA, A>)  (NOT A(1-A)*gA!)
    gS = A * (gA - jnp.sum(gA * A, -1, keepdims=True))
    gq = jnp.einsum("bhtj,bhji->bhti", gS, k_) / rt
    gk = jnp.einsum("bhtj,bhti->bhji", gS, q) / rt

    # ---- curvature (diagonal propagation)
    cA = jnp.einsum("bhti,bhji->bhtj", Cyh, v * v)   # <c_t, v_j^2>
    cV = jnp.einsum("bhtj,bhti->bhji", A * A, Cyh)   # sum_t A^2 c_t
    cS = cA * (A * (1.0 - A)) ** 2                   # softmax row diag^2
    cq = jnp.einsum("bhtj,bhji->bhti", cS, k_ * k_) / dh
    ck = jnp.einsum("bhtj,bhti->bhji", cS, q * q) / dh

    def merge(a, b, c):
        def flat(t):
            return t.transpose(0, 2, 1, 3).reshape(B * T, H * dh)
        return jnp.concatenate([flat(a), flat(b), flat(c)], axis=-1)

    return ({"y": merge(gq, gk, gv), "aux": merge(cq, ck, cV)},)

mhsa.defvjp(_mhsa_fwd, _mhsa_bwd)

# ----------------------------- CE loss --------------------------------------
@jax.custom_vjp
def ce_lm(out, t):
    return _ce_lm_fwd(out, t)[0]

def _ce_lm_fwd(out, t):
    Y = out["y"]
    L = -jnp.mean(jnp.take_along_axis(
        jax.nn.log_softmax(Y), t[:, None].astype(int), axis=1))
    return L, (Y, t)

def _ce_lm_bwd(res, ct):
    Y, t = res
    p = jax.nn.softmax(Y, -1)
    onehot = jax.nn.one_hot(t, Y.shape[-1])
    g = (p - onehot) / Y.shape[0]
    c = p * (1.0 - p)            # EXACT diagonal of the CE GGN  diag(p)-pp^T
    return ({"y": g, "aux": c}, None)

ce_lm.defvjp(_ce_lm_fwd, _ce_lm_bwd)

# ----------------------------- dual forward ---------------------------------
def forward_dual(P, x, y):
    """P = wrapped params. Returns (loss, As). Curvature rides the cotangent."""
    lin, norms = P["lin"], P["norms"]
    B, T = x.shape
    N = B * T
    As = []
    h = eng.wrap_input(lin[0]["W"]["val"][x].reshape(N, -1))
    freq = jnp.bincount(x.reshape(-1), length=lin[0]["W"]["val"].shape[0]) / N
    As.append(jnp.diag(freq))

    def dense_w(i, u):
        X = u["y"]
        As.append((X.T @ X) / X.shape[0])
        b = lin[i]["b"]
        if b is None:      # biasless linear (head): fake zero-bias, cotangent
            m = lin[i]["W"]["val"].shape[0]         # is discarded by autodiff
            b = {"val": jnp.zeros((m,)), "curv": jnp.zeros((m,))}
        return eng.dense(u, lin[i]["W"], b)

    idx = 1
    for li in range(N_LAYER):
        a = dense_w(idx, rmsn(h, norms["val"][2 * li])); idx += 1
        o = mhsa(a, T, N_HEAD)
        h = eng.residual(h, dense_w(idx, o)); idx += 1
        m = gelu_act(dense_w(idx, rmsn(h, norms["val"][2 * li + 1]))); idx += 1
        h = eng.residual(h, dense_w(idx, m)); idx += 1
    logits = dense_w(idx, rmsn(h, norms["val"][2 * N_LAYER]))
    L = ce_lm(logits, y.reshape(-1))
    return L, As

def loss_fn_dual(Pw, x, y):
    return forward_dual(Pw, x, y)

# ----------------------------- NatCurv optimizer ----------------------------
class NatCurvG:
    """Newton-style: W <- W - lr * m_hat / d_hat, d = EMA of the native
    curvature diag (exact GGN diagonal), with per-tensor floor + Sophia-style
    trust clip. Embedding + biases + norms: AdamW. This is optimizers.NatCurv
    (mode='newton') ported to the GPT structure."""
    use_as = True
    use_curv = True

    def __init__(self, lr, wd=0.01, b1=0.9, b2=0.999, floor_frac=0.05,
                 min_curv=1e-8, clip_ratio=0.02, adam_lr=3e-3):
        self.lr, self.wd = lr, wd
        self.b1, self.b2 = b1, b2
        self.floor_frac, self.min_curv = floor_frac, min_curv
        self.clip_ratio = clip_ratio
        self.adam_lr = adam_lr

    def init(self, P):
        st = []
        for i, l in enumerate(P["lin"]):
            adam_emb = (i == 0)               # embedding: no curvature rule
            st.append({"m": jnp.zeros_like(l["W"]),
                       "d": jnp.zeros_like(l["W"]),
                       "aW": ({"m": jnp.zeros_like(l["W"]),
                               "v": jnp.zeros_like(l["W"])}
                              if adam_emb else None),
                       "b": None if l["b"] is None else
                            {"m": jnp.zeros_like(l["b"]),
                             "v": jnp.zeros_like(l["b"])}})
        return {"t": jnp.int32(0), "lin": st, "misc": _misc_state(P)}

    def apply(self, P, G, st, As=None, Cv=None):
        t = st["t"] + 1
        lr = _lr_t(self.lr, t)
        alr = _lr_t(self.adam_lr, t)
        lin, nst = [], []
        for i, (l, gl, sl, cl) in enumerate(zip(P["lin"], G["lin"],
                                                st["lin"], Cv["lin"])):
            W, gW, cW = l["W"], gl["W"], cl["W"]
            if sl["aW"] is not None:                       # embedding: AdamW
                uW, mW_, vW_ = adam_upd(gW, sl["aW"]["m"], sl["aW"]["v"], t,
                                         0.9, 0.999, 1e-8)
                W = W * (1 - alr * self.wd) - alr * uW
                nW, aW, m, d = None, {"m": mW_, "v": vW_}, None, None
            else:                                          # native Newton
                m = self.b1 * sl["m"] + (1 - self.b1) * gW
                d = self.b2 * sl["d"] + (1 - self.b2) * cW
                m_hat = m / (1 - self.b1 ** t)
                d_hat = d / (1 - self.b2 ** t)
                floor = self.floor_frac * jnp.mean(d_hat) + self.min_curv
                d_safe = jnp.maximum(d_hat, floor)
                upd = m_hat / d_safe
                rms_u = jnp.sqrt(jnp.mean(upd * upd)) + 1e-30
                max_u = self.clip_ratio * (jnp.sqrt(jnp.mean(W * W)) + 1e-30) \
                    / lr
                upd = upd * jnp.minimum(1.0, max_u / rms_u)
                W = W * (1 - lr * self.wd) - lr * upd
                nW, aW = None, None
            b, nb = l["b"], sl["b"]
            if b is not None:
                ub, mb_, vb_ = adam_upd(gl["b"], sl["b"]["m"], sl["b"]["v"], t,
                                        0.9, 0.999, 1e-8)
                b = b - lr * ub
                nb = {"m": mb_, "v": vb_}
            lin.append({"W": W, "b": b})
            nst.append({"m": m, "d": d, "aW": aW, "b": nb})
        um, sm = _upd_misc(G["norms"], st["misc"], t, alr)
        norms = P["norms"] - alr * um
        return {"lin": lin, "norms": norms}, \
               {"t": t, "lin": nst, "misc": sm}


if __name__ == "__main__":
    main()
