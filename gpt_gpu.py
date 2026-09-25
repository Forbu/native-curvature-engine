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

N_LAYER, D, N_HEAD, CTX = 4, 256, 4, 256
BATCH = int(os.environ.get("BATCH", "128"))
STEPS, LOG_EVERY, WARMUP = 3000, 250, 100

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
    def apply(self, P, G, st, As=None):
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
    def apply(self, P, G, st, As=None):
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
        ("AdamW",      lambda lr: AdamWOpt(lr),              [3e-3, 1e-2]),
        ("Muon",       lambda lr: MuonOpt(lr),               [0.02, 0.05]),
        ("SOAP-2s",    lambda lr: SOAPG(lr, hybrid=False),   [1e-2, 3e-2]),
        ("HybridSOAP", lambda lr: SOAPG(lr, hybrid=True),    [1e-2, 3e-2]),
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

if __name__ == "__main__":
    main()
