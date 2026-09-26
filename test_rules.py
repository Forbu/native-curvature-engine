"""Isolated verification of the transformer curvature rules: each custom_vjp
rule's gradient (val slot) must match plain autodiff exactly."""
import jax
import jax.numpy as jnp
import numpy as np
import gpt_gpu as g
import engine as eng

key = jax.random.PRNGKey(0)


def next_key():
    global key
    key, k = jax.random.split(key)
    return k


def rel(a, b):
    return float(jnp.abs(a - b).max() /
                 (jnp.abs(b).max() + 1e-30))


# ---------------------------------------------------------------- RMSNorm ---
B, T, d = 3, 8, 16
h = jax.random.normal(next_key(), (B * T, d)) * 2
gam = jnp.ones((d,)) + 0.1 * jax.random.normal(next_key(), (d,))
lam = jax.random.normal(next_key(), (B * T, d))

def plain_rms(h_, g_, lam_):
    inv = 1 / jnp.sqrt(jnp.mean(h_ * h_, -1, keepdims=True) + 1e-5)
    return jnp.sum(g_ * (h_ * inv) * lam_)

r_h = jax.grad(plain_rms, 0)(h, gam, lam)
r_g = jax.grad(plain_rms, 1)(h, gam, lam)

def dual_rms(hw):
    return jnp.sum(g.rmsn(hw, gam)["y"] * lam)

out = jax.grad(dual_rms)(eng.wrap_input(h))["y"]
print(f"rmsn  dh rel {rel(out, r_h):.2e}", end="  ")

def dual_rms_g(h_plain, gam_):
    hw = eng.wrap_input(h_plain)
    return jnp.sum(g.rmsn(hw, gam_)["y"] * lam)

print(f"dg rel {rel(jax.grad(dual_rms_g, 1)(h, gam), r_g):.2e}")

# ------------------------------------------------------------------- GELU ---
z = jax.random.normal(next_key(), (B * T, d)) * 2
lam2 = jax.random.normal(next_key(), (B * T, d))

def plain_gelu(z_):
    return jnp.sum(jax.nn.gelu(z_, approximate=True) * lam2)

rz = jax.grad(plain_gelu)(z)

def dual_gelu(z_plain):
    return jnp.sum(g.gelu_act(eng.wrap_input(z_plain))["y"] * lam2)

print(f"gelu  dz rel {rel(jax.grad(dual_gelu)(z), rz):.2e}")

# -------------------------------------------------------------- attention ---
H = 4
dh = d // H
qkv = jax.random.normal(next_key(), (B * T, 3 * d)) * 0.5

def plain_mhsa(qkv_, T_, H_):
    Y3 = qkv_
    N = Y3.shape[0]
    B_ = N // T_
    dh_ = Y3.shape[1] // (3 * H_)
    def heads(a):
        return a.reshape(B_, T_, H_, a.shape[-1] // H_).transpose(0, 2, 1, 3)
    q, k_, v = (heads(Y3[:, :dh_ * H_]), heads(Y3[:, dh_ * H_:2 * dh_ * H_]),
                heads(Y3[:, 2 * dh_ * H_:]))
    att = q @ k_.transpose(0, 1, 3, 2) / jnp.sqrt(dh_)
    att = jnp.where(jnp.tril(jnp.ones((T_, T_), bool))[None, None], att, -1e30)
    A = jax.nn.softmax(att, -1)
    y = (A @ v).transpose(0, 2, 1, 3).reshape(N, H_ * dh_)
    return y

lam3 = jax.random.normal(next_key(), (B * T, d))

def plain_att_l(qkv_):
    return jnp.sum(plain_mhsa(qkv_, T, H) * lam3)

rq = jax.grad(plain_att_l)(qkv)

def dual_att_l(qkvw):
    return jnp.sum(g.mhsa(qkvw, T, H)["y"] * lam3)

print(f"mhsa  dqkv rel {rel(jax.grad(dual_att_l)(eng.wrap_input(qkv))['y'], rq):.2e}")

# ---------------------------------------------------------------------- CE ---
V = 37
logits = jax.random.normal(next_key(), (B * T, V)) * 3
t = jax.random.randint(next_key(), (B * T,), 0, V)

def plain_ce(l_):
    return -jnp.mean(jnp.take_along_axis(
        jax.nn.log_softmax(l_), t[:, None].astype(int), 1))

rl = jax.grad(plain_ce)(logits)

def dual_ce(lw):
    return g.ce_lm(lw, t)

cot = jax.grad(dual_ce)(eng.wrap_input(logits))["y"]
print(f"ce    dlogits rel {rel(cot, rl):.2e}")

# ------------------------------------------------- full-model grad compare ---
tr, va, Voc = g.get_data()
P = g.init_params(jax.random.PRNGKey(0), Voc)
bt = g.Batches(tr, 4, 128, 0)
x, y = bt.next()
(L0, _), Gp = jax.value_and_grad(g.loss_fn, has_aux=True)(P, x, y)
(L1, _), Gd = jax.value_and_grad(g.loss_fn_dual, has_aux=True)(
    eng.wrap_tree(P), x, y)
gd, cv = eng.split_dual_grad(Gd)
print(f"full model: L {float(L0):.5f} vs {float(L1):.5f}")
for i, (lp, ld) in enumerate(zip(Gp["lin"], gd["lin"])):
    for p in ("W", "b"):
        if lp[p] is None:
            continue
        r = rel(ld[p], lp[p])
        if r > 1e-4:
            print(f"  layer {i:2d} {p}: rel {r:.2e}  <<<< MISMATCH")
print("(no output above => all layers < 1e-4 relative)")
