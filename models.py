"""Shared model definitions: engine (wrapped) and plain-autodiff (reference).

Both forward passes implement the exact same network so tests can compare
the engine's dual output against textbook autodiff.
"""

import jax
import jax.numpy as jnp
import engine as eng


def init_params(sizes, key, scale=1.0):
    """List of layers {'W': [out, in], 'b': [out]} in plain (unwrapped) form."""
    params, keys = [], jax.random.split(key, len(sizes) - 1)
    for k, (n_in, n_out) in zip(keys, zip(sizes[:-1], sizes[1:])):
        W = jax.random.normal(k, (n_out, n_in)) * (scale / n_in ** 0.5)
        params.append({"W": W, "b": jnp.zeros(n_out)})
    return params


# ----------------------------- engine models ------------------------------

def mlp_engine(pw, X, act="relu"):
    a = eng.wrap_input(X)
    for l in pw:
        a = eng.dense(a, l["W"], l["b"])
        if act == "relu":
            a = eng.relu(a)
        else:
            a = eng.tanh_act(a)
    return a  # wrapped logits


def resnet_engine(pw, X, act="relu", n_blocks=None):
    """ResNet MLP:  h = relu(dense(X));  K x [h += relu(dense2(relu(dense1(h))))];
    logits = dense(h).  pw = [head] + [l1, l2] * K + [out]."""
    A = eng.relu if act == "relu" else eng.tanh_act
    a = A(eng.dense(eng.wrap_input(X), pw[0]["W"], pw[0]["b"]))
    for i in range(n_blocks):
        l1, l2 = pw[1 + 2 * i], pw[2 + 2 * i]
        h = A(eng.dense(a, l1["W"], l1["b"]))
        a = eng.residual(a, A(eng.dense(h, l2["W"], l2["b"])))
    return eng.dense(a, pw[-1]["W"], pw[-1]["b"])


# ----------------------------- plain models -------------------------------

def _act(Y, name):
    return jnp.maximum(Y, 0.0) if name == "relu" else jnp.tanh(Y)


def mlp_plain(p, X, act="relu"):
    h = X
    for l in p:
        h = _act(h @ l["W"].T + l["b"], act)
    return h


def resnet_plain(p, X, act="relu", n_blocks=None):
    a = _act(X @ p[0]["W"].T + p[0]["b"], act)
    for i in range(n_blocks):
        l1, l2 = p[1 + 2 * i], p[2 + 2 * i]
        h = _act(a @ l1["W"].T + l1["b"], act)
        a = a + _act(h @ l2["W"].T + l2["b"], act)
    return a @ p[-1]["W"].T + p[-1]["b"]


def resnet_engine_trace(pw, X, act="relu", n_blocks=None):
    """Same as resnet_engine but also returns every dense layer's input
    activations (for exact GGN input factors  A = E[x xᵀ])."""
    A = eng.relu if act == "relu" else eng.tanh_act
    inputs = []
    a = eng.wrap_input(X)
    inputs.append(a["y"])
    a = A(eng.dense(a, pw[0]["W"], pw[0]["B" if False else "b"]))
    for i in range(n_blocks):
        l1, l2 = pw[1 + 2 * i], pw[2 + 2 * i]
        inputs.append(a["y"])
        h = A(eng.dense(a, l1["W"], l1["b"]))
        inputs.append(h["y"])
        a = eng.residual(a, A(eng.dense(h, l2["W"], l2["b"])))
    inputs.append(a["y"])
    return eng.dense(a, pw[-1]["W"], pw[-1]["b"]), inputs
