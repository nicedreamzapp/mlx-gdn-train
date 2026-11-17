"""Chunked gated delta rule vs mlx_lm's reference per-token recurrence.

Forward y and final state, plus gradients wrt q, k, v, g, beta and the initial state.
Run with MLX_ENABLE_TF32=0 to see full-fp32 agreement; the M5 GPU defaults to TF32
for float32 matmuls, which caps fp32 agreement near 1e-3 (still below the bf16 noise floor).
Error columns: max abs error, and rel = max abs error / max |reference|.
In bf16 we also print the reference's own bf16-vs-fp32 error as the noise floor.
"""

import sys

import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_ops

from gdn_chunk import chunk_gated_delta


def make_inputs(B, T, Hk, Hv, Dk, Dv, seed, repeat=False):
    mx.random.seed(seed)
    q = mx.random.normal((B, T, Hk, Dk))
    k = mx.random.normal((B, T, Hk, Dk))
    if repeat:
        # runs of repeated tokens (tabs/whitespace in real email data): near-identical keys,
        # beta ~ 1 and almost no decay -- the case that breaks naive chunk inversions
        k = k[:, :1] + 0.01 * k
    # as in qwen3_5: k = l2norm(k), q = l2norm(q) / sqrt(Dk)
    k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    q = q / mx.linalg.norm(q, axis=-1, keepdims=True) / Dk**0.5
    v = mx.random.normal((B, T, Hv, Dv))
    a = mx.random.normal((B, T, Hv))
    A_log = mx.log(mx.random.uniform(0, 16, (Hv,)))
    g = -mx.exp(A_log) * mx.logaddexp(a + 1.0, 0.0)  # log decay, as in the model
    beta = mx.sigmoid(mx.random.normal((B, T, Hv)))
    if repeat:
        g = -1e-3 * mx.abs(a)
        beta = mx.sigmoid(4 + mx.random.normal((B, T, Hv)))
    s0 = 0.1 * mx.random.normal((B, Hv, Dv, Dk))
    return [q, k, v, g, beta, s0]


def run(fn, args, dtype):
    q, k, v, g, beta, s0 = args
    q, k, v, beta = (t.astype(dtype) for t in (q, k, v, beta))
    return fn(q, k, v, g, beta, s0)


# "g" below is the log-decay (what the model's gradient actually flows through:
# g = exp(-exp(A_log) * softplus(a + dt_bias))); the reference gets exp(g_log).
def ref_fn(q, k, v, g_log, beta, s0):
    return gated_delta_ops(q, k, v, mx.exp(g_log), beta, s0)


def new_fn(q, k, v, g_log, beta, s0):
    return chunk_gated_delta(q, k, v, g_log, beta, s0)


def loss_and_grads(fn, args, dtype, wy, ws):
    def loss(q, k, v, g, beta, s0):
        y, s = fn(q, k, v, g, beta, s0)
        return (y.astype(mx.float32) * wy).sum() + (s * ws).sum()

    cast = [a.astype(dtype) if i in (0, 1, 2, 4) else a for i, a in enumerate(args)]
    grads = mx.grad(loss, argnums=list(range(6)))(*cast)
    mx.eval(grads)
    return grads


def err(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    ab = mx.abs(a - b).max().item()
    return ab, ab / max(mx.abs(b).max().item(), 1e-30)


NAMES = ["q", "k", "v", "g_log", "beta", "state0"]


def check(shape, dtype, grads=True, seed=0, repeat=False):
    B, T, Hk, Hv, Dk, Dv = shape
    args = make_inputs(*shape, seed, repeat)
    yr, sr = run(ref_fn, args, dtype)
    yn, sn = run(new_fn, args, dtype)
    mx.eval(yr, sr, yn, sn)
    rows = [("y", *err(yn, yr)), ("state", *err(sn, sr))]
    if dtype == mx.bfloat16:
        y32, s32 = run(ref_fn, args, mx.float32)
        floor_y = err(yr, y32)[1]
    else:
        floor_y = None
    if grads:
        wy = mx.random.normal(yr.shape)
        ws = mx.random.normal(sr.shape)
        gr = loss_and_grads(ref_fn, args, dtype, wy, ws)
        gn = loss_and_grads(new_fn, args, dtype, wy, ws)
        rows += [("d" + n, *err(a, b)) for n, a, b in zip(NAMES, gn, gr)]
    tag = ("REPEATED-KEYS " if repeat else "") + f"B={B} T={T} Hk={Hk} Hv={Hv} Dk={Dk} Dv={Dv} {str(dtype).split('.')[-1]}"
    worst = max(r[2] for r in rows)
    print(f"\n{tag}  worst rel={worst:.2e}" + (f"  (ref bf16-vs-fp32 floor on y: {floor_y:.2e})" if floor_y else ""))
    for name, ab, rel in rows:
        print(f"  {name:8s} max_abs={ab:.3e}  rel={rel:.3e}")
    return worst


if __name__ == "__main__":
    shapes = [
        ((1, 64, 1, 1, 32, 32), True),
        ((2, 100, 2, 6, 32, 32), True),  # GQA x3, T not multiple of 64
        ((2, 130, 4, 4, 64, 32), True),  # Dk != Dv
        ((1, 300, 16, 48, 128, 128), True),  # the model's real linear-attn dims
        ((1, 2048, 16, 48, 128, 128), False),  # real dims, full training length, forward only
        ((1, 1000, 16, 48, 128, 128), True),  # real dims, long, grads
        ((1, 500, 16, 48, 128, 128), True, True),  # real dims, adversarial repeated keys
    ]
    worst = {}
    for dtype in (mx.float32, mx.bfloat16):
        for shape, g, *rep in shapes:
            w = check(shape, dtype, grads=g, repeat=bool(rep))
            worst[str(dtype)] = max(worst.get(str(dtype), 0), w)
    print("\nworst relative error:", worst)
    sys.exit(0)
