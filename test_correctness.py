"""Chunked gated delta rule vs mlx_lm's reference per-token recurrence.

Forward y and final state, plus gradients wrt q, k, v, g, beta and the initial state.
Run with MLX_ENABLE_TF32=0 to see full-fp32 agreement; the M5 GPU defaults to TF32
for float32 matmuls, which caps fp32 agreement near 1e-3 (still below the bf16 noise floor).
Error columns: max abs error, and rel = max abs error / max |reference|.
In bf16 we also print the reference's own bf16-vs-fp32 error as the noise floor.

Packed rows (SEGMENTED / PACKED cases): a row holding several examples with seg_start marks
must give the same y, state and gradients as running every example on its own from a zero
state. Checked on the kernel (vs mlx_lm's recurrence run per segment) and on a small random
qwen3_5 model through gdn_patch (GDN state + conv reset, block-causal attention): logits and
parameter gradients of the packed row vs the sum over separate rows.
"""

import sys

import mlx.core as mx
import mlx.nn as nn
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


def seg_from_bounds(T, bounds):
    import numpy as np

    s = np.zeros(T, dtype=bool)
    for b in bounds:
        s[b] = True
    return mx.array(s)[None]


def ref_segmented(q, k, v, g_log, beta, s0, seg):
    """mlx_lm's recurrence run separately on each segment (zero state at every boundary)."""
    import numpy as np

    T = q.shape[1]
    starts = [i for i in range(T) if bool(np.array(seg)[0, i])]
    edges = sorted(set([0] + starts + [T]))
    ys, state = [], s0
    for a, b in zip(edges[:-1], edges[1:]):
        if a in starts:
            state = mx.zeros_like(s0)
        y, state = gated_delta_ops(q[:, a:b], k[:, a:b], v[:, a:b], mx.exp(g_log[:, a:b]), beta[:, a:b], state)
        ys.append(y)
    return mx.concatenate(ys, axis=1), state


def check_segments(shape, dtype, bounds, grads=True, seed=3):
    B, T, Hk, Hv, Dk, Dv = shape
    args = make_inputs(*shape, seed)
    seg = seg_from_bounds(T, bounds)
    seg_b = mx.broadcast_to(seg, (B, T))
    ref = lambda q, k, v, g, beta, s0: ref_segmented(q, k, v, g, beta, s0, seg_b)
    new = lambda q, k, v, g, beta, s0: chunk_gated_delta(q, k, v, g, beta, s0, seg_start=seg_b)
    yr, sr = run(ref, args, dtype)
    yn, sn = run(new, args, dtype)
    mx.eval(yr, sr, yn, sn)
    rows = [("y", *err(yn, yr)), ("state", *err(sn, sr))]
    floor_y = err(yr, run(ref, args, mx.float32)[0])[1] if dtype == mx.bfloat16 else None
    if grads:
        wy = mx.random.normal(yr.shape)
        ws = mx.random.normal(sr.shape)
        gr = loss_and_grads(ref, args, dtype, wy, ws)
        gn = loss_and_grads(new, args, dtype, wy, ws)
        rows += [("d" + n, *err(a, b)) for n, a, b in zip(NAMES, gn, gr)]
    worst = max(r[2] for r in rows)
    print(f"\nSEGMENTED bounds={bounds} B={B} T={T} Hk={Hk} Hv={Hv} Dk={Dk} Dv={Dv} {str(dtype).split('.')[-1]}  worst rel={worst:.2e}"
          + (f"  (ref bf16-vs-fp32 floor on y: {floor_y:.2e})" if floor_y else ""))
    for name, ab, rel in rows:
        print(f"  {name:8s} max_abs={ab:.3e}  rel={rel:.3e}")
    return worst


def tiny_qwen35(seed=0, layers=8):
    """Small random qwen3_5 text model: 6 GDN layers + 2 full-attention layers, fp32."""
    from mlx_lm.models import qwen3_5

    cfg = dict(model_type="qwen3_5", hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
               num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=97, rms_norm_eps=1e-6,
               linear_num_value_heads=6, linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
               linear_conv_kernel_dim=4, full_attention_interval=4, tie_word_embeddings=False,
               rope_parameters={"rope_type": "default", "rope_theta": 100000, "partial_rotary_factor": 0.25})
    mx.random.seed(seed)
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(cfg))
    # the default init is tiny for a 64-wide net; use N(0, 0.05) so layers actually do something
    from mlx.utils import tree_flatten, tree_unflatten

    params = {}
    for k, v in tree_flatten(model.parameters()):
        if k.endswith("A_log") or k.endswith("dt_bias") or "norm" in k:
            params[k] = v
        else:
            params[k] = 0.05 * mx.random.normal(v.shape)
    model.update(tree_unflatten(list(params.items())))
    mx.eval(model.parameters())
    return model


def check_packed_model(lengths, dtype, seed=0):
    """Packed row of random token examples vs each example alone, through gdn_patch."""
    import numpy as np
    from mlx.utils import tree_flatten
    import gdn_patch

    gdn_patch.enable()
    model = tiny_qwen35(seed)
    if dtype == mx.bfloat16:  # like mlx_lm's cast_predicate: everything bf16 except A_log
        from mlx.utils import tree_unflatten

        model.update(tree_unflatten([(k, v if k.endswith("A_log") else v.astype(mx.bfloat16))
                                     for k, v in tree_flatten(model.parameters())]))
    model.train()
    T = sum(lengths)
    rng = np.random.default_rng(seed)
    toks = rng.integers(0, 97, size=T, dtype=np.int32)
    wts = rng.random(T).astype(np.float32)  # per-token loss weights (like a loss mask)
    bounds = list(np.cumsum([0] + lengths[:-1]))
    seg = seg_from_bounds(T, bounds)
    x = mx.array(toks)[None]
    tgt = mx.array(np.roll(toks, -1))[None]
    w = mx.array(wts)[None]

    def loss_packed(model):
        gdn_patch.set_segments(seg)
        lg = model(x).astype(mx.float32)
        gdn_patch.set_segments(None)
        return (nn.losses.cross_entropy(lg, tgt) * w).sum(), lg

    def loss_separate(model):
        tot, lgs = 0.0, []
        for a, L in zip(bounds, lengths):
            lg = model(x[:, a : a + L]).astype(mx.float32)
            tot = tot + (nn.losses.cross_entropy(lg, tgt[:, a : a + L]) * w[:, a : a + L]).sum()
            lgs.append(lg)
        return tot, mx.concatenate(lgs, axis=1)

    (lp, lg_p), gp = nn.value_and_grad(model, loss_packed)(model)
    (ls, lg_s), gs = nn.value_and_grad(model, loss_separate)(model)
    mx.eval(lp, lg_p, gp, ls, lg_s, gs)
    rows = [("logits", *err(lg_p, lg_s)), ("loss", abs(lp.item() - ls.item()), abs(lp.item() - ls.item()) / abs(ls.item()))]
    gp, gs = dict(tree_flatten(gp)), dict(tree_flatten(gs))
    gnum = max(err(gp[k], gs[k])[1] for k in gs if mx.abs(gs[k]).max().item() > 0)
    worst_name = max((k for k in gs if mx.abs(gs[k]).max().item() > 0), key=lambda k: err(gp[k], gs[k])[1])
    rows.append(("grads(all params, worst)", *err(gp[worst_name], gs[worst_name])))
    others = [k for k in gs if mx.abs(gs[k]).max().item() > 0 and not k.endswith(("A_log", "dt_bias"))]
    wo = max(others, key=lambda k: err(gp[k], gs[k])[1])
    rows.append(("grads(no A_log/dt_bias)", *err(gp[wo], gs[wo])))
    floor = gfloor = None
    if dtype == mx.bfloat16:  # the reference's own bf16-vs-fp32 noise, logits and grads
        m32 = tiny_qwen35(seed); m32.train()
        gdn_patch.set_segments(None)
        (_, lg32), g32 = nn.value_and_grad(m32, loss_separate)(m32)
        mx.eval(lg32, g32)
        floor = err(lg_s, lg32)[1]
        g32 = dict(tree_flatten(g32))
        gfloor = max(err(gs[k], g32[k])[1] for k in gs if mx.abs(g32[k]).max().item() > 0)
        rows.append(("ref grad floor bf16-vs-fp32 (worst tensor)", 0.0, gfloor))
    # also: the plain (unsegmented) packed row must NOT match, or the test proves nothing
    gdn_patch.set_segments(None)
    lg_plain = model(x).astype(mx.float32)
    leak = err(lg_plain, lg_s)[1]
    worst = max(r[2] for r in rows if not r[0].startswith("ref grad floor"))
    print(f"\nPACKED MODEL lengths={lengths} T={T} {str(dtype).split('.')[-1]}  worst rel={worst:.2e}"
          + (f"  (ref bf16-vs-fp32 floor on logits: {floor:.2e})" if floor else "")
          + f"  [unsegmented packing differs by rel {leak:.2e}: the leak the fix removes]")
    for name, ab, rel in rows:
        print(f"  {name:40s} max_abs={ab:.3e}  rel={rel:.3e}")
    print(f"  worst grad tensor: {worst_name}")
    assert leak > 1e-2, "unsegmented packed row matched the separate runs; test is not exercising the boundary"
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
    print("\nworst relative error (plain rows):", worst)

    # packed rows: boundaries at 0, chunk-aligned (64, 128), mid-chunk, several per chunk, last token
    seg_cases = [
        ((1, 64, 1, 1, 32, 32), [0, 20, 45]),
        ((2, 100, 2, 6, 32, 32), [0, 64, 70, 71, 99]),
        ((2, 130, 4, 4, 64, 32), [37, 64, 128]),
        ((1, 300, 16, 48, 128, 128), [0, 5, 64, 100, 101, 250]),
        ((1, 500, 16, 48, 128, 128), [128, 129, 300]),
    ]
    worst_seg = {}
    for dtype in (mx.float32, mx.bfloat16):
        for shape, bounds in seg_cases:
            w = check_segments(shape, dtype, bounds)
            worst_seg[str(dtype)] = max(worst_seg.get(str(dtype), 0), w)
    print("\nworst relative error (segmented kernel vs per-segment reference):", worst_seg)

    worst_model = {}
    for dtype in (mx.float32, mx.bfloat16):
        for lengths in ([37, 70, 21], [10, 12, 9, 33, 64, 2, 70], [64, 64, 64]):
            w = check_packed_model(lengths, dtype)
            worst_model[str(dtype)] = max(worst_model.get(str(dtype), 0), w)
    print("\nworst relative error (packed qwen3_5 model vs separate examples):", worst_model)
    sys.exit(0)
