"""Chunkwise-parallel (WY) gated delta rule in pure MLX ops.

Same math as mlx_lm.models.gated_delta.gated_delta_ops (the per-token recurrence):

    S_t = g_t * S_{t-1}
    S_t = S_t + k_t (beta_t * (v_t - S_t k_t))^T        (state kept as [Dv, Dk] in mlx_lm)
    y_t = S_t q_t

but computed 64 tokens at a time, so the graph has ~T/64 small sequential steps
instead of T, and MLX autograd gives a GPU backward with O(T/64) stored states.

Inputs follow mlx_lm's layout:
    q, k   [B, T, Hk, Dk]   (already rms-normed / scaled by the caller, as in qwen3_5)
    v      [B, T, Hv, Dv]
    g_log  [B, T, Hv]       log of the per-step decay (<= 0)
    beta   [B, T, Hv]
    state  [B, Hv, Dv, Dk]  or None
    seg_start [B, T] bool, optional: True on the first token of every packed example. The
           recurrent state is reset to zero there (decay 0 on that step), so each example in a
           packed row sees only its own tokens. Implemented with exact masks on the chunk math
           (same-segment pairs, "no boundary yet" / "no boundary after" per token), not with a
           huge negative log-decay, so fp32 agreement stays at ~1e-5. None = plain path.
Returns y [B, T, Hv, Dv] (dtype of q) and the final state [B, Hv, Dv, Dk] (float32).
"""

from typing import Optional, Tuple

import mlx.core as mx

CHUNK = 64


_level_masks = {}


def _masks(C):
    if C not in _level_masks:
        i = mx.arange(C)[:, None]
        j = mx.arange(C)[None, :]
        ms, s = [], 1
        while s < C:
            ms.append((i // (2 * s) == j // (2 * s)) & (i // s != j // s))
            s *= 2
        _level_masks[C] = ms
    return _level_masks[C]


def _unit_lower_inverse(A: mx.array, C: int) -> mx.array:
    """(I + A)^{-1} for strictly lower-triangular A, by recursive 2x2 block inversion
    done on full CxC matrices: with D = inverse of the s-sized diagonal blocks and O the
    lower-left blocks that join pairs of them, the 2s-sized inverse is D - D @ O @ D.
    log2(C) levels, 2 batched matmuls each, all on the GPU.
    (A Neumann/doubling product (I-N)(I+N^2)... is NOT used: with near-identical keys,
    e.g. runs of repeated tokens, N^32 reaches ~1e17 and cancels to NaN.)"""
    D = mx.eye(C, dtype=A.dtype)
    for m in _masks(C):
        O = mx.where(m, A, 0.0)
        D = D - D @ O @ D
    return D


def chunk_gated_delta(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g_log: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    seg_start: Optional[mx.array] = None,
    chunk: int = CHUNK,
) -> Tuple[mx.array, mx.array]:
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    out_dtype = q.dtype
    C = chunk

    if (r := Hv // Hk) > 1:
        q = mx.repeat(q, r, -2)
        k = mx.repeat(k, r, -2)

    f32 = mx.float32
    q, k, v = q.astype(f32), k.astype(f32), v.astype(f32)
    g_log, beta = g_log.astype(f32), beta.astype(f32)
    seg = None
    if seg_start is not None:
        seg = seg_start.astype(mx.bool_)
        if seg.ndim == 1:
            seg = mx.broadcast_to(seg[None], (B, T))
        # the boundary token's own decay never acts on anything (the state it would decay is
        # zeroed), so drop it from the cumulative sums: cleaner numerics, exact zero gradient
        g_log = mx.where(seg[..., None], 0.0, g_log)

    pad = (-T) % C
    if pad:
        # zero k/v/beta and zero log-decay leave the state untouched
        q, k, v = (mx.pad(t, [(0, 0), (0, pad), (0, 0), (0, 0)]) for t in (q, k, v))
        g_log, beta = (mx.pad(t, [(0, 0), (0, pad), (0, 0)]) for t in (g_log, beta))
        if seg is not None:
            seg = mx.pad(seg, [(0, 0), (0, pad)])
    N = (T + pad) // C

    # [B, T, H, D] -> [B, H, N, C, D]
    def chunks(t):
        return t.reshape(B, N, C, Hv, t.shape[-1]).transpose(0, 3, 1, 2, 4)

    q, k, v = chunks(q), chunks(k), chunks(v)
    g_log = g_log.reshape(B, N, C, Hv).transpose(0, 3, 1, 2)  # [B, H, N, C]
    beta = beta.reshape(B, N, C, Hv).transpose(0, 3, 1, 2)

    G = mx.cumsum(g_log, axis=-1)  # within-chunk cumulative log decay
    G_last = G[..., -1:]  # [B, H, N, 1]

    incl = mx.tril(mx.ones((C, C), dtype=mx.bool_))
    strict = mx.tril(mx.ones((C, C), dtype=mx.bool_), k=-1)
    if seg is not None:
        # chunk-local segment index per token: [B, 1, N, C] (broadcasts over heads)
        cnt = mx.cumsum(seg.reshape(B, N, C).astype(mx.int32), axis=-1)[:, None]
        same = cnt[..., :, None] == cnt[..., None, :]  # [B, 1, N, C, C] same example
        incl = incl & same
        nb = (cnt == 0).astype(f32)  # no boundary at or before token i: may read the carried state
        nl = (cnt == cnt[..., -1:]).astype(f32)  # no boundary after token j: feeds the next state
        nochunk = (cnt[..., -1:] == 0).astype(f32)  # [B, 1, N, 1] no boundary in this chunk
    diff = G[..., :, None] - G[..., None, :]  # G_i - G_j
    L = mx.exp(mx.where(incl, diff, -mx.inf))  # decay from j to i, 0 above diagonal / across examples

    kT = k.swapaxes(-1, -2)
    kb = k * beta[..., None]
    A = mx.where(strict, (kb @ kT) * L, 0.0)
    Tm = _unit_lower_inverse(A, C)  # (I + A)^{-1}

    eG = mx.exp(G)  # decay from chunk start to token i
    eGl = mx.exp(G_last - G)  # decay from token j to chunk end
    dlast = mx.exp(G_last)[..., None]  # [B, H, N, 1, 1]
    if seg is not None:
        eG = eG * nb
        eGl = eGl * nl
        dlast = dlast * nochunk[..., None]

    u = Tm @ (v * beta[..., None])  # [B, H, N, C, Dv]
    w = Tm @ (kb * eG[..., None])  # [B, H, N, C, Dk]

    qg = q * eG[..., None]  # for the inter-chunk read-out
    Aqk = (q @ kT) * L  # intra-chunk causal scores incl. diagonal
    kdec = (k * eGl[..., None]).swapaxes(-1, -2)  # [B, H, N, Dk, C]

    # recurrence over chunks; S kept as [Dk, Dv]
    if state is None:
        S = mx.zeros((B, Hv, Dk, Dv), dtype=f32)
    else:
        S = state.astype(f32).swapaxes(-1, -2)
    S_in, v_new = [], []
    for n in range(N):
        S_in.append(S)
        vn = u[:, :, n] - w[:, :, n] @ S
        v_new.append(vn)
        S = S * dlast[:, :, n] + kdec[:, :, n] @ vn
    S_in = mx.stack(S_in, axis=2)  # [B, H, N, Dk, Dv]
    v_new = mx.stack(v_new, axis=2)  # [B, H, N, C, Dv]

    y = qg @ S_in + Aqk @ v_new  # [B, H, N, C, Dv]
    y = y.transpose(0, 2, 3, 1, 4).reshape(B, N * C, Hv, Dv)[:, :T]
    return y.astype(out_dtype), S.swapaxes(-1, -2)
