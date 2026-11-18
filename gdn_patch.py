"""Drop-in: route mlx_lm's qwen3_5 (and qwen3_next / qwen3_5_moe) linear-attention
TRAINING path through the chunked gated delta rule. No edits to site-packages.

    import gdn_patch; gdn_patch.enable()     # before or after mlx_lm.load, before training

Off switch: set MLX_GDN_CHUNK=0 in the environment (enable() then does nothing),
or call gdn_patch.disable() at runtime.

Inference is untouched: when the model is not in training mode mlx_lm already uses its
fast Metal kernel (use_kernel=True), and we pass that straight through. Calls with a
padding mask (batched inference) also go to the original.

Packed rows (several examples in one sequence) are separated per example when you tell the
patch where each example starts:

    gdn_patch.set_segments(seg_start)   # [B, T] bool, True on the first token of each example
    loss, grads = value_and_grad(model, x, ...)
    gdn_patch.set_segments(None)        # back to plain rows

While segments are set (qwen3_5 only, training path, no KV cache):
  * every Gated DeltaNet layer resets its recurrent state at each example start
    (chunk_gated_delta(..., seg_start=...)),
  * its depthwise causal conv1d (kernel 4) does not read the previous example's last 3 tokens
    (the conv state is zero at each example start),
  * the full-attention layers get a block-diagonal causal boolean mask, so a token attends
    only to earlier tokens of its own example.
RoPE positions are left as they are: attention scores depend only on relative position, so a
masked example at offset p gives the same output as the same example at offset 0 (up to
rounding). Result: a packed row gives the same outputs and gradients as running each example
on its own (test_correctness.py, "PACKED" cases).
"""

import importlib
import os
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from gdn_chunk import chunk_gated_delta

_TARGETS = ["mlx_lm.models.qwen3_5", "mlx_lm.models.qwen3_next", "mlx_lm.models.qwen3_5_moe"]
_originals = {}
_stats = {"chunked_calls": 0, "segmented_calls": 0}
_ctx = {"seg_start": None}


def set_segments(seg_start: Optional[mx.array]):
    """seg_start: [B, T] (or [T]) bool/int, True on the first token of every packed example,
    or None to turn segment separation off."""
    if seg_start is not None:
        seg_start = mx.array(seg_start) if not isinstance(seg_start, mx.array) else seg_start
        seg_start = seg_start.astype(mx.bool_)
        if seg_start.ndim == 1:
            seg_start = seg_start[None]
    _ctx["seg_start"] = seg_start


def segments():
    return _ctx["seg_start"]


def block_causal_mask(seg_start: mx.array) -> mx.array:
    """[B, 1, T, T] bool: causal and same example. Row i always keeps the diagonal."""
    ids = mx.cumsum(seg_start.astype(mx.int32), axis=1)  # [B, T]
    T = ids.shape[1]
    causal = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    same = ids[:, :, None] == ids[:, None, :]  # [B, T, T]
    return (causal[None] & same)[:, None]


def _make_patched(orig):
    def gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
        if use_kernel or mask is not None:
            return orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)
        _stats["chunked_calls"] += 1
        beta = mx.sigmoid(b)
        # log of mlx_lm's compute_g, without the exp/log round trip
        g_log = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
        seg = _ctx["seg_start"]
        if seg is not None:
            _stats["segmented_calls"] += 1
        return chunk_gated_delta(q, k, v, g_log, beta, state, seg_start=seg)

    return gated_delta_update


def _segmented_conv(conv: nn.Conv1d, x: mx.array, seg_start: mx.array) -> mx.array:
    """Depthwise causal conv1d (no bias, padding 0 on an input left-padded by K-1 zeros), but a
    tap that would read a token from an earlier example reads zero instead. Same numbers as
    conv(concat(zeros, x)) when seg_start marks only token 0."""
    W = conv.weight  # [C, K, 1]
    K = W.shape[1]
    ids = mx.cumsum(seg_start.astype(mx.int32), axis=1)  # [B, T]
    B, T, _ = x.shape
    out = x * W[:, K - 1, 0]
    for d in range(1, K):
        src = mx.pad(x[:, : T - d], [(0, 0), (d, 0), (0, 0)])
        prev = mx.pad(ids[:, : T - d], [(0, 0), (d, 0)], constant_values=-1)
        ok = (prev == ids)[..., None]
        out = out + mx.where(ok, src, 0.0) * W[:, K - 1 - d, 0]
    return out


def _make_patched_gdn_call(orig_call):
    def __call__(self, inputs, mask=None, cache=None):
        seg = _ctx["seg_start"]
        if seg is None or cache is not None or mask is not None or not self.training:
            return orig_call(self, inputs, mask, cache)
        B, S, _ = inputs.shape
        if self.sharding_group is not None:
            return orig_call(self, inputs, mask, cache)
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)
        conv_out = nn.silu(_segmented_conv(self.conv1d, qkv, seg))
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        mod = importlib.import_module("mlx_lm.models.qwen3_5")
        out, _ = mod.gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, None, None, use_kernel=False)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))

    return __call__


def _make_patched_attn_mask(orig_fn):
    def create_attention_mask(h, cache=None, window_size=None, return_array=False):
        seg = _ctx["seg_start"]
        if seg is None or cache is not None:
            return orig_fn(h, cache, window_size, return_array)
        if seg.shape[1] != h.shape[1]:
            raise ValueError(f"segments length {seg.shape[1]} != sequence length {h.shape[1]}")
        return block_causal_mask(seg)

    return create_attention_mask


def enable():
    if os.environ.get("MLX_GDN_CHUNK", "1") == "0":
        print("[gdn_patch] MLX_GDN_CHUNK=0: chunked path OFF, using mlx_lm's per-token training path")
        return False
    for name in _TARGETS:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if name in _originals or not hasattr(mod, "gated_delta_update"):
            continue
        _originals[name] = mod.gated_delta_update
        mod.gated_delta_update = _make_patched(mod.gated_delta_update)
    # per-example separation inside packed rows (qwen3_5 only: that is what we train)
    q35 = importlib.import_module("mlx_lm.models.qwen3_5")
    if "qwen3_5.GatedDeltaNet.__call__" not in _originals:
        _originals["qwen3_5.GatedDeltaNet.__call__"] = q35.GatedDeltaNet.__call__
        q35.GatedDeltaNet.__call__ = _make_patched_gdn_call(q35.GatedDeltaNet.__call__)
    if "qwen3_5.create_attention_mask" not in _originals:
        _originals["qwen3_5.create_attention_mask"] = q35.create_attention_mask
        q35.create_attention_mask = _make_patched_attn_mask(q35.create_attention_mask)
    print(f"[gdn_patch] chunked gated-delta training path ON for {sorted(k for k in _originals if k.startswith('mlx_lm'))}; "
          "packed-row separation (GDN state + conv reset, block-causal attention) via set_segments()")
    return True


def disable():
    q35 = importlib.import_module("mlx_lm.models.qwen3_5")
    for name, fn in _originals.items():
        if name == "qwen3_5.GatedDeltaNet.__call__":
            q35.GatedDeltaNet.__call__ = fn
        elif name == "qwen3_5.create_attention_mask":
            q35.create_attention_mask = fn
        else:
            importlib.import_module(name).gated_delta_update = fn
    _originals.clear()
    _ctx["seg_start"] = None


def stats():
    return dict(_stats)
