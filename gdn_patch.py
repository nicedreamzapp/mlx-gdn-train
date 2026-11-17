"""Drop-in: route mlx_lm's qwen3_5 (and qwen3_next / qwen3_5_moe) linear-attention
TRAINING path through the chunked gated delta rule. No edits to site-packages.

    import gdn_patch; gdn_patch.enable()     # before or after mlx_lm.load, before training

Off switch: set MLX_GDN_CHUNK=0 in the environment (enable() then does nothing),
or call gdn_patch.disable() at runtime.

Inference is untouched: when the model is not in training mode mlx_lm already uses its
fast Metal kernel (use_kernel=True), and we pass that straight through. Calls with a
padding mask (batched inference) also go to the original.
"""

import importlib
import os

import mlx.core as mx
import mlx.nn as nn

from gdn_chunk import chunk_gated_delta

_TARGETS = ["mlx_lm.models.qwen3_5", "mlx_lm.models.qwen3_next", "mlx_lm.models.qwen3_5_moe"]
_originals = {}
_stats = {"chunked_calls": 0}


def _make_patched(orig):
    def gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
        if use_kernel or mask is not None:
            return orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)
        _stats["chunked_calls"] += 1
        beta = mx.sigmoid(b)
        # log of mlx_lm's compute_g, without the exp/log round trip
        g_log = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
        return chunk_gated_delta(q, k, v, g_log, beta, state)

    return gated_delta_update


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
    print(f"[gdn_patch] chunked gated-delta training path ON for {sorted(_originals)}")
    return True


def disable():
    for name, fn in _originals.items():
        importlib.import_module(name).gated_delta_update = fn
    _originals.clear()


def stats():
    return dict(_stats)
