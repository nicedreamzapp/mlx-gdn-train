"""Real-model check on real pilot text, old per-token path vs chunked, training mode:
  1. loss over 512 tokens (forward only)
  2. gradients of one linear-attention layer's A_log / dt_bias (flow through the gated delta rule)
     over 128 tokens with grad checkpointing (the old path's backward is the memory hog)."""
import json, os, sys
os.environ.setdefault("MLX_ENABLE_TF32", "0")
import mlx.core as mx, mlx.nn as nn
from mlx_lm.utils import load
from mlx_lm.tuner.trainer import default_loss, grad_checkpoint
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gdn_patch
sys.path.insert(0, "/Users/dtribe/SongForgeM5")
from mem_client import reserve
_lease = reserve("gdn-bench", 72, ttl=900, honest=False); _lease.__enter__()
mx.set_cache_limit(2 * 1024**3)
model, tok = load(os.path.expanduser("~/.cache/huggingface/hub/models--donedynamics--Qwen3.8-27B-heretic-VL-MLX-bf16/snapshots/71c53db66b6a7ad02b6fa29b618e9cb02e2bf43a"))
buf = []
for line in open("/Users/dtribe/life-model/runs/pilot-20261117-0845/data/train.jsonl"):
    buf += tok.apply_chat_template(json.loads(line)["messages"], tokenize=True)
    if len(buf) > 513: break
LAYER = 0
model.freeze(); model.layers[LAYER].linear_attn.unfreeze(keys=["A_log", "dt_bias"], recurse=False); model.train()
grad_checkpoint(model.layers[0])
vg = nn.value_and_grad(model, default_loss)
out = {}
for name in ("old", "new"):
    if name == "new": gdn_patch.enable()
    l512, _ = default_loss(model, mx.array([buf[:513]]), mx.array([[1, 512]])); mx.eval(l512)
    (l, _), g = vg(model, mx.array([buf[:129]]), mx.array([[1, 128]])); mx.eval(l, g)
    g = dict(nn.utils.tree_flatten(g))
    ga = [v for k, v in g.items() if k.endswith("A_log")][0].astype(mx.float32)
    gd = [v for k, v in g.items() if k.endswith("dt_bias")][0].astype(mx.float32)
    out[name] = (l512.item(), ga, gd)
    print(f"{name}: loss512={l512.item():.6f} loss128={l.item():.6f} peak={mx.get_peak_memory()/1e9:.1f}GB", flush=True)
for i, n in ((1, "dA_log"), (2, "ddt_bias")):
    a, b = out["new"][i], out["old"][i]
    print(f"{n}: max_abs={mx.abs(a - b).max().item():.3e} rel={(mx.abs(a - b).max() / mx.abs(b).max()).item():.3e}")
print(f"loss512 diff={abs(out['new'][0] - out['old'][0]):.2e}  chunked calls={gdn_patch.stats()['chunked_calls']}")
