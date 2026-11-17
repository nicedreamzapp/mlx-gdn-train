"""Real-model checks, part 2:
  A. noise floor: 512-token loss with mlx_lm's own Metal inference kernel vs its per-token ops path
     vs the chunked path (all three implement the same math; kernel-vs-ops = bf16 noise floor).
  B. NaN hunt at 2048 tokens (bench row 0): kernel path vs chunked path, first layer that goes non-finite.
  C. one real GatedDeltaNet layer on real hidden states, fp32 weights, old vs chunked: output and
     grads wrt A_log, dt_bias, in_proj_a/b weights."""
import json, os, sys
os.environ.setdefault("MLX_ENABLE_TF32", "0")
import mlx.core as mx, mlx.nn as nn
from mlx_lm.utils import load
from mlx_lm.tuner.trainer import default_loss
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gdn_patch
sys.path.insert(0, "/Users/dtribe/SongForgeM5")
from mem_client import reserve
_lease = reserve("gdn-bench", 72, ttl=900, honest=False); _lease.__enter__()
try:
    mx.set_cache_limit(2 * 1024**3)
    model, tok = load(os.path.expanduser("~/.cache/huggingface/hub/models--donedynamics--Qwen3.8-27B-heretic-VL-MLX-bf16/snapshots/71c53db66b6a7ad02b6fa29b618e9cb02e2bf43a"))
    buf = []
    for line in open("/Users/dtribe/life-model/runs/pilot-20261117-0845/data/train.jsonl"):
        buf += tok.apply_chat_template(json.loads(line)["messages"], tokenize=True)
        if len(buf) > 2049: break
    model.freeze(); model.train()
    las = [l.linear_attn for l in model.layers if l.is_linear]
    def kernel_mode(on):
        for la in las: la.eval() if on else la.train()
    def loss(n):
        l, _ = default_loss(model, mx.array([buf[:n + 1]]), mx.array([[1, n]])); return l.item()
    # A
    kernel_mode(True); lk = loss(512)
    kernel_mode(False); lo = loss(512)
    gdn_patch.enable(); ln = loss(512)
    print(f"A 512-token loss: metal-kernel={lk:.6f} ops(old train path)={lo:.6f} chunked={ln:.6f} | "
          f"|kernel-ops|={abs(lk-lo):.2e} |chunked-ops|={abs(ln-lo):.2e} |chunked-kernel|={abs(ln-lk):.2e}", flush=True)
    # B
    def first_bad(chunked):
        if not chunked: kernel_mode(True)
        h = model.language_model.model.embed_tokens(mx.array([buf[:2048]])) if hasattr(model, "language_model") else model.model.embed_tokens(mx.array([buf[:2048]]))
        from mlx_lm.models.base import create_attention_mask
        fam = create_attention_mask(h, None)
        for i, l in enumerate(model.layers):
            h = l(h, mask=None if l.is_linear else fam, cache=None); mx.eval(h)
            if not mx.isfinite(h).all().item():
                kernel_mode(False); return i, l.is_linear
        kernel_mode(False); return None, None
    print(f"B 2048 tokens: loss kernel={(kernel_mode(True), loss(2048))[1]:.4f}", flush=True); kernel_mode(False)
    print(f"B 2048 tokens: loss chunked={loss(2048):.4f}", flush=True)
    print("B first non-finite layer: kernel", first_bad(False), " chunked", first_bad(True), flush=True)
    # C
    la = model.layers[0].linear_attn
    emb = (model.language_model.model.embed_tokens if hasattr(model, "language_model") else model.model.embed_tokens)
    x = model.layers[0].input_layernorm(emb(mx.array([buf[:1024]]))).astype(mx.float32)
    la.set_dtype(mx.float32); la.unfreeze(); la.train()
    w = mx.random.normal((1, 1024, x.shape[-1]))
    keys = ["A_log", "dt_bias", "in_proj_a.weight", "in_proj_b.weight", "in_proj_qkv.weight"]
    def f(params):
        la.update(params); return (la(x) * w).sum()
    res = {}
    for name in ("old", "new"):
        if name == "old": gdn_patch.disable()
        else: gdn_patch.enable()
        val, g = mx.value_and_grad(f)(la.trainable_parameters()); y = la(x); mx.eval(val, g, y)
        res[name] = (y, dict(nn.utils.tree_flatten(g)))
    yo, go = res["old"]; yn, gn = res["new"]
    print(f"C real layer0 fp32, 1024 real tokens: y rel={(mx.abs(yn-yo).max()/mx.abs(yo).max()).item():.2e}")
    for k in keys:
        a, b = gn[k], go[k]
        print(f"  d{k}: rel={(mx.abs(a-b).max()/mx.abs(b).max()).item():.2e}")
finally:
    _lease.__exit__(None, None, None)
