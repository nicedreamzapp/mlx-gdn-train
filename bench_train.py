"""Short LoRA training benchmark on the real 27B base (bf16): old per-token path vs chunked.

    python bench_train.py --path new --steps 10
    python bench_train.py --path old --steps 3

Batch 1, one packed 2048-token sequence per step built from the pilot's train.jsonl,
LoRA rank 32 on every linear layer of every block, grad checkpointing, AdamW.
Same MLX cache limit as the pilot (2 GB) so the process footprint is comparable.
Prints one JSON line with per-step seconds, tokens/s, MLX peak and the process
phys_footprint (current and lifetime max, what forge_guard / Activity Monitor see).
"""

import argparse
import ctypes
import json
import os
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--path", choices=["old", "new"], required=True)
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--tf32", action="store_true", help="leave the M5's default TF32 fp32 matmuls on")
ap.add_argument("--model", default=os.path.expanduser(
    "~/.cache/huggingface/hub/models--donedynamics--Qwen3.8-27B-heretic-VL-MLX-bf16/snapshots/"
    "71c53db66b6a7ad02b6fa29b618e9cb02e2bf43a"))
ap.add_argument("--data", default="/Users/dtribe/life-model/runs/pilot-20261117-0845/data/train.jsonl")
ap.add_argument("--out", default=None)
a = ap.parse_args()
if not a.tf32:
    os.environ["MLX_ENABLE_TF32"] = "0"  # exact fp32 inside the chunked kernel; model weights are bf16 anyway

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.tuner.trainer import default_loss, grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gdn_patch

# register with forge_guard so it knows this process and its size (Song Forge still wins)
sys.path.insert(0, "/Users/dtribe/SongForgeM5")
from mem_client import reserve
_lease = reserve("gdn-bench", 72, ttl=1800, honest=False)
_lease.__enter__()

# ---- process footprint poller (proc_pid_rusage, RUSAGE_INFO_V4) ----
_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
_buf = (ctypes.c_uint64 * 64)()


def footprint():
    _libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(_buf))
    # v4 layout: uuid(16B)=2 u64, then user,sys,pkg_idle,intr,pageins,wired,resident,phys_footprint(idx 9)
    # ... ri_lifetime_max_phys_footprint is field index 2+... see <libproc.h>; idx 30 in u64 words
    return _buf[9], _buf[30]


_fp_max = [0]


def _poll():
    while True:
        cur, _ = footprint()
        _fp_max[0] = max(_fp_max[0], cur)
        time.sleep(0.25)


threading.Thread(target=_poll, daemon=True).start()

mx.set_cache_limit(2 * 1024**3)
if a.path == "new":
    assert gdn_patch.enable()

t0 = time.time()
model, tok = load(a.model)
load_s = time.time() - t0
model.freeze()
linear_to_lora_layers(model, len(model.layers), {"rank": 32, "scale": 2.0, "dropout": 0.05})
n_train = sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))
model.train()
grad_checkpoint(model.layers[0])

# ---- data: pack pilot examples into fixed seq-length rows ----
rows, buf = [], []
with open(a.data) as f:
    for line in f:
        msgs = json.loads(line)["messages"]
        buf += tok.apply_chat_template(msgs, tokenize=True)
        while len(buf) >= a.seq + 1:
            rows.append(buf[: a.seq + 1])
            buf = buf[a.seq + 1:]
        if len(rows) >= a.steps:
            break

opt = optim.AdamW(learning_rate=1e-5)
loss_vg = nn.value_and_grad(model, default_loss)
steps = []
for i in range(a.steps):
    batch = mx.array([rows[i]])
    lengths = mx.array([[1, a.seq]])
    mx.reset_peak_memory()
    t = time.time()
    (loss, ntoks), grads = loss_vg(model, batch, lengths)
    opt.update(model, grads)
    mx.eval(loss, model.trainable_parameters(), opt.state)
    dt = time.time() - t
    cur, life = footprint()
    s = {"step": i, "sec": round(dt, 2), "tok_s": round(a.seq / dt, 1), "loss": round(loss.item(), 4),
         "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2), "footprint_gb": round(cur / 1e9, 2),
         "footprint_max_gb": round(max(_fp_max[0], life) / 1e9, 2)}
    steps.append(s)
    print(json.dumps(s), flush=True)

warm = steps[1:] if len(steps) > 1 else steps
res = {"path": a.path, "tf32": a.tf32, "seq": a.seq, "steps": len(steps), "load_s": round(load_s, 1),
       "trainable_params_M": round(n_train / 1e6, 1),
       "sec_per_step_warm": round(sum(s["sec"] for s in warm) / len(warm), 2),
       "tok_s_warm": round(a.seq * len(warm) / sum(s["sec"] for s in warm), 1),
       "mlx_peak_gb": max(s["mlx_peak_gb"] for s in steps),
       "footprint_max_gb": max(s["footprint_max_gb"] for s in steps),
       "chunked_calls": gdn_patch.stats()["chunked_calls"], "per_step": steps}
print("RESULT " + json.dumps(res), flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
