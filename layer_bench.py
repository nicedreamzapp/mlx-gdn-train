"""One real-size GatedDeltaNet layer (Qwen3.8-27B dims), T=2048, forward+backward. Old vs chunked."""
import sys, time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5
import gdn_patch

args = qwen3_5.TextModelArgs(model_type="qwen3_5_text", hidden_size=5120, num_hidden_layers=1,
    intermediate_size=17408, num_attention_heads=24, num_key_value_heads=4, rms_norm_eps=1e-6,
    vocab_size=1000, linear_num_value_heads=48, linear_num_key_heads=16, linear_key_head_dim=128,
    linear_value_head_dim=128, linear_conv_kernel_dim=4, head_dim=256, full_attention_interval=4)
layer = qwen3_5.GatedDeltaNet(args); layer.set_dtype(mx.bfloat16); layer.train()
x = (0.5 * mx.random.normal((1, int(sys.argv[2]) if len(sys.argv) > 2 else 2048, 5120))).astype(mx.bfloat16)
w = mx.random.normal(x.shape).astype(mx.bfloat16)
def loss(params, x):
    layer.update(params); return (layer(x).astype(mx.float32) * w).sum()
vg = mx.value_and_grad(loss)
if sys.argv[1] == "new": gdn_patch.enable()
mx.eval(layer.parameters())
for i in range(4):
    mx.reset_peak_memory(); t = time.time()
    l, g = vg(layer.trainable_parameters(), x); mx.eval(l, g)
    print(f"{sys.argv[1]} iter {i}: {time.time()-t:.3f}s  peak {mx.get_peak_memory()/1e9:.2f} GB")
