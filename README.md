# mlx-gdn-train

This folder makes LoRA training of Qwen3.8-27B (mlx_lm `qwen3_5`, 48 Gated DeltaNet layers + 16
full-attention layers) fast enough and small enough to run on the M5 (128 GB). It uses MLX only and
does not edit site-packages.

## The problem

mlx_lm 0.31.2 (and 0.32.2) only use their Metal gated-delta kernel for inference
(`use_kernel=not self.training`). In training they run `gated_delta_ops`, a Python loop with one
compiled step per token. A 2048-token sequence through 48 layers makes about 100k tiny graph nodes,
and autograd keeps a fp32 `[48,128,128]` state for every token. In this morning's pilot that gave
15-40 tok/s and an 88-103 GB process footprint, and forge_guard stopped it.

## What it does

- `gdn_chunk.py`: `chunk_gated_delta(q, k, v, g_log, beta, state)` computes the same recurrence
  64 tokens at a time with the chunkwise WY form (the algorithm behind flash-linear-attention's
  `chunk_gated_delta_rule`). Each chunk is a batched 64x64 problem: decay masks, `(I+A)^-1` by
  recursive 2x2 block inversion (log2(64)=6 levels of matmuls on the GPU), then
  `u = T(beta*v)` and `w = T(beta*k*exp(G))`. A sequential loop runs over chunks (32 steps for
  2048 tokens), and the read-out is batched afterwards. It is written in plain MLX ops, so
  autograd produces the GPU backward and stores one state per chunk. It needed no custom VJP and
  no Metal kernel. Internal math is fp32, and the output dtype matches q.
  It handles GQA head grouping (Hv=48, Hk=16, same `repeat` as mlx_lm), lengths that are not a
  multiple of 64 (zero padding with log-decay 0 leaves the state unchanged), and an initial state.
- `gdn_patch.py`: `enable()` swaps `gated_delta_update` in `mlx_lm.models.qwen3_5` (and
  `qwen3_next`) for a wrapper. The wrapper takes the chunked path only when mlx_lm asks for the
  training path (`use_kernel=False`, no mask). Inference and masked batched calls go to the
  original untouched. It computes the log-decay directly (`-exp(A_log)*softplus(a+dt_bias)`),
  with no exp/log round trip. Turn it off with `MLX_GDN_CHUNK=0` or `gdn_patch.disable()`.
- `lora_chunked.py`: a drop-in for `mlx_lm.lora` with the patch on.

An early version used a Neumann/doubling product for the chunk inverse. It passed random-data
tests but gave NaN loss on real email text, because runs of repeated tab/whitespace tokens make
keys nearly identical and `N^32` reaches about 1e17 before it cancels. The block inversion
replaced it, and the test suite now includes an adversarial repeated-keys case.

## Correctness (`test_correctness.py`, outputs in `results/test_tf32_*.txt`)

The tests compare against mlx_lm's own `gated_delta_ops` on y, the final state, and gradients wrt
q, k, v, log-decay, beta and the initial state. They cover 7 shapes: tiny, GQA x3 with T=100,
Dk!=Dv with T=130, the real head dims (16/48 heads x 128) at T=300, 1000 and 2048 (2048 is
forward only), and the repeated-keys case. rel = max abs error / max |reference|.

- **float32** (with `MLX_ENABLE_TF32=0`): worst rel error over every output and gradient is
  **4.5e-5**. Typical values are 1e-6 to 1e-5. The real-dims T=1000 case gives y 6e-6, state 1.2e-5,
  and grads 3e-7 to 9e-6.
- **bfloat16**: worst rel error is **8.5e-3**. The reference's own bf16-vs-fp32 error on y is
  4.4e-3 to 1.5e-2, so the chunked path agrees with the reference to within bf16 rounding.
- **The M5 runs fp32 matmuls as TF32 by default.** With TF32 on, fp32 agreement is limited to
  about 2e-3 (7e-3 on repeated keys). `MLX_ENABLE_TF32=0` costs no measurable speed here (88 ms per
  layer either way), so `lora_chunked.py` and the benchmark turn it off.
- **Real model, real pilot text** (`model_check2.py`, `results/model_check2.log`): layer 0's
  GatedDeltaNet with real weights in fp32 on 1024 real tokens gives output rel 2.7e-7, and grads
  wrt A_log / dt_bias / in_proj_a / in_proj_b / in_proj_qkv agree to rel 2e-6 to 3e-5.
  The full bf16 model's 512-token loss is 8.3830 on the per-token path and 8.4010 chunked, a
  difference of 0.018. mlx_lm's own Metal kernel differs from its own per-token path by 0.0087,
  so the bf16 noise across 64 layers is the same order. At 2048 tokens no layer goes non-finite.

## Benchmark (real 27B bf16 base, LoRA r32 on every linear layer, batch 1, grad checkpointing, AdamW)

`bench_train.py`, with logs and JSON in `results/`. Packed rows of the pilot's train.jsonl were
used, with the same 2 GB MLX cache cap as the pilot. Footprint is the process `phys_footprint`,
which is what forge_guard sees.

| path | seq | sec/step | tok/s | MLX peak | process footprint max |
|---|---|---|---|---|---|
| chunked (new) | 2048 | 16.3 | **126** | 64.2 GB | **68.1 GB** |
| chunked (new) | 512 | 3.2 | **157** | 59.4 GB | 61.4 GB |
| per-token (old) | 512 | 24.6 | 20.9 | 68.0 GB | 71.4 GB |
| per-token (old), pilot | <=768 | - | 15-40 | 69.5 GB | 88-103 GB |

The old path was not run at 2048. One real-size layer's forward+backward at 2048 tokens
(`layer_bench.py`) peaks at 41 GB on the old path and takes 1.7 s, against 3.3 GB and 0.088 s
chunked. On top of the 54 GB model that would push the process past ~95 GB, which is the level
where forge_guard kills it. At equal length (512) the chunked path is 7.5x faster. At 2048 it is
about 6-8x faster than the pilot's measured 15-40 tok/s, and the footprint drops from 88-103 GB to
about 68 GB. Loss fell normally over the 10 steps (10.5 to 1.8 at 2048).

Projected full run (8.3M tokens x 2 epochs, the pilot's estimate) at batch 1: about **18 h** packed
at 2048 (126 tok/s) and about **15 h** at ~512-token rows (157 tok/s). That assumes rows are
packed or batched to fill a few hundred tokens or more. The pilot's raw examples average ~217
tokens, and stock `mlx_lm.lora` neither packs them nor batches without padding, so per-step
overhead would make it slower.

## How to enable

```bash
# exactly like mlx_lm.lora, with the chunked training path on
~/.local/mlx-server/bin/python /Users/dtribe/Desktop/PROJECTS/mlx-gdn-train/lora_chunked.py -c config.yaml
# off switch (stock mlx_lm path)
MLX_GDN_CHUNK=0 ~/.local/mlx-server/bin/python .../lora_chunked.py -c config.yaml
```
or in your own training script: `sys.path.insert(0, ".../mlx-gdn-train"); import gdn_patch; gdn_patch.enable()`.

Re-run the tests with `MLX_ENABLE_TF32=0 ~/.local/mlx-server/bin/python test_correctness.py`
(about 10 s).

Run every model-sized job under a forge_guard lease (`mem_client.reserve`, as the scripts here
do). An unleased process that crosses about 100 GB gets SIGSTOPped and then SIGTERMed. A process
killed while it holds a lease leaves the lease behind until its TTL runs out. Release it with
`curl -X DELETE 127.0.0.1:8790/api/reserve?id=<id>`.
