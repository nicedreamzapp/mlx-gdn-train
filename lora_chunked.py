"""mlx_lm.lora with the chunked gated-delta training path switched on.

    ~/.local/mlx-server/bin/python /Users/dtribe/Desktop/PROJECTS/mlx-gdn-train/lora_chunked.py -c config.yaml

Same arguments as `mlx_lm.lora`. MLX_GDN_CHUNK=0 turns the patch off (stock per-token path).
Also turns off the M5's TF32 fp32 matmuls (exact fp32 inside the chunk math; no measured
speed cost; the model itself is bf16) unless MLX_ENABLE_TF32 is already set, and caps MLX's
free-buffer cache at 2 GB like the pilot so the process footprint stays honest.
"""

import os
import runpy
import sys

os.environ.setdefault("MLX_ENABLE_TF32", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx  # noqa: E402

import gdn_patch  # noqa: E402

mx.set_cache_limit(int(os.environ.get("MLX_CACHE_LIMIT_GB", "2")) * 1024**3)
gdn_patch.enable()
sys.argv = ["mlx_lm.lora"] + sys.argv[1:]
runpy.run_module("mlx_lm.lora", run_name="__main__")
