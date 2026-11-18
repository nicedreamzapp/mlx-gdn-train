"""Does the installed mlx's fused mx.fast.gated_delta_update have a correct gradient wrt the
decay (gamma)? mlx issue #4614 reported a wrong g-gradient on the chunked path at Hk=16 (rel
error ~880), fixed by PR #4792 on 3 Nov 2026. This compares the fused op's gradients against
mlx_lm's per-token recurrence (gated_delta_ops) at the real head layout and at a small one.
"""
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_ops
from test_correctness import make_inputs, err, NAMES

print("mlx", mx.__version__, "fused op:", hasattr(mx.fast, "gated_delta_update"))
for shape in [(1, 100, 2, 6, 32, 32), (1, 300, 16, 48, 128, 128)]:
    B, T, Hk, Hv, Dk, Dv = shape
    q, k, v, g_log, beta, s0 = make_inputs(*shape, 0)
    g = mx.exp(g_log)
    # fused op takes the initial state as [B, Hv, Dk, Dv]? try mlx_lm's [B, Hv, Dv, Dk] first
    def fused(q, k, v, g, beta, s0):
        out = mx.fast.gated_delta_update(q, k, v, g, beta, s0)
        return out[0], out[1]
    def ref(q, k, v, g, beta, s0):
        return gated_delta_ops(q, k, v, g, beta, s0)
    try:
        yf, sf = fused(q, k, v, g, beta, s0); mx.eval(yf, sf)
    except Exception as e:
        print(shape, "fused op call failed:", str(e)[:200]); continue
    yr, sr = ref(q, k, v, g, beta, s0); mx.eval(yr, sr)
    print(f"\nshape B={B} T={T} Hk={Hk} Hv={Hv} D={Dk}: fwd y rel {err(yf, yr)[1]:.2e}, state shape fused {sf.shape} ref {sr.shape}")
    wy = mx.random.normal(yr.shape)
    def lf(q, k, v, g, beta, s0): return (fused(q, k, v, g, beta, s0)[0] * wy).sum()
    def lr(q, k, v, g, beta, s0): return (ref(q, k, v, g, beta, s0)[0] * wy).sum()
    try:
        gf = mx.grad(lf, argnums=[0, 1, 2, 3, 4])(q, k, v, g, beta, s0); mx.eval(gf)
    except Exception as e:
        print("  fused VJP failed:", str(e)[:300]); continue
    gr = mx.grad(lr, argnums=[0, 1, 2, 3, 4])(q, k, v, g, beta, s0); mx.eval(gr)
    for n, a, b in zip(["q", "k", "v", "gamma", "beta"], gf, gr):
        print(f"  d{n:6s} rel {err(a, b)[1]:.3e}")
