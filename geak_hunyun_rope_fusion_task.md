Implement the fused Hunyun HY3 attention-preparation operators for MI308X/gfx942:

- `rope_norm_store_kv`
- `rope_norm_store_kv_fp8`

These are fusion operators, not a generic "build hpc" task.  The operators must be
implemented in this repo and exposed through the existing Python extension module
as `hpc.rope_norm_store_kv(...)`, `hpc.rope_norm_store_kv_fp8(...)`, and
`hpc.QuantType(...)` so that the provided tests can call them.

Fusion target:

```text
reshape + RMSNorm(Q) + RMSNorm(K) + RoPE(Q/K) + paged KV cache store
+ optional Hadamard + FP8 quantization + FP8 cache store
```

Required Python APIs:

- `hpc.rope_norm_store_kv(...)`
- `hpc.rope_norm_store_kv_fp8(...)`
- `hpc.QuantType(...)`

The `hpc` module is only the required public API surface.  The core deliverable
is the fused kernel implementation behind these APIs.

Correctness is defined by:

```bash
python3 -m pytest -sv ./op_tests/test_hunyun_rope_norm_store_kv.py
```

Performance is defined by:

```bash
python3 ./op_tests/bench_rope_norm_store_kv.py --suite quick
```

The benchmark prints `PERF_SUM_US=<value>`; lower is better.

GEAK is launched with:

```bash
python3 ./op_tests/run_hunyun_rope_geak_eval.py
```

That wrapper reports `CORRECTNESS_PASS`, `BENCHMARK_PASS`, baseline timing, and
target timing while always returning exit code 0.  This is intentional because
the initial repository may not expose the target `hpc` APIs yet.  A valid final
solution must print `CORRECTNESS_PASS=1`, `BENCHMARK_PASS=1`, `TARGET_MISSING=0`
or no `TARGET_MISSING` line, and a finite `PERF_SUM_US`.

Functional requirements:

- Support prefill mode.
- Support decode mode.
- Support MTP decode mode.
- Support `qk_norm_policy = 0, 1, 2`.
  - `0`: no RMSNorm.
  - `1`: RoPE then RMSNorm.
  - `2`: RMSNorm then RoPE.
- Support `quant_policy = 0, 1, 2, 3` for `rope_norm_store_kv_fp8`.
  - `0`: dynamic per-token per-head Q/K quantization, per-head V scale.
  - `1`: dynamic per-token per-head Q quantization, static K/V scale.
  - `2`: static Q/K/V quantization using caller-provided `q_scale_inv`.
  - `3`: dynamic per-token per-head Q/K, per-head V scale, and Hadamard on Q/K before quantization.
- Support BF16 input `qkv`.
- Support BF16 cache path for `rope_norm_store_kv`.
- Support FP8 e4m3 cache path for `rope_norm_store_kv_fp8`.
- Support paged KV cache using `kvcache_indices`.
- Support the shapes used by `op_tests/test_rope.py`:
  - `(num_q_heads=8, num_kv_heads=1, qk_head_dim=128)`
  - `(num_q_heads=64, num_kv_heads=8, qk_head_dim=128)`
  - `num_req = 7, 16`
  - `kv_block_size = 64`
  - `v_head_dim = qk_head_dim`

Implementation guidance:

- Prioritize correctness first.  Do not optimize before correctness passes.
- Fuse operations to avoid unnecessary intermediate global memory traffic.
- Minimize separate kernel launches where possible.
- In prefill, handle variable `q_index` ranges and per-request suffix lengths.
- In decode/MTP decode, handle padded batches and ignore padded rows.
- Preserve public API and tensor semantics.
- Keep a simple fallback path if needed, but optimize the hot path used by the benchmark.
- Optimize for AMD MI308X/gfx942.

Selection rule:

- Correctness test must pass.
- `PERF_SUM_US` from the benchmark is the optimization metric.
- Report per-case median latencies and final `PERF_SUM_US`.
