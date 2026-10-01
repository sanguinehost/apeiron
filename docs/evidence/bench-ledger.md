# Benchmark ledger

Consolidated from the per-run receipts in `docs/evidence/bench/` (27 runs, 54 files). One
section per run; the id is the original filename, so old links resolve as `bench-ledger.md#<id>`.
Machine receipts are in `bench-ledger.json` (`runs[]`).

| run | kind |
|---|---|
| [1790593468-q35-sliced](bench-ledger.md#1790593468-q35-sliced) | q35-sliced |
| [1790593686-ab-head-slices](bench-ledger.md#1790593686-ab-head-slices) | ab-head-slices |
| [1790594041-ab-ecores](bench-ledger.md#1790594041-ab-ecores) | ab-ecores |
| [1790596585-ab-prefill-cols-gemm](bench-ledger.md#1790596585-ab-prefill-cols-gemm) | ab-prefill-cols-gemm |
| [1790607104-astra-batched-baseline](bench-ledger.md#1790607104-astra-batched-baseline) | astra-batched-baseline |
| [1790607289-astra-core-budget](bench-ledger.md#1790607289-astra-core-budget) | astra-core-budget |
| [1790608521-astra-write-traversal](bench-ledger.md#1790608521-astra-write-traversal) | astra-write-traversal |
| [1790610276-astra-swiglu-batch](bench-ledger.md#1790610276-astra-swiglu-batch) | astra-swiglu-batch |
| [1790611132-astra-swiglu-owned](bench-ledger.md#1790611132-astra-swiglu-owned) | astra-swiglu-owned |
| [1790612485-astra-projection-baseline](bench-ledger.md#1790612485-astra-projection-baseline) | astra-projection-baseline |
| [1790616920-ns-r001](bench-ledger.md#1790616920-ns-r001) | ns-r001 |
| [1790618878-ns-r002](bench-ledger.md#1790618878-ns-r002) | ns-r002 |
| [1790620216-ns-r003](bench-ledger.md#1790620216-ns-r003) | ns-r003 |
| [1790622516-ns-r004](bench-ledger.md#1790622516-ns-r004) | ns-r004 |
| [1790624325-ns-r005](bench-ledger.md#1790624325-ns-r005) | ns-r005 |
| [1790626263-ns-r006](bench-ledger.md#1790626263-ns-r006) | ns-r006 |
| [1790626949-ns-r007](bench-ledger.md#1790626949-ns-r007) | ns-r007 |
| [1790633894-ns-r009](bench-ledger.md#1790633894-ns-r009) | ns-r009 |
| [1790668367-ns-r012](bench-ledger.md#1790668367-ns-r012) | ns-r012 |
| [1790678074-ns-r015](bench-ledger.md#1790678074-ns-r015) | ns-r015 |
| [1790680301-ns-r016](bench-ledger.md#1790680301-ns-r016) | ns-r016 |
| [1790690080-ns-r019](bench-ledger.md#1790690080-ns-r019) | ns-r019 |
| [1790692313-ns-r020](bench-ledger.md#1790692313-ns-r020) | ns-r020 |
| [1790700697-ns-r023](bench-ledger.md#1790700697-ns-r023) | ns-r023 |
| [1790710920-ns-r026](bench-ledger.md#1790710920-ns-r026) | ns-r026 |
| [1790713685-ns-r027](bench-ledger.md#1790713685-ns-r027) | ns-r027 |
| [1790717185-ns-r028](bench-ledger.md#1790717185-ns-r028) | ns-r028 |

<a id="1790593468-q35-sliced"></a>

# Engine benchmark `q35-sliced`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/q35-t5/prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `74c037b1cfc79b0d6be6d3907398fe926d78d12d` with 224 dirty paths; src sha256 `1a019cd4b898598a4c5b05463347f62e6991ede6dfc31fc8f5d2f47ee360a5d0`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 71.9 / 84.1 / 88.4 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 113.74 | 110.83 | 132.47 |
| llama.cpp pp (best threads) | 470.88 | 446.79 | 587.78 |
| ratio pp (per round) | 0.24 | 0.22 | 0.27 |
| apeiron tg | 58.61 | 51.27 | 63.84 |
| llama.cpp tg (best threads) | 46.76 | 46.47 | 54.21 |
| ratio tg (per round) | 1.18 | 1.10 | 1.26 |
| apeiron spec (e2e) | 62.99 | 53.05 | 66.29 |
| spec acceptance | 0.70 | 0.70 | 0.70 |
| apeiron pp busy cores | 23.78 | 23.73 | 23.80 |
| apeiron tg busy cores | 23.62 | 23.48 | 23.68 |
| mean MHz before pp | 3868.27 | 2802.52 | 4097.59 |
| mean MHz before tg | 4470.08 | 4257.18 | 4761.88 |

### Traced phases

#### pp: 200 tokens, 8.58 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 26 | - | - |
| norm | 0.214 | 2.5% | 1274 | - | - |
| residual | 0.049 | 0.6% | 1248 | - | - |
| attn.proj_in | 0.324 | 3.8% | 156 | 13.4 | 97.0 |
| attn.core | 1.071 | 12.5% | 156 | - | - |
| attn.proj_out | 0.157 | 1.8% | 156 | 11.1 | 80.2 |
| affine.proj_in | 1.396 | 16.3% | 468 | 15.0 | 108.6 |
| affine.conv | 0.504 | 5.9% | 468 | - | - |
| affine.heads | 1.559 | 18.2% | 468 | - | - |
| affine.proj_out | 0.438 | 5.1% | 468 | 11.9 | 86.3 |
| ffn | 2.801 | 32.7% | 624 | 13.0 | 94.3 |
| head | 0.029 | 0.3% | 1 | 46.7 | 44.0 |
| untraced in pass | 0.012 | 0.1% | 0 | - | - |
| outside passes | 0.021 | 0.2% | 0 | - | - |

#### tg: 64 tokens, 17.35 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.215 | 1.2% | 3136 | - | - |
| residual | 0.056 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.714 | 4.1% | 384 | 46.8 | 44.1 |
| attn.core | 0.671 | 3.9% | 384 | - | - |
| attn.proj_out | 0.244 | 1.4% | 384 | 54.7 | 51.5 |
| affine.proj_in | 3.155 | 18.2% | 1152 | 51.1 | 48.0 |
| affine.conv | 0.653 | 3.8% | 1152 | - | - |
| affine.heads | 1.887 | 10.9% | 1152 | - | - |
| affine.proj_out | 0.764 | 4.4% | 1152 | 52.5 | 49.4 |
| ffn | 4.658 | 26.8% | 1536 | 60.3 | 56.7 |
| head | 4.289 | 24.7% | 64 | 63.0 | 59.3 |
| untraced in pass | 0.041 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |

#### spec: 64 tokens, 16.89 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 38 | - | - |
| norm | 0.250 | 1.5% | 1862 | - | - |
| residual | 0.060 | 0.4% | 1824 | - | - |
| attn.proj_in | 0.604 | 3.6% | 266 | 38.3 | 69.7 |
| attn.core | 0.850 | 5.0% | 266 | - | - |
| attn.proj_out | 0.270 | 1.6% | 266 | 34.3 | 62.3 |
| affine.proj_in | 2.184 | 12.9% | 684 | 43.8 | 81.3 |
| affine.conv | 0.710 | 4.2% | 684 | - | - |
| affine.heads | 2.331 | 13.8% | 684 | - | - |
| affine.proj_out | 0.633 | 3.7% | 684 | 37.6 | 69.9 |
| ffn | 4.419 | 26.2% | 950 | 39.3 | 72.6 |
| head | 3.588 | 21.2% | 75 | 50.5 | 88.5 |
| restore | 0.475 | 2.8% | 11 | - | - |
| untraced in pass | 0.273 | 1.6% | 0 | - | - |
| outside passes | 0.243 | 1.4% | 0 | - | - |

<a id="1790593686-ab-head-slices"></a>

# A/B `ab-head-slices`

- pp 200, tg 64, 5 rounds, arm order rotated per round; quiet gate busy < 0.05: every round passed

| arm | env | pp tok/s | tg tok/s | spec tok/s | pp ratio | tg ratio | spec ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| whole | `APEIRON_HEAD_SLICES=1` | 125.75 | 65.33 | 68.37 | 1.000 | 1.000 | 1.000 |
| s4 | `APEIRON_HEAD_SLICES=4` | 136.69 | 66.52 | 72.06 | 1.083 | 1.033 | 1.055 |
| s8 | `APEIRON_HEAD_SLICES=8` | 136.85 | 66.75 | 73.96 | 1.071 | 1.051 | 1.071 |

Ratios are the median of per-round ratios to the first arm.

<a id="1790594041-ab-ecores"></a>

# A/B `ab-ecores`

- pp 200, tg 64, 5 rounds, arm order rotated per round; quiet gate busy < 0.05: every round passed

| arm | env | pp tok/s | tg tok/s | spec tok/s | pp ratio | tg ratio | spec ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| p8e16 | `` | 124.59 | 63.59 | NaN | 1.000 | 1.000 | NaN |
| p8e0 | `APEIRON_ECORES=0` | 127.98 | 64.83 | NaN | 0.976 | 1.006 | NaN |
| p8e8 | `APEIRON_ECORES=8` | 128.32 | 58.57 | NaN | 1.030 | 0.909 | NaN |

Ratios are the median of per-round ratios to the first arm.

<a id="1790596585-ab-prefill-cols-gemm"></a>

# A/B `ab-prefill-cols-gemm`

- pp 200, tg 16, 3 rounds, arm order rotated per round; quiet gate busy < 0.05: every round passed

| arm | env | pp tok/s | tg tok/s | spec tok/s | pp ratio | tg ratio | spec ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| c8 | `APEIRON_PREFILL_COLS=8` | 191.88 | 65.72 | NaN | 1.000 | 1.000 | NaN |
| c16 | `APEIRON_PREFILL_COLS=16` | 209.12 | 65.09 | NaN | 1.096 | 0.990 | NaN |
| c32 | `APEIRON_PREFILL_COLS=32` | 214.96 | 65.86 | NaN | 1.120 | 1.001 | NaN |
| c64 | `APEIRON_PREFILL_COLS=64` | 220.56 | 65.33 | NaN | 1.138 | 0.993 | NaN |
| c200 | `APEIRON_PREFILL_COLS=200` | 221.80 | 66.43 | NaN | 1.158 | 1.010 | NaN |

Ratios are the median of per-round ratios to the first arm.

<a id="1790607104-astra-batched-baseline"></a>

# Engine benchmark `astra-batched-baseline`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/q35-t5/prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `39298a1c858a11397b7a9e3d638c5a0b796b4a47` with 6 dirty paths; src sha256 `96c75f9773af6cb9d0f0ef2536cc7c0c8ccb86020759fe614c214a706300c8de`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 65.8 / 88.6 / 88.1 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 224.99 | 222.14 | 236.69 |
| llama.cpp pp (best threads) | 523.54 | 497.13 | 574.82 |
| ratio pp (per round) | 0.43 | 0.39 | 0.48 |
| apeiron tg | 62.71 | 62.44 | 64.65 |
| llama.cpp tg (best threads) | 49.90 | 46.88 | 50.81 |
| ratio tg (per round) | 1.27 | 1.25 | 1.33 |
| apeiron spec (e2e) | 37.95 | 37.67 | 39.14 |
| spec acceptance | 0.24 | 0.24 | 0.24 |
| apeiron pp busy cores | 23.74 | 23.70 | 23.77 |
| apeiron tg busy cores | 23.77 | 23.73 | 23.82 |
| mean MHz before pp | 2819.17 | 2207.62 | 4254.25 |
| mean MHz before tg | 4675.39 | 4634.62 | 4829.34 |

### Traced phases

#### pp: 200 tokens, 4.33 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.064 | 1.5% | 245 | - | - |
| residual | 0.051 | 1.2% | 240 | - | - |
| attn.proj_in | 0.115 | 2.6% | 30 | 7.3 | 274.4 |
| attn.core | 0.355 | 8.2% | 30 | - | - |
| attn.proj_out | 0.057 | 1.3% | 30 | 5.9 | 222.4 |
| affine.proj_in | 0.514 | 11.9% | 90 | 7.8 | 294.8 |
| affine.conv | 0.519 | 12.0% | 90 | - | - |
| affine.heads | 1.323 | 30.5% | 90 | - | - |
| affine.proj_out | 0.164 | 3.8% | 90 | 6.1 | 229.5 |
| ffn | 1.125 | 26.0% | 120 | 6.2 | 234.9 |
| head | 0.025 | 0.6% | 1 | 53.3 | 50.2 |
| untraced in pass | 0.007 | 0.2% | 0 | - | - |
| outside passes | 0.012 | 0.3% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 44.7) | 1.137 | 26.3% | 25464 | - | - |

#### tg: 64 tokens, 15.91 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.227 | 1.4% | 3136 | - | - |
| residual | 0.056 | 0.4% | 3072 | - | - |
| attn.proj_in | 0.631 | 4.0% | 384 | 52.9 | 49.8 |
| attn.core | 0.601 | 3.8% | 384 | - | - |
| attn.proj_out | 0.218 | 1.4% | 384 | 61.4 | 57.8 |
| affine.proj_in | 2.702 | 17.0% | 1152 | 59.6 | 56.1 |
| affine.conv | 0.975 | 6.1% | 1152 | - | - |
| affine.heads | 1.621 | 10.2% | 1152 | - | - |
| affine.proj_out | 0.638 | 4.0% | 1152 | 62.9 | 59.2 |
| ffn | 4.095 | 25.7% | 1536 | 68.6 | 64.5 |
| head | 4.100 | 25.8% | 64 | 65.9 | 62.0 |
| untraced in pass | 0.039 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 49.3) | 2.355 | 14.8% | 19712 | - | - |

#### spec: 64 tokens, 25.21 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 52 | - | - |
| norm | 0.581 | 2.3% | 2548 | - | - |
| residual | 0.096 | 0.4% | 2496 | - | - |
| attn.proj_in | 0.878 | 3.5% | 364 | 36.1 | 63.6 |
| attn.core | 0.722 | 2.9% | 364 | - | - |
| attn.proj_out | 0.337 | 1.3% | 364 | 37.7 | 66.4 |
| affine.proj_in | 3.581 | 14.2% | 936 | 36.5 | 68.1 |
| affine.conv | 1.178 | 4.7% | 936 | - | - |
| affine.heads | 2.502 | 9.9% | 936 | - | - |
| affine.proj_out | 0.980 | 3.9% | 936 | 33.2 | 62.0 |
| ffn | 6.450 | 25.6% | 1300 | 36.8 | 67.6 |
| head | 5.861 | 23.3% | 103 | 42.3 | 74.4 |
| restore | 1.620 | 6.4% | 39 | - | - |
| untraced in pass | 0.258 | 1.0% | 0 | - | - |
| outside passes | 0.160 | 0.6% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 92.7) | 6.991 | 27.7% | 29896 | - | - |

<a id="1790607289-astra-core-budget"></a>

# A/B `astra-core-budget`

- pp 200, tg 64, 5 rounds, arm order rotated per round; quiet gate busy < 0.05: every round passed

| arm | env | pp tok/s | tg tok/s | spec tok/s | pp ratio | tg ratio | spec ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | `` | 271.38 | 69.15 | NaN | 1.000 | 1.000 | NaN |
| p8 | `APEIRON_ECORES=0` | 230.91 | 67.56 | NaN | 0.849 | 0.979 | NaN |
| p4 | `APEIRON_ECORES=0 APEIRON_THREADS=4` | 137.92 | 47.84 | NaN | 0.514 | 0.693 | NaN |
| p4-slices8 | `APEIRON_ECORES=0 APEIRON_THREADS=4 APEIRON_HEAD_SLICES=8` | 133.99 | 47.97 | NaN | 0.494 | 0.708 | NaN |

Ratios are the median of per-round ratios to the first arm.

<a id="1790608521-astra-write-traversal"></a>

# A/B `astra-write-traversal`

- pp 200, tg 64, 5 rounds, arm order rotated per round; quiet gate busy < 0.05: every round passed

| arm | env | pp tok/s | tg tok/s | spec tok/s | pp ratio | tg ratio | spec ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| row | `APEIRON_WRITE_COLUMN_MAJOR=0` | 273.61 | 69.26 | NaN | 1.000 | 1.000 | NaN |
| column | `APEIRON_WRITE_COLUMN_MAJOR=1` | 273.28 | 69.01 | NaN | 1.000 | 0.996 | NaN |
| p4-row | `APEIRON_ECORES=0 APEIRON_THREADS=4 APEIRON_WRITE_COLUMN_MAJOR=0` | 140.41 | 49.41 | NaN | 0.511 | 0.710 | NaN |
| p4-column | `APEIRON_ECORES=0 APEIRON_THREADS=4 APEIRON_WRITE_COLUMN_MAJOR=1` | 140.58 | 48.23 | NaN | 0.514 | 0.699 | NaN |

Ratios are the median of per-round ratios to the first arm.

<a id="1790610276-astra-swiglu-batch"></a>

# A/B `astra-swiglu-batch`

- pp 200, tg 64, 5 rounds, arm order rotated per round; quiet gate busy < 0.05: every round passed

| arm | env | pp tok/s | tg tok/s | spec tok/s | pp ratio | tg ratio | spec ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| columns | `APEIRON_BATCH_SWIGLU=0` | 273.23 | 68.99 | 49.51 | 1.000 | 1.000 | 1.000 |
| batched | `APEIRON_BATCH_SWIGLU=1` | 269.90 | 68.83 | 47.26 | 0.987 | 0.999 | 0.948 |
| p4-columns | `APEIRON_ECORES=0 APEIRON_THREADS=4 APEIRON_BATCH_SWIGLU=0` | 140.23 | 46.59 | 23.27 | 0.513 | 0.679 | 0.473 |
| p4-batched | `APEIRON_ECORES=0 APEIRON_THREADS=4 APEIRON_BATCH_SWIGLU=1` | 138.55 | 48.18 | 23.17 | 0.507 | 0.714 | 0.469 |

Ratios are the median of per-round ratios to the first arm.

<a id="1790611132-astra-swiglu-owned"></a>

# A/B `astra-swiglu-owned`

- pp 200, tg 64, 5 rounds, arm order rotated per round; quiet gate busy < 0.05: every round passed

| arm | env | pp tok/s | tg tok/s | spec tok/s | pp ratio | tg ratio | spec ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| columns | `APEIRON_BATCH_SWIGLU=0` | 267.27 | 68.86 | 48.15 | 1.000 | 1.000 | 1.000 |
| owned | `APEIRON_BATCH_SWIGLU=1` | 272.67 | 67.96 | 47.32 | 1.027 | 0.987 | 0.992 |
| p4-columns | `APEIRON_ECORES=0 APEIRON_THREADS=4 APEIRON_BATCH_SWIGLU=0` | 139.51 | 49.97 | 23.49 | 0.522 | 0.724 | 0.491 |
| p4-owned | `APEIRON_ECORES=0 APEIRON_THREADS=4 APEIRON_BATCH_SWIGLU=1` | 140.14 | 49.25 | 23.40 | 0.521 | 0.714 | 0.485 |

Ratios are the median of per-round ratios to the first arm.

<a id="1790612485-astra-projection-baseline"></a>

# Engine benchmark `astra-projection-baseline`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/Workspace/apeiron/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `2d0b0ef13f21eccc008d860bd57a9d325dde9660` with 6 dirty paths; src sha256 `e932d0e81bc736d0be102cc3f9c1d3d6928f7a1251ed68fc1bb352fcfb402186`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 67.9 / 89.8 / 86.9 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 256.65 | 255.31 | 261.94 |
| llama.cpp pp (best threads) | 576.42 | 571.67 | 594.19 |
| ratio pp (per round) | 0.45 | 0.43 | 0.46 |
| apeiron tg | 67.31 | 62.65 | 69.30 |
| llama.cpp tg (best threads) | 52.37 | 51.11 | 55.68 |
| ratio tg (per round) | 1.28 | 1.16 | 1.32 |
| apeiron spec (e2e) | 47.43 | 44.37 | 49.72 |
| spec acceptance | 0.24 | 0.24 | 0.24 |
| apeiron pp busy cores | 23.70 | 23.68 | 23.74 |
| apeiron tg busy cores | 23.76 | 23.61 | 23.85 |
| mean MHz before pp | 2318.02 | 1312.87 | 3412.77 |
| mean MHz before tg | 4671.61 | 4451.82 | 4814.95 |

### Traced phases

#### pp: 200 tokens, 3.77 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.065 | 1.7% | 245 | - | - |
| residual | 0.050 | 1.3% | 240 | - | - |
| attn.proj_in | 0.105 | 2.8% | 30 | 7.9 | 298.6 |
| attn.core | 0.270 | 7.2% | 30 | - | - |
| attn.proj_out | 0.051 | 1.3% | 30 | 6.6 | 248.5 |
| affine.proj_in | 0.469 | 12.4% | 90 | 8.6 | 323.3 |
| affine.conv | 0.442 | 11.7% | 90 | - | - |
| affine.heads | 1.113 | 29.5% | 90 | - | - |
| affine.proj_out | 0.150 | 4.0% | 90 | 6.7 | 252.5 |
| ffn | 1.015 | 26.9% | 120 | 6.9 | 260.3 |
| head | 0.025 | 0.7% | 1 | 55.0 | 51.8 |
| untraced in pass | 0.006 | 0.2% | 0 | - | - |
| outside passes | 0.012 | 0.3% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 44.7) | 0.668 | 17.7% | 25464 | - | - |

#### tg: 64 tokens, 15.05 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.189 | 1.3% | 3136 | - | - |
| residual | 0.048 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.626 | 4.2% | 384 | 53.4 | 50.2 |
| attn.core | 0.587 | 3.9% | 384 | - | - |
| attn.proj_out | 0.231 | 1.5% | 384 | 57.8 | 54.4 |
| affine.proj_in | 2.689 | 17.9% | 1152 | 59.9 | 56.4 |
| affine.conv | 0.820 | 5.4% | 1152 | - | - |
| affine.heads | 1.470 | 9.8% | 1152 | - | - |
| affine.proj_out | 0.625 | 4.2% | 1152 | 64.1 | 60.4 |
| ffn | 3.971 | 26.4% | 1536 | 70.7 | 66.5 |
| head | 3.752 | 24.9% | 64 | 72.0 | 67.8 |
| untraced in pass | 0.035 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 49.3) | 2.473 | 16.4% | 19712 | - | - |

#### spec: 64 tokens, 21.92 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 52 | - | - |
| norm | 0.572 | 2.6% | 2548 | - | - |
| residual | 0.093 | 0.4% | 2496 | - | - |
| attn.proj_in | 0.737 | 3.4% | 364 | 43.0 | 75.8 |
| attn.core | 0.694 | 3.2% | 364 | - | - |
| attn.proj_out | 0.319 | 1.5% | 364 | 39.7 | 70.0 |
| affine.proj_in | 3.051 | 13.9% | 936 | 42.9 | 80.0 |
| affine.conv | 1.102 | 5.0% | 936 | - | - |
| affine.heads | 2.263 | 10.3% | 936 | - | - |
| affine.proj_out | 0.844 | 3.9% | 936 | 38.6 | 71.9 |
| ffn | 5.611 | 25.6% | 1300 | 42.3 | 77.8 |
| head | 4.773 | 21.8% | 103 | 51.9 | 91.3 |
| restore | 1.568 | 7.2% | 39 | - | - |
| untraced in pass | 0.136 | 0.6% | 0 | - | - |
| outside passes | 0.152 | 0.7% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 92.7) | 4.233 | 19.3% | 29896 | - | - |

<a id="1790616920-ns-r001"></a>

# Engine benchmark `ns-r001`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/ns/integ/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `2eca8145d9a5a907de04ac2be85c3529a76a4e84` with 0 dirty paths; src sha256 `1aff5eeea28b754e83911145f6230991887d4e3306e8c3b8cd6697322d58d44d`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 72.6 / 86.3 / 92.5 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 266.09 | 263.33 | 270.21 |
| llama.cpp pp (best threads) | 588.69 | 577.35 | 597.48 |
| ratio pp (per round) | 0.45 | 0.44 | 0.46 |
| apeiron tg | 70.06 | 69.97 | 70.37 |
| llama.cpp tg (best threads) | 55.63 | 55.28 | 60.13 |
| ratio tg (per round) | 1.26 | 1.17 | 1.27 |
| apeiron pp busy cores | 23.71 | 23.64 | 23.77 |
| apeiron tg busy cores | 23.84 | 23.80 | 23.87 |
| mean MHz before pp | 1844.27 | 1743.01 | 2783.14 |
| mean MHz before tg | 4417.55 | 4014.43 | 4543.80 |

### Traced phases

#### pp: 200 tokens, 3.62 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.060 | 1.7% | 245 | - | - |
| residual | 0.050 | 1.4% | 240 | - | - |
| attn.proj_in | 0.099 | 2.7% | 30 | 8.5 | 318.8 |
| attn.core | 0.275 | 7.6% | 30 | - | - |
| attn.proj_out | 0.050 | 1.4% | 30 | 6.7 | 253.8 |
| affine.proj_in | 0.453 | 12.5% | 90 | 8.9 | 334.5 |
| affine.conv | 0.432 | 12.0% | 90 | - | - |
| affine.heads | 1.018 | 28.2% | 90 | - | - |
| affine.proj_out | 0.145 | 4.0% | 90 | 6.9 | 259.6 |
| ffn | 0.993 | 27.5% | 120 | 7.1 | 266.1 |
| head | 0.024 | 0.7% | 1 | 55.4 | 52.1 |
| untraced in pass | 0.006 | 0.2% | 0 | - | - |
| outside passes | 0.011 | 0.3% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 44.6) | 0.643 | 17.8% | 25554 | - | - |

#### tg: 64 tokens, 14.23 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 64 | - | - |
| norm | 0.180 | 1.3% | 3136 | - | - |
| residual | 0.047 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.575 | 4.0% | 384 | 58.2 | 54.7 |
| attn.core | 0.551 | 3.9% | 384 | - | - |
| attn.proj_out | 0.201 | 1.4% | 384 | 66.4 | 62.5 |
| affine.proj_in | 2.480 | 17.4% | 1152 | 65.0 | 61.1 |
| affine.conv | 0.784 | 5.5% | 1152 | - | - |
| affine.heads | 1.345 | 9.5% | 1152 | - | - |
| affine.proj_out | 0.592 | 4.2% | 1152 | 67.8 | 63.8 |
| ffn | 3.799 | 26.7% | 1536 | 73.9 | 69.6 |
| head | 3.645 | 25.6% | 64 | 74.1 | 69.8 |
| untraced in pass | 0.032 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 47.5) | 2.065 | 14.5% | 20864 | - | - |

<a id="1790618878-ns-r002"></a>

# Engine benchmark `ns-r002`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/ns/integ/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `67fd6c96ccf22a33280262a8368121609c03105d` with 0 dirty paths; src sha256 `ff5d151fd69b65184ac666f9238686a9de2922ced1bd1aac7ac1be04b32359a7`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 67.0 / 87.1 / 93.1 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 293.46 | 286.40 | 297.30 |
| llama.cpp pp (best threads) | 589.97 | 577.65 | 596.53 |
| ratio pp (per round) | 0.50 | 0.48 | 0.51 |
| apeiron tg | 70.14 | 69.49 | 70.16 |
| llama.cpp tg (best threads) | 57.10 | 53.57 | 57.87 |
| ratio tg (per round) | 1.23 | 1.21 | 1.30 |
| apeiron pp busy cores | 23.71 | 23.54 | 23.75 |
| apeiron tg busy cores | 23.84 | 23.83 | 23.86 |
| mean MHz before pp | 2328.33 | 1304.74 | 3410.30 |
| mean MHz before tg | 4348.00 | 4276.07 | 4888.35 |

### Traced phases

#### pp: 200 tokens, 3.23 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.061 | 1.9% | 245 | - | - |
| residual | 0.048 | 1.5% | 240 | - | - |
| attn.proj_in | 0.073 | 2.3% | 30 | 11.4 | 430.8 |
| attn.core | 0.264 | 8.2% | 30 | - | - |
| attn.proj_out | 0.043 | 1.3% | 30 | 7.8 | 294.5 |
| affine.proj_in | 0.317 | 9.8% | 90 | 12.7 | 478.5 |
| affine.conv | 0.444 | 13.8% | 90 | - | - |
| affine.heads | 1.008 | 31.2% | 90 | - | - |
| affine.proj_out | 0.124 | 3.8% | 90 | 8.1 | 304.5 |
| ffn | 0.803 | 24.9% | 120 | 8.7 | 328.9 |
| head | 0.024 | 0.7% | 1 | 56.6 | 53.3 |
| untraced in pass | 0.006 | 0.2% | 0 | - | - |
| outside passes | 0.011 | 0.3% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 46.0) | 0.656 | 20.3% | 27042 | - | - |

#### tg: 64 tokens, 14.47 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.187 | 1.3% | 3136 | - | - |
| residual | 0.049 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.588 | 4.1% | 384 | 56.8 | 53.5 |
| attn.core | 0.560 | 3.9% | 384 | - | - |
| attn.proj_out | 0.201 | 1.4% | 384 | 66.4 | 62.5 |
| affine.proj_in | 2.526 | 17.5% | 1152 | 63.7 | 60.0 |
| affine.conv | 0.803 | 5.6% | 1152 | - | - |
| affine.heads | 1.369 | 9.5% | 1152 | - | - |
| affine.proj_out | 0.598 | 4.1% | 1152 | 67.1 | 63.1 |
| ffn | 3.835 | 26.5% | 1536 | 73.2 | 68.9 |
| head | 3.719 | 25.7% | 64 | 72.6 | 68.4 |
| untraced in pass | 0.034 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 47.5) | 2.139 | 14.8% | 20864 | - | - |

<a id="1790620216-ns-r003"></a>

# Engine benchmark `ns-r003`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/ns/integ/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `d3cbd4fbea3c759222314fba81b89cda795d4f88` with 0 dirty paths; src sha256 `a7d92bec630e22ae7681ad26672bee62bcf6b8273dcb3c6df7bd033f3c48dc2c`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 67.3 / 86.6 / 66.4 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 299.50 | 239.97 | 302.58 |
| llama.cpp pp (best threads) | 585.48 | 576.38 | 591.96 |
| ratio pp (per round) | 0.51 | 0.41 | 0.52 |
| apeiron tg | 67.09 | 63.85 | 69.36 |
| llama.cpp tg (best threads) | 53.97 | 52.65 | 56.45 |
| ratio tg (per round) | 1.24 | 1.18 | 1.32 |
| apeiron pp busy cores | 23.68 | 23.65 | 23.70 |
| apeiron tg busy cores | 23.78 | 23.67 | 23.80 |
| mean MHz before pp | 2068.77 | 1210.32 | 3039.93 |
| mean MHz before tg | 4730.38 | 4387.83 | 4803.54 |

### Traced phases

#### pp: 200 tokens, 3.20 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.062 | 1.9% | 245 | - | - |
| residual | 0.048 | 1.5% | 240 | - | - |
| attn.proj_in | 0.076 | 2.4% | 30 | 11.0 | 416.0 |
| attn.core | 0.280 | 8.7% | 30 | - | - |
| attn.proj_out | 0.044 | 1.4% | 30 | 7.6 | 285.6 |
| affine.proj_in | 0.336 | 10.5% | 90 | 12.0 | 451.2 |
| affine.conv | 0.303 | 9.5% | 90 | - | - |
| affine.heads | 1.049 | 32.8% | 90 | - | - |
| affine.proj_out | 0.135 | 4.2% | 90 | 7.4 | 280.1 |
| ffn | 0.828 | 25.9% | 120 | 8.5 | 319.3 |
| head | 0.024 | 0.7% | 1 | 56.5 | 53.2 |
| untraced in pass | 0.006 | 0.2% | 0 | - | - |
| outside passes | 0.011 | 0.3% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 46.0) | 0.714 | 22.3% | 27042 | - | - |

#### tg: 64 tokens, 14.76 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.182 | 1.2% | 3136 | - | - |
| residual | 0.049 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.584 | 4.0% | 384 | 57.2 | 53.9 |
| attn.core | 0.565 | 3.8% | 384 | - | - |
| attn.proj_out | 0.294 | 2.0% | 384 | 45.4 | 42.8 |
| affine.proj_in | 2.549 | 17.3% | 1152 | 63.2 | 59.5 |
| affine.conv | 0.734 | 5.0% | 1152 | - | - |
| affine.heads | 1.461 | 9.9% | 1152 | - | - |
| affine.proj_out | 0.627 | 4.2% | 1152 | 64.0 | 60.2 |
| ffn | 3.984 | 27.0% | 1536 | 70.5 | 66.3 |
| head | 3.693 | 25.0% | 64 | 73.2 | 68.9 |
| untraced in pass | 0.034 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 47.5) | 2.472 | 16.7% | 20864 | - | - |

<a id="1790622516-ns-r004"></a>

# Engine benchmark `ns-r004`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/ns/integ/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `decba339d355ef7dba207aaaec8c84d1df4c24fb` with 0 dirty paths; src sha256 `9e4d128c1c13af4fc45009219d3fb0590d36073036c5b07d5d4a80b276b88a6b`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 74.1 / 86.4 / 89.9 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 285.32 | 236.46 | 299.66 |
| llama.cpp pp (best threads) | 514.25 | 469.07 | 569.32 |
| ratio pp (per round) | 0.53 | 0.46 | 0.61 |
| apeiron tg | 64.83 | 61.31 | 67.76 |
| llama.cpp tg (best threads) | 52.99 | 51.07 | 56.15 |
| ratio tg (per round) | 1.17 | 1.12 | 1.32 |
| apeiron pp busy cores | 23.59 | 23.36 | 23.70 |
| apeiron tg busy cores | 23.77 | 23.72 | 23.79 |
| mean MHz before pp | 3035.20 | 1957.75 | 4661.13 |
| mean MHz before tg | 4701.49 | 4564.97 | 4787.04 |

### Traced phases

#### pp: 200 tokens, 4.76 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.110 | 2.3% | 245 | - | - |
| residual | 0.061 | 1.3% | 240 | - | - |
| attn.proj_in | 0.091 | 1.9% | 30 | 9.2 | 344.9 |
| attn.core | 0.399 | 8.4% | 30 | - | - |
| attn.proj_out | 0.103 | 2.2% | 30 | 3.3 | 122.7 |
| affine.proj_in | 0.544 | 11.4% | 90 | 7.4 | 278.4 |
| affine.conv | 0.467 | 9.8% | 90 | - | - |
| affine.heads | 1.489 | 31.3% | 90 | - | - |
| affine.proj_out | 0.212 | 4.5% | 90 | 4.7 | 177.9 |
| ffn | 1.227 | 25.8% | 120 | 5.7 | 215.3 |
| head | 0.031 | 0.7% | 1 | 43.0 | 40.5 |
| untraced in pass | 0.008 | 0.2% | 0 | - | - |
| outside passes | 0.012 | 0.2% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 75.4) | 1.838 | 38.7% | 3642 | - | - |

#### tg: 64 tokens, 16.72 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.250 | 1.5% | 3136 | - | - |
| residual | 0.061 | 0.4% | 3072 | - | - |
| attn.proj_in | 0.667 | 4.0% | 384 | 50.1 | 47.2 |
| attn.core | 0.625 | 3.7% | 384 | - | - |
| attn.proj_out | 0.287 | 1.7% | 384 | 46.5 | 43.8 |
| affine.proj_in | 2.882 | 17.2% | 1152 | 55.9 | 52.6 |
| affine.conv | 0.926 | 5.5% | 1152 | - | - |
| affine.heads | 1.830 | 10.9% | 1152 | - | - |
| affine.proj_out | 0.691 | 4.1% | 1152 | 58.1 | 54.6 |
| ffn | 4.220 | 25.2% | 1536 | 66.5 | 62.6 |
| head | 4.237 | 25.3% | 64 | 63.8 | 60.0 |
| untraced in pass | 0.039 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 47.5) | 3.455 | 20.7% | 20864 | - | - |

<a id="1790624325-ns-r005"></a>

# Engine benchmark `ns-r005`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/ns/integ/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `a6d0620c8ce5a6ff01e8aa6178af6936eb89de1e` with 0 dirty paths; src sha256 `b2fb30b3d03fd686191f7dcb9bcda2e38355a03174dccf8b05178f9f89bc875c`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 68.5 / 87.9 / 91.2 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 331.70 | 325.83 | 336.08 |
| llama.cpp pp (best threads) | 582.33 | 570.83 | 590.58 |
| ratio pp (per round) | 0.57 | 0.56 | 0.58 |
| apeiron tg | 69.58 | 67.38 | 69.89 |
| llama.cpp tg (best threads) | 54.68 | 53.52 | 55.34 |
| ratio tg (per round) | 1.27 | 1.23 | 1.30 |
| apeiron pp busy cores | 23.65 | 23.46 | 23.70 |
| apeiron tg busy cores | 23.82 | 23.77 | 23.84 |
| mean MHz before pp | 2738.97 | 1208.19 | 3937.26 |
| mean MHz before tg | 4766.42 | 4394.43 | 4821.70 |

### Traced phases

#### pp: 200 tokens, 2.88 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.061 | 2.1% | 245 | - | - |
| residual | 0.049 | 1.7% | 240 | - | - |
| attn.proj_in | 0.068 | 2.4% | 30 | 12.3 | 464.7 |
| attn.core | 0.275 | 9.6% | 30 | - | - |
| attn.proj_out | 0.036 | 1.3% | 30 | 9.2 | 347.0 |
| affine.proj_in | 0.304 | 10.6% | 90 | 13.2 | 497.9 |
| affine.conv | 0.300 | 10.4% | 90 | - | - |
| affine.heads | 1.037 | 36.0% | 90 | - | - |
| affine.proj_out | 0.102 | 3.6% | 90 | 9.8 | 368.9 |
| ffn | 0.601 | 20.9% | 120 | 11.7 | 439.9 |
| head | 0.026 | 0.9% | 1 | 51.2 | 48.2 |
| untraced in pass | 0.006 | 0.2% | 0 | - | - |
| outside passes | 0.011 | 0.4% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 64.8) | 0.512 | 17.8% | 3162 | - | - |

#### tg: 64 tokens, 14.35 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.180 | 1.3% | 3136 | - | - |
| residual | 0.045 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.582 | 4.1% | 384 | 57.5 | 54.1 |
| attn.core | 0.557 | 3.9% | 384 | - | - |
| attn.proj_out | 0.205 | 1.4% | 384 | 65.2 | 61.3 |
| affine.proj_in | 2.513 | 17.5% | 1152 | 64.1 | 60.3 |
| affine.conv | 0.676 | 4.7% | 1152 | - | - |
| affine.heads | 1.396 | 9.7% | 1152 | - | - |
| affine.proj_out | 0.609 | 4.2% | 1152 | 65.9 | 62.0 |
| ffn | 3.876 | 27.0% | 1536 | 72.4 | 68.2 |
| head | 3.671 | 25.6% | 64 | 73.6 | 69.3 |
| untraced in pass | 0.034 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 47.5) | 2.116 | 14.7% | 20864 | - | - |

<a id="1790626263-ns-r006"></a>

# Engine benchmark `ns-r006`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/ns/integ/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `4b3051a3b55c76670815bb771fa2722c82d7e46f` with 0 dirty paths; src sha256 `e63944e4673f29525ba8a77e64a50fdc25ec51bee22ab4ee5bc884bdc322979d`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 71.9 / 87.3 / 91.0 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 331.11 | 326.58 | 338.30 |
| llama.cpp pp (best threads) | 584.62 | 578.53 | 590.56 |
| ratio pp (per round) | 0.56 | 0.56 | 0.58 |
| apeiron tg | 68.14 | 67.32 | 69.91 |
| llama.cpp tg (best threads) | 53.79 | 53.05 | 54.93 |
| ratio tg (per round) | 1.27 | 1.24 | 1.32 |
| apeiron pp busy cores | 23.63 | 23.55 | 23.66 |
| apeiron tg busy cores | 23.79 | 23.75 | 23.84 |
| mean MHz before pp | 2464.67 | 1900.32 | 4067.69 |
| mean MHz before tg | 4722.34 | 4647.46 | 4849.54 |

### Traced phases

#### pp: 200 tokens, 2.84 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.063 | 2.2% | 245 | - | - |
| residual | 0.050 | 1.8% | 240 | - | - |
| attn.proj_in | 0.062 | 2.2% | 30 | 13.5 | 506.6 |
| attn.core | 0.265 | 9.3% | 30 | - | - |
| attn.proj_out | 0.035 | 1.2% | 30 | 9.6 | 361.6 |
| affine.proj_in | 0.274 | 9.6% | 90 | 14.7 | 554.0 |
| affine.conv | 0.310 | 10.9% | 90 | - | - |
| affine.heads | 1.029 | 36.3% | 90 | - | - |
| affine.proj_out | 0.097 | 3.4% | 90 | 10.3 | 387.5 |
| ffn | 0.609 | 21.5% | 120 | 11.5 | 434.0 |
| head | 0.024 | 0.9% | 1 | 56.0 | 52.7 |
| untraced in pass | 0.006 | 0.2% | 0 | - | - |
| outside passes | 0.011 | 0.4% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 56.7) | 0.543 | 19.2% | 2610 | - | - |

#### tg: 64 tokens, 14.57 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.198 | 1.4% | 3136 | - | - |
| residual | 0.049 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.597 | 4.1% | 384 | 56.0 | 52.7 |
| attn.core | 0.568 | 3.9% | 384 | - | - |
| attn.proj_out | 0.212 | 1.5% | 384 | 63.2 | 59.5 |
| affine.proj_in | 2.550 | 17.5% | 1152 | 63.2 | 59.4 |
| affine.conv | 0.725 | 5.0% | 1152 | - | - |
| affine.heads | 1.403 | 9.6% | 1152 | - | - |
| affine.proj_out | 0.614 | 4.2% | 1152 | 65.4 | 61.5 |
| ffn | 3.887 | 26.7% | 1536 | 72.2 | 68.0 |
| head | 3.733 | 25.6% | 64 | 72.4 | 68.1 |
| untraced in pass | 0.035 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 47.5) | 2.015 | 13.8% | 20864 | - | - |

<a id="1790626949-ns-r007"></a>

# Engine benchmark `ns-r007`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/ns/integ/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `129122d33294576b0043a26dcf2416e32f51cc28` with 0 dirty paths; src sha256 `5ba2368963c038c3dbcd8aed710a2569d6a209a6f42059df0000096d36fdaebb`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 83.3 / 85.7 / 91.7 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 339.72 | 330.68 | 356.59 |
| llama.cpp pp (best threads) | 578.45 | 574.43 | 602.67 |
| ratio pp (per round) | 0.58 | 0.55 | 0.62 |
| apeiron tg | 69.82 | 67.66 | 70.65 |
| llama.cpp tg (best threads) | 54.38 | 52.90 | 56.02 |
| ratio tg (per round) | 1.27 | 1.25 | 1.30 |
| apeiron pp busy cores | 23.66 | 23.35 | 23.66 |
| apeiron tg busy cores | 23.80 | 23.77 | 23.82 |
| mean MHz before pp | 2533.28 | 1973.35 | 3520.68 |
| mean MHz before tg | 4581.60 | 4546.70 | 4688.49 |

### Traced phases

#### pp: 200 tokens, 2.97 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.068 | 2.3% | 245 | - | - |
| residual | 0.050 | 1.7% | 240 | - | - |
| attn.proj_in | 0.074 | 2.5% | 30 | 11.3 | 426.8 |
| attn.core | 0.294 | 9.9% | 30 | - | - |
| attn.proj_out | 0.039 | 1.3% | 30 | 8.5 | 319.2 |
| affine.proj_in | 0.296 | 9.9% | 90 | 13.6 | 512.8 |
| affine.conv | 0.322 | 10.8% | 90 | - | - |
| affine.heads | 1.045 | 35.1% | 90 | - | - |
| affine.proj_out | 0.104 | 3.5% | 90 | 9.7 | 363.8 |
| ffn | 0.635 | 21.4% | 120 | 11.1 | 416.3 |
| head | 0.029 | 1.0% | 1 | 46.2 | 43.5 |
| untraced in pass | 0.006 | 0.2% | 0 | - | - |
| outside passes | 0.011 | 0.4% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 56.7) | 0.771 | 25.9% | 2610 | - | - |

#### tg: 64 tokens, 14.75 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.180 | 1.2% | 3136 | - | - |
| residual | 0.047 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.593 | 4.0% | 384 | 56.3 | 53.0 |
| attn.core | 0.556 | 3.8% | 384 | - | - |
| attn.proj_out | 0.259 | 1.8% | 384 | 51.6 | 48.6 |
| affine.proj_in | 2.563 | 17.4% | 1152 | 62.8 | 59.1 |
| affine.conv | 0.774 | 5.2% | 1152 | - | - |
| affine.heads | 1.348 | 9.1% | 1152 | - | - |
| affine.proj_out | 0.629 | 4.3% | 1152 | 63.8 | 60.0 |
| ffn | 4.007 | 27.2% | 1536 | 70.1 | 65.9 |
| head | 3.756 | 25.5% | 64 | 71.9 | 67.7 |
| untraced in pass | 0.033 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 47.5) | 2.647 | 18.0% | 20864 | - | - |

<a id="1790633894-ns-r009"></a>

# Engine benchmark `ns-r009`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/ns/integ/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `bb33653f4458507d3458c9031a616226562ce723` with 0 dirty paths; src sha256 `9ddd6f3bff99540071872db77d6dadd9d33d40b392200b757d0ca3618808b11a`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 72.7 / 84.9 / 88.7 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 27.78 | 18.16 | 334.14 |
| llama.cpp pp (best threads) | 557.91 | 533.31 | 588.97 |
| ratio pp (per round) | 0.05 | 0.03 | 0.57 |
| apeiron tg | 5.16 | 3.40 | 69.81 |
| llama.cpp tg (best threads) | 52.96 | 52.48 | 53.81 |
| ratio tg (per round) | 0.10 | 0.06 | 1.30 |
| apeiron pp busy cores | 6.61 | 3.74 | 23.34 |
| apeiron tg busy cores | 7.09 | 4.21 | 23.84 |
| mean MHz before pp | 2518.75 | 2124.72 | 3154.48 |
| mean MHz before tg | 4205.64 | 2031.85 | 4399.05 |

### Traced phases

#### pp: 200 tokens, 2.72 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.060 | 2.2% | 245 | - | - |
| residual | 0.050 | 1.8% | 240 | - | - |
| attn.proj_in | 0.061 | 2.3% | 30 | 13.6 | 512.8 |
| attn.core | 0.273 | 10.0% | 30 | - | - |
| attn.proj_out | 0.033 | 1.2% | 30 | 10.0 | 375.9 |
| affine.proj_in | 0.267 | 9.8% | 90 | 15.1 | 568.7 |
| affine.conv | 0.306 | 11.3% | 90 | - | - |
| affine.heads | 0.925 | 34.0% | 90 | - | - |
| affine.proj_out | 0.097 | 3.6% | 90 | 10.3 | 387.6 |
| ffn | 0.606 | 22.3% | 120 | 11.6 | 436.0 |
| head | 0.024 | 0.9% | 1 | 56.3 | 53.0 |
| untraced in pass | 0.006 | 0.2% | 0 | - | - |
| outside passes | 0.011 | 0.4% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 56.7) | 0.547 | 20.1% | 2610 | - | - |

#### tg: 64 tokens, 14.44 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.188 | 1.3% | 3136 | - | - |
| residual | 0.048 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.583 | 4.0% | 384 | 57.4 | 54.0 |
| attn.core | 0.561 | 3.9% | 384 | - | - |
| attn.proj_out | 0.205 | 1.4% | 384 | 65.1 | 61.3 |
| affine.proj_in | 2.559 | 17.7% | 1152 | 62.9 | 59.2 |
| affine.conv | 0.700 | 4.8% | 1152 | - | - |
| affine.heads | 1.320 | 9.1% | 1152 | - | - |
| affine.proj_out | 0.611 | 4.2% | 1152 | 65.7 | 61.8 |
| ffn | 3.914 | 27.1% | 1536 | 71.7 | 67.5 |
| head | 3.715 | 25.7% | 64 | 72.7 | 68.4 |
| untraced in pass | 0.034 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 47.5) | 2.167 | 15.0% | 20864 | - | - |

> **INVALID (host contention):** lead-dispatched experiments (hyperbolic-attention probe runs on 24 CPU threads and LM training with tokenization) ran during this receipt; load average ≈31, apeiron busy cores median 6.6. The quiet gate sampled only at round start. Not a measurement of the engine.

<a id="1790668367-ns-r012"></a>

# Engine benchmark `ns-r012`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/ns/integ/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `54eb0ff307c89a0dce5dd14de3d85098a61b8061` with 0 dirty paths; src sha256 `a0b0c082bc9885ae178f7d9c688f03923ae8ed953d9764fe1887bc6fe4038011`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 74.9 / 84.5 / 87.0 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 323.75 | 311.97 | 333.76 |
| llama.cpp pp (best threads) | 563.23 | 505.50 | 567.31 |
| ratio pp (per round) | 0.59 | 0.55 | 0.66 |
| apeiron tg | 64.16 | 60.40 | 68.16 |
| llama.cpp tg (best threads) | 51.52 | 49.71 | 52.02 |
| ratio tg (per round) | 1.29 | 1.17 | 1.34 |
| apeiron pp busy cores | 23.50 | 23.32 | 23.59 |
| apeiron tg busy cores | 23.64 | 22.97 | 23.76 |
| mean MHz before pp | 3034.77 | 1730.23 | 3726.67 |
| mean MHz before tg | 4762.01 | 4703.04 | 4803.93 |

### Traced phases

#### pp: 200 tokens, 2.91 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.064 | 2.2% | 245 | - | - |
| residual | 0.051 | 1.7% | 240 | - | - |
| attn.proj_in | 0.063 | 2.2% | 30 | 13.2 | 496.4 |
| attn.core | 0.290 | 10.0% | 30 | - | - |
| attn.proj_out | 0.044 | 1.5% | 30 | 7.5 | 283.1 |
| affine.proj_in | 0.288 | 9.9% | 90 | 14.0 | 526.9 |
| affine.conv | 0.330 | 11.3% | 90 | - | - |
| affine.heads | 0.974 | 33.5% | 90 | - | - |
| affine.proj_out | 0.103 | 3.5% | 90 | 9.8 | 368.1 |
| ffn | 0.663 | 22.8% | 120 | 10.6 | 398.7 |
| head | 0.024 | 0.8% | 1 | 56.1 | 52.8 |
| untraced in pass | 0.006 | 0.2% | 0 | - | - |
| outside passes | 0.011 | 0.4% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 56.7) | 0.695 | 23.9% | 2610 | - | - |

#### tg: 64 tokens, 15.61 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.184 | 1.2% | 3136 | - | - |
| residual | 0.047 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.637 | 4.1% | 384 | 52.5 | 49.4 |
| attn.core | 0.568 | 3.6% | 384 | - | - |
| attn.proj_out | 0.280 | 1.8% | 384 | 47.7 | 44.9 |
| affine.proj_in | 2.752 | 17.6% | 1152 | 58.5 | 55.1 |
| affine.conv | 0.781 | 5.0% | 1152 | - | - |
| affine.heads | 1.495 | 9.6% | 1152 | - | - |
| affine.proj_out | 0.631 | 4.0% | 1152 | 63.6 | 59.8 |
| ffn | 4.289 | 27.5% | 1536 | 65.5 | 61.6 |
| head | 3.911 | 25.1% | 64 | 69.1 | 65.0 |
| untraced in pass | 0.036 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 47.5) | 3.419 | 21.9% | 20864 | - | - |

<a id="1790678074-ns-r015"></a>

# Engine benchmark `ns-r015`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/ns/integ/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `509f8e2830f3388bbbd7c41527e84b63831e27ca` with 0 dirty paths; src sha256 `b9e0f526c55af1cbac3f0828b9c191c07b1bc45fd28ccfd21f27fc90aa84094e`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 76.3 / 83.8 / 85.6 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 337.32 | 323.99 | 359.61 |
| llama.cpp pp (best threads) | 566.57 | 543.51 | 576.15 |
| ratio pp (per round) | 0.60 | 0.56 | 0.63 |
| apeiron tg | 65.68 | 63.26 | 66.92 |
| llama.cpp tg (best threads) | 51.42 | 50.38 | 52.16 |
| ratio tg (per round) | 1.28 | 1.23 | 1.29 |
| apeiron pp busy cores | 23.54 | 23.16 | 23.61 |
| apeiron tg busy cores | 23.67 | 22.99 | 23.77 |
| mean MHz before pp | 3446.92 | 1854.10 | 3726.50 |
| mean MHz before tg | 4662.98 | 4621.27 | 4712.82 |

### Traced phases

#### pp: 200 tokens, 2.84 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.076 | 2.7% | 245 | - | - |
| residual | 0.050 | 1.8% | 240 | - | - |
| attn.proj_in | 0.061 | 2.2% | 30 | 13.6 | 513.7 |
| attn.core | 0.207 | 7.3% | 30 | - | - |
| attn.proj_out | 0.034 | 1.2% | 30 | 10.0 | 374.7 |
| affine.proj_in | 0.280 | 9.8% | 90 | 14.4 | 541.9 |
| affine.conv | 0.349 | 12.3% | 90 | - | - |
| affine.heads | 1.004 | 35.3% | 90 | - | - |
| affine.proj_out | 0.106 | 3.7% | 90 | 9.4 | 354.7 |
| ffn | 0.634 | 22.3% | 120 | 11.1 | 416.5 |
| head | 0.024 | 0.9% | 1 | 55.7 | 52.5 |
| untraced in pass | 0.006 | 0.2% | 0 | - | - |
| outside passes | 0.011 | 0.4% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 56.7) | 0.692 | 24.3% | 2610 | - | - |

#### tg: 64 tokens, 15.41 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.184 | 1.2% | 3136 | - | - |
| residual | 0.046 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.595 | 3.9% | 384 | 56.2 | 52.9 |
| attn.core | 0.480 | 3.1% | 384 | - | - |
| attn.proj_out | 0.212 | 1.4% | 384 | 63.0 | 59.3 |
| affine.proj_in | 2.608 | 16.9% | 1152 | 61.8 | 58.1 |
| affine.conv | 0.868 | 5.6% | 1152 | - | - |
| affine.heads | 1.474 | 9.6% | 1152 | - | - |
| affine.proj_out | 0.646 | 4.2% | 1152 | 62.1 | 58.5 |
| ffn | 4.334 | 28.1% | 1536 | 64.8 | 61.0 |
| head | 3.921 | 25.5% | 64 | 68.9 | 64.9 |
| untraced in pass | 0.035 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 47.5) | 3.202 | 20.8% | 20864 | - | - |

<a id="1790680301-ns-r016"></a>

# Engine benchmark `ns-r016`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/ns/integ/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `b0a4942a11a655e65cd5bb572b6af0db020e33fc` with 0 dirty paths; src sha256 `aa596166ccdbd94d69ff8d275a4e13cf666bf199f2f72d20298d5a308f87318c`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 69.8 / 84.7 / 89.5 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 340.51 | 332.58 | 344.30 |
| llama.cpp pp (best threads) | 567.84 | 543.75 | 571.99 |
| ratio pp (per round) | 0.60 | 0.58 | 0.63 |
| apeiron tg | 65.41 | 62.15 | 66.15 |
| llama.cpp tg (best threads) | 51.12 | 49.90 | 53.33 |
| ratio tg (per round) | 1.26 | 1.21 | 1.32 |
| apeiron pp busy cores | 23.53 | 23.33 | 23.57 |
| apeiron tg busy cores | 23.70 | 23.07 | 23.74 |
| mean MHz before pp | 3192.36 | 1793.09 | 4173.68 |
| mean MHz before tg | 4692.20 | 4622.22 | 4805.04 |

### Traced phases

#### pp: 200 tokens, 2.85 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.063 | 2.2% | 245 | - | - |
| residual | 0.052 | 1.8% | 240 | - | - |
| attn.proj_in | 0.071 | 2.5% | 30 | 11.8 | 443.1 |
| attn.core | 0.210 | 7.4% | 30 | - | - |
| attn.proj_out | 0.034 | 1.2% | 30 | 9.8 | 368.0 |
| affine.proj_in | 0.302 | 10.6% | 90 | 13.3 | 502.4 |
| affine.conv | 0.335 | 11.8% | 90 | - | - |
| affine.heads | 0.978 | 34.3% | 90 | - | - |
| affine.proj_out | 0.105 | 3.7% | 90 | 9.6 | 360.0 |
| ffn | 0.655 | 23.0% | 120 | 10.7 | 403.2 |
| head | 0.026 | 0.9% | 1 | 51.6 | 48.5 |
| untraced in pass | 0.006 | 0.2% | 0 | - | - |
| outside passes | 0.011 | 0.4% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 56.7) | 0.627 | 22.0% | 2610 | - | - |

#### tg: 64 tokens, 15.10 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.191 | 1.3% | 3136 | - | - |
| residual | 0.048 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.638 | 4.2% | 384 | 52.4 | 49.3 |
| attn.core | 0.499 | 3.3% | 384 | - | - |
| attn.proj_out | 0.267 | 1.8% | 384 | 50.2 | 47.2 |
| affine.proj_in | 2.683 | 17.8% | 1152 | 60.0 | 56.5 |
| affine.conv | 0.734 | 4.9% | 1152 | - | - |
| affine.heads | 1.411 | 9.3% | 1152 | - | - |
| affine.proj_out | 0.654 | 4.3% | 1152 | 61.3 | 57.7 |
| ffn | 4.101 | 27.2% | 1536 | 68.5 | 64.4 |
| head | 3.835 | 25.4% | 64 | 70.4 | 66.3 |
| untraced in pass | 0.037 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 47.5) | 2.535 | 16.8% | 20864 | - | - |

<a id="1790690080-ns-r019"></a>

# Engine benchmark `ns-r019`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/ns/integ/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `fff03d47d47e922a347a8580301dda3a2c33bd7e` with 0 dirty paths; src sha256 `e404013ace475feac2fbcad3c7bc7429bf542b359fc29e53bc3250f54c4a40fa`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 72.6 / 86.0 / 88.3 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 340.88 | 324.59 | 356.87 |
| llama.cpp pp (best threads) | 577.81 | 532.88 | 582.03 |
| ratio pp (per round) | 0.59 | 0.56 | 0.67 |
| apeiron tg | 64.74 | 57.52 | 67.47 |
| llama.cpp tg (best threads) | 51.06 | 50.56 | 52.08 |
| ratio tg (per round) | 1.24 | 1.14 | 1.32 |
| apeiron pp busy cores | 23.55 | 23.25 | 23.60 |
| apeiron tg busy cores | 23.70 | 21.74 | 23.76 |
| mean MHz before pp | 1970.95 | 1874.60 | 2926.52 |
| mean MHz before tg | 4703.12 | 4538.32 | 4788.36 |

### Traced phases

#### pp: 200 tokens, 2.87 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.066 | 2.3% | 245 | - | - |
| residual | 0.055 | 1.9% | 240 | - | - |
| attn.proj_in | 0.080 | 2.8% | 30 | 10.5 | 394.2 |
| attn.core | 0.194 | 6.8% | 30 | - | - |
| attn.proj_out | 0.035 | 1.2% | 30 | 9.6 | 362.0 |
| affine.proj_in | 0.288 | 10.0% | 90 | 14.0 | 527.2 |
| affine.conv | 0.313 | 10.9% | 90 | - | - |
| affine.heads | 1.001 | 34.9% | 90 | - | - |
| affine.proj_out | 0.131 | 4.6% | 90 | 7.7 | 288.7 |
| ffn | 0.652 | 22.8% | 120 | 10.8 | 405.2 |
| head | 0.032 | 1.1% | 1 | 42.8 | 40.2 |
| untraced in pass | 0.007 | 0.2% | 0 | - | - |
| outside passes | 0.013 | 0.4% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 56.7) | 0.597 | 20.8% | 2610 | - | - |

#### tg: 64 tokens, 15.60 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.189 | 1.2% | 3136 | - | - |
| residual | 0.047 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.619 | 4.0% | 384 | 54.0 | 50.8 |
| attn.core | 0.495 | 3.2% | 384 | - | - |
| attn.proj_out | 0.310 | 2.0% | 384 | 43.1 | 40.6 |
| affine.proj_in | 2.631 | 16.9% | 1152 | 61.2 | 57.6 |
| affine.conv | 0.800 | 5.1% | 1152 | - | - |
| affine.heads | 1.495 | 9.6% | 1152 | - | - |
| affine.proj_out | 0.672 | 4.3% | 1152 | 59.7 | 56.2 |
| ffn | 4.359 | 27.9% | 1536 | 64.4 | 60.6 |
| head | 3.941 | 25.3% | 64 | 68.5 | 64.5 |
| untraced in pass | 0.036 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 47.5) | 3.066 | 19.7% | 20864 | - | - |

<a id="1790692313-ns-r020"></a>

# Engine benchmark `ns-r020`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/ns/integ/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `9c3766704b26f8f8fa041e1a4648260906456cb4` with 0 dirty paths; src sha256 `6790c05e85bd630ab891fef9e9b350759a949d4912f4a0e57bb562678a146d69`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 67.5 / 84.4 / 87.8 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 323.00 | 296.72 | 329.47 |
| llama.cpp pp (best threads) | 552.48 | 523.99 | 558.48 |
| ratio pp (per round) | 0.60 | 0.54 | 0.62 |
| apeiron tg | 65.95 | 62.86 | 66.92 |
| llama.cpp tg (best threads) | 51.98 | 51.40 | 53.81 |
| ratio tg (per round) | 1.24 | 1.20 | 1.28 |
| apeiron pp busy cores | 23.50 | 23.13 | 23.55 |
| apeiron tg busy cores | 23.74 | 23.10 | 23.79 |
| mean MHz before pp | 2361.24 | 2044.44 | 3948.15 |
| mean MHz before tg | 4550.95 | 4549.68 | 4554.05 |

### Traced phases

#### pp: 200 tokens, 2.85 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.077 | 2.7% | 245 | - | - |
| residual | 0.050 | 1.8% | 240 | - | - |
| attn.proj_in | 0.062 | 2.2% | 30 | 13.4 | 504.8 |
| attn.core | 0.189 | 6.6% | 30 | - | - |
| attn.proj_out | 0.036 | 1.3% | 30 | 9.2 | 346.8 |
| affine.proj_in | 0.292 | 10.2% | 90 | 13.8 | 519.7 |
| affine.conv | 0.341 | 12.0% | 90 | - | - |
| affine.heads | 0.974 | 34.2% | 90 | - | - |
| affine.proj_out | 0.111 | 3.9% | 90 | 9.0 | 340.3 |
| ffn | 0.679 | 23.8% | 120 | 10.3 | 389.2 |
| head | 0.017 | 0.6% | 1 | 79.3 | 74.7 |
| untraced in pass | 0.006 | 0.2% | 0 | - | - |
| outside passes | 0.012 | 0.4% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 56.7) | 0.703 | 24.7% | 2611 | - | - |

#### tg: 64 tokens, 14.93 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.185 | 1.2% | 3136 | - | - |
| residual | 0.047 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.600 | 4.0% | 384 | 55.7 | 52.5 |
| attn.core | 0.493 | 3.3% | 384 | - | - |
| attn.proj_out | 0.253 | 1.7% | 384 | 52.8 | 49.7 |
| affine.proj_in | 2.708 | 18.1% | 1152 | 59.5 | 56.0 |
| affine.conv | 0.909 | 6.1% | 1152 | - | - |
| affine.heads | 1.578 | 10.6% | 1152 | - | - |
| affine.proj_out | 0.725 | 4.9% | 1152 | 55.3 | 52.1 |
| ffn | 4.372 | 29.3% | 1536 | 64.2 | 60.4 |
| head | 3.018 | 20.2% | 64 | 89.5 | 84.2 |
| untraced in pass | 0.035 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 47.4) | 3.478 | 23.3% | 20928 | - | - |

<a id="1790700697-ns-r023"></a>

# Engine benchmark `ns-r023`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/ns/integ/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `41a3b37ba09833c6b3a675e6ad88d4af7f4080cc` with 0 dirty paths; src sha256 `757c8e58bba00d617428410a23b2ddd5aab0d876260889369c6ed81977579e43`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 77.6 / 85.0 / 86.7 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 322.29 | 286.58 | 348.10 |
| llama.cpp pp (best threads) | 583.55 | 532.32 | 590.29 |
| ratio pp (per round) | 0.58 | 0.49 | 0.65 |
| apeiron tg | 69.00 | 68.38 | 72.06 |
| llama.cpp tg (best threads) | 51.68 | 50.02 | 53.70 |
| ratio tg (per round) | 1.36 | 1.28 | 1.39 |
| apeiron pp busy cores | 23.54 | 20.15 | 23.57 |
| apeiron tg busy cores | 23.81 | 23.81 | 23.86 |
| mean MHz before pp | 3221.75 | 2663.52 | 4409.38 |
| mean MHz before tg | 4585.34 | 4574.30 | 4591.95 |

### Traced phases

#### pp: 200 tokens, 2.63 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.062 | 2.4% | 245 | - | - |
| residual | 0.051 | 1.9% | 240 | - | - |
| attn.proj_in | 0.056 | 2.1% | 30 | 14.9 | 562.7 |
| attn.core | 0.187 | 7.1% | 30 | - | - |
| attn.proj_out | 0.032 | 1.2% | 30 | 10.4 | 391.1 |
| affine.proj_in | 0.256 | 9.7% | 90 | 15.7 | 591.5 |
| affine.conv | 0.295 | 11.2% | 90 | - | - |
| affine.heads | 0.960 | 36.4% | 90 | - | - |
| affine.proj_out | 0.099 | 3.8% | 90 | 10.1 | 379.5 |
| ffn | 0.600 | 22.8% | 120 | 11.7 | 440.6 |
| head | 0.017 | 0.6% | 1 | 80.4 | 75.6 |
| untraced in pass | 0.006 | 0.2% | 0 | - | - |
| outside passes | 0.012 | 0.4% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 56.7) | 0.509 | 19.3% | 2611 | - | - |

#### tg: 64 tokens, 13.96 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.192 | 1.4% | 3136 | - | - |
| residual | 0.048 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.615 | 4.4% | 384 | 54.3 | 51.2 |
| attn.core | 0.495 | 3.5% | 384 | - | - |
| attn.proj_out | 0.229 | 1.6% | 384 | 58.4 | 55.0 |
| affine.proj_in | 2.601 | 18.6% | 1152 | 61.9 | 58.3 |
| affine.conv | 0.724 | 5.2% | 1152 | - | - |
| affine.heads | 1.391 | 10.0% | 1152 | - | - |
| affine.proj_out | 0.626 | 4.5% | 1152 | 64.0 | 60.3 |
| ffn | 3.975 | 28.5% | 1536 | 70.6 | 66.5 |
| head | 3.020 | 21.6% | 64 | 89.5 | 84.2 |
| untraced in pass | 0.038 | 0.3% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 47.4) | 2.344 | 16.8% | 20928 | - | - |

<a id="1790710920-ns-r026"></a>

# Engine benchmark `ns-r026`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/ns/integ/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `b4d828765e6136c4e9721242e43c7563ff1a9808` with 0 dirty paths; src sha256 `814bcea1bea0fce8ac4d6f7832d0544cb874064ad8692e8065c5463d769fc1dc`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 71.8 / 84.1 / 86.4 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 349.47 | 331.78 | 355.84 |
| llama.cpp pp (best threads) | 551.54 | 544.94 | 577.83 |
| ratio pp (per round) | 0.62 | 0.57 | 0.65 |
| apeiron tg | 68.72 | 63.70 | 69.76 |
| llama.cpp tg (best threads) | 51.24 | 50.84 | 52.46 |
| ratio tg (per round) | 1.34 | 1.21 | 1.37 |
| apeiron pp busy cores | 23.44 | 22.69 | 23.54 |
| apeiron tg busy cores | 23.81 | 23.25 | 23.84 |
| mean MHz before pp | 2780.91 | 2206.27 | 3975.07 |
| mean MHz before tg | 4562.66 | 4525.79 | 4643.31 |

### Traced phases

#### pp: 200 tokens, 2.67 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.066 | 2.5% | 245 | - | - |
| residual | 0.051 | 1.9% | 240 | - | - |
| attn.proj_in | 0.047 | 1.8% | 30 | 17.8 | 668.5 |
| attn.core | 0.199 | 7.5% | 30 | - | - |
| attn.proj_out | 0.034 | 1.3% | 30 | 9.7 | 367.0 |
| affine.proj_in | 0.188 | 7.1% | 90 | 21.4 | 806.2 |
| affine.conv | 0.318 | 11.9% | 90 | - | - |
| affine.heads | 0.997 | 37.4% | 90 | - | - |
| affine.proj_out | 0.094 | 3.5% | 90 | 10.6 | 400.6 |
| ffn | 0.636 | 23.9% | 120 | 11.0 | 415.2 |
| head | 0.017 | 0.6% | 1 | 80.2 | 75.4 |
| untraced in pass | 0.006 | 0.2% | 0 | - | - |
| outside passes | 0.011 | 0.4% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 70.3) | 0.608 | 22.8% | 2083 | - | - |

#### tg: 64 tokens, 14.57 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.192 | 1.3% | 3136 | - | - |
| residual | 0.048 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.629 | 4.3% | 384 | 53.2 | 50.0 |
| attn.core | 0.531 | 3.6% | 384 | - | - |
| attn.proj_out | 0.212 | 1.5% | 384 | 63.0 | 59.3 |
| affine.proj_in | 2.741 | 18.8% | 1152 | 58.8 | 55.3 |
| affine.conv | 0.743 | 5.1% | 1152 | - | - |
| affine.heads | 1.474 | 10.1% | 1152 | - | - |
| affine.proj_out | 0.675 | 4.6% | 1152 | 59.4 | 55.9 |
| ffn | 4.269 | 29.3% | 1536 | 65.8 | 61.9 |
| head | 3.022 | 20.7% | 64 | 89.4 | 84.1 |
| untraced in pass | 0.035 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 47.4) | 3.041 | 20.9% | 20928 | - | - |

<a id="1790713685-ns-r027"></a>

# Engine benchmark `ns-r027`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/ns/integ/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `91461a9d6d7eaf9b3decb22d560a71c2e3b6f367` with 0 dirty paths; src sha256 `578e27fb4f0cf86130b2fb86934b1542a8f0ca5d67983e0c09501e0a0b0856d9`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 78.4 / 86.2 / 91.3 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 356.75 | 350.71 | 365.20 |
| llama.cpp pp (best threads) | 560.51 | 554.62 | 581.89 |
| ratio pp (per round) | 0.63 | 0.61 | 0.65 |
| apeiron tg | 68.76 | 59.72 | 70.84 |
| llama.cpp tg (best threads) | 50.60 | 50.04 | 52.00 |
| ratio tg (per round) | 1.36 | 1.19 | 1.39 |
| apeiron pp busy cores | 23.42 | 23.30 | 23.51 |
| apeiron tg busy cores | 23.79 | 21.53 | 23.85 |
| mean MHz before pp | 2780.33 | 1738.26 | 4333.67 |
| mean MHz before tg | 4561.37 | 4445.51 | 4616.70 |

### Traced phases

#### pp: 200 tokens, 2.59 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.071 | 2.8% | 245 | - | - |
| residual | 0.050 | 1.9% | 240 | - | - |
| attn.proj_in | 0.042 | 1.6% | 30 | 19.7 | 741.7 |
| attn.core | 0.184 | 7.1% | 30 | - | - |
| attn.proj_out | 0.031 | 1.2% | 30 | 10.9 | 409.5 |
| affine.proj_in | 0.194 | 7.5% | 90 | 20.7 | 780.5 |
| affine.conv | 0.310 | 12.0% | 90 | - | - |
| affine.heads | 0.979 | 37.8% | 90 | - | - |
| affine.proj_out | 0.112 | 4.3% | 90 | 9.0 | 337.8 |
| ffn | 0.582 | 22.5% | 120 | 12.1 | 454.3 |
| head | 0.017 | 0.7% | 1 | 80.2 | 75.5 |
| untraced in pass | 0.006 | 0.2% | 0 | - | - |
| outside passes | 0.011 | 0.4% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 70.3) | 0.548 | 21.2% | 2083 | - | - |

#### tg: 64 tokens, 14.15 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.187 | 1.3% | 3136 | - | - |
| residual | 0.046 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.597 | 4.2% | 384 | 56.0 | 52.7 |
| attn.core | 0.490 | 3.5% | 384 | - | - |
| attn.proj_out | 0.216 | 1.5% | 384 | 62.0 | 58.3 |
| affine.proj_in | 2.696 | 19.1% | 1152 | 59.7 | 56.2 |
| affine.conv | 0.748 | 5.3% | 1152 | - | - |
| affine.heads | 1.439 | 10.2% | 1152 | - | - |
| affine.proj_out | 0.623 | 4.4% | 1152 | 64.4 | 60.6 |
| ffn | 4.098 | 29.0% | 1536 | 68.5 | 64.5 |
| head | 2.975 | 21.0% | 64 | 90.8 | 85.5 |
| untraced in pass | 0.035 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 47.4) | 2.727 | 19.3% | 20928 | - | - |

<a id="1790717185-ns-r028"></a>

# Engine benchmark `ns-r028`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/ns/integ/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `446414de165d340848c135527d5a2fdf85bff892` with 0 dirty paths; src sha256 `46a5b1746eb7cc87bbf6c7f2da2087f3e732111261d815b0a8377fd6eb95170f`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 74.7 / 85.3 / 83.9 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 376.37 | 350.55 | 387.06 |
| llama.cpp pp (best threads) | 561.13 | 551.34 | 585.43 |
| ratio pp (per round) | 0.64 | 0.63 | 0.70 |
| apeiron tg | 69.91 | 65.79 | 71.35 |
| llama.cpp tg (best threads) | 50.31 | 48.57 | 51.44 |
| ratio tg (per round) | 1.40 | 1.28 | 1.47 |
| apeiron pp busy cores | 23.57 | 23.53 | 23.61 |
| apeiron tg busy cores | 23.87 | 23.79 | 23.89 |
| mean MHz before pp | 2751.00 | 1824.17 | 4081.86 |
| mean MHz before tg | 4550.03 | 4418.44 | 4550.12 |

### Traced phases

#### pp: 200 tokens, 2.51 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.061 | 2.4% | 245 | - | - |
| residual | 0.050 | 2.0% | 240 | - | - |
| attn.proj_in | 0.047 | 1.9% | 30 | 17.7 | 667.1 |
| attn.core | 0.189 | 7.5% | 30 | - | - |
| attn.proj_out | 0.032 | 1.3% | 30 | 10.6 | 399.4 |
| affine.proj_in | 0.189 | 7.5% | 90 | 21.3 | 803.1 |
| affine.conv | 0.244 | 9.7% | 90 | - | - |
| affine.heads | 0.967 | 38.5% | 90 | - | - |
| affine.proj_out | 0.094 | 3.7% | 90 | 10.7 | 402.2 |
| ffn | 0.602 | 24.0% | 120 | 11.7 | 438.7 |
| head | 0.020 | 0.8% | 1 | 67.9 | 63.9 |
| untraced in pass | 0.006 | 0.2% | 0 | - | - |
| outside passes | 0.011 | 0.4% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 70.3) | 0.441 | 17.5% | 2083 | - | - |

#### tg: 64 tokens, 14.06 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.189 | 1.3% | 3136 | - | - |
| residual | 0.048 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.638 | 4.5% | 384 | 52.4 | 49.3 |
| attn.core | 0.456 | 3.2% | 384 | - | - |
| attn.proj_out | 0.217 | 1.5% | 384 | 61.6 | 58.0 |
| affine.proj_in | 2.642 | 18.8% | 1152 | 61.0 | 57.4 |
| affine.conv | 0.694 | 4.9% | 1152 | - | - |
| affine.heads | 1.425 | 10.1% | 1152 | - | - |
| affine.proj_out | 0.640 | 4.6% | 1152 | 62.6 | 58.9 |
| ffn | 4.099 | 29.2% | 1536 | 68.5 | 64.5 |
| head | 2.971 | 21.1% | 64 | 90.9 | 85.6 |
| untraced in pass | 0.035 | 0.3% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 47.4) | 2.283 | 16.2% | 20928 | - | - |

