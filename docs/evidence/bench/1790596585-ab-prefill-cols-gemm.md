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
