# A/B `astra-core-budget`

- pp 200, tg 64, 5 rounds, arm order rotated per round; quiet gate busy < 0.05: every round passed

| arm | env | pp tok/s | tg tok/s | spec tok/s | pp ratio | tg ratio | spec ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | `` | 271.38 | 69.15 | NaN | 1.000 | 1.000 | NaN |
| p8 | `APEIRON_ECORES=0` | 230.91 | 67.56 | NaN | 0.849 | 0.979 | NaN |
| p4 | `APEIRON_ECORES=0 APEIRON_THREADS=4` | 137.92 | 47.84 | NaN | 0.514 | 0.693 | NaN |
| p4-slices8 | `APEIRON_ECORES=0 APEIRON_THREADS=4 APEIRON_HEAD_SLICES=8` | 133.99 | 47.97 | NaN | 0.494 | 0.708 | NaN |

Ratios are the median of per-round ratios to the first arm.
