# A/B `astra-swiglu-batch`

- pp 200, tg 64, 5 rounds, arm order rotated per round; quiet gate busy < 0.05: every round passed

| arm | env | pp tok/s | tg tok/s | spec tok/s | pp ratio | tg ratio | spec ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| columns | `APEIRON_BATCH_SWIGLU=0` | 273.23 | 68.99 | 49.51 | 1.000 | 1.000 | 1.000 |
| batched | `APEIRON_BATCH_SWIGLU=1` | 269.90 | 68.83 | 47.26 | 0.987 | 0.999 | 0.948 |
| p4-columns | `APEIRON_ECORES=0 APEIRON_THREADS=4 APEIRON_BATCH_SWIGLU=0` | 140.23 | 46.59 | 23.27 | 0.513 | 0.679 | 0.473 |
| p4-batched | `APEIRON_ECORES=0 APEIRON_THREADS=4 APEIRON_BATCH_SWIGLU=1` | 138.55 | 48.18 | 23.17 | 0.507 | 0.714 | 0.469 |

Ratios are the median of per-round ratios to the first arm.
