# A/B `astra-swiglu-owned`

- pp 200, tg 64, 5 rounds, arm order rotated per round; quiet gate busy < 0.05: every round passed

| arm | env | pp tok/s | tg tok/s | spec tok/s | pp ratio | tg ratio | spec ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| columns | `APEIRON_BATCH_SWIGLU=0` | 267.27 | 68.86 | 48.15 | 1.000 | 1.000 | 1.000 |
| owned | `APEIRON_BATCH_SWIGLU=1` | 272.67 | 67.96 | 47.32 | 1.027 | 0.987 | 0.992 |
| p4-columns | `APEIRON_ECORES=0 APEIRON_THREADS=4 APEIRON_BATCH_SWIGLU=0` | 139.51 | 49.97 | 23.49 | 0.522 | 0.724 | 0.491 |
| p4-owned | `APEIRON_ECORES=0 APEIRON_THREADS=4 APEIRON_BATCH_SWIGLU=1` | 140.14 | 49.25 | 23.40 | 0.521 | 0.714 | 0.485 |

Ratios are the median of per-round ratios to the first arm.
