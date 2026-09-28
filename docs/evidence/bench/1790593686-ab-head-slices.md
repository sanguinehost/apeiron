# A/B `ab-head-slices`

- pp 200, tg 64, 5 rounds, arm order rotated per round; quiet gate busy < 0.05: every round passed

| arm | env | pp tok/s | tg tok/s | spec tok/s | pp ratio | tg ratio | spec ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| whole | `APEIRON_HEAD_SLICES=1` | 125.75 | 65.33 | 68.37 | 1.000 | 1.000 | 1.000 |
| s4 | `APEIRON_HEAD_SLICES=4` | 136.69 | 66.52 | 72.06 | 1.083 | 1.033 | 1.055 |
| s8 | `APEIRON_HEAD_SLICES=8` | 136.85 | 66.75 | 73.96 | 1.071 | 1.051 | 1.071 |

Ratios are the median of per-round ratios to the first arm.
