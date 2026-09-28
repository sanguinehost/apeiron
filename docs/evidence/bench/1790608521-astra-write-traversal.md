# A/B `astra-write-traversal`

- pp 200, tg 64, 5 rounds, arm order rotated per round; quiet gate busy < 0.05: every round passed

| arm | env | pp tok/s | tg tok/s | spec tok/s | pp ratio | tg ratio | spec ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| row | `APEIRON_WRITE_COLUMN_MAJOR=0` | 273.61 | 69.26 | NaN | 1.000 | 1.000 | NaN |
| column | `APEIRON_WRITE_COLUMN_MAJOR=1` | 273.28 | 69.01 | NaN | 1.000 | 0.996 | NaN |
| p4-row | `APEIRON_ECORES=0 APEIRON_THREADS=4 APEIRON_WRITE_COLUMN_MAJOR=0` | 140.41 | 49.41 | NaN | 0.511 | 0.710 | NaN |
| p4-column | `APEIRON_ECORES=0 APEIRON_THREADS=4 APEIRON_WRITE_COLUMN_MAJOR=1` | 140.58 | 48.23 | NaN | 0.514 | 0.699 | NaN |

Ratios are the median of per-round ratios to the first arm.
