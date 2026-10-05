# Map bench baseline (`benches/map.rs`)

Command: `cargo bench --bench map --locked -- --check` (under `systemd-run --user --scope -p MemoryMax=9G`,
`timeout 570`), on the live tree at `8a80703` plus the bench itself; i9-14900KF; the box was running
two other build lanes (contended; each cell is the mean of 7 interleaved rounds with the min–max spread).
Exit 1: 40 cell-incumbent pairs exceed ratio 1.0. Grade: measured.

Arms: `apeiron::freq::Table<u64, i64>` (the executed body, vendored hashtrinity), `hashbrown 0.15.5` +
`rustc-hash 2.1.3` `FxBuildHasher`, `hashbrown` + `twox-hash 2.1.4` XXH3-64. Correctness oracle: identical
observable results across arms before timing (exit 2 on mismatch).

Reading: apeiron wins most `dense`, `high-bits` and `low-byte` cells; it loses `random` keys 2–3× at
n = 100000 on present-hit, absent-miss, insert and build; `absent-miss` n = 100000 `dense` is pathological
(a miss sweeping the contiguous dense block; addressed by APEIRON-HASHDOS-CAP).

```text
git: 8a80703916b650cba686fad007308d11e9161a0d dirty: true sha256(src/hashtrinity/mod.rs): 413f51d66ff3ff96b24a13f1d160e928b2aed96d45f1fa40fce329a9f9d75b84 sha256(src/freq.rs): cacdd492579645979475dc46b5d9a9da201763c088e4c846dc73ed0bcba7fdac rustc: rustc 1.97.0-nightly (17584a181 2026-04-13) cpu: Intel(R) Core(TM) i9-14900KF target_dir: target

| column         |       n | dist      | apeiron ns |   hb_fx ns |       ap/fx [min, max] | hb_xxh3 ns |     ap/xxh3 [min, max] |
|----------------|---------|-----------|------------|------------|------------------------|------------|------------------------|
| present-hit    |    1000 | dense     |       1.29 |       2.57 |  0.504 [0.488, 0.584] |       4.13 |  0.313 [0.305, 0.362] |
| present-hit    |  100000 | dense     |       0.62 |       2.32 |  0.283 [0.146, 0.495] |       3.47 |  0.180 [0.153, 0.300] |
| present-hit    | 1000000 | dense     |       1.22 |       9.51 |  0.127 [0.085, 0.207] |      19.41 |  0.061 [0.043, 0.084] |
| present-hit    |    1000 | random    |       1.89 |       1.13 |  1.658 [1.154, 2.921] |       1.94 |  0.973 [0.670, 1.811] |
| present-hit    |  100000 | random    |       6.55 |       2.34 |  2.797 [2.699, 2.926] |       3.59 |  1.827 [1.709, 1.941] |
| present-hit    | 1000000 | random    |       7.02 |      11.45 |  0.699 [0.325, 0.942] |      20.84 |  0.426 [0.160, 0.602] |
| present-hit    |    1000 | high-bits |       1.21 |       1.17 |  1.034 [1.007, 1.119] |       2.01 |  0.603 [0.562, 0.762] |
| present-hit    |  100000 | high-bits |       1.06 |       2.03 |  0.522 [0.496, 0.564] |       3.38 |  0.314 [0.296, 0.331] |
| present-hit    | 1000000 | high-bits |       7.44 |       8.45 |  0.880 [0.648, 2.011] |      11.25 |  0.680 [0.382, 1.665] |
| present-hit    |     256 | low-byte  |       0.88 |       1.14 |  0.775 [0.746, 0.809] |       1.88 |  0.469 [0.431, 0.607] |
| absent-miss    |    1000 | dense     |       0.48 |       0.87 |  0.547 [0.535, 0.574] |       2.80 |  0.194 [0.076, 0.233] |
| absent-miss    |  100000 | dense     |   43243.68 |       7.70 | 11215.319 [2104.316, 17185.809] |      15.11 | 3884.429 [1589.963, 6229.691] |
| absent-miss    | 1000000 | dense     |       0.97 |       2.70 |  0.358 [0.319, 0.399] |       7.47 |  0.130 [0.117, 0.145] |
| absent-miss    |    1000 | random    |       5.02 |       2.62 |  1.899 [1.448, 3.014] |       3.51 |  1.421 [1.078, 2.309] |
| absent-miss    |  100000 | random    |      18.64 |       5.73 |  3.252 [3.141, 3.320] |       7.38 |  2.526 [2.481, 2.575] |
| absent-miss    | 1000000 | random    |      15.52 |       5.20 |  2.993 [2.167, 3.554] |       7.14 |  2.214 [1.509, 3.139] |
| absent-miss    |    1000 | high-bits |       3.64 |       2.00 |  1.825 [1.779, 1.907] |       3.88 |  0.938 [0.897, 1.005] |
| absent-miss    |  100000 | high-bits |       5.01 |       1.72 |  2.928 [2.589, 3.367] |       7.20 |  0.697 [0.632, 0.735] |
| absent-miss    | 1000000 | high-bits |      13.86 |       3.06 |  4.414 [2.692, 6.713] |      10.78 |  1.445 [0.414, 2.741] |
| absent-miss    |     256 | low-byte  |       4.75 |       2.02 |  2.356 [1.704, 5.508] |       3.75 |  1.269 [0.894, 2.963] |
| insert-fresh   |    1000 | dense     |       2.15 |       4.73 |  0.454 [0.447, 0.460] |       5.87 |  0.366 [0.358, 0.369] |
| insert-fresh   |  100000 | dense     |       2.18 |       6.84 |  0.319 [0.295, 0.381] |       7.08 |  0.308 [0.282, 0.361] |
| insert-fresh   | 1000000 | dense     |       6.43 |      23.49 |  0.281 [0.187, 0.322] |      23.69 |  0.272 [0.247, 0.294] |
| insert-fresh   |    1000 | random    |       3.85 |       4.36 |  0.878 [0.704, 1.338] |       5.24 |  0.737 [0.541, 1.181] |
| insert-fresh   |  100000 | random    |      12.86 |       6.97 |  1.853 [1.668, 2.065] |       7.33 |  1.757 [1.563, 1.892] |
| insert-fresh   | 1000000 | random    |      31.68 |      20.52 |  1.547 [1.435, 1.694] |      21.15 |  1.500 [1.357, 1.629] |
| insert-fresh   |    1000 | high-bits |       2.31 |       4.74 |  0.487 [0.465, 0.543] |       4.19 |  0.551 [0.538, 0.605] |
| insert-fresh   |  100000 | high-bits |       5.89 |       5.33 |  1.172 [0.718, 1.414] |       5.92 |  1.025 [0.781, 1.187] |
| insert-fresh   | 1000000 | high-bits |      31.32 |      19.93 |  1.572 [1.471, 1.652] |      20.43 |  1.537 [1.414, 1.646] |
| insert-fresh   |     256 | low-byte  |       2.51 |       4.92 |  0.509 [0.478, 0.608] |       5.43 |  0.461 [0.421, 0.577] |
| build          |    1000 | dense     |       5.47 |      11.53 |  0.474 [0.465, 0.499] |      14.21 |  0.385 [0.362, 0.394] |
| build          |  100000 | dense     |       5.17 |      10.84 |  0.477 [0.452, 0.491] |      13.80 |  0.375 [0.355, 0.390] |
| build          | 1000000 | dense     |      20.95 |      35.74 |  0.586 [0.569, 0.608] |      40.82 |  0.514 [0.491, 0.530] |
| build          |    1000 | random    |      21.01 |      15.83 |  1.320 [1.139, 1.819] |      15.49 |  1.342 [1.079, 1.734] |
| build          |  100000 | random    |      29.17 |      14.04 |  2.091 [1.690, 2.260] |      14.71 |  1.987 [1.775, 2.073] |
| build          | 1000000 | random    |      78.78 |      42.26 |  1.882 [1.526, 2.854] |      43.12 |  1.831 [1.469, 2.560] |
| build          |    1000 | high-bits |      14.32 |      13.86 |  1.032 [0.950, 1.233] |      15.05 |  0.949 [0.902, 1.044] |
| build          |  100000 | high-bits |      12.75 |      10.81 |  1.173 [0.722, 3.195] |      13.70 |  0.920 [0.588, 2.476] |
| build          | 1000000 | high-bits |      54.32 |      42.91 |  1.358 [0.736, 2.808] |      55.24 |  1.010 [0.638, 1.446] |
| build          |     256 | low-byte  |       7.46 |      12.66 |  0.586 [0.485, 0.802] |      14.90 |  0.496 [0.429, 0.612] |
| update         |    1000 | dense     |       1.75 |       3.24 |  0.542 [0.440, 0.720] |       3.91 |  0.447 [0.417, 0.600] |
| update         |  100000 | dense     |       2.07 |       5.90 |  0.351 [0.338, 0.378] |       7.63 |  0.273 [0.242, 0.302] |
| update         | 1000000 | dense     |       2.07 |      17.22 |  0.121 [0.112, 0.140] |      25.33 |  0.082 [0.076, 0.090] |
| update         |    1000 | random    |       4.33 |       3.53 |  1.223 [0.916, 2.113] |       4.47 |  0.969 [0.724, 1.701] |
| update         |  100000 | random    |      12.12 |       5.87 |  2.068 [1.983, 2.219] |       7.34 |  1.652 [1.570, 1.717] |
| update         | 1000000 | random    |      16.73 |      20.32 |  0.826 [0.744, 0.918] |      24.51 |  0.684 [0.633, 0.741] |
| update         |    1000 | high-bits |       2.70 |       3.59 |  0.753 [0.689, 0.825] |       4.38 |  0.616 [0.559, 0.677] |
| update         |  100000 | high-bits |       4.16 |       5.81 |  0.716 [0.708, 0.723] |       7.38 |  0.563 [0.553, 0.574] |
| update         | 1000000 | high-bits |      15.59 |      22.07 |  0.713 [0.534, 0.861] |      20.85 |  0.800 [0.532, 1.338] |
| update         |     256 | low-byte  |       2.81 |       3.81 |  0.739 [0.656, 0.800] |       4.62 |  0.609 [0.560, 0.702] |
| remove-present |    1000 | dense     |       1.83 |       3.76 |  0.486 [0.475, 0.497] |       5.23 |  0.351 [0.318, 0.367] |
| remove-present |  100000 | dense     |       1.54 |       4.51 |  0.358 [0.296, 0.516] |      10.53 |  0.146 [0.139, 0.158] |
| remove-present | 1000000 | dense     |       1.74 |      17.19 |  0.101 [0.094, 0.112] |      22.48 |  0.077 [0.073, 0.086] |
| remove-present |    1000 | random    |       3.63 |       3.93 |  0.929 [0.627, 1.599] |       4.67 |  0.766 [0.560, 1.198] |
| remove-present |  100000 | random    |      11.88 |       9.66 |  1.235 [1.119, 1.375] |      12.05 |  0.988 [0.908, 1.076] |
| remove-present | 1000000 | random    |      17.36 |      18.61 |  0.936 [0.821, 1.015] |      23.06 |  0.756 [0.691, 0.870] |
| remove-present |    1000 | high-bits |       2.06 |       3.68 |  0.561 [0.538, 0.606] |       4.68 |  0.441 [0.424, 0.474] |
| remove-present |  100000 | high-bits |       3.35 |       5.91 |  0.566 [0.557, 0.570] |      11.81 |  0.284 [0.271, 0.318] |
| remove-present | 1000000 | high-bits |      18.12 |      21.81 |  0.835 [0.710, 0.950] |      22.09 |  0.826 [0.729, 0.963] |
| remove-present |     256 | low-byte  |       1.79 |       3.05 |  0.586 [0.551, 0.647] |       4.39 |  0.407 [0.379, 0.485] |

Check failed: 40 cell-ratio thresholds exceeded:
  FAIL: present-hit n=1000 dist=random: ratio vs hb_fx = 1.658 > 1.0
  FAIL: present-hit n=100000 dist=random: ratio vs hb_fx = 2.797 > 1.0
  FAIL: present-hit n=100000 dist=random: ratio vs hb_xxh3 = 1.827 > 1.0
  FAIL: present-hit n=1000 dist=high-bits: ratio vs hb_fx = 1.034 > 1.0
  FAIL: absent-miss n=100000 dist=dense: ratio vs hb_fx = 11215.319 > 1.0
  FAIL: absent-miss n=100000 dist=dense: ratio vs hb_xxh3 = 3884.429 > 1.0
  FAIL: absent-miss n=1000 dist=random: ratio vs hb_fx = 1.899 > 1.0
  FAIL: absent-miss n=1000 dist=random: ratio vs hb_xxh3 = 1.421 > 1.0
  FAIL: absent-miss n=100000 dist=random: ratio vs hb_fx = 3.252 > 1.0
  FAIL: absent-miss n=100000 dist=random: ratio vs hb_xxh3 = 2.526 > 1.0
  FAIL: absent-miss n=1000000 dist=random: ratio vs hb_fx = 2.993 > 1.0
  FAIL: absent-miss n=1000000 dist=random: ratio vs hb_xxh3 = 2.214 > 1.0
  FAIL: absent-miss n=1000 dist=high-bits: ratio vs hb_fx = 1.825 > 1.0
  FAIL: absent-miss n=100000 dist=high-bits: ratio vs hb_fx = 2.928 > 1.0
  FAIL: absent-miss n=1000000 dist=high-bits: ratio vs hb_fx = 4.414 > 1.0
  FAIL: absent-miss n=1000000 dist=high-bits: ratio vs hb_xxh3 = 1.445 > 1.0
  FAIL: absent-miss n=256 dist=low-byte: ratio vs hb_fx = 2.356 > 1.0
  FAIL: absent-miss n=256 dist=low-byte: ratio vs hb_xxh3 = 1.269 > 1.0
  FAIL: insert-fresh n=100000 dist=random: ratio vs hb_fx = 1.853 > 1.0
  FAIL: insert-fresh n=100000 dist=random: ratio vs hb_xxh3 = 1.757 > 1.0
  FAIL: insert-fresh n=1000000 dist=random: ratio vs hb_fx = 1.547 > 1.0
  FAIL: insert-fresh n=1000000 dist=random: ratio vs hb_xxh3 = 1.500 > 1.0
  FAIL: insert-fresh n=100000 dist=high-bits: ratio vs hb_fx = 1.172 > 1.0
  FAIL: insert-fresh n=100000 dist=high-bits: ratio vs hb_xxh3 = 1.025 > 1.0
  FAIL: insert-fresh n=1000000 dist=high-bits: ratio vs hb_fx = 1.572 > 1.0
  FAIL: insert-fresh n=1000000 dist=high-bits: ratio vs hb_xxh3 = 1.537 > 1.0
  FAIL: build n=1000 dist=random: ratio vs hb_fx = 1.320 > 1.0
  FAIL: build n=1000 dist=random: ratio vs hb_xxh3 = 1.342 > 1.0
  FAIL: build n=100000 dist=random: ratio vs hb_fx = 2.091 > 1.0
  FAIL: build n=100000 dist=random: ratio vs hb_xxh3 = 1.987 > 1.0
  FAIL: build n=1000000 dist=random: ratio vs hb_fx = 1.882 > 1.0
  FAIL: build n=1000000 dist=random: ratio vs hb_xxh3 = 1.831 > 1.0
  FAIL: build n=1000 dist=high-bits: ratio vs hb_fx = 1.032 > 1.0
  FAIL: build n=100000 dist=high-bits: ratio vs hb_fx = 1.173 > 1.0
  FAIL: build n=1000000 dist=high-bits: ratio vs hb_fx = 1.358 > 1.0
  FAIL: build n=1000000 dist=high-bits: ratio vs hb_xxh3 = 1.010 > 1.0
  FAIL: update n=1000 dist=random: ratio vs hb_fx = 1.223 > 1.0
  FAIL: update n=100000 dist=random: ratio vs hb_fx = 2.068 > 1.0
  FAIL: update n=100000 dist=random: ratio vs hb_xxh3 = 1.652 > 1.0
  FAIL: remove-present n=100000 dist=random: ratio vs hb_fx = 1.235 > 1.0
error: bench failed, to rerun pass `--bench map`

Caused by:
  process didn't exit successfully: `/home/socol/Workspace/apeiron/target/release/deps/map-40d57cefe8315168 --check --json /tmp/claude-1000/-home-socol-Workspace-apeiron/0c0fd0d8-6dda-4eb1-afe9-7296a5d3b119/scratchpad/bench-live.jsonl --bench` (exit status: 1)
```
