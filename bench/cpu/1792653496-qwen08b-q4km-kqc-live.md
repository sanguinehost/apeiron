# Engine benchmark `qwen08b-q4km-kqc-live`

- model `~/kld08/qwen08b-Q4_K_M.gguf` (sha256 `31b465c975b41b88707b9759536e098c917a252d0345e59431c0e27d7d99d352`)
- prompt ids `lead-1022/SPEED35/ids-README.txt` (sha256 `c39464bda16bb7808b875f0bddf140799af378ef4ffe6418f78577163c7ebd3d`), pp 512, tg 128, 5 rounds
- git `5064c50032d524d78ec72ecef54159b7bea1a12e` with 21 dirty paths; src sha256 `7119ff186ccd9b5567cf7930b54376d128b5c989bc7d2e8b540e4abb4d61d1a8`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 67.3 / 85.6 / 92.4 GB/s at 8 / 16 / 32 threads
- llama.cpp `60bc8866b`, threads 16,32; comparator gpu_info `` backends `CPU`
- quiet gate busy < 1: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 658.10 | 631.96 | 661.68 |
| llama.cpp pp (best threads) | 632.67 | 596.73 | 659.62 |
| ratio pp (per round) | 1.02 | 1.00 | 1.10 |
| apeiron tg | 84.19 | 79.75 | 84.84 |
| llama.cpp tg (best threads) | 60.39 | 59.81 | 61.06 |
| ratio tg (per round) | 1.39 | 1.33 | 1.40 |
| apeiron pp busy cores | 23.92 | 23.89 | 23.95 |
| apeiron tg busy cores | 23.92 | 23.89 | 23.92 |
| mean MHz before pp | 4287.50 | 4156.18 | 4415.38 |
| mean MHz before tg | 4270.15 | 4151.83 | 4406.48 |

### Traced phases

#### pp: 512 tokens, 1.56 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.003 | 0.2% | 2 | - | - |
| norm | 0.057 | 3.6% | 98 | - | - |
| residual | 0.051 | 3.3% | 96 | - | - |
| attn.proj_in | 0.059 | 3.8% | 12 | 2.2 | 533.3 |
| attn.core | 0.072 | 4.6% | 12 | - | - |
| attn.proj_out | 0.028 | 1.8% | 12 | 1.8 | 444.6 |
| affine.proj_in | 0.256 | 16.4% | 36 | 2.5 | 592.2 |
| affine.conv | 0.112 | 7.2% | 36 | - | - |
| affine.heads | 0.160 | 10.3% | 36 | - | - |
| affine.proj_out | 0.088 | 5.6% | 36 | 1.8 | 430.6 |
| ffn | 0.658 | 42.2% | 48 | 1.7 | 401.4 |
| head | 0.007 | 0.4% | 1 | 80.0 | 75.3 |
| untraced in pass | 0.005 | 0.3% | 0 | - | - |
| outside passes | 0.004 | 0.2% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 214.8) | 0.216 | 13.9% | 916 | - | - |

#### tg: 128 tokens, 11.97 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.005 | 0.0% | 128 | - | - |
| norm | 0.151 | 1.3% | 6272 | - | - |
| residual | 0.057 | 0.5% | 6144 | - | - |
| attn.proj_in | 0.653 | 5.5% | 768 | 51.2 | 48.2 |
| attn.core | 0.272 | 2.3% | 768 | - | - |
| attn.proj_out | 0.229 | 1.9% | 768 | 58.3 | 54.8 |
| affine.proj_in | 2.542 | 21.2% | 2304 | 63.4 | 59.6 |
| affine.conv | 0.311 | 2.6% | 2304 | - | - |
| affine.heads | 0.536 | 4.5% | 2304 | - | - |
| affine.proj_out | 0.646 | 5.4% | 2304 | 62.1 | 58.5 |
| ffn | 3.341 | 27.9% | 3072 | 84.0 | 79.1 |
| head | 3.181 | 26.6% | 128 | 84.9 | 79.9 |
| untraced in pass | 0.045 | 0.4% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 43.2) | 1.940 | 16.2% | 28672 | - | - |
