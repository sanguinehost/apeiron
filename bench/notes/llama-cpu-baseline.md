# llama.cpp CPU baseline: the pinned numbers an engine receipt quotes

Status: `measured` for every row, with the host, command and CSV values kept here so a later
receipt does not re-run llama.cpp to answer "where do we stand". The llama.cpp rows are append-only:
a new engine receipt quotes them and adds only its engine row; llama.cpp is re-measured only when
the pin, host, model or prompt tuple changes, and then the new row is added with its CSV line. The
engine position table is a snapshot, updated by the receipt that measures it.

Every row here is **Qwen3-0.6B-Base F16** (596 M params); none is a 7B or a `Q4_K_M` row.

At this note: box cold 1,827-token prefill is **1.59–1.82× behind** llama.cpp at 1/4/8 threads for
the shipped one-dispatch build; its HEAD (`bd5e02dd`) measures **1.70–2.04× behind** across the
windows in this note, and width 1 is the one shipped row that regresses. The laptop's last paired
cold turn is **3.04–4.01× behind** in a loaded window; laptop decode is **1.22× ahead in that
loaded window** (1.47× in the clean one). The tables below carry the rows, the windows and the
commands.

## Pin and artifacts

- Source `~/llama.cpp` at commit `049326a00` (`git -C ~/llama.cpp log -1` is the provenance line),
  Release build with `-DGGML_NATIVE=ON -DGGML_LTO=OFF`, targets `llama-bench` and
  `llama-completion`.
- Model `~/models/qwen3-0.6b-base-f16/Qwen3-0.6B-Base.f16.gguf`: **1,192,230,912 B** of tensors
  (llama-bench's model size; the file is 1,198,182,592 B), **596,049,920 params**, F16.
- Box: `ubuntu@139.99.223.161`, 8 vCPU EPYC Milan, 8 sockets × 1 core × 1 thread, 31.3 GiB,
  256 MiB L3, no AVX-512, no AVX-VNNI, no cpufreq governor.
- Laptop: i5-10210U, 4C/8T, 15 W, 6 MiB L3, AVX2/BMI2/POPCNT.

## Box: prefill, `-p 1827 -n 0 -r 3`

```sh
~/llama.cpp/build/bin/llama-bench -m ~/models/qwen3-0.6b-base-f16/Qwen3-0.6B-Base.f16.gguf \
  -p 1827 -n 0 -t <1|4|8> -r 3 -o csv
```

Canonical row (`/tmp/opencode/final-evidence.log`, the hour of the receipt's earlier A/B):

| threads | tok/s | ± stddev | ms/token |
|:---:|---:|---:|---:|
| 1 | 74.933822 | 0.057259 | 13.35 |
| 4 | 262.897104 | 0.248647 | 3.80 |
| 8 | 460.261463 | 1.632670 | 2.17 |

Two other hours of the same tuple on the box: 74.18 / 262.49 / 465.64 ± 0.46 / 2.05 / 0.48
(`/tmp/opencode/gate-final.log`) and 74.50 / 262.99 / 463.11 ± 0.43 / 1.29 / 1.85
(`/tmp/opencode/ab.log`), so the across-hour range is 74.18–74.93, 262.49–262.99 and
460.26–465.64, a 0.2–1.2% spread across the three hours. The engine's cold 1,827-position
prefill at block 1024 is compared against the canonical row in `bench/cpu/1795593118-nq-planes.md`
§ Position against llama.cpp; that receipt's ship-gate session did not re-run llama.cpp, so its
engine row is compared across hours, with the range above as the uncertainty.

## Box: decode, `-p 0 -n 64 -r 3`

```sh
~/llama.cpp/build/bin/llama-bench -m ~/models/qwen3-0.6b-base-f16/Qwen3-0.6B-Base.f16.gguf \
  -p 0 -n 64 -t <1|4|8> -r 3 -o csv
```

| threads | tok/s | ± stddev | ms/token |
|:---:|---:|---:|---:|
| 1 | 26.400796 | 0.023332 | 37.88 |
| 4 | 64.221545 | 0.122860 | 15.57 |
| 8 | 93.851521 | 1.753149 | 10.65 |

One hour (`/tmp/opencode/ab-planes.log` lines 200–202, `llama-bench f16 decode tg64`). The box has
no llama.cpp turn row: `llama-bench` times
pp and tg separately and does not run them as one process, so the laptop `llama-completion` table
below is the turn baseline. This decode row is generation from an empty prompt; it is not the
same workload as decode after a 1,827-token prefix, so no engine/llama ratio is computed across
those two rows. The engine's decode-after-prefix rows live in the receipt that measures them.

## Laptop: paired cold turn, `llama-completion --prompt-cache`

```sh
# P is the 1,838-token text of the engine's `--len 1827` prompt (llama-p1.txt); the 187-token
# delta is llama-p2.txt. The binary is ~/Workspace/llamacpp-arelath at 049326a, the same pin.
rm -f ~/models/nq-scratch/llama-f16.cache          # cold pass: no prompt cache
~/Workspace/llamacpp-arelath/build/bin/llama-completion \
  -m ~/models/qwen3-0.6b-base-f16/Qwen3-0.6B-Base.f16.gguf \
  --prompt-cache ~/models/nq-scratch/llama-f16.cache \
  -c 2048 -t 4 -ngl 0 --no-warmup --no-display-prompt \
  -n 16 --temp 0 --seed 1 -p "$(cat "$P")"
```

1,838-token prompt, 187-token delta, three alternating rounds against the engine on
`skeleton-q8head.gguf` + `nq06-v3.nqp` (`bench/cpu/1795503411-nq06-tiled-attention.md`; llama
reports `eval time` over 15 runs because the 16-token generation replays the last prompt token, so
both sides quote wall per prefilled token):

| turn | llama.cpp | window |
|---|---:|---|
| cold prefill | 11.69 ms/token (85.6 tok/s) | 89 °C, another lane's `rustc` |
| 211-token tail | 15.76 ms/token | same |
| cold 16-token turn | 23.39 s | same |
| tail 16-token turn | 4.72 s | same |
| decode | 8.24 tok/s (8.08–8.36) | same |
| cold prefill | 8.62 ms/token (116.0 tok/s) | clean, load 0.85, 54 °C (`1795498875`) |
| 211-token tail | 10.82 ms/token | same (`1795498875`) |
| decode | 10.01 tok/s (8.77–10.28) | same (`1795498875`) |

Between the windows both sides slow together: the engine 60.911 → 85.729 s cold (1.41×) and
llama.cpp 15.836 → 21.478 s (1.36×), so the paired ratio moves 3.87× → 4.01× (1.04×). The
engine build also changed between the two tables (pooled projection → tiled attention), so the
windows do
not decompose into a thermal factor alone; each row is quoted with its window and build, and a
clean-window absolute is not inferred from the loaded one.

## Position of the engine at this note

Box, cold 1,827-position prefill, block 1024, `APEIRON_PREFILL_COLS=1024`
(`bench/cpu/1795593118-nq-planes.md`):

| threads | llama.cpp tok/s | engine HEAD `0f65d137` | ratio | engine + planes (`bd5e02dd`) | ratio |
|:---:|---:|---:|---:|---:|---:|
| 1 | 74.93 | 42.68 | 1.76× behind | 43.55 | 1.72× behind |
| 4 | 262.90 | 128.13 | 2.05× behind | 150.12 | 1.75× behind |
| 8 | 460.26 | 173.80 | 2.65× behind | 225.27 | 2.04× behind |

The engine rows are the three-pair means of `bench/cpu/1795593118-nq-planes.md`, which supersede
the earlier one-run snapshot; the HEAD row is the committed `nq.rs` (`b9a1004f…`) rebuilt as
`~/bin/prefix_solve.head`, so HEAD moves with the run's spread too.

Laptop, loaded 89 °C window (`1795503411`, engine tiled-attention build):

| turn | engine | llama.cpp | ratio |
|---|---:|---:|---:|
| cold prefill, per token | 46.93 ms | 11.69 ms | 4.01× behind |
| 211-token tail, per token | 59.18 ms | 15.76 ms | 3.76× behind |
| cold 16-token turn | 87.37 s | 23.39 s | 3.73× behind |
| tail 16-token turn | 14.34 s | 4.72 s | 3.04× behind |
| decode | 10.03 tok/s | 8.24 tok/s | engine 1.22× ahead |

Laptop, clean window (`1795498875`, engine pooled-projection build):

| turn | engine | llama.cpp | ratio |
|---|---:|---:|---:|
| cold prefill, per token | 33.34 ms | 8.62 ms | 3.87× behind |
| 211-token tail, per token | 54.7 ms | 10.82 ms | 5.05× behind |
| cold 16-token turn | 62.00 s | 17.43 s | 3.56× behind |
| tail 16-token turn | 12.74 s | 3.50 s | 3.64× behind |
| decode | 14.67 tok/s | 10.01 tok/s | engine 1.47× ahead |

The laptop engine rows predate the plane change; they are the last paired laptop runs, not the
current build, and the two tables use different engine builds. The box rows are the current
comparison.

## How to use it

- A receipt that claims parity or a win quotes this pin, host, model and tuple, and the row it
  compares against. It does not re-run llama.cpp unless the tuple changed.
- A new llama.cpp row is appended with the CSV line (tok/s and stddev), the host, the thread count
  and the hour's condition, so the old row stays legible.
- The engine row is same-host; same-hour interleaving is required for a ratio, and a cross-host or
  cross-build ratio is labeled as such.

## The one-dispatch prefill window

The engine's pooled projection loop changed, so llama.cpp was re-run in the same window for a paired
ratio: same pin (`049326a00`), host (8 vCPU EPYC Milan), model (F16 `Qwen3-0.6B-Base.f16.gguf`) and
tuple (`-p 1827 -n 0 -r 1`). Another repository's `cargo` build shared the box (load 4.4–7.6);
2026-11-25 10:14–10:44 UTC (`/tmp/opencode/final-gate.log`, `/tmp/opencode/final-rows.log`).

| threads | llama.cpp tok/s, six or three reps | mean | this note's pinned row |
|:---:|---|---:|---:|
| 1 | 74.86, 74.53, 74.78 | 74.72 | 74.93 |
| 4 | 266.59, 268.19, 262.19 | 265.66 | 262.90 |
| 8 | 471.29, 466.48, 467.28, 463.82, 460.23, 463.56 | 465.45 | 460.26 |

The means sit inside 1.8% of the pinned rows and the six width-8 reps span 2.4% (460.23–471.29);
the pinned rows are not superseded, they are bracketed.

| threads | llama.cpp tok/s | engine HEAD `bd5e02dd` tok/s | ratio | engine + one dispatch + parallel add `84a9a772` | ratio |
|:---:|---:|---:|---:|---:|---:|
| 1 | 74.72 | 43.59 | 1.71× behind | 41.25 | 1.81× behind |
| 4 | 265.66 | 150.72 | 1.76× behind | 164.53 | 1.61× behind |
| 8 | 465.45 | 229.39 | 2.03× behind | 276.23 | **1.69× behind** |

The engine rows are the `/tmp/opencode/final-attr.log` window — two interleaved pairs at width 1 and
three at widths 4 and 8 — and the right-hand column is the `84a9a772` build (one dispatch plus the
parallel residual add), not the shipped `merge4` rows below. The width-8 llama wall is 3.88–3.97 s
against the engine's 7.965 s (HEAD) and 6.614 s (change). The number is 1.69× inside 1.67–1.71× as
the llama band moves.

Provenance note: `~/bin/prefix_solve.head` (`3886b48e…`, `nq.rs` `b9a1004f…`) is commit
`0f65d137`, one commit before the committed planes change (`bd5e02dd…`, `final5` `d62ad6d4…`).
A screen that quotes 10.45 s as "HEAD" at 1,827 is quoting the older commit; HEAD's wall is
7.965 s in this window.

## The same comparison in the quiet window, with the parallel add

The window above was shared with another repository's `cargo` build. The box went quiet (load 0.34
with no other job) and the change was timed there with the parallel residual add: same pin, host,
model and tuple,
2026-11-25 10:50–11:10 UTC (`/tmp/opencode/ship3-timed.log`). llama.cpp was interleaved between the
engine pairs, so this is a paired row for the add arm, not a replacement for the pinned rows.

| threads | llama.cpp tok/s | mean | engine HEAD `bd5e02dd` tok/s | ratio | engine + one dispatch + parallel add `aa1a6448` tok/s | ratio |
|:---:|---|---:|---:|---:|---:|---:|
| 1 | 74.10, 74.12 | 74.11 | 43.39 | 1.71× behind | 40.58 | 1.83× behind |
| 4 | 258.40, 262.24, 261.12 | 260.59 | 150.67 | 1.73× behind | 164.66 | 1.58× behind |
| 8 | 456.73, 455.53, 466.91 | 459.72 | 227.94 | 2.02× behind | 276.41 | **1.66× behind** |

The quiet llama rows are 0.8–1.9% below the window above, so the ratio moves 1.69× → 1.66× on the
llama row alone, the engine's width-8 rate being unchanged (276.23 → 276.41 tok/s) while its own
HEAD row moves 7.965 → 8.015 s. The width-8 llama wall is 3.91–4.01 s against the engine's 8.015 s
(HEAD) and 6.610 s (change).

The provenance note above stands: `~/bin/prefix_solve.head` is not HEAD. The quiet-window binary,
`merge3`, is the one-dispatch change **with** a parallel `add_cols` (`cols.rs` `653a594f…`); it is
not shipped, and its rows are kept here as the measured size of that deferred change. The shipped
rows are in the next section.

## After the parallel add was reverted

The band-parallel residual add above changes `step::cols::add_cols`, which the extraction carries
and `proof/corpus/Serve/Block/Cols2.lean` proves it stops at the first refusing column
(`word_add_cols_refuses`); a pool of bands runs every band, so the `Err` post-state is not the proved
one. It is not shipped. The shipped binary is the one-dispatch change alone, re-timed in a third
window with the same pin, host, model and tuple, 2026-11-25 11:20–11:45 UTC
(`/tmp/opencode/final2-timed.log`).

| threads | llama.cpp tok/s | mean | engine HEAD `bd5e02dd` tok/s | ratio | engine one dispatch `eae8ce74` tok/s | ratio | + parallel add `aa1a6448` tok/s | ratio |
|:---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 73.59, 74.40 | 73.99 | 43.42 | 1.70× behind | 40.64 | 1.82× behind | 40.92 | 1.81× behind |
| 4 | 259.79, 257.87 | 258.83 | 151.11 | 1.71× behind | 162.32 | 1.59× behind | 165.65 | 1.56× behind |
| 8 | 449.56, 464.31, 456.63 | 456.83 | 232.62 | 1.96× behind | 262.61 | **1.74× behind** | 275.91 | 1.66× behind |

The width-8 llama wall is 3.94–4.06 s against the engine's 7.854 s (HEAD), 6.957 s (shipped) and
6.622 s (with the add). The llama means drift 0.6–1.9% between the three windows of this note, so a
cross-engine ratio is quoted inside the llama band: 1.74× is 1.71–1.77× as the width-8 reps move.
The shipped change is 1.13× at eight, 1.07× at four and 0.936× at one; the parallel add would have
been another 0.335 s at eight and does not fix width 1 (44.96 → 44.65 s).
