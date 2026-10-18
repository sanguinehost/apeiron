Row: `unified-carrier-level-as-Bi4NM1lt3LKJy0qsUykdBZ` (UNIFIED-CARRIER W0).

# Unified carrier: per-carrier site inventory (UC-W0)

Source revision inventoried: `f544f26` (worktree `/home/socol/oc-tmp/wt-w0`).
Commands are `rg`/`wc`/`sed`; no `find`/`grep` (shadowed on this box, `AGENTS.md` §4).

## 1. Carrier enums

| enum | path:line | variants |
|---|---|---|
| weight carrier | `src/gpu/simt/mod.rs:507` | `F32, F16, Q8, Q4, Q2, Q3` (6) |
| CPU numeric dial | `src/engine/mixer.rs:299` | `Q32, F32` (2) — a *different* enum, private to `src/engine/`, not the simt carrier |

The task names the simt enum. The engine enum is inventoried separately because its arms are a
2-way fixed-vs-float dial (`swiglu`, `conv`, `AffineHead`, `KvStorage`), a different parameter.

## 2. Files referencing the simt carrier

23 files in the simt/gpu tree contain a literal carrier variant. Tree totals (`src/gpu/simt`,
`src/gpu/nn`, `src/gpu`, `tests/gpu_*`, `tests/simt_*`, `tests/spirv_*`): **38 files,
24 664 physical lines**. Six further files reference `carrier` as a value only
(`src/gpu/nn/{coop,glue,prefill_glue,norm,spec,attn_dp4a}.rs`).

Per-file `Carrier::` hit count (simt files):

| file | hits | lines |
|---|---|---|
| `src/gpu/nn/spec.rs` | 37 | 1056 |
| `src/gpu/simt/mod.rs` | 27 | 984 |
| `src/gpu/simt/superblock.rs` | 8 | 2123 |
| `src/gpu/simt/gemv.rs` | 8 | 1527 |
| `src/gpu/decode/state.rs` | 5 | 323 |
| `src/gpu/decode/pack.rs` | 4 | 239 |
| `src/gpu/simt/dp4a.rs` | 3 | 1687 |
| `src/gpu/nn/attn/attn_dp4a.rs` | 3 | 510 |
| `src/gpu/decode/layer.rs` | 3 | 513 |
| `tests/spirv_storage_race.rs` | 10 | 534 |
| `tests/gpu_decode_step.rs` | 3 | 169 |
| `tests/gpu_coopmat_gemm.rs` | 3 | 343 |
| `tests/gpu_prefill_match.rs` | 2 | 315 |
| `tests/gpu_gemv_match.rs` | 2 | 239 |
| `tests/simt_spirv_val.rs` | 2 | 122 |
| `src/gpu/nn/coopmat_gemm.rs` | 1 | 143 |
| `src/gpu/nn/norm.rs` | 1 | 726 |
| `src/gpu/decode/prefill.rs` | 1 | 723 |
| `src/gpu/decode.rs` | 1 | 456 |
| `src/gpu/mod.rs` | 1 | 42 |
| `src/gpu/nn/tests.rs` | 1 | 117 |
| `tests/gpu_spv_check.rs` | 1 | 778 |
| `tests/gpu_layer_match.rs` | 1 | 200 |

`src/gpu/nn/{coop,glue,prefill_glue}.rs` also take `carrier: crate::simt::Carrier` as a parameter
type (no literal arm).

`match`/branch sites switching on the simt enum — **only one literal `match`**:

| site | kind | arms |
|---|---|---|
| `src/gpu/simt/gemv.rs:460` | `match carrier` | `F32` `{}`, `F16` (UnpackHalf2x16), `Q3` (panic), `Q8\|Q4\|Q2` (shift/mask) — 35 lines (460–494) |
| `src/gpu/nn/attn/attn_dp4a.rs:235` | `carrier == Q3` bool | 2-way |
| `src/gpu/nn/norm.rs:171` | `carrier == F32` | 2-way |
| `src/gpu/nn/spec.rs:528,559` | `Q3`/`Q2` equality | refusal rules |
| `src/gpu/nn/coopmat_gemm.rs:107`, `src/gpu/decode/prefill.rs:87`, `src/gpu/simt/superblock.rs:896` | `matches!(carrier, Q8\|Q4\|Q2)` | refusal / run selector |

No `.comp` file references the Rust enum; the SPIR-V carrier is a `layout(constant_id=2) PACK`.

## 3. Groups

| group | files | carriers with a distinct body | physical lines / carrier body | parameterisable by `b`? |
|---|---|---|---|---|
| **GEMV (GPU DP4A)** | `src/gpu/simt/dp4a.rs` (`gemv_dp4a_module_moe_sc` 132–903; `gemv_dp4a_batch_module` 960–1413), `src/gpu/nn/attn/attn_dp4a.rs` (`dp4a_chart_fold` 212–465), `src/gpu/simt/superblock.rs` (`gemv_dp4a_sb_module_ex` 850–1789) | none — one body, `pack()`-derived | 771 + 453 + 253 + 940 | **already parameterised** — `codes_per_word/dots_per_word/cpb/wbits/byte_mask/split` from `carrier.pack()` at `dp4a.rs:166–171` and `964–969`, `attn_dp4a.rs:241–246`; superblock `pack/width/planes/wpu` at `superblock.rs:900–903`. The same 6-line derivation is **copied 4×** |
| **GEMM / prefill (GPU)** | `src/gpu/spirv/coopmat_gemm.comp` (279), `src/gpu/spirv/int8_coopmat_q8.comp` (210), `src/gpu/spirv/q8_tile.comp` (152), `src/gpu/nn/coopmat_gemm.rs` (143), `src/gpu/decode/prefill.rs` | `int8_coopmat_q8.comp`, `q8_tile.comp` are Q8-only copies | 210 + 152 | `coopmat_gemm.comp` **already parameterised** (`SPEC_PACK` const, `:66–73`); the two Q8 sources are **copy per carrier** (same ABI/geometry, Q8 fixed). `q8_tile` (scalar `dotPacked4x8`) vs `int8_coopmat_q8` (int8 coopmat MMA) differ by **instruction path** |
| **CPU chart projection** | `src/engine/chart/mod.rs` (`q8act_q8_0/q5_0/q4_k/q6_k_row_fast` 1056–1079; `FastBlockDecoder` impls 519/596/684/810), `chart/x86.rs`, `chart/portable.rs` | 4 decoder impls (Q8_0 74, Q5_0 81, Q4_K 119, Q6_K 93 lines) | ~367 + 22 dispatch | **already parameterised** by the `FastBlockDecoder` trait (W2c record, epic line 145). Decoders differ by GGML block layout, not by one `b` |
| **weight pack/unpack** | `src/gpu/decode/pack.rs` (`pack_proj_f32` 96–145 **parameterised**; `pack_q4_k_raw` 148–236 hardcodes `pack=8, width=4`), `src/gpu/simt/superblock.rs` (`q2sb_gpu_chart` 1812–1832 (21), `q4_sub8_gpu_chart` 1853–1884 (32), `q3_planes_gpu_chart` 1998–2020 (23)), `src/format/quant.rs`, `src/format/packing.rs` | `pack_q4_k_raw` **copy per carrier** (Q4); `q2sb`/`q4_sub8`/`q3_planes` are three layout bodies, `q2sb` and `q3`-plane-A byte-identical | 89 + 76 | `pack_proj_f32` parameterisable; the three packers **copy per carrier**; `format/quant.rs` per-format `q*_codes`/`dequant` (~380) genuinely distinct GGML layouts |
| **attention weight reads** | `src/gpu/nn/attn/attn_dp4a.rs` | none | 0 | **already parameterised** (`dp4a_chart_fold`) |
| **KV / embedding reads** | `src/gpu/decode/{state,layer}.rs` | none (pass-through; `head_carrier()` maps Q4→Q8 at `state.rs:54`) | 0 | carrier value threaded, no per-carrier body |
| **dequant helpers** | `src/format/quant.rs` (897), `src/engine/num/dequant_lowbit.rs` (914) | one per GGML/lowbit format | ~380 + ~350 | **genuinely distinct** — GGML block layouts (two-bit planes, 5/6-bit high planes), not one `b` |
| **tests** | `tests/gpu_prefill_match.rs` (Q4/Q8 only), `tests/gpu_coopmat_gemm.rs` (`[Q8,Q4,Q2]` at :246), `tests/gpu_decode_step.rs` (Q8/Q4), `tests/simt_spirv_val.rs` (`[F32,F16,Q8,Q4]`, `[Q8,Q4,Q2]`), `tests/gpu_gemv_match.rs` (Q4) | prefill arms Q4/Q8 only | — | arms parameterised by carrier; Q2 already instantiable |

## 4. Ranking — (lines removable) × (hot-path weight), hardest first

| rank | group | removable lines | hot-path weight | verdict |
|---|---|---|---|---|
| 1 | GEMM/prefill SPIR-V (`q8_tile.comp` 152 + `int8_coopmat_q8.comp` 210) | ~362 | prefill (per-prompt, the measured `pp512` number) | **unify first**; both are Q8-only instances of the parameterised `coopmat_gemm.comp`. `q8_tile`↔`int8_coopmat_q8` differ by instruction path; each differs from `coopmat_gemm.comp` by the `PACK` const only |
| 2 | GEMV DP4A derivation copies | ~18 (3× the 6-line block) | decode (per-token, the served path) | low line count, but the 4× copy of `wbits/mask/cpb` is the exact `b`-parameter the epic proves; move into `Carrier` |
| 3 | weight pack/unpack (`pack_q4_k_raw`; `q2sb`/`q4_sub8`/`q3_planes`) | ~89 + ~76 | one-off at load | `pack_q4_k_raw` collapses into `pack_proj_f32`; `q2sb`/`q3` plane-A share the expression |
| 4 | `src/gpu/simt/gemv.rs:460` direct/chart match | ~35 | decode, non-served (DP4A is the served path) | F32/F16 vs chart: **genuinely distinct** instruction path (FMA vs `OpSUDot`) |
| 5 | CPU chart face | 0 | CPU prefill | **already unified** (W2c) |
| 6 | dequant helpers / GGML formats | 0 | load | **genuinely distinct**; not one `b` |

**Unify first: the prefill GEMM SPIR-V.** File set (3): `src/gpu/spirv/coopmat_gemm.comp`,
`src/gpu/decode/prefill.rs`, `tests/gpu_prefill_match.rs`. `int8_coopmat_q8.comp` and
`q8_tile.comp` are registered dispatch contracts (`tools/dispatch/contracts.json`, Lean
`proof/dispatch/contracts/{Int8CoopmatSpirv,Q8TileSpirv}.lean`) with **no Rust consumer**
(`rg 'include_bytes' src` → only `coopmat_gemm.spv`); deleting them is a `retire_check` +
contract-retirement question, not a file edit.

## 5. Totals

- simt-carrier tree: **38 files, 24 664 physical lines**; 28 reference the enum.
- Identifiable lines inside carrier-specific arms/bodies: **≈1 755**
  (engine chart decoders ~367; `format/quant.rs` ~380; `dequant_lowbit.rs` ~350;
  per-carrier SPIR-V sources 362; `pack_q4_k_raw` 89; superblock packers 76;
  `gemv.rs` match 35; `impl Carrier` parameter table 96).
- The canonical parameter table (`src/gpu/simt/mod.rs:543–638`, 96 lines) is the single source the
  copies derive from; it is the `b`-indexed surface the epic extends, not a copy.

## 6. What could not be determined

- Whether `int8_coopmat_q8.comp` / `q8_tile.comp` are dead (retainable baselines) or required by a
  live `admit.py` contract: they have no `include_bytes!` and no `glslc` invocation in `src`, but a
  populated `tools/dispatch/contracts.json` entry and a CUDA comparison gate. Retirement is a
  `retire_check` decision, not decidable from source alone.
- Whether the `q4_sub8` two-level scale face and `dm` single-level face are one `b` with a second
  scale level or a second dial (epic open question line 86); the inventory shows the scale layout
  is a **separate** body from the code layout.
- Per-arm exact line counts for `attn_dp4a::dp4a_chart_fold` and `norm::gemv_norm_module_sc`: the
  bodies are parameterised, not per-carrier, so no arm boundary exists to count.

## Reproduce

```sh
rg -l 'Carrier::(F32|F16|Q8|Q4|Q2|Q3)' src/gpu/simt src/gpu/nn src/gpu tests --glob '*.rs' | wc -l   # 23
rg -c  'Carrier::(F32|F16|Q8|Q4|Q2|Q3)' src/gpu/simt src/gpu/nn src/gpu tests --glob '*.rs'
rg -n 'match (carrier|self\.carrier)' src --glob '*.rs'   # 1 simt at src/gpu/simt/gemv.rs:460; 5 engine
rg -n 'include_bytes!' src --glob '*.rs'                  # only coopmat_gemm.spv
```
