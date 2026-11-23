# The 8B fit at ≤2 h: the measured split, and the one row that clears

Criterion: **Q3 campaign cost**. No part of Q3's PPL or 0-shot bar is advanced here. Grade:
`measured` for the layer split (the log exists, `bench/notes/2026-11-23-admm-split.md`), `anchor`
for the source findings (each names its line), `model` for every wall-clock projection — the L4 is
off, so no projection below is measured.

The result, in one sentence: **the ≤2 h gate is cleared only if the BF16 fused ADMM lands near
12 s/layer; without it the floor implied by the measured split is 5.3 h.** The 30-minute ideal is
not reachable even on Argon's own numbers (33.1 min), and the term that stops it is the per-layer
advance and tape residual, not the fit.

## What was checked against source

| claim | verdict | artifact |
|---|---|---|
| `LimbGemm::bind` retained a whole buffer set on every capacity growth | **true** | before the change, `bind` replaced `self.bound` with no release; `Bound` (`src/gpu/limb.rs:48`) has no `Drop`. The fix frees the six buffers; the descriptor set, pipeline, layout and per-buffer command pools still leak (`Vk::command_buffer` makes a pool per buffer, `src/gpu/vulkan.rs:2676`) |
| frozen slots are materialised on every step | **true** | `FrontTune::record` (`src/gpu/nn/train/front.rs:946`) and `BlockTune::record` (`src/gpu/nn/train/block.rs:802`) record `materialise` whenever `factor[k]`; `set_stage` sets `factor` from the persistent converted set (`front.rs:1113`), and `train` gates only Adam on the front (`:966`) and Adam plus `factor_bwd` on the block (`block.rs:827`). The front records `factor_bwd` for frozen slots anyway (`front.rs:961`) |
| `admm_budget_l23_down` licenses `I = 100` on one 7B tensor | **true, narrow** | `proof/corpus/Learn/Quant/NanoQuant.lean:356`: `blk.23.ffn_down`, r=768, seed 0, a 1024-step sketched tune at s=256. After the same 1024 steps the 100-iteration factor is **0.00118 worse** (0.02157 against 0.02039) in 1/4 the ADMM wall. The 8B seven-linear extrapolation is unmeasured |
| `kl_l23_down`: the selected gate code beats W2 in KL | **true, at 2.6% more bits** | 1.175 against 1.1451 b/w; 53.3× Q4_K_M against 78.8× (`NanoQuant.lean:173`). The constants come from logs outside the tree; the Lean file checks their arithmetic only |
| `factorize` discards the input Gram when `out < inn` | **false** | `src/learn/nanoquant.rs:627,696` runs on `wᵀ` with both statistics swapped. The only preconditioner argument is the diagonal; the dense `r×r` Gram is formed inside `x_update` (`syrk_q32`, `:574`) and inverted at `:594`, so there is nothing to discard. The branch condition matches the driver's `is_transpose` flag (`bench/gpu/campaign/ref-r1.py:77`); the reference's branch body is not in this tree |

The frozen-slot cost: `materialise` is `out·inn·r` multiply-adds per slot per step. At
`12288×4096`, `r = 3040` that is 1.53e11, against 1.03e11 for a 2048-column forward of the same
linear. One frozen slot's rematerialisation is larger than the forward it feeds, and the 7-slot
loop carries every slot already converted. The joint tune below removes the term; caching it only
masks it.

## The measured layer split (4,587.4 s, layer 0)

`bench/notes/2026-11-23-admm-split.md`, from `trace-round.log:40-189`:

| phase | wall s | share |
|---|---:|---:|
| ADMM, seven linears at `I=400` | 1,820.3 | 39.7% |
| `tune_nonfact`, seven rounds | 1,297.4 | 28.3% |
| `tune_fact`, seven rounds | 1,365.0 | 29.8% |
| teacher and advance | 21.0 | 0.5% |
| residual (tape dump, streams, setup) | 83.7 | 1.8% |
| **layer** | **4,587.4** | 36 layers = **45.87 h** |

The two tune phases are **58.0%**, the ADMM **39.7%**. Inside the ADMM, `inv` is 20.4% of the
layer wall (52–69% of each linear's loop); removing it entirely is a 1.256× layer ceiling, not the
1.49–1.9× a kernel estimate suggests. That ceiling is what bounds the exact carrier.

## Every projection, with its terms named

The gate is 2 h = 7,200 s = **200 s/layer**. Each row changes only the terms named; the rest are
the measured column. `ref` is the measured PyTorch ADMM rate (`ref-r1-gate.log`, 47.37 s at
`I=400` for gate_proj, scaled); `fused` is Argon's single-submit BF16 model at 62 TFLOP/s. Argon's
report is `/tmp/opencode/argon-quant/out.log`, outside this tree, so its constants are `model`
pointers, not artifacts.

| row | ADMM | tune | advance+residual | layer | 36 layers | vs gate |
|---|---:|---:|---:|---:|---:|---|
| measured, `I=400` | 1,820.3 | 2,662.4 | 104.7 | 4,587.4 | 45.87 h | — |
| `I=100` | 455.1 | 2,662.4 | 104.7 | 3,222.2 | 32.2 h | fails |
| `+` joint tune 64.6 | 455.1 | 64.6 | 104.7 | 624.4 | 6.24 h | fails |
| `+` 3-limb Cholesky, `inv` 14× | 368.9 | 64.6 | 104.7 | 538.2 | 5.38 h | fails |
| `+` BF16 at `ref`, `S_f=1024` | 34.8 | 64.6 | 104.7 | 204.1 | 2.04 h | **fails by 2%** |
| `+` BF16 at `ref`, `S_f=512` | 34.8 | 32.3 | 104.7 | 171.8 | 1.72 h | passes 14% |
| `+` BF16 `fused`, `S_f=1024` | 11.95 | 64.6 | 104.7 | 181.3 | 1.81 h | passes 9% |
| `+` BF16 `fused`, `S_f=512` | 11.95 | 32.3 | 104.7 | 149.0 | 1.49 h | passes 26% |
| `+` fused eval 10.95, `S_f=512` | 11.95 | 32.3 | 10.95 | 55.2 | 33.1 min | passes 2 h; misses 30 min |

Three corrections to the headline numbers:

- **"1.22 h" is not a row of its own.** It is `34.8 + 64.6 + 22.9`: the conservative ADMM with
  the *fused* base (22.9 = 11.95 + 10.95). With the measured advance it is 2.04 h and **fails**.
  A consistent conservative row needs `S_f=512` (1.72 h).
- **"33.1 min" is the total, but of a 55.2 s layer whose advance is 10.95 s** — a 9.6× cut of the
  measured 104.7 s, with no artifact. On the measured advance the same fused row is 1.49 h.
- **Tier "exact Q32.32 at 1.94 h" is impossible.** Argon's 79.6 s ADMM needs a further 4.55×
  below the `inv`→0 floor; the measured `inv` share caps the layer at 1.256×, so the floor implied
  by the measured split at `I=100` with the joint tune and the measured advance is
  **5.32 h** (`362.2 + 64.6 + 104.7`). The exact carrier cannot reach 2 h.

The 30-minute target is therefore not a fit-kernel problem. After `11.95 + 32.3 = 44.25 s` of
ADMM and tune, the budget for everything else is 5.75 s; the measured teacher-and-advance alone is
21.0 s. The 33.1-minute row needs the 104.7 s advance and residual cut to 10.95 s — **below the
measured teacher-and-advance term on its own** — and no lever in the note does that.

## Levers, by measured share of the layer

1. **Joint factored tune** (58.0%): factor all seven linears from the layer's cached activations,
   then one 1,024-step pass. The forward is `y = ((x ⊙ s_pre)·Vᵀ)·Uᵀ ⊙ s_post`, cost
   `T_step·r·(o+n)` instead of `T_step·o·n + 3·o·n·r`. Projected `2,662.4 → 64.6 s/layer` at
   `S_f=1024` (`32.3` at `S_f=512`). This removes the term that `formE_ceiling`
   (`NanoQuant.lean:87`) bounds at 2.33× for a step that keeps forming `E`; the theorem does
   not license the new schedule, and the upstream 30-minute claim has no artifact in this tree.
2. **`I = 100`** (licensed on one tensor): ADMM `1,820.3 → 455.1` s/layer. The block-tune test now
   reads `APEIRON_BT_ADMM_ITERS` (`tests/learn_block_tune.rs:134`).
3. **Device Cholesky + `trsm`** replaces `spd_inverse`: the inverse is `⅔r³` at `INV_LIMBS=6`;
   the solve is `⅓r³` plus `trsm`, 14× fewer inverse MMAs. Measured `inv` is 20.4% of the layer,
   so the layer saving is bounded by 1.256× — 5.38 h at `I=100`, not 1.94 h. The two `r×r`
   matrices are 148 MB in f64, 74 MB in FP32; the 8,388 MiB figure is Argon's six-limb inverse
   tree, not two matrices.
4. **BF16 coopmat in ADMM**: at the measured reference rate, 34.8 s/layer at `I=100` (2.04 h with
   the measured advance — a fail); at the fused 62 TFLOP/s model, 11.95 s/layer (1.81 h — a
   pass). The gap between those two rows is 2.9× and it is the whole gate. It is unmeasured.

The `bind` fix is necessary but not the layer-2 cause: `bench/gpu/campaign/campaign-per-layer.sh:3`
attributes that OOM to allocator growth across layers after a **20.1 GiB peak**, and fixes it with
one process per layer. The leaked sets are a real defect and are unquantified as a cause.

## Verification obligation per candidate

- **Cholesky for `spd_inverse`** does not keep the inverse bit-exact: the explicit inverse and an
  `L Lᵀ` solve round differently, and a limb Cholesky needs an integer square root the current
  path never takes. The obligation is an error bound on the substituted solve over the pack's `H`
  family, plus a device equality test where the arithmetic is exact (unit-diagonal triangular
  systems, power-of-two scalings). Aeneas cannot model the Vulkan dispatch; only an extracted
  host triangular solve could be `proved`.
- **BF16 coopmat** changes the carrier. No theorem licenses it as equivalent to Q32.32. The
  obligation is a forward error bound on the ADMM iterate plus a scored output (`relH`, then
  PPL). What cannot be proved is that the BF16 iterate reaches the same fixed point.
- **The fold closed form** is refused as an ETA. `src/learn/fold/fit/` holds the block quantisers,
  `eigh` and the corrector, not the Hessian-ordered error-feedback sweep Argon cites; the only
  measured attempt at that mechanism is a synthetic prototype whose Gram-energy ratio moved
  0.903 → 0.968 (`bench/notes/fit-cost.md:279–290`, 192×256, rank 128, higher is worse). The
  `c6x29-sel` KL anchor is real but comes from logs outside the tree.

## The measurement order when a host exists

```sh
# 1. One set live after growth, not two (skips without cooperative matrices).
cargo test --release --locked --features gpu --lib gpu::limb::tests::rebind_releases_the_old_capacity

# 2. I=100 on one layer, same kernels. --ignored: block_loop is #[ignore].
APEIRON_NQ_LOOP=admm100 APEIRON_NQ_BLOCKS=0..1 APEIRON_BT_ADMM_ITERS=100 \
  cargo test --release --locked --features gpu --test learn_block_tune -- --ignored --nocapture

# 3. The joint tune, one layer, all 7 slots: the 58.0% term.
#    Gate A: layer <= 200 s before any ≤2 h row is quoted.
#    Gate B: layer <= 55.2 s before any 33-minute row is quoted.
```

Until step 2 and step 3 print, the only measured rows are the first one and the phase split. Every
other number above is `model`, and the gate is decided by a 2.9× kernel factor that has never run
on the L4.
