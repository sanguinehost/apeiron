# The NQ proof ladder, bottom up

Status: an audit of the committed extraction and bridge at HEAD `735a7b06` plus the working tree.
The pinned Aeneas/Charon/Lean toolchain is absent, so no row below is a fresh `regen.sh --check`;
"extracted" is read from the committed `proof/ApeironWord.lean`, and "proved" names the theorem and
its `#audit`. The extraction is **stale** (§1), so the ladder is an ordering of artifacts, not a
chain that closes today. The ternary cell and its binary fallback are design only: their source,
test and theorem do not exist, and their rows are `model`.

Criteria: W4 (`PLAN.md:79`); D5 (`PLAN.md:160-161`) for the twin rule.

## 1 · Toolchain, drift, and what was not run

| item | state |
|---|---|
| `target/aeneas/` | absent |
| missing executables | `target/aeneas/aeneas`, `target/aeneas/charon`, `target/aeneas/charon-driver` |
| missing model | `target/aeneas/backends/lean` |
| `proof/regen.sh --check` | not run; it exits 2 on the first missing tool, `target/aeneas/charon` (`proof/regen.sh:65-68`); the translator `target/aeneas/aeneas` is also missing |
| bootstrap | not run, per the task |

**Extraction drift: yes.** Two independent artifacts establish it.

1. The committed generated file was written at `7292279f` (2026-10-20,
   `git log -1 -- proof/ApeironWord.lean`). It defines
   `kern.limb.MAX_INNER := 512 * quant.Q8_0_QK` (`proof/ApeironWord.lean:24932`), while the source
   at HEAD `b6cf11d5` (2026-11-16) defines `MAX_INNER = 1024 * Q8_0_QK`
   (`src/engine/kern/limb.rs:86`). The generated `Source:` span says `lines 82:0-82:39`; the source
   is at line 86. `src/engine/kern/gemm.rs` (2026-10-22) and `src/engine/kern/matmat.rs`
   (2026-11-02) also postdate the generated file. `src/cell.rs` (2026-10-14) does not.
2. `proof/README.md:563-568` records both `--check` runs at `6c724104` exiting 1 with 415 diff
   hunks, from the `limb.rs` and `gemm.rs` changes. HEAD is later, and the working tree also
   modifies `limb.rs`, `nq.rs` and `learn/nanoquant.rs`.

The `Cell<i64>` rung is unaffected (its source predates the generated file), but the freshness
artifact as a whole is red, and the limb/q8 rungs below are the ones that moved.

### Bootstrap cost (measured, not run here)

| phase | command | time | memory / disk |
|---|---|---|---|
| tools | `proof/bootstrap.sh` | 49 s on the recorded 32-thread host, aeneas `dune build` ≈42 s (`proof/README.md:719-727`) | opam switch from `proof/opam-switch.export` at `/tmp/apeiron-opam`, Rust `nightly-2026-08-18` with `rustc-dev`+`miri`, binutils `strip`; source checkouts and model |
| extraction check | `proof/regen.sh --check` | 505.4–537.2 s; Aeneas phase 510 s | peak RSS 13.7 GB; killed at 580 s under a 12 GB cgroup (`proof/README.md:546-561`) |
| bridge | `cd proof/corpus && lake build` | cold 9302 jobs, exceeds one 580 s window; Mathlib cold >1 h (`proof/README.md:388-390,742`) | Lean build products several GB (`AGENTS.md`, Gates) |

This host has 6.7 GiB RAM (1.7 GiB available) and 24 GB free on `/`, with 2 visible cores. The
13.7 GB Aeneas phase cannot run here at all; the task's 2 GiB cap is not the binding limit. The
ladder's freshness and any `lake build` run on the larger box:

```sh
proof/bootstrap.sh
proof/regen.sh --check
cd proof/corpus && lake build
```

## 2 · The ladder

| # | rung | source / generated | extracted | theorem artifact | grade | leaves open |
|---|---|---|---|---|---|---|
| 1 | `Cell<i64>` mul, norm | `src/cell.rs:127-163`; `ApeironWord.lean:1026,1041` | yes | `proof/corpus/WordRefinement.lean:45` `word_mul_denote`, `:56` `word_norm_denote`, `:110` `word_norm_mul`, `:146` `word_mul_conj`; audit `proof/corpus/AxiomAudit/Core.lean:202-203` | **proved**, standard cone | nothing in the ring algebra |
| 2 | limb placement / product | `src/engine/kern/limb.rs`; `ApeironWord.lean:28116` `Layout.placement`, `:28651` `split_digits`; roots `proof/extract/ports/limb_port.rs`, `proof/regen.sh:702-705` | yes | `proof/corpus/Learn/Limb.lean:581` `abs_limbPre_sub_exactPre_le` is an analytic bound over a real model, not a body refinement; no `word_limb_*` theorem exists | **proved** (analytic) / **model** as refinement | `word_limb_{gemm,syrk,gemv,rowdot}_denote`; digit-bound induction; accumulator-exactness premise; freshness |
| 3 | `matmat_q8act` control flow | `src/engine/kern/matmat.rs:165`; `ApeironWord.lean:26403` and following loops | control flow yes; cores no | equality tests `sign_dot` `src/engine/kern/nq.rs:1436`, q8 `:1539`; no refinement theorem | **tested** (cores) / none (caller) | root the safe twins; theorem; AVX2 cores stay opaque |
| 4 | q8 reader in `nq.rs` | `src/engine/kern/nq.rs:421` `code_select_twin`, `:137` `sign_dot_i64`; `kern.nq` absent from the generated file (0 occurrences) | **no** | none; `proof/README.md:582-591` records the debt | none | a `#[path]` root; `word_code_select_denote` |
| 5 | NQ factorization: `fro2`, `spd_inverse`, `export` | `src/learn/nanoquant.rs:291,401,762` | **no**; no `#[path]`, no root, no `nanoquant` in `proof/extract/` | `fro2` test `src/learn/nanoquant/tests.rs:27` (uncommitted); `spd_inverse` tests `:46,156,195`; `export` none; `proof/corpus/Learn/Quant/NanoQuant.lean:9-12` says "No clause here is linked to an extracted body … W4" | **model** (`fro2`, `export`) / **tested** (`spd_inverse` names) | extraction boundary, spec, roots |
| 6 | ternary cell | no source | **no** | proposed `word_ternary_dot_denote` (§7) | **model** | twin, test, root, theorem, audit |
| 6b | binary activation cell: `popcount(w ⊕ a)`, one popcount per 64 terms; p1 bound 62.9 tok/s, front-end ~40+; the rate floor if ternary fails Q2 and binary activation quality passes | no source | **no** | none; the `loop_words` binary×binary oracle is prior art only (§7) | **model** | spec, twin, boundary equality test, quality verdict; Q2 **not** claimed |
| 7 | scale epilogue | `src/engine/num/value.rs:9`; `ApeironWord.lean:5952` `round_shift_32`, `:44425` `mul_q32` | yes | `proof/corpus/Serve/Engine/Step.lean:349` `word_mul_q32_ok`; audit `proof/corpus/AxiomAudit/Serve.lean:153` | **proved** for `mul_q32` | the NQ row's composed epilogue (`low + corr + res`, then `i64::try_from`) has no theorem |

Read the table bottom up: the first rung with no theorem is 2 (limb), and the first rung with no
extracted body is 4 (the q8 reader in `nq.rs`). Rungs 6 and 6b are not rungs yet: neither has a
source.

## 3 · `Cell<i64>`: extracted and proved

`src/cell.rs` `Cell::mul` and `Cell::norm_form` are in the extraction unit by
`#[path = "../../src/cell.rs"]` (`proof/extract/lib.rs:16-17`) and rooted at
`word_mul`/`word_norm` (`proof/extract/lib.rs:188-195`). The generated bodies are
`cell.Cell.mul` (`proof/ApeironWord.lean:1041`) and `cell.Cell.norm_form` (`:1026`); both are the
real generic bodies instantiated at the `i64` ring, not restatements.

The bridge is `proof/corpus/WordRefinement.lean` over `ZMod (2^64)`:

```lean
theorem word_mul_denote (a : cell.Cell I64) (s : cell.Sigma) (b : cell.Cell I64) :
    (do let c ← word_mul a s b; ok (denoteCell c)) =
      ok (Apeiron.Trichotomy.AnyRing.ringMul (denoteSigma s) (denoteCell a) (denoteCell b))
```

`word_norm_denote` is the same shape against `ringNormForm`; `word_norm_mul` and `word_mul_conj`
transfer the canonical laws. The audit is `#audit word_mul_denote` and `#audit word_norm_denote`
at `proof/corpus/AxiomAudit/Core.lean:202-203`; the audit macro throws on any axiom outside
`{propext, Classical.choice, Quot.sound}` (`Core.lean:190-196`). This is the only rung where the
extracted body, the theorem and the audit all exist.

## 4 · limb placement and product: extracted, unproved

The four products are extracted and rooted (`proof/regen.sh:702-705`):

- `word_limb_gemm_q32` → `gemm_q32` (`src/engine/kern/limb.rs:160`);
- `word_limb_syrk_q32` → `syrk_q32` (`:198`);
- `word_limb_gemv_q32` → `gemv_q32`;
- `word_limb_rowdot_q32` → `rowdot_q32` (`:533`).

Their call closure translates the real `Layout::placement` (`ApeironWord.lean:28116`), the serial
`product_rows`, `split_digits` (`:28651`) and the transpose. The AVX2 digit split is the one
admitted opaque item: `kern.limb.simd.split_block_avx2` is an `axiom` (`:28575`), its scalar twin
is the extracted `split_digits` loop in `split_block`, and `proof/boundary-axioms.txt` admits it
under the `simd-twin` class with a `tested` grade.

What is missing is the refinement, not the extraction:

- `proof/corpus/Learn/Limb.lean` proves the analytic bound `G` on one block
  (`abs_limbPre_sub_exactPre_le`, line 581; specialized `limb_q_error_down`,
  `limb_d_error_down`). Its header states the two premises that are **not** theorems: the digit
  and residual bounds of `split_digits` (`hα`/`hβ`/`hρ`) and that the `i64`/`i128` accumulator is
  exactly `T` (`hT`); the AVX-VNNI core is `tested` against `exact_dot`, not extracted. It does not
  claim a refinement of the AVX-VNNI body or the `i64` non-overflow side condition.
- No `word_limb_*_denote` theorem exists in `proof/corpus`, and no extracted-body `limb` headline
  is audited. (`Learn.Limb`'s headlines are the analytic bound over the model, not a refinement;
  `grep -rn 'word_limb' proof --include=*.lean` returns only the generated `kern.limb` bodies.)

The ordering consequence: a `gemm_q32` refinement would compose the limb placement with the q8
core, whose equality is `tested` (`proof/README.md:589-591`). The theorem cannot be stronger than
that premise.

## 5 · `matmat_q8act` and the q8 reader

`matmat_q8act` (`src/engine/kern/matmat.rs:165`) **is** extracted as control flow: the generated
file has `kern.matmat.matmat_q8act` and its loops (`ApeironWord.lean:26403`), `row_accs`,
`pooled_rows` and `matvec_q8act` (the oracle). The portable row bodies are extracted too
(`kern.matmat.portable_u8_i8_dot`, `portable_q5_0_dot`). The fast path's arithmetic is not: the
AVX2 row cores (`crate::chart::q8_0_row_kcols_avx2`,
`crate::kern::matmat::q5_0_block_kdots_avx2`, `crate::chart::x86`,
`crate::kern::gemm::x86`) are opaque, so the generated branch that feeds `pooled` selects a body
whose arithmetic is `tested`, not `proved` (`proof/regen.sh:251-293`, `proof/README.md:582-588`).

`src/engine/kern/nq.rs` is behind `#[path = "../../src/engine/kern/mod.rs"]`
(`proof/extract/lib.rs:1127-1128`), and `src/engine/kern/mod.rs:12` declares `pub(crate) mod nq;`.
The module is therefore textually in the translation unit, but **no `--start-from` root reaches
it**: `kern.nq` occurs 0 times in `proof/ApeironWord.lean`, as do `code_select_twin`,
`sign_dot_i64` and `sign_dot_avx2`. The extent is: included by `#[path]`, not translated.

The missing twin is `code_select_twin` (`src/engine/kern/nq.rs:421`), the safe fold of the q8
selection `Σ_b S_b·Σ_{bit=1} codes[j]`; its intrinsic counterpart is `code_select_avx2`
(`:596`). The second is `sign_dot_i64` (`:137`), the safe twin of `sign_dot_avx2` (`:183`). Both
twins are private fns and neither is rooted. What exists is the pair of equality tests:

- `the_vector_source_equals_the_scalar_source` (`src/engine/kern/nq.rs:1436`) compares
  `sign_dot_avx2` with the `sign_dot!` macro (not with `sign_dot_i64` as a separate body);
- `the_code_source_equals_the_scalar_source` (`:1539`) compares `code_select_avx2` with both
  `code_select_twin` and the dequantized `sign_dot!`, at lengths
  `{0,1,7,31,32,33,63,64,65,127,128,129,1000,1023}` and exponents `{0,1,12,40}`.

Grade `tested` for each equality; no `proved` claim. `proof/README.md:589-591` states the debt
exactly: `sign_dot_i64` is a safe twin but is not a root of `proof/extract/lib.rs`, so no
refinement theorem covers it; the `Q4_K` lane has no safe twin at all, only the oracle equality.

## 6 · NQ factorization (`fro2`, `spd_inverse`, `export`)

**Not extracted.** `src/learn/nanoquant.rs` is not named in `proof/regen.sh`'s source list and is
not pulled by `#[path]` in `proof/extract/`; `grep -rn nanoquant proof/extract` is empty. There is
no `word_nq_*` root and no generated `nanoquant` definition. The canonical file
`proof/corpus/Learn/Quant/NanoQuant.lean:9-12` says so: "No clause here is linked to an extracted
body … (the Aeneas link is open, W4)".

### `fro2`

`fro2` (`src/learn/nanoquant.rs:291`) is the `f64` Frobenius norm `Σ m_i²`. The rescale is the
working-tree change:

```rust
let (mut scale, mut unscale) = (1.0f64, 1.0f64);
while row * scale * scale > 4.611_686_018_427e18 { scale *= 0.5; unscale *= 4.0; }
let q = grid_map(m.len(), &|c| m[c] * scale)?;
let d = rowdot_q32(&q, &q, rows, cols, LIMBS)?;
Ok(d.iter().map(|&w| from_grid(w)).sum::<f64>() * unscale)
```

Exactness statement. `scale = 2^{-k}` and `unscale = 2^{2k} = scale^{-2}`; both are powers of two
and exactly representable in `f64`, and multiplication by a power of two changes only the
exponent, so `scale`/`unscale` introduce no arithmetic error. The identity is

```text
fro2(m) = 2^{2k} · Σ_i from_grid(to_grid(m_i · 2^{-k}))²
```

`to_grid` rounds `x·2^{32}` to the nearest integer (`src/learn/solve/q32.rs:61-67`), so the one
rounding left is the per-entry Q32.32 grid at the scaled exponent; `unscale` is exact. The rescale
changes **where the grid rounds**, not the stored factor: `export` (`:762`) uses `fro2` only for
the balance factors `na`, `nb` (`bf = (nb/na)^{1/2}`) and the `relH` denominator, while the
signs/scales come from `a`, `b` after the same divisions. Without the rescale, a row of `a` on 7B
`ffn_down` is 5.6×`i64::MAX` in Q32.32 and `rowdot_q32` refuses `Range` before `export` runs.
The test `fro2_carries_a_row_past_the_word_range` (`src/learn/nanoquant/tests.rs:27`) asserts
`|got − want| ≤ 1e-6·want` on a 3×3424 matrix; it is **uncommitted**, and `PLAN.md:75` records the
test as queued. Grade `model`, not `tested`, until that test has a run receipt.

### `spd_inverse` and `export`

`spd_inverse` (`:401`) is the block recursion over `i64` words: `inv_factor_upper` at the base,
then `gemm_q32`/`syrk_q32` products, `btx[c] >> 1` symmetrisation and checked assembly. It calls
the extracted limb products, but the recursion, the base inverse and the `AtomicBool`/pool
boundary are outside the extraction. Tests: `spd_inverse_inverts` (`tests.rs:46`),
`spd_inverse_matches_serial` (`:156`), `spd_inverse_inline_matches_pooled` (`:195`); these are the
artifact that would make the claim `tested`.

`export` (`:762`) divides out the statistics, balances `A`/`B`, applies the column scale, takes
`s_pre`/`s_post` as mean magnitudes and `U`/`V` as signs (zero → `+1`), and emits `NqFactor`. It
uses `fro2` twice and sums the column norms in `f64`; it has no unit test of its own. Grade `model`.

The whole row is `model` as a refinement: the quality verdicts in `Learn.Quant.NanoQuant` are
`model`/`measured` inputs, not extracted-body claims.

## 7 · The ternary cell: the proposed rung

There is no ternary cell in `src/` yet: `grep -rn 'ternary_dot' src` finds only the GPU
`TernaryDP4A` vocabulary, and `bench/notes/trichotomy-cell.md` does not exist. Prior art for a
bitmask dot is the NQ test oracle `loop_words` (`tests/learn_nanoquant.rs:476`): it packs two ±1
sign rows into `u64` words and computes `r − 2·popcount(a ⊕ b)` per entry (`:528`). It is
binary×binary, not ternary: no zero trit and no per-block `p`/`n` counts. It is a test helper
that fills a `.param` file for a comparison harness, not a production path. Its arithmetic is
checked against the scalar sign sum by `loop_words_matches_the_factor` (`:546`) at one length,
`r = 70`, on a 5×7 factor: two packed words with 6 padding bits, and one latent zero pins
`sign(0) = +1`. It is not checked at 0, 1, 63, 64, 65, 127 or 128, so it is not the boundary
evidence the ternary twin needs.

### The safe twin and the identity

The twin, per `PLAN.md:85-92`, is safe Rust first:

```rust
fn ternary_dot(w: u64, p: u64, n: u64, len: usize) -> i128
```

with `w` the 64-bit weight sign word, `p`/`n` the positive/negative activation masks, `p & n = 0`,
and bits `≥ len` padding. The identity, under the `sign_dot!` convention (`s_i = 1 − 2·bit_i(w)`,
bit 1 is `−1`; `src/engine/kern/nq.rs:25-29,108-127`), is

```text
Σ_{i<len} s_i·t_i  =  2·popcount(w & n) − 2·popcount(w & p)
                      + popcount(p) − popcount(n)
```

with `t_i = +1` on `p`, `−1` on `n`, `0` elsewhere. `PLAN.md:37-39` states this same identity
under the same convention, and records the check: exhaustive over all 3- and 4-term cells and
200,000 random 64-term cells. The convention that must not drift is the crate's `U`/`V` sign,
`sign(0) = +1`, i.e. `s_i = 2·bit_i(w) − 1`, which negates this identity. The Lean statement and
the equality test must pick one convention; the test should compare against `sign_dot!` on the
same word, which fixes it.

### The Lean theorem (proposed, `model`)

The popcount identity is not in Aeneas's std model, so the extracted body is the bit-recursive
twin; `u64::count_ones` (or the AVX2 `popcnt` cell) is the intrinsic fast path with a `tested`
equality, the same shape as `sign_dot_i64` / `sign_dot_avx2` (`PLAN.md:88-92`). The theorem over
the twin, in `proof/corpus/WordRefinement.lean` (or a new `TernaryRefinement.lean`):

```lean
def ternaryDot (w p n len : Nat) : ℤ :=
  ∑ i in Finset.range len,
    (1 - 2 * ((w >>> i) &&& 1)) * (((p >>> i) &&& 1) - ((n >>> i) &&& 1))

theorem word_ternary_dot_denote (w p n : Std.U64) (len : Std.Usize)
    (hlen : len.val ≤ 64) (hdisj : p &&& n = 0#u64) :
    (do let r ← word_ternary_dot w p n len; ok r) =
      ok (ternaryDot w.val p.val n.val len.val)

theorem ternary_popcount_identity (w p n : Nat) (h : p &&& n = 0) :
    ternaryDot w p n 64 =
      2 * (w &&& n).popcount - 2 * (w &&& p).popcount
        + p.popcount - n.popcount
```

The second theorem is the arithmetic law the fast path relies on; it needs no Aeneas body. The
first is the refinement of the extracted twin. Audit expectation: both under
`{propext, Classical.choice, Quot.sound}` and no boundary axiom (the cell touches no `f32`, `f64`
or pool entry); `ternary_popcount_identity` should need only `propext` in fact, but the closed
cone it is audited against is the same three.

### Refusal bounds

1. `len ≤ 64` per word; a longer row iterates words. Bits `≥ len` are padding and ignored, the
   `sign_dot!` contract.
2. `p & n = 0`. The quantizer must establish it; the twin should take it as a checked
   precondition, not assume it.
3. The dot itself is in `[−64, 64]`, so it never overflows `i32`/`i64`/`i128`. There is no refusal
   inside the 64-term cell.
4. Scale epilogue: the block sum is `Σ_b 2^{e_b}·d_b` with `d_b ∈ [−64, 64]` and `e_b` the
   per-block power-of-two activation scale. Admit it only when `64·Σ_b 2^{e_b} ≤ i64::MAX` — the
   `4064` bound of `CodeTerms::new` (`src/engine/kern/nq.rs:510-515`) with `64` in place of `4064`
   — else keep the exact reader. The final `mul_q32` / `i64::try_from` refuses `Overflow`
   (`NqError::Overflow`), as `project_with` does.
5. The weight word is the pack's sign word; the cell must use the same bit convention as the
   `U`/`V` plane it replaces, or negate once, as `project_rows` does for the residual
   (`src/engine/kern/nq.rs:1026-1029`).

### The binary activation fallback (rung 6b)

`popcount(w ⊕ a)`, with `w` the weight sign word and `a` the flat ±1 activation word (both
`1 − 2·bit`), is the same cell with one mask instead of two:
`Σ_{i<len} s_i·t_i = len − 2·popcount(w ⊕ a)` for `len ≤ 64` with zero padding, one `popcnt` per
64 terms. The port bound is 62.9 tok/s at the `PLAN.md:44-45` disclosure — that figure is quoted
for two popcounts per 64 terms, so it is a floor for one — and the shorter loop puts the front end
near 40+ tok/s (estimate, not measured). It is the rate floor if the ternary activation fails Q2
and a binary activation passes it; it does not claim Q2, and no binary-activation quality
measurement exists in the tree. The obligations have the same shape as the ternary cell: a safe
twin, a boundary equality test over the lengths in §8, and a `word_binary_dot_denote` theorem;
`len ≤ 64` with zero padding replaces the `p & n = 0` precondition.

## 8 · Gap list

| gap | current grade | missing artifact | command that produces it |
|---|---|---|---|
| limb body refinement | **proved** analytic / **model** as refinement | `word_limb_gemm_denote` (and syrk/gemv/rowdot) plus the `split_digits` digit induction and the accumulator-exactness premise that `Learn/Limb.lean` names | root it, `proof/regen.sh --check`, then `cd proof/corpus && lake build` |
| q8 selection refinement | **tested** (equality only) | root `code_select_twin` and `sign_dot_i64` in `proof/extract/lib.rs`; `WordRefinement` theorem over the generated wrapper | same two commands |
| q8 AVX2 core | **tested** | nothing provable until the twin is rooted; the core stays an opaque `simd-twin` axiom | `cargo test` for the named equality, already `src/engine/kern/nq.rs:1539` |
| NQ factorization | **model** | a `#[path]`/root for `nanoquant` and a spec for `fro2`/`spd_inverse`/`export`; the `fro2` rescale currently uncommitted, its test queued | commit the W0 change, run `cargo test --lib learn::nanoquant`, then root + regen |
| ternary twin | **model** | `ternary_dot`, its equality test over `{0,1,7,8,9,31,32,33,63,64,65,127,128,129,1000,1023}`, a root, the two theorems above | `cargo test` for the twin, then `proof/regen.sh --check`, `lake build`, audit |
| scale epilogue composition | **proved** only for `mul_q32` (`word_mul_q32_ok`, audited `AxiomAudit/Serve.lean:153`) | the NQ row's `low + corr + res` checked add and `i64::try_from` step | a theorem over the real `project_rows` body; needs the NQ root first |
| extraction freshness | **measured stale** | a regenerated `ApeironWord.lean` from the current sources | `proof/bootstrap.sh`; `proof/regen.sh --check` on a host with >14 GB |

The first two rows are the bottom of the ladder; the ternary row is above them because it replaces
the q8 reader, and its theorem cannot be stated against a body that does not exist.

## 9 · Debt and the `unsafe` rule

`proof/README.md:582-591` is the debt list:

- `sign_dot_avx2` has the safe twin `sign_dot_i64` and the equality test
  `the_vector_source_equals_the_scalar_source`, but `sign_dot_i64` is not a root, so no refinement
  theorem covers it;
- the `Q4_K` packed lane `q4_k_row_lanes_avx2` has only the oracle equality
  `pooled_q4_k_matches_per_column`, no safe twin; it is `tested` debt;
- the limb AVX2 split has the scalar `split_digits` twin and a `tested` boundary entry
  (`kern.limb.simd.split_block_avx2` in `proof/boundary-axioms.txt`).

**The ternary cell adds no `unsafe` without a twin.** There is no ternary source yet, so nothing is
added at all. When the cell lands under D5 (`PLAN.md:160-161`), the order is: safe
`ternary_dot` and its equality test first; then the intrinsic `count_ones`/`popcnt` body; then the
root for the twin; then `word_ternary_dot_denote`. If the intrinsic lands without the rooted twin,
it repeats the `sign_dot_i64` debt exactly: a test but no theorem. The AVX2 `popcnt` cell will be
an opaque `simd-twin` axiom in `proof/boundary-axioms.txt`; the refinement statement is over the
twin, never over the intrinsic (AGENTS.md, "Unsafe").

## 10 · What this note does not establish

It is an audit of artifacts by file and line, not a rebuilt extraction: no `lake build` ran, no
`regen.sh --check` ran, and no Lean theorem was rechecked here. The `proved` grades are the grades
recorded by the existing `#audit` and README; the ternary theorem is a proposed statement, not a
proof. The note makes no claim about rustc/LLVM code generation, the kernel ABI, the AVX2/AVX-VNNI
bodies, the pool boundary, `f64` behavior under the host FPU, or the GPU.
