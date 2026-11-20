# The 64-term ternary cell: the parabolic trit, an exact mask identity, and the rate it projects

Status: **exploratory, opened by the agent in W2 rather than requested, and refuted on quality.**
The design is retained as a record and a negative result, not as a product path. The 0.6B PPL arm
scores 77,741,390 against the same-binary q8act control's 72.988013
(`bench/cpu/1794834226-ternary-ppl.md`).

Criterion: **T1** (decode) and **Q2** (quality screen), PLAN workstream W2. Grades, by part:

| part | grade | artifact |
|---|---|---|
| the trichotomy theorems | `proved` | `classification`, `trichotomy_complete`, `channel_iff_fst_of_disc_zero`, `nilpotent_of_disc_zero`, `noZeroDivisors_iff_disc_neg` |
| identity, quantiser, correction arithmetic | `measured` | `python3 /tmp/opencode/ternary_check.py`, sha256 `2d293af7a43c9845a02d22a4f8e78b117f2a61d9d1277240e3a1d47f5292b18d` |
| the reading that puts the trit on the parabolic face | `anchor` | `proof/corpus/WordRefinement.lean:29-34` (`cell.Sigma.Parabolic`) |
| instruction counts, tok/s, byte ceilings | `model` | `bench/cpu/1794822551-nq-loop-ports.md` §4 |
| the reference PPL pair and its commands | `measured` | `bench/cpu/1794822341-nq06-ppl-vs-q4.md`, protocol `cargo build --release --locked --example ppl` |

Scope: Qwen3-0.6B-Base, the v3 pack (0.978 b/w, 53,815,472 B), 196 linears. The cell is **not in
the tree**: this note is a design, no engine file changes, and no speedup is claimed. `S = 2^k` is
the stored power-of-two block scale, `M = 64` the level in code units, `τ ∈ {−1,0,+1}` the trit.

## 1. The trichotomy: a third value needs a square-zero direction

`Cell = ℝ × ℝ`, `mul σ (a,b) (c,d) = (ac + σbd, ad + bc)`, `normForm σ (a,b) = a² − σb²`
(`proof/canonical/Apeiron/Trichotomy/Basic.lean:107-162`). The corpus proves the three cases are
the whole list: `classification` (Basic.lean:925) puts every two-direction real algebra in this
form with `σ ∈ {−1,0,1}`, and `trichotomy_complete` (Basic.lean:1033) decides the zero-divisor
question by `sign σ` for every such algebra. The three faces differ in exactly the place a
three-valued activation needs:

| σ | species | channels | zero divisors | a third state |
|---|---|---|---|---|
| `> 0` | split-complex | two, `upper ≠ lower` | two lines; a channel reads each as nonzero | must pick a channel; the other disagrees, so dropping it is approximate |
| `0` | dual numbers | one, `upper = lower = fst` | exactly the nilpotent line `b·e` | `e = (0,1)`: reads `0` in the only channel |
| `< 0` | complex | none (`no_channel_of_disc_neg`, `Channels.lean:77`) | none (`noZeroDivisors_iff_disc_neg`, Basic.lean:735) | no scalar dot is definable |

At `σ = 0`: `e² = 0` and `e ≠ 0` (`nilpotent_of_disc_zero`, Basic.lean:807), and `fst` is the
only unital algebra map to `ℝ` (`channel_iff_fst_of_disc_zero`, `Channels.lean:90`). The three
states are `(1,0)`, `(−1,0)`, `(0,1)`; `fst` sends them to `+1`, `−1`, `0`. For any weight `(s,0)`
with `s = ±1`: `(s,0)·(0,1) = (0,s)`, and `(0,1)·(0,1) = (0,0)` — the first coordinate is zero in
both, and the second is what no channel reads. **Dropping a zero trit is exact in the scalar the
dot is taken in.** Above zero the same zero divisor is seen by one channel, so the drop would be an
approximation; below zero there is no zero divisor to spend a third state on. The parabolic face
is the unique one, not a convenient one.

The same polynomial identities hold over `Z/2^64` (`proof/canonical/Apeiron/Trichotomy/AnyRing.lean`),
where the units are the odd elements (`exists_inverse_iff_isUnit_ringNormForm`, AnyRing.lean:373),
and the word kernel already carries the three-way dial (`cell.Sigma` with `Parabolic = 0`,
`proof/corpus/WordRefinement.lean:29-34`). This is not a real-number analogy laid over a machine
kernel.

The third state is not spent on the weight. A weight is stored once and read once per token, 7.94G
weight reads/token; Q1 is 1.174 b/w, and three levels cost `log₂ 3 = 1.585` b/w before scales. A
nilpotent weight has `normForm 0` (`normForm_zero_iff_channel_zero`, `NullBasis.lean:581`), hence is
not a unit (`isUnit_iff_normForm_ne_zero`, Basic.lean:695): the rank factorisation could not
normalise or invert it. `±1` are the units with `normForm 1`, which is what the pack stores. The
activation is streamed, not counted in Q1's denominator, so the third state belongs there.

## 2. The identity, and the sign the code actually uses

With `w` the 64-bit sign word, `p`/`n` the disjoint positive/negative trit masks, `len ≤ 64`,
`pc` = popcount, and

```text
ε_i = 1 − 2·bit_i(w),          t_i = [p_i] − [n_i]

Σ_{i<len} ε_i t_i = 2·pc(w & n) − 2·pc(w & p) + pc(p) − pc(n)          (p & n = 0)
```

Proof: three allowed states per term, no induction and no case on `w`.

| `p_i n_i` | `t_i` | `ε_i t_i` | `2b[n] − 2b[p] + [p] − [n]` |
|---:|---:|---|---|
| `00` | `0` | `0` | `0` |
| `10` | `+1` | `1 − 2b` | `1 − 2b` |
| `01` | `−1` | `2b − 1` | `2b − 1` |

Sum over `i`; `pc(w&p)` and `pc(w&n)` count exactly the `b`'s in the two rows. The corollary the
loop uses: with `d = pc(w&n) − pc(w&p)` and `c = pc(p) − pc(n)`, the block sum is `2d + c`; `c` is
independent of `w`, hence of the row, and `d` costs one `sub`.

**Sign correction.** `sign_dot!` documents and implements `ε = 1 − 2·bit` (`nq.rs:108-128`), so
bit 1 → `−1`. The pack writer is the opposite: `pack_signs` sets bit 1 for `+1`
(`examples/nq_fit.rs:176-187`), and the module header says so (`nq.rs:25-29`). A cell fed the
pack's words therefore returns the negation of the stored-sign dot. It cancels where `U` and `V`
are read as a product and is negated once where a plane is read alone; `nq.rs:1032-1035` is exactly
that negation for the residual dot. `PLAN.md:37` calls `ε` "the pack's convention"; it is the
kernel macro's, and the identity above is stated in the macro's. The activation masks are written
by the quantiser, not the pack, and are unaffected.

**Exhaustive check** (`python3 /tmp/opencode/ternary_check.py`, ~10 s):

```text
3-term cells exhaustive: 216
4-term cells exhaustive: 1296
64-term random cells: 200000
tail lengths: 0..64, 65,127,128,129,1000,1023 (200 each), w padding arbitrary
quantiser: c in [-127,127], M=64/128, t=sign(c)[2|c|>M], r=c-Mt, |r|<=M-1, exact, midpoint-optimal
correction: 20000 random 4-outlier blocks, 16-entry table exact
ALL OK
```

The 3- and 4-term cells are every `(w,p,n)` per term with `p & n = 0` — six states per term,
`6³` and `6⁴` — so the identity is checked at every state of a small cell, not sampled. The check
is of the formula and the quantiser arithmetic, not of a compiled kernel.

**Tail and padding.** For `len < 64`, `p` and `n` are zero above `len`; `w`'s bits above `len` are
arbitrary, because `w & p = w & n = 0` there. `pc(p)` and `pc(n)` then count only real terms. A
packer that leaves `p`, `n` set above `len` adds phantom terms to both sides; zeroing them is the
packer's contract. `w` padding is not a contract: any value gives the same answer. All 0.6B and
7B linear widths are multiples of 64 (1024, 2048, 3072, 3584), so the tail is a fallback, not the
hot path; the equality test covers it anyway.

## 3. Quantiser, residual, and the correction table

Per 64-code block, `S = 2^k` is the smallest power of two with `127·S ≥ absmax` — the rule of
`src/engine/kern/pow2.rs:22` `act_scale` that `quant_act` already applies at `ACT_QK = 32`
(`src/engine/chart/mod.rs:68,262`) — and the code is `c = round(x/S)`, `|c| ≤ 127`. The trit and
residual are

```text
τ = sign(c)  if 2|c| > M  else 0            M = 64
r = c − M·τ                                 |r| ≤ M − 1 = 63
S·c = 2^{k+6}·τ + 2^k·r                     exact, for every code
```

`2|c| > M` is the midpoint rule for the level `M·S`, and it is what makes the residual payload seven
bits, signed. `M = 64` is the top bit of the 7-bit magnitude; the level is
`64·S ∈ [0.50, 1.01]·absmax`, so the cell keeps the sign and the top bit of the code and drops the
rest. The 0.6B arm measures what that costs. If it fails, the one-line lever is a per-block `m` byte
with `M = 2^m` and the two exponents added when the table is built: `+1 B` per block (`+0.125`
b/term) and **no** extra row-loop instruction. That lever is pre-registered here so a failed arm
does not become a redesign.

`k ∈ {0,1,2,4}` exact int8 terms per block. "Exact" means the stored term reproduces `S·c_i`
bit-for-bit: `2^{k+6}τ_i + 2^k r_i = S·c_i` by the identity above, so an outlier is the q8 term,
not an approximation of it. The rest of the block is the level, and the error is
`2^k·Σ_{i∉O} r_i`, at most `(64−k)·63·S` and in practice the block's own dropped residual mass.

| `k` | positions | residual bits | table entries | bytes / 64 | bits / term |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | — | **17** | **2.125** |
| 1 | 6 | 7 | 2 | 18.625 | 2.328 |
| 2 | 12 | 14 | 4 | 20.25 | 2.531 |
| 4 | 24 | 28 | 16 | **23.5** | **2.938** |

17 B = 16 B for `p`,`n` plus 1 B for the exponent (`k+1`, zero for an all-zero block). The layout is
fixed-`k`, so a block with fewer nonzero residuals stores zeros and the row loop has no branch.

**Exact correction without a second row loop.** The correction is
`Σ_{j∈O} ε_j r_j`, a function of the `k` weight bits at the outlier positions and nothing else.
Build a table of `2^k` entries per block, once per activation vector in the pack step:

```text
table_b[s] = ( Σ_j (1 − 2·bit_j(s))·r_j ) << (k_b + 6)        s ∈ [0, 2^k)
```

pre-scaled by `2^{k_b+6}`, so the row loop adds one entry and never shifts the correction. The
index is `idx = pext(w, posmask)` — `pext` gives `bit_j(w)` at index bit `j` — and the row cost is
`pext` plus `add acc, [table_b + idx*8]`, two instructions per block, independent of `k`: 1 and 4
outliers cost the same. At `k = 4` the table is 16×8 = 128 B per block, L1-resident, and read once
per row. Its construction is `O(2^k)` per block per vector, outside the row loop; at `out = 1024`
rows it is under 1% of the row work. The 64-entry portable index (`and` with `posmask`, 512 B
table) replaces `pext` at +1 static instruction; it is the arm for hosts where `pext` is microcoded
(Zen 1/2: 7 uops, 18–19 cycles) and for a dispatch that cannot prove BMI2 cheap.

**Scale folding and range.** The level contribution is `d << (k_b + 7)` (the `×2` of `2d + c` is
inside the shift), and the row-independent part is one constant per activation vector,
`C = Σ_b c_b << (k_b + 6)`, added once per row. No per-block constant enters the row loop. The
accumulator is `i64` when the vector's precomputed bound fits and `i128` otherwise — the dial
`sign_dot!` already has (`nq.rs:108-128`). The table entries are bounded by `4·63·2^{k+6}`, and the
`i64` route is admitted only when the vector's bound covers them.

**Storage per activation vector.** 2.125–2.938 b/term, against 10 b/term for the q8 carrier (one
`i64` per 32 plus one `i8` per element, `src/engine/chart/mod.rs:68-78`): 3.4–4.7× smaller. Over the
0.6B's 286,720 activation terms per token (seven vectors per layer × 28) that is 76–105 KB/token,
at most 0.011% of the 0.998 GB weight stream, so the carrier does not move Q1.

## 4. Instruction budget and the rate it projects

Per 64 terms per output row, block-outer, with `p`, `n`, `posmask`, the table base and the shift
count resident:

| # | instruction | role | uops | port |
|---:|---|---|---:|---|
| 1 | `mov (%rdi), %rax` | `w` | 1 | p23 |
| 2 | `pext %r12, %rax, %rcx` | outlier index (k>0) | 1 | p1 |
| 3 | `mov %rax, %rdx` | copy | 0 | — |
| 4 | `and %r8, %rdx` | `w & p` | 1 | p015 |
| 5 | `and %r9, %rax` | `w & n` | 1 | p015 |
| 6 | `popcnt %rdx, %rdx` | `pc(w & p)` | 1 | p1 |
| 7 | `popcnt %rax, %rax` | `pc(w & n)` | 1 | p1 |
| 8 | `sub %rdx, %rax` | `d` | 1 | p0156 |
| 9 | `shlx %r10, %rax, %rax` | `×2^{k+7}` | 1 | p06 |
| 10 | `add %rax, %r11` | row accumulator | 1 | p0156 |
| 11 | `add (%rbx,%rcx,8), %r11` | pre-scaled correction (k>0) | 2 | p23+p0156 |

Pure: 9 static / 8 exec uops. With 4 outliers: 11 static / 10 exec. Loop control is charged to each
row (3 per 4-block tile), giving **9.75** and **11.75** static per 64 (8.5 and 10.5 exec); the
`add C` epilogue is one instruction per row. The 4-row tile is at the GPR limit; one spilled
register per block adds 2 instructions, lands the pure arm on the residual arm's 22.5, and is still
above 20.

| layout | static / 64 | exec / 64 | 2.1 IPC | 3.0 IPC |
|---|---:|---:|---:|---:|
| pure, 4-row tile | 9.75 | 8.5 | **27.1** | **38.7** |
| 4/64 outliers, 4-row tile | **11.75** | 10.5 | **22.5** | **32.1** |

4 cores × 3.9 GHz = 15.6 G cycles/s; 32.76 G instructions/s at 2.1 IPC, 46.8 at 3.0;
7.94G terms/token; IPC is applied to the static count, as `PLAN.md:17-22` does, and the ceilings
below are the port and front-end checks.
`32.76 × 64 / 11.75 / 7.94 = 22.5`.

Ceilings, per block: `popcnt` is one per cycle on port 1, so the pure cell is 2.0 cycles →
7.8 G blocks/s → 499.2 G terms/s → **62.9 tok/s**; the correction adds `pext`, also p1, 3.0 cycles →
**41.9**. The front end is 8.75/4 = 2.19 cycles (pure) and 10.75/4 = 2.69 (with correction), the
scheduler 8.5/4 and 10.5/4; both sit above the IPC projections, so the table's 22.5/32.1 is the
conservative rate, not the port ceiling. At 2.1 IPC the **instruction issue binds** (22.5 against
28.0 bytes); at 3.0 the **byte stream binds** (28.0 against 32.1). Crossover: 2.62 IPC with the
correction, 2.17 pure. A row-outer loop that reloads `p`, `n` and the exponent per block per row
costs 14.75 static (17.9 tok/s at 2.1), which is why the loop is block-outer.

Bytes: 0.998 GB/token (1 bit/term = 0.9925 GB plus 0.6% scales and headers) at the measured
27.97 GB/s gives **28.03 tok/s**. Against T1: 30 tok/s at 2k needs 29.94 GB/s = **1.07× the
measured bandwidth** and 2.80 IPC at 11.75/64 — refused on bytes before the kernel is considered.
24 tok/s at 200k needs 23.95 GB/s of *weights alone* (85.6% of measured, leaving under 4.0 GB/s for
KV and the head, which are not in the 0.998) and 2.24 IPC — at 2.1 IPC the cell gives 22.5, 6.4%
short of 24, with the KV read still to pay. The residual arm clears PLAN's 20 tok/s target at 2.1 IPC
(22.5, 1.12×) and does not clear T1's 24 or 30 on this device.

This is not a performance receipt. The phase split has not run for this cell: it is not in the
tree, and on the 0.6B fixture the tied `F16` head owned 82% of a decode token, capping any
projection-side change at 1.22× there; moving that head to `Q8_0` took 3.44 → 21.42 tok/s
(`bench/cpu/1794813322-q8-head-fixture.md`). A 7B split is unmeasured. The gate is
`examples/phase_split --target` on the built cell, and it has not run.

## 5. The safe twin and the Lean obligation

The twin lives in a new `src/engine/kern/ternary.rs`, declared in `src/engine/kern/mod.rs`, which
`proof/extract/lib.rs:1127` already includes by `#[path]`. No wrapper is needed: the signature is
concrete.

```rust
/// Popcount without `u64::count_ones`: the pinned Aeneas model names
/// `core.num.I64.count_ones`, not `core.num.U64.count_ones`.
/// O(len) time, O(1) space; one shift per bit.
pub fn popcount_u64(x: u64) -> u64 {
    let mut bits = x;
    let mut count = 0u64;
    while bits != 0 {
        count += bits & 1;
        bits >>= 1;
    }
    count
}

/// Σ (1 − 2·bit_i(w))·t_i over the first `len` terms, t from the masks.
/// `p & n = 0` is the contract; `len` above 64 saturates at the full word.
/// O(1) time and space: four popcounts, no loop over terms.
pub fn ternary_dot(w: u64, p: u64, n: u64, len: usize) -> i128 {
    let mask = if len >= 64 { u64::MAX } else { (1u64 << len) - 1 };
    let (p, n) = (p & mask, n & mask);
    let wp = i128::from(popcount_u64(w & p));
    let wn = i128::from(popcount_u64(w & n));
    2 * wn - 2 * wp + i128::from(popcount_u64(p)) - i128::from(popcount_u64(n))
}
```

Equality test lengths, from PLAN W2: `{0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 129,
1000, 1023}`. At each length the test runs the twin against a per-term loop and, once the cell
exists, the `popcnt` cell against the twin, over random `w` with arbitrary padding above `len` and
random disjoint `p`, `n`. The cases land on the word and block boundaries (0, 1, 8/9, 32/33, 64/65)
and on the multiword tail (127–129, 1000, 1023, where the mask saturates and the function is total).
One extra case sets `p`, `n` above `len` and checks that the twin equals the masked sum — that is
the twin's masking, not the cell's contract.

Lean: `ternary_dot_denote`, in `proof/corpus/WordRefinement.lean` (namespace `apeiron_word`, the
file that already imports `Apeiron.Trichotomy.AnyRing`), by induction on `len` and the three-state
case split of §2, in the monadic shape of `word_point_flux_denote` (WordRefinement.lean:265). The
function is monadic because the variable shift is (`let i ← m >>> shift`,
`proof/ApeironWord.lean:8094`), not because the formula has a failure case:

```text
(do let r ← ternary_dot w p n len; ok r)
  = ok (Σ i in range len, (1 − 2·bit_i w) · t_i)          with p &&& n = 0
```

Axiom expectation: `{propext, Classical.choice, Quot.sound}` and nothing else — no `sorryAx`, no
`Lean.ofReduceBool`. The `#audit` line goes in `proof/corpus/AxiomAudit.lean`. Two gaps are named,
not hidden. `u64::count_ones` has no model in
`proof/patches/aeneas-lean-model.patch:154-171` (which defines `core.num.I64.count_ones`), so the
twin does not call it; `(x as i64).count_ones()` is modelled and already used at
`proof/ApeironWord.lean:23442`, and is a legitimate shorter twin if a trial extraction accepts the
cast, but the loop is the specified body because it adds no extraction risk. And `popcnt` is
`core::arch`, which Aeneas does not model: the intrinsic path is `tested` by
`cargo test --workspace --all-features --locked` against the twin over the same lengths, and no
Lean theorem covers it — the AGENTS unsafe rule, the shape of `sign_dot_i64` → `sign_dot_avx2`. A
mutation of the twin's `2*wn − 2*wp` to `2*wp − 2*wn` must redden the test.

## 6. The 0.6B experiment

Three arms, same token stream, two lengths. The Q4 and q8act rows already exist; only the ternary
row is new.

| arm | model | kernel |
|---|---|---|
| Q4_K_M | `Qwen3-0.6B-Base-Q4_K_M.gguf` | `q8act` |
| NQ v3 | `skeleton-q8head.gguf` + `/tmp/opencode/nq06-v3.nqp` | `q8act` |
| NQ v3 | same | **`ternary`** (new `--kernel` value) |

```sh
~/Workspace/tools/agent-guard run -- cargo build --release --locked --example ppl

~/Workspace/tools/agent-guard run -- target/release/examples/ppl \
  --model /home/lucas/models/qwen3-0.6b-base-f16/Qwen3-0.6B-Base-Q4_K_M.gguf \
  --tokenizer /home/lucas/models/qwen3-0.6b-base-e8/tokenizer.json \
  --data /home/lucas/models/wikitext2/wikitext-2-raw/wiki.test.raw \
  --tokens <2048|8192> --skip 0 --ctx 512 --head-batch 16 --kernel q8act

~/Workspace/tools/agent-guard run -- target/release/examples/ppl \
  --model /home/lucas/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack /tmp/opencode/nq06-v3.nqp --tokenizer /home/lucas/models/qwen3-0.6b-base-e8/tokenizer.json \
  --data /home/lucas/models/wikitext2/wikitext-2-raw/wiki.test.raw \
  --tokens <2048|8192> --skip 0 --ctx 512 --head-batch 16 --kernel <q8act|ternary>
```

`--tokens 2048` scores 2,044 positions; `--tokens 8192` scores 8,176. Reference PPL
(`bench/cpu/1794822341-nq06-ppl-vs-q4.md`): Q4 18.118871 / 21.715296, v3 q8act 74.296766 /
54.806671. The criterion arithmetic:

| tokens | 1.5× Q4 | 1.25× q8act | q8act ÷ 1.5×Q4 |
|---:|---:|---:|---:|
| 2,044 | 27.178 | 92.871 | 2.73× |
| 8,176 | 32.573 | 68.508 | 1.68× |

**Kill criterion, read against the numbers.** The 1.5× Q4 clause cannot be reached by a recoding of
the q8 codes: the q8 reader is already 4.10× Q4 at 2,044 and 2.52× at 8,176, so passing that clause
requires the ternary arm to beat the q8 reader by 2.73× or 1.68×, which a coarsening does not do.
The live clause is **PPL ≤ 1.25× the q8 reader**, i.e. ≤ 92.871 and ≤ 68.508. The instruction clause
of 8 static / 64 terms is met only under the core-only accounting, and only by the unscaled cell:
2 `and` + 2 `popcnt` + 2 `add` = 6, plus `pext` and the table add = 8 at `k = 4`. This cell spends
`sub` and `shlx` on the per-block exponent, so its core count is 9 and its loop count 11.75 (pure 7
and 9.75). The clause refuses it as specified; the arm still runs because 11.75 projects 22.5 tok/s
at 2.1 IPC, above PLAN's 20 target. Quote the counting rule with the number.

A pass at 1.25× is a quality floor for the cell, not Q2: the 0.6B is a screen at ctx 512, and the
7B 1.174 pack that Q2 needs is unbuilt. A fail retires the cell to the int4-activation fallback
(`PLAN.md:51-53`) and is the useful result.

Files that would change, none of them in this turn:

| file | change |
|---|---|
| `src/engine/kern/ternary.rs` (new) | `popcount_u64`, `ternary_dot`, the `popcnt` cell |
| `src/engine/kern/mod.rs` | `mod ternary;` so the extraction's `kern` include carries it |
| `src/engine/kern/nq.rs` | the cell dispatch (owned by K; not edited here) |
| `src/engine/chart/mod.rs` | `quant_act64`: scale, codes, masks, outliers, pre-scaled table |
| `src/engine/step.rs` | `KernelMode::Ternary` |
| `src/engine/loader/nq.rs` | the ternary arm in the stage dispatch |
| `examples/ppl.rs` | `--kernel ternary` |
| `proof/corpus/WordRefinement.lean` | `ternary_dot_denote`; `proof/corpus/AxiomAudit.lean` the `#audit` |
| `bench/cpu/<id>-ternary-ppl.md` | the receipt |

## 7. What this does not cover

- **The cell is not in the tree and no speedup is measured.** §4 is a static count over a loop no
  compiler has emitted; rustc may spill (the 4-row tile is at the GPR limit), paying the 2
  instructions that separate the pure and residual arms, or fail to fold the shifts.
- **No phase split, no pool, no head, no KV.** The 4-core projection is a linear extrapolation; the
  0.6B fixture's dominant term is the head, so the cell cannot move that fixture's rate by more
  than 1.22×.
- **No PPL for the ternary arm.** The 1.25× clause, the `M` byte lever and the outlier dial are
  untested on any model. The 7B point that Q2 needs does not exist.
- **The sign convention** is read from source, not from a compiled cell: `sign_dot!`'s `ε` and
  `pack_signs`' bit are cited, and the negation is argued, not run.
- **The Lean theorem does not exist yet.** The trichotomy theorems are `proved` (`classification`,
  `trichotomy_complete`); the denotation theorem `ternary_dot_denote` is a statement of work, and
  the `popcnt` path is `tested` by `cargo test --workspace --all-features --locked` against the
  twin, with no Lean theorem over it.
- **The byte bound uses 0.998 GB/token as specified**, weight stream only; masks, table, head, KV,
  the 200k context and the bandwidth under four concurrent readers are outside it.
