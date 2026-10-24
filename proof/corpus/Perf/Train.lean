import ApeironWord
import Mathlib.Tactic

/-!
# The RMSNorm backward kernel's traffic model (P83)

`ApeironWord` is imported for the FORM-ORPHAN reachability edge: this cost model prices a GPU
kernel that is waived from extraction (its SPIR-V builder and dispatch are the `gpu`/`ffi`
classes), so no extracted definition is used here.

`src/gpu/nn/norm_bwd.rs` dispatches one workgroup per row. One dispatch over `t` rows of width
`w` reads `x`, `gain`, `dy` and writes `dx`, `dγ_rows`: four `t·w` tensors and the `w`-wide gain,
four bytes per f32.

    bytes(t, w) = (4·t·w + w) · 4

At the device's 960 GB/s the memory floor is `bytes / 960·10^9` seconds: 7.34 MB and 7.65 µs at
`t = 512`, `w = 896`. The arithmetic is about ten flops per element, two orders below that floor
at the machine's fp32 rate (a `model` input from P83).

The second dispatch (`rmsnorm_dgamma_module`, P85) replaces the host row fold: it reads `dγ_rows`
(`t·w` f32) and writes `dγ` (`w` f32), `(t·w + w)·4` bytes, and removes the host readback of the
`t·w` rows. What is proved here: the two rows' cost equations, expanded. The memory floors
themselves are roofline instances at a declared machine bandwidth, not kernel properties
(obligation 11); no theorem here produces a byte or a microsecond.
-/

namespace Perf.Train

/-- Bytes moved by one `rmsnorm_bwd` dispatch at `t` rows of width `w`: `x`, `gain`, `dy` read;
    `dx`, `dγ_rows` written, four bytes per element. -/
def rmsnormBwdBytes (t w : Nat) : Nat := (4 * t * w + w) * 4

/-- The P83 cost equation in expanded form: `16·t·w + 4·w` bytes. -/
theorem rmsnormBwdBytes_eq (t w : Nat) : rmsnormBwdBytes t w = 16 * (t * w) + 4 * w := by
  unfold rmsnormBwdBytes
  ring

/-- Bytes moved by a `rmsnorm_dgamma` reduction dispatch at `t` rows of width `w`: `dγ_rows`
    read (`t·w` f32) and `dγ` written (`w` f32). -/
def rmsnormDgammaReduceBytes (t w : Nat) : Nat := (t * w + w) * 4

/-- The P85 cost equation in expanded form: `4·t·w + 4·w` bytes. -/
theorem rmsnormDgammaReduceBytes_eq (t w : Nat) :
    rmsnormDgammaReduceBytes t w = 4 * (t * w) + 4 * w := by
  unfold rmsnormDgammaReduceBytes
  ring

/-- Words read by one naive `linear_bwd_input` dispatch at `n` rows, input width `i`, output width
    `o`: each of the `n·i` outputs reads its `dy` row (`o` words) and its `w` column (`o` words),
    and the `n·i` outputs are written. -/
def linearBwdInputReads (t i o : Nat) : Nat := 2 * t * i * o + t * i

/-- The P86 cost equation in expanded form: `2·n·i·o + n·i` words. -/
theorem linearBwdInputReads_eq (t i o : Nat) :
    linearBwdInputReads t i o = 2 * t * i * o + t * i := rfl

/-- Words read by one naive `linear_bwd_weight` dispatch at `n` rows, input width `i`, output width
    `o`: each of the `o·i` outputs reads its `dy` column (`n` words) and its `x` column (`n`
    words), and the `o·i` outputs are written. -/
def linearBwdWeightReads (t i o : Nat) : Nat := 2 * t * i * o + o * i

/-- The P87 cost equation in expanded form: `2·n·o·i + o·i` words. -/
theorem linearBwdWeightReads_eq (t i o : Nat) :
    linearBwdWeightReads t i o = 2 * t * i * o + o * i := rfl

/-- Words read by one naive causal-attention `dQ` dispatch at `t` tokens, `nh` query heads and
    head width `hd`: each of the `t·nh·hd` outputs reads its `dout` row and its `v` row once per
    key `j ≤ i`, `2·t·hd` words per thread, and the outputs are written. -/
def attnBwdQReads (t nh hd : Nat) : Nat := 2 * t * t * nh * hd * hd + t * nh * hd

/-- The P88 cost equation in expanded form: `2·t²·nh·hd² + t·nh·hd` words. -/
theorem attnBwdQReads_eq (t nh hd : Nat) :
    attnBwdQReads t nh hd = 2 * t * t * nh * hd * hd + t * nh * hd := rfl

/-- Words moved by the materialised-dS causal-attention backward (P90): dispatch 2 writes
    `dS[h,i,j] = P[h,i,j]·(dP_ij − rowsum[h·t+i])` and dispatches 3 and 4 read it instead of
    recomputing `dP_ij`.

    Upper bound over the causal half, `nh` in place of `nkv·rep` for the kv outputs:

    * `ds`: `nh·t·t` threads; a thread reads `2·hd` words of `dout`/`v`, one `P` word and one
      `rowsum` word (`2·hd + 2`) and writes one `dS` word: `nh·t·t·(2·hd + 2) + nh·t·t`.
    * `dq`: `t·nh·hd` outputs; each reads at most `t` `dS` words and one `K` word and writes one
      output: `t·nh·hd·(t + 2)`.
    * `dk`/`dv`: each output walks `rep·t` `(h,i)` pairs, reading one `dS`, one `Q`, one `dout`
      and one `P` word (`4`) and writing two words: `4·t·t·nh·hd + 2·t·nh·hd`.

    The repeated dot-product traffic falls from `4·t²·nh·hd²` (two in-thread `dP` recomputations)
    to `2·t²·nh·hd` (`dout`/`v` in `ds`): a factor `2·hd`. -/
def attnBwdDsReads (t nh hd : Nat) : Nat :=
  nh * t * t * (2 * hd + 2) + nh * t * t + t * nh * hd * (t + 2) + 4 * t * t * nh * hd +
    2 * t * nh * hd

/-- The P90 cost equation in expanded form as defined above. -/
theorem attnBwdDsReads_eq (t nh hd : Nat) :
    attnBwdDsReads t nh hd =
      nh * t * t * (2 * hd + 2) + nh * t * t + t * nh * hd * (t + 2) + 4 * t * t * nh * hd +
        2 * t * nh * hd := rfl

/-- Words read by one tiled `linear_bwd_input` dispatch at `t` rows, input width `i`, output
    width `o`: each 16-row tile of `dy` is read once per column tile (`i/16` of them), each
    16-column tile of `w` once per row tile (`t/16`), and the `t·i` outputs are written. -/
def denseBwdTiledReads (t i o : Nat) : Nat := t * o * (i / 16) + o * i * (t / 16) + t * i

/-- The P89 cost equation in expanded form: `t·o·⌊i/16⌋ + o·i·⌊t/16⌋ + t·i` words, 16× below the
    naive `2·t·i·o + t·i` of P86 at square multiples of 16 (`t·i·o/8` against `2·t·i·o`). -/
theorem denseBwdTiledReads_eq (t i o : Nat) :
    denseBwdTiledReads t i o = t * o * (i / 16) + o * i * (t / 16) + t * i := rfl

/-- Words read by one `adam_update` dispatch at `n` elements: `p`, `g`, `m`, `v` read and `p`,
    `m`, `v` written, seven words per element. -/
def adamUpdateReads (n : Nat) : Nat := 7 * n

/-- The P91 cost equation: `7·n` words. -/
theorem adamUpdateReads_eq (n : Nat) : adamUpdateReads n = 7 * n := rfl

/-- Words read and written by one tiled `linear_fwd` dispatch at `t` rows, input width `i`, output
    width `o`: each 16-row tile of `x` is read once per column tile (`o/16` of them), each 16-row
    tile of `w` once per row tile (`t/16`), and the `t·o` outputs are written. -/
def denseFwdTiledReads (t i o : Nat) : Nat := t * i * (o / 16) + o * i * (t / 16) + t * o

/-- The P92 cost equation in expanded form: `t·i·⌊o/16⌋ + o·i·⌊t/16⌋ + t·o` words, 16× below the
    naive `2·t·o·i + t·o` (each output re-reads its x row and w row) at square multiples of 16
    (`t·i·o/8` against `2·t·i·o`). -/
theorem denseFwdTiledReads_eq (t i o : Nat) :
    denseFwdTiledReads t i o = t * i * (o / 16) + o * i * (t / 16) + t * o := rfl

/-- Words moved by one causal-attention forward dispatch (P93): one workgroup per `(i,h)` query
    row, the `t` scores of the row in shared memory, the causal half summed over `j ≤ i`.

    * phase 1 reads `hd` words of `q` per row and `i·hd` of `k`: `t·nh·hd + nh·t²·hd/2`.
    * phase 3 reads `i·hd` of `v` and writes `hd` of `out`: `nh·t²·hd/2 + t·nh·hd`.
    * the `P` row is written: `nh·t²/2`.

    The materialised alternative writes the `nh·t·t` scores and re-reads them for the max, the
    partition function and every `(i,h,d)` value mix: `3·nh·t²·hd/2 + 2·nh·t² + 2·t·nh·hd` words,
    so the fused row removes the two extra `nh·t²·hd/2` passes and the score traffic. -/
def attnFwdReads (t nh hd : Nat) : Nat :=
  nh * t * t * hd / 2 + 2 * t * nh * hd + nh * t * t / 2 + t * nh * hd

/-- The P93 cost equation in expanded form as defined above. -/
theorem attnFwdReads_eq (t nh hd : Nat) :
    attnFwdReads t nh hd =
      nh * t * t * hd / 2 + 2 * t * nh * hd + nh * t * t / 2 + t * nh * hd := rfl

/-- Words read by one attention-backward row-sum dispatch (P94): one workgroup per `(h,i)` query
    row, `nh·t` workgroups, each reading the `i+1` causal key rows once — `2·hd + 1` words per
    key (`dO` row, `V` row, `P` word) — and writing one `rowsum` word.

    The traffic is unchanged from the single-workgroup form: `Σ_i (i+1)·(2·hd+1)·nh` words.
    Only the grid changes, from `⌈t·nh/256⌉` workgroups to `nh·t`. -/
def attnRowsumReads (t nh hd : Nat) : Nat := nh * (t * (t + 1) * hd + t * (t + 1) / 2)

/-- The P94 cost equation in expanded form as defined above. -/
theorem attnRowsumReads_eq (t nh hd : Nat) :
    attnRowsumReads t nh hd = nh * (t * (t + 1) * hd + t * (t + 1) / 2) := rfl

/-- Words moved by the P102 tiled attention-backward row sum and its partial reduce, both
    dispatches: `⌈t/16⌉·t·hd` staged `dO` words and `⌈t/16⌉·t·hd` staged `V` words per head
    (each valid row once per tile axis), one `P` word per causal pair (`t·(t+1)/2` per head),
    `t·⌈t/16⌉` partials written and read back per head, and `t` `rowsum` words written per head.
    The P94 identity `attnRowsumReads` is the historical pre-tiling record and does not price the
    staged tiles or the reduce. -/
def attnBwdRowsumTiledReads (t nh hd : Nat) : Nat :=
  let tiles := (t + 15) / 16
  2 * nh * tiles * t * hd + nh * (t * (t + 1) / 2) + 2 * nh * t * tiles + nh * t

/-- The P102 cost equation in expanded form as defined above: the tiles read each operand word
    `⌈t/16⌉` times and the reduce adds the partial-write and partial-read word counts. -/
theorem attnBwdRowsumTiledReads_eq (t nh hd : Nat) :
    attnBwdRowsumTiledReads t nh hd =
      2 * nh * ((t + 15) / 16) * t * hd + nh * (t * (t + 1) / 2)
        + 2 * nh * t * ((t + 15) / 16) + nh * t := rfl

/-- Words moved by one `swiglu_fwd` dispatch at `n` elements: `gate` and `up` read, `out`
    written, three words per element, one thread per element. -/
def swigluFwdReads (n : Nat) : Nat := 3 * n

/-- The P96 forward cost equation: `3·n` words. -/
theorem swigluFwdReads_eq (n : Nat) : swigluFwdReads n = 3 * n := rfl

/-- Words moved by one `swiglu_bwd` dispatch at `n` elements: `gate`, `up` and `dy` read,
    `dgate` and `dup` written, five words per element, one thread per element. -/
def swigluBwdReads (n : Nat) : Nat := 5 * n

/-- The P96 backward cost equation: `5·n` words. -/
theorem swigluBwdReads_eq (n : Nat) : swigluBwdReads n = 5 * n := rfl

/-- Words moved by one KL-head dispatch (P95) over `r` rows of vocabulary `v`: one workgroup per
    row strides the vocabulary three times — the maxima pass reads `z_s` and `z_t` (`2·v`), the
    partition-function pass reads them again (`2·v`), and the loss/dz pass reads them a third time
    (`2·v`) — so `6·v` words are read per row; the single `loss` word per row is written here, and
    the `dz` row adds `v` more written words. -/
def klHeadReads (r v : Nat) : Nat := r * (6 * v + 1)

/-- The P95 cost equation in expanded form: `r·(6·v + 1)` words (reads plus the loss word; `dz`
    adds `r·v`). -/
theorem klHeadReads_eq (r v : Nat) : klHeadReads r v = r * (6 * v + 1) := rfl

/-- Words moved by one attention-backward dK/dV dispatch (P97) at `t` keys, `nh` query heads,
    `nkv` KV heads and head width `hd`: one workgroup of 256 threads per `(j,g,32-column d-tile)`
    of `t·nkv·⌈hd/32⌉`, each thread owning one output column `d < hd` and one of eight reduction
    groups. Group `grp` walks the `rep = nh/nkv` query heads and, inside each, the rows
    `i = j+grp, j+grp+8, … < t`, reading one `ds`, one `p`, one `q` and one `dout` word per
    `(h,i)` pair — `4·hd` words per pair across the `hd` lanes, so a full tile reads
    `32·4·rep·(t−j)` and the `⌈hd/32⌉` tiles of one `(j,g)` read `4·hd·rep·(t−j)`. The grid reads
    `Σ_j nkv·hd·4·rep·(t−j) = 2·hd·nh·t·(t+1)` words and writes `2·t·nkv·hd` (`dk`, `dv`).

    The one-thread-per-output form read and wrote the same words; the shared `8·32`-word partial
    arrays per output are workgroup memory, not traffic. At `(64,4,2,32)` the count is
    1,064,960 + 8,192 words. -/
def attnDkdvReads (t nh nkv hd : Nat) : Nat :=
  2 * hd * nh * t * (t + 1) + 2 * t * nkv * hd

/-- The P97 cost identity in the expanded form as defined above. -/
theorem attnDkdvReads_eq (t nh nkv hd : Nat) :
    attnDkdvReads t nh nkv hd = 2 * hd * nh * t * (t + 1) + 2 * t * nkv * hd := rfl

/-- Words moved by one row-batched `rmsnorm_fwd` dispatch (P98) over `t` rows of width `w`
    (`t ≥ 1`, `w ≥ 1`, grid `t`, LocalSize 256): each lane caches its `⌈w/256⌉` x words in
    registers, so the dispatch reads `t·w` (`x`) plus `w` (`γ`) and writes `t·w` (`out`) —
    `2·t·w + w` words. -/
def rmsnormFwdReads (t w : Nat) : Nat := 2 * t * w + w

/-- The P98 cost identity in the expanded form as defined above. -/
theorem rmsnormFwdReads_eq (t w : Nat) : rmsnormFwdReads t w = 2 * t * w + w := rfl

/-- Words read and written by one `rope_fwd` dispatch at `t` rows, `nh` heads and head width
    `hd`: each of the `t·nh·hd/2` threads reads two words of `x` and one `cos` and one `sin`
    word, and writes two words of `out` — `x` and `out` are `t·nh·hd` words each and the two
    half-tables `t·hd/2` each. -/
def ropeFwdReads (t nh hd : Nat) : Nat := 2 * t * nh * hd + t * hd

/-- The P100 forward cost equation: `2·t·nh·hd + t·hd` words. -/
theorem ropeFwdReads_eq (t nh hd : Nat) :
    ropeFwdReads t nh hd = 2 * t * nh * hd + t * hd := rfl

/-- Words read and written by one `rope_bwd` dispatch (a second definition, the same count: the
    adjoint reads `dy` and the two half-tables and writes `dx`, one word per component). -/
def ropeBwdReads (t nh hd : Nat) : Nat := 2 * t * nh * hd + t * hd

/-- The P100 adjoint cost equation: `2·t·nh·hd + t·hd` words. -/
theorem ropeBwdReads_eq (t nh hd : Nat) :
    ropeBwdReads t nh hd = 2 * t * nh * hd + t * hd := rfl

/-- Words moved by the tiled attention-backward `ds` dispatch (P99): one 16×16 output tile per
    workgroup, `nh·⌈t/16⌉²` workgroups of 256 threads, the tile's `dout` rows and `v` rows staged
    in two `16·hd`-word shared arrays.

    Workgroup `(h, ti, tj)` stages `min(16, t−16·ti)·hd` words of `dout` and
    `min(16, t−16·tj)·hd` of `v` — a partial last tile stages only its valid rows, the rest of the
    shared array stays zero — then reads one `p` and one `rowsum` word per causal thread and writes
    one `ds` word per valid thread (`j > i` writes the zero, so every valid thread writes).

    Summed over the exact grid: `Σ_ti min(16, t−16·ti) = t`, so each operand tile axis reads
    `⌈t/16⌉·t·hd` words per head; the causal threads partition all `t(t+1)/2` pairs of a head, so
    `p`/`rowsum` read `t(t+1)` words; the `ds` writes are the `t²` valid threads. Total
    `2·nh·⌈t/16⌉·t·hd + nh·t·(t+1) + nh·t·t` words. The shared stores are workgroup memory, not
    traffic.

    The untiled form (P90) moved `nh·t·t·(2·hd+2) + nh·t·t` words, so the tile removes a factor
    `⌈t/16⌉` from the operand traffic: 1,097,728 → 88,560 words at `(t, nh, hd) = (64, 4, 32)`. -/
def attnBwdDsTiledReads (t nh hd : Nat) : Nat :=
  2 * nh * ((t + 15) / 16) * t * hd + nh * t * (t + 1) + nh * t * t

/-- The P99 cost equation in expanded form as defined above. -/
theorem attnBwdDsTiledReads_eq (t nh hd : Nat) :
    attnBwdDsTiledReads t nh hd =
      2 * nh * ((t + 15) / 16) * t * hd + nh * t * (t + 1) + nh * t * t := rfl

/-- Compute dispatches recorded by one `Trainer::step`: 47 on the layer tape (the forward's 17
    and the backward's 30) plus the nine Adam updates. The 111 compute->compute barriers (56
    after a dispatch and 55 between) and the nine buffer copies record no dispatch.

    Counted from the `rec_step` sequence in `src/gpu/nn/train/mod.rs`; a guard on the composition,
    not a hardware quantity: it moves only when the tape does. -/
def layerStepDispatches : Nat := 56

/-- The P101 dispatch count, by `rfl`. -/
theorem layerStepDispatches_eq : layerStepDispatches = 56 := rfl

/-- Host wall of one `Trainer::step`, in microseconds: device work `device`, `builds` pipeline
    builds at `build` µs each, and queue wait `queue`. The model, not a hardware theorem: the step
    records 56 dispatches and then waits, so its host wall is the device span plus whatever the
    host spent outside the queue. Before the Adam pipeline cache `builds = 9` on every step, because
    the bias-correction constants were baked per step ordinal into the module. -/
def stepHostMicros (device builds : Nat) (build queue : Nat) : Nat :=
  device + builds * build + queue

/-- The cache removes exactly the nine per-step builds: for every device span, build cost and
    queue wait, `stepHostMicros d 9 b q − stepHostMicros d 0 b q = 9·b`.

    Grade `model`: arithmetic over the named quantities. The constants below are the `measured`
    inputs; this theorem produces no microsecond. -/
theorem cacheSavesNineBuilds (d b q : Nat) :
    stepHostMicros d 9 b q - stepHostMicros d 0 b q = 9 * b := by
  unfold stepHostMicros
  omega

/-- Measured per-pipeline Adam build cost, in µs. The gate run (RTX 5080, driver 615.78.08) ran
    three nine-build sets of 29,180.0 / 27,858.4 / 27,068.3 µs, mean 28,035.6 µs, so 3,115 µs per
    pipeline; the three single-pipeline runs measured 9,642.5 / 6,938.5 / 7,759.5 µs (mean
    8,113.5 µs), above the per-pipeline share of a nine-build set. Three gate runs of the same
    command measured 23,104.9 / 28,035.6 / 23,136.1 µs per nine (2,567 / 3,115 / 2,571 µs per
    pipeline): the driver's shader cache and clock state move the constant, the identity does not.

    Command: `cargo test --release --features gpu --test gpu_train_timeline -- --nocapture
    --test-threads=1` (the `CONTROL` line). Grade `measured`. -/
def measuredAdamBuildMicros : Nat := 3115

/-- Measured pre-fix host wall of one step, in µs: 23,857 µs at
    `t = 64, i = 896, h = 4864, v = 64, nh = 14, nkv = 2, hd = 64`, where nine fresh Adam
    pipelines were built on every step. Grade `measured` (the spike's in-buffer/by-submission
    campaign, not this identity's arithmetic). -/
def preFixStepHostMicros : Nat := 23857

/-- Measured post-fix host wall of one campaign-A step, in µs (2,994.7 µs mean over seven traced
    steps; campaign B 3,224.0 µs). Grade `measured`. -/
def measuredPostFixHostMicros : Nat := 2995

/-- The post-fix acceptance target, in µs. -/
def postFixTargetMicros : Nat := 5000

/-- The measured per-pipeline cost substituted into the identity: nine builds account for
    `9·3,115 = 28,035` µs of the pre-fix wall, and the cache removes all of it. -/
theorem cacheSavesNineBuildsAtMeasured (d q : Nat) :
    stepHostMicros d 9 measuredAdamBuildMicros q -
        stepHostMicros d 0 measuredAdamBuildMicros q =
      9 * measuredAdamBuildMicros :=
  cacheSavesNineBuilds d _ q

/-- The measured post-fix wall is below the 5 ms target. -/
theorem measuredPostFixBelowTarget : measuredPostFixHostMicros < postFixTargetMicros := by
  decide

/-! ## Block reconstruction tune (P123)

`src/gpu/nn/train/block.rs` `BlockTune` tunes the four sign factors of one layer (`o`, `gate`, `up`,
`down`, 1.1456 b/w) against the teacher's block output on the student's own input stream:
`y = h + (silu(n·Wgᵀ) ⊙ n·Wuᵀ)·Wdᵀ`, `h = x + a·Woᵀ`, straight-through sign, Adam, 8 epochs over
35 × 512 calibration tokens, layer by layer. Units: millionths of nats (held-out KL, chunks 8–9) and
seconds of wall. -/

/-- Held-out KL against f16 (`kl-bt24-89.log`): ADMM factors alone, block-tuned, and Q4_K_M.
Grade `measured`. -/
def klAdmm : Nat := 4_083_521
def klBlockTuned : Nat := 1_264_101
def klQ4KM : Nat := 49_140

/-- Whole-trunk tune wall, 24 layers (`bt-tune-all.log`). Grade `measured`. -/
def tuneWallSecs : Nat := 236

/-- The block tune cuts KL 3.23×; it remains 25.7× above Q4_K_M; the 24-layer tune fits the 600 s
ceiling. -/
theorem blockTune_fit :
    klBlockTuned < klAdmm ∧ klAdmm * 100 / klBlockTuned = 323 ∧
    klQ4KM < klBlockTuned ∧ klBlockTuned * 10 / klQ4KM = 257 ∧ tuneWallSecs < 600 := by
  decide

/-! ## Device student stream (P124)

`src/gpu/nn/train/mixer.rs` `LayerFwd` runs the layer forward on the device, so the calibration set
grows to 256 C4 chunks at 4 epochs. Units as in P123. -/

/-- Held-out KL against f16 with the device stream and 256 C4 chunks (`kl-c4n256.log`).
Grade `measured`. -/
def klDeviceStream : Nat := 1_185_169

/-- The two resumable rounds, layers 0–11 and 12–23, in whole seconds (`c4n256-a.log`,
`c4n256-b.log`). Grade `measured`. -/
def deviceRoundSecs : Nat × Nat := (382, 388)

/-- The device stream cuts KL 1.066× below P123 and remains 24.1× above Q4_K_M. Each round fits
the 580 s round bound; the 769 s total is above the 600 s ceiling, an open defect of the run. -/
theorem blockTuneDevice_fit :
    klDeviceStream < klBlockTuned ∧ klBlockTuned * 1000 / klDeviceStream = 1066 ∧
    klDeviceStream * 10 / klQ4KM = 241 ∧ deviceRoundSecs.1 < 580 ∧ deviceRoundSecs.2 < 580 ∧
    600 < deviceRoundSecs.1 + deviceRoundSecs.2 := by
  decide

/-! ## Register-tiled tape GEMM (P125)

`src/gpu/nn/fwd/lin.rs` `tiled_gemm` serves the linear forward and both adjoints: a 128 × 128 output
tile per workgroup, a 16-wide reduction slice, 8 × 8 register accumulators per lane. Units: tens of
microseconds of device time at `(n, i, o) = (2048, 1024, 3584)`, means of 5 interleaved old/new
pairs, in the order forward, input adjoint, weight adjoint. -/

/-- Device time of the one-output-per-lane 16 × 16 body (P89/P92). Grade `measured`. -/
def gemmOldTenMicros : Nat × Nat × Nat := (384, 321, 362)

/-- Device time of the register-tiled body. Grade `measured`. -/
def gemmNewTenMicros : Nat × Nat × Nat := (66, 69, 64)

/-- Shared loads per FMA fall by `2·BM·BN / (16·(BM + BN))` = 8 at `BM = BN = 128`. -/
theorem tiledGemm_sharedCut : 2 * 128 * 128 / (16 * (128 + 128)) = 8 := by decide

/-- Every op is at least 4× faster at the production shape: the registered prediction. -/
theorem tiledGemm_fit :
    4 * gemmNewTenMicros.1 ≤ gemmOldTenMicros.1 ∧
    4 * gemmNewTenMicros.2.1 ≤ gemmOldTenMicros.2.1 ∧
    4 * gemmNewTenMicros.2.2 ≤ gemmOldTenMicros.2.2 := by
  decide

/-! ## Gated DeltaNet backward (P126)

`src/gpu/nn/train/mixer.rs` `DeltaBwd`: mode 0 recomputes the forward and stores every pre-update
state `M_{t-1}` (`T · NV · 128²` f32), mode 1 reads it back in the reverse scan. Units: microseconds
of device time at `T = 2048`, `NV = 16`, means of 5 runs on the live tree. -/

/-- Device time of the forward `delta_scan`. Grade `measured`. -/
def deltaFwdMicros : Nat := 1406

/-- Device time of the three backward modes. Grade `measured`. -/
def deltaBwdMicros : Nat := 15271

/-- Bytes of the state store, written once in mode 0 and read once in mode 1. -/
def deltaBwdStoreBytes : Nat := 2 * (2048 * 16 * 128 * 128 * 4)

/-- The registered 3× wall bound is refuted, and it lay below the design's own traffic floor:
the state store alone takes `4.29 GB / 960 GB/s` = 4473 µs, above `3 · 1406`. The backward
measures 10× the forward and 3× that floor. Correctness clauses are `tested` (P126). -/
theorem deltaBwd_fit :
    3 * deltaFwdMicros < deltaBwdMicros ∧
    deltaBwdStoreBytes / 960_000 = 4473 ∧
    3 * deltaFwdMicros < deltaBwdStoreBytes / 960_000 ∧
    10 * deltaFwdMicros ≤ deltaBwdMicros ∧ deltaBwdMicros < 11 * deltaFwdMicros ∧
    3 * (deltaBwdStoreBytes / 960_000) ≤ deltaBwdMicros := by
  decide

/-! ## Per-linear EM interleave of the block tune (P127)

`tests/learn_block_tune.rs` under `APEIRON_BT_SCHED=em`, `src/gpu/nn/train/block.rs` `set_stage`,
the schedule of NanoQuant `compress_model.py:83-101`: every slot of `o`/`gate`/`up`/`down` starts
dense at the teacher's weight; for each linear in turn, one epoch tunes the slots still dense, ADMM
factorises that linear from its tuned weight, and one epoch tunes the slots factorised so far. The
joint arm factorises all four first and tunes them together. Units: held-out block error ×10⁵ per
layer, layers 0–3, c4n256; wall in tenths of a second. -/

/-- Held-out block error of the joint arm, layers 0–3 (`c4n256-a.log:6-11`). Grade `measured`. -/
def emJointHeld : List Nat := [19744, 20142, 20521, 22642]

/-- Held-out block error of the EM arm, layers 0–3, same 256 chunks; one dense and one factor
epoch per linear (8 epochs) against the joint arm's 4. Grade `measured`. -/
def emInterleaveHeld : List Nat := [17437, 17110, 17606, 19661]

/-- Wall of the layers 0–3 round under the EM schedule on the live tree (lane runs 231.8 and
235.8 s). Grade `measured`. -/
def emRoundDeciseconds : Nat := 2393

/-- The registered prediction: layer 0 below the joint arm and the round inside 580 s. Each of
the four layers is at least 11% below its joint value (`em · 100 ≤ joint · 89`). -/
theorem emInterleave_fit :
    emInterleaveHeld.head! < emJointHeld.head! ∧
    emRoundDeciseconds ≤ 5800 ∧
    (List.zip emInterleaveHeld emJointHeld).all (fun p => p.1 * 100 ≤ p.2 * 89) := by
  decide

/-! ## Gated softmax attention-map backward (P128)

`src/gpu/nn/train/attn_bwd.rs` `AttnBwd`, shader `src/gpu/spirv/train/attn_map_bwd.comp`: the FA2
recompute-from-lse backward (Dao 2023, Algorithm 2), dq per query row, dk and dv per key column.
Units: microseconds of device time at `T = 2048`, `NH = 8`, `NKV = 2`, `HD = 256`, means of 5 runs
on the live tree (`lead-1024/l2a/live-timing.log`). -/

/-- Device time of the forward `attn_map`. Grade `measured`. -/
def attnFwdMicros : Nat := 2115

/-- Device time of the three backward modes. Grade `measured`. -/
def attnBwdMicros : Nat := 22044

/-- Matmul-equivalents per pass: the forward's QK and PV; the backward's mode 0 QK, PV, dA·V, dS·K
and mode 1 QK, dA·V, dSᵀ·Q, Pᵀ·dA. -/
def attnFwdMatmuls : Nat := 2
def attnBwdMatmuls : Nat := 8

/-- The registered wall band [3, 6] is refuted: the backward is 10.4× the forward, above the
falsifier's 8. Per matmul-equivalent the backward runs at least 2.5× slower than the forward, so the
gap is the backward kernel's rate, not its arithmetic. Correctness clauses are `tested` (P128). -/
theorem attnBwd_fit :
    8 * attnFwdMicros < attnBwdMicros ∧
    10 * attnFwdMicros ≤ attnBwdMicros ∧ attnBwdMicros < 11 * attnFwdMicros ∧
    5 * attnFwdMicros * attnBwdMatmuls ≤ 2 * attnBwdMicros * attnFwdMatmuls := by
  decide

/-! ## Device-resident block-tune streams (P129)

`src/gpu/nn/train/block.rs` `Stream`, `chunk_into`, `chunk_from`, `BlockTune::set_cols`,
`src/gpu/nn/train/mixer.rs` `LayerFwd::run_dev`: the activation, teacher and target streams of one
layer live on the device, and each step copies its chunks device to device inside the step's own
command buffer, one `submit_wait` per step. Units: hundredths of a second per layer, layer 0, mean of
3 runs, base 78de1af2 against the installed tree (`lead-1024/r/live-ga.log`, lane attribution). -/

/-- Host/device transfer terms of one layer before and after (gather, upload memcpy and submit,
forward download submit and memcpy, scatter, loss read). Grade `measured`. -/
def residentTransferBefore : Nat := 1440
def residentTransferAfter : Nat := 58

/-- Whole per-layer wall before and after; the ADMM term is not in this ledger. Grade `measured`. -/
def residentLayerBefore : Nat := 6230
def residentLayerAfter : Nat := 4050

/-- The registered layer wall of about 32 s is refuted: the measured 40.5 s keeps the tape's own
29 ms per step, so the excess over the prediction is compute, not transfer. The transfer term falls
at least 20-fold and the layer wall falls by at least a third. Output identity (32 of 32 factor
files, both state files) is `tested`, not proved. -/
theorem residentStreams_fit :
    20 * residentTransferAfter ≤ residentTransferBefore ∧
    residentLayerAfter < residentLayerBefore ∧
    3 * residentLayerAfter ≤ 2 * residentLayerBefore ∧
    3200 < residentLayerAfter := by
  decide

/-- Ungated softmax attention (Qwen3) on the device attention map: bit 2 of the `mode` word drops
the `sigmoid(gate)` factor from the attend stage and its gradient from the backward. Device errors
against the f64 reference at `T = 2048`, `NH = 16`, `NKV = 8`, `HD = 128`, in units of 1e-9, under
the 1e-3 bound. Grade `measured`; gated bit identity (`cmp` of the dump) is `tested`. -/
def ungatedFwdErrNano : Nat := 221
def ungatedDqfErrNano : Nat := 9195
def ungatedBoundNano : Nat := 1000000

theorem ungatedAttn_fit :
    ungatedFwdErrNano < ungatedBoundNano ∧
    ungatedDqfErrNano < ungatedBoundNano := by
  decide

/-- Same-harness reference arms on Qwen3-0.6B-Base (NanoQuant `evaluate_model`: wikitext-2 perplexity
and the mean of six 0-shot accuracies), in units of 1e-4 (accuracy) and 1e-3 (perplexity); P131.
Grade `measured`. NanoQuant as shipped at 1.18 b/w misses both bars against Q4_K_M: perplexity
above 1.10 x and accuracy below 0.90 x. -/
def q4kmPplMilli : Nat := 13763
def nqRefPplMilli : Nat := 24332
def q4kmAccE4 : Nat := 5556
def nqRefAccE4 : Nat := 4235

theorem nqReference_missesBars :
    10 * nqRefPplMilli > 11 * q4kmPplMilli ∧
    10 * nqRefAccE4 < 9 * q4kmAccE4 := by
  decide

/-- The device block tune for Qwen3 (P132). The ungated softmax face on the device layer forward:
relative error of the layer output against the CPU engine at layer 0 and of the attention output at
layer 3 of Qwen3-0.6B-Base, 512 columns, in units of 1e-5, under the 3e-2 bound; the control (mode bit
2 dropped) is above it. The step of `BlockTune` (o/gate/up/down, `T = 2048`) measures 14.8 ms against
a GEMM floor of 7.7 ms (N = 256, 64 steps, 2 runs), 1.38e5 token-passes per second; the reference's
whole-block loop runs about 2.9e5. Grade `measured`; the slot refactor's tape identity is `tested`. -/
def q3LayerOutErrE5 : Nat := 1556
def q3AttnOutErrE5 : Nat := 2475
def q3LayerBoundE5 : Nat := 3000
def q3ControlErrE5 : Nat := 97230
def blockStepMicro : Nat := 14800
def blockFloorMicro : Nat := 7700
def refTokPerS : Nat := 290000
def ourTokPerS : Nat := 138000

theorem qwen3BlockTune_fit :
    q3LayerOutErrE5 < q3LayerBoundE5 ∧
    q3AttnOutErrE5 < q3LayerBoundE5 ∧
    q3LayerBoundE5 < q3ControlErrE5 ∧
    blockFloorMicro < blockStepMicro ∧
    blockStepMicro < 2 * blockFloorMicro ∧
    2 * ourTokPerS < refTokPerS := by
  decide

/-- The o-proj input adjoint of `BlockTune` (P133): `da = dh·Wo`, one `linear_bwd_input_dispatch`
after the `dwt(0)` weight gradient, the gradient the q/k/v slots of the reference loop read
(`compress_model.py:83-101`). Worst relative error against f64 central differences of the host loss,
in units of 1e-9, under the 1e-4 bound; the control (`wt(1)` for `wt(0)`) is above it. The added
GEMM is `2·T·a·d` flops, 8.6 GFLOP at `T = 2048`, `a = 2048`, `d = 1024` (Qwen3-0.6B), on a step
floor of 161 GFLOP. Grade: the errors `tested`, the cost `model`. -/
def daErrNano : Nat := 179
def daBoundNano : Nat := 100000
def daControlErrNano : Nat := 3290000000
def daGflopTenths : Nat := 86
def blockFloorGflopTenths : Nat := 1610

theorem blockTuneAdjoint_fit :
    daErrNano < daBoundNano ∧
    daBoundNano < daControlErrNano ∧
    10 * daGflopTenths < blockFloorGflopTenths := by
  decide

/-- The q/k/v front of the device block tune (P134): `FrontTune` gradients of the q, k and v factor
slots against the f64 host reference, which itself matches central differences; worst relative
errors in units of 1e-9 under the 1e-3 bound, and the control (the k and v weight gradients swapped)
above it. The added GEMM work, `6·T·d·(nh + 2·nkv)·hd` at `T = 2048`, `d = 1024`, `nh = 16`,
`nkv = 8`, `hd = 128`, in tenths of a GFLOP. Grade: the errors `tested`, the cost `model`. -/
def frontFdErrNano : Nat := 549
def frontFdBoundNano : Nat := 1000
def frontQErrNano : Nat := 600
def frontKErrNano : Nat := 530
def frontVErrNano : Nat := 295
def frontBoundNano : Nat := 1000000
def frontControlErrNano : Nat := 2540000000
def frontGflopTenths : Nat := 515

theorem frontTune_fit :
    frontFdErrNano < frontFdBoundNano ∧
    frontQErrNano < frontBoundNano ∧
    frontKErrNano < frontBoundNano ∧
    frontVErrNano < frontBoundNano ∧
    frontBoundNano < frontControlErrNano ∧
    6 * 2048 * 1024 * (16 + 2 * 8) * 128 / 100000000 = frontGflopTenths := by
  decide

/-- The tensor-core training GEMMs (P135): `linear_{fwd,bwd_input,bwd_weight}_coop_dispatch` in
`src/gpu/nn/bwd/coop.rs`, bf16 operands and an f32 accumulator on `16x16x16` cooperative matrices.
Worst relative error over the 15 products (five Qwen3-0.6B shapes, three products) against an f64
host product, in units of 1e-6, under the 4e-3 bf16 bound; the control (the last K tile dropped)
above it. Device time of the 15 calls at `n = 2048`, in microseconds, f32 against coop, mean of 5
runs: the coop sum is under half the f32 sum. Grade: the errors `tested`, the times `measured`. -/
def coopErrMicro : Nat := 2106
def coopBoundMicro : Nat := 4000
def coopControlErrMicro : Nat := 176000
def coopF32Micros : Nat := 6849
def coopMicros : Nat := 2876

theorem trainCoopGemm_fit :
    coopErrMicro < coopBoundMicro ∧
    coopBoundMicro < coopControlErrMicro ∧
    2 * coopMicros < coopF32Micros := by
  decide

/-- The reference tune step on the device tapes (P136): NanoQuant's optimiser (`optimi`, betas
`(0.9, 0.99)`, eps `1e-6`, debiased-β form), its `CosineAnnealingLR` folded into the bias word by
`β₁ᵗ' = 1 − (1−β₁ᵗ)/c`, and its importance-weighted loss `Σ imp[c]·(y−yt)²` in `block_map` modes 4
and 5. Worst relative errors in units of 1e-9: `p − p0` over 256 scheduled steps against an f64
host replica of `optimi/adam.py:371-394` under 1e-4, with the control (`β₂ = 0.999`) above it; the
weighted `dy` against f64 under 1e-4, with the control (`D` dropped from mode 4) above it. Grade:
`tested`; no time is claimed. -/
def optimiErrNano : Nat := 22200
def optimiBoundNano : Nat := 100000
def optimiControlErrNano : Nat := 26300000
def impDyErrNano : Nat := 121
def impDyBoundNano : Nat := 100000
def impDyControlErrNano : Nat := 2640000000

theorem refTuneStep_fit :
    optimiErrNano < optimiBoundNano ∧
    optimiBoundNano < optimiControlErrNano ∧
    impDyErrNano < impDyBoundNano ∧
    impDyBoundNano < impDyControlErrNano := by
  decide

/-- The block-loop driver (P137): `tests/learn_block_tune.rs` `block_loop` on block 0, slots q and
v, factors injected from the reference's dump. Worst per-epoch deviation from the reference's own
`losses.json`, in per-mille, against the 10% band: q fact and v fact inside it, v nonfact outside
it (the refuted clause); the control (importance all ones) at 122× the reference. The ADMM arm's
recon norm in units of 1e-4, ours against the reference's. Grade: `measured`. -/
def driveBandMille : Nat := 100
def driveQFactMille : Nat := 27
def driveVFactMille : Nat := 13
def driveVNonfactMille : Nat := 193
def driveControlMille : Nat := 122000
def driveArmQ : Nat := 2595
def driveArmQRef : Nat := 2595
def driveArmV : Nat := 3042
def driveArmVRef : Nat := 3041

theorem blockLoopDrive_fit :
    driveQFactMille < driveBandMille ∧
    driveVFactMille < driveBandMille ∧
    driveBandMille < driveVNonfactMille ∧
    driveBandMille < driveControlMille ∧
    driveArmQ = driveArmQRef ∧
    driveArmV - driveArmVRef ≤ 1 := by
  decide

/-- The tiled attention backward (P138): `attn_map_bwd.comp` modes 0 and 1 on f16 cooperative
matrices with an f32 accumulator. Worst gradient error over the gated and ungated device tests
against f64, in units of 1e-6, under the 5e-3 bound of the precision verdict; the control (mode 0's
last key tile dropped) above it. Device time per call at `T = 2048`, `n_ctx = 1024`, `nh 16`, `nkv
8`, `hd 128`, in microseconds, mean of 5 runs: before 58160, after 8280, under the 10 ms
prediction and more than 7× faster. Grade: the errors `tested`, the times `measured`. -/
def attnBwdErrMicro : Nat := 1176
def attnBwdBoundMicro : Nat := 5000
def attnBwdControlErrMicro : Nat := 997800
def attnBwdBeforeMicros : Nat := 58160
def attnBwdAfterMicros : Nat := 8280
def attnBwdPredMicros : Nat := 10000

theorem attnBwdTile_fit :
    attnBwdErrMicro < attnBwdBoundMicro ∧
    attnBwdBoundMicro < attnBwdControlErrMicro ∧
    attnBwdAfterMicros < attnBwdPredMicros ∧
    7 * attnBwdAfterMicros < attnBwdBeforeMicros := by
  decide

/-- One 2048-token attention chunk (P139): `attn_map.comp` holds `MAX_N_CTX = 2048` scores, the
reference's calibration sample whole (`NanoQuant` `main.py:38`). Teacher relative error to the
reference's `b0/yt0.bf16` in units of 1e-6: 34000 at two 1024-token halves, 5203 at one chunk, under
the 1e-2 prediction. Worst ungated attention-backward gradient at `n_ctx = 2048` under the 5e-3
bound; device time per call 14830 µs, inside the predicted 14000–20000 µs. Block-0 per-epoch loss
gap to the reference, in per-mille: q fact 39 and v fact 3 inside the 100 band, v nonfact 179
outside it (our f32 update below the reference's bf16 one). Grade: errors and losses `tested`,
times `measured`. -/
def nctxTeacherBeforeMicro : Nat := 34000
def nctxTeacherAfterMicro : Nat := 5203
def nctxTeacherPredMicro : Nat := 10000
def nctxBwdErrMicro : Nat := 921
def nctxBwdMicrosLo : Nat := 14000
def nctxBwdMicros : Nat := 14830
def nctxBwdMicrosHi : Nat := 20000
def nctxQFactMille : Nat := 39
def nctxVFactMille : Nat := 3
def nctxVNonfactMille : Nat := 179

theorem nctx2048_fit :
    nctxTeacherAfterMicro < nctxTeacherPredMicro ∧
    6 * nctxTeacherAfterMicro < nctxTeacherBeforeMicro ∧
    nctxBwdErrMicro < attnBwdBoundMicro ∧
    nctxBwdMicrosLo < nctxBwdMicros ∧ nctxBwdMicros < nctxBwdMicrosHi ∧
    nctxQFactMille < driveBandMille ∧
    nctxVFactMille < driveBandMille ∧
    driveBandMille < nctxVNonfactMille := by
  decide

/-- The chained input adjoint (P140): `FrontTune` computes `dn1 = dqf·Wq + dkf·Wk + dv·Wv`, then
`dx = rmsnorm_bwd(x, g1, dn1) + dres` with `dres` the block's `dh`, so the KD cotangent reaches every
layer's scales (`NanoQuant` `compress_model.py:213-230`). Two chained layers against f64 central
differences, relative error in units of 1e-6: `dx0` worst 361 over 8 indices and the layer-0 scale
gradients worst whole-vector 663, both under the 1000 prediction; the control (the `dx += dres` add
deleted) 1420000. Grade: `tested`. -/
def kdChainDxMicro : Nat := 361
def kdChainScaleMicro : Nat := 663
def kdChainBoundMicro : Nat := 1000
def kdChainControlMicro : Nat := 1420000

theorem kdChain_fit :
    kdChainDxMicro < kdChainBoundMicro ∧
    kdChainScaleMicro < kdChainBoundMicro ∧
    kdChainBoundMicro < kdChainControlMicro := by
  decide

end Perf.Train
