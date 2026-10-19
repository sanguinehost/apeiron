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

end Perf.Train
