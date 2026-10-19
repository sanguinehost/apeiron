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

end Perf.Train
