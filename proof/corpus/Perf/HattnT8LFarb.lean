import Perf.HattnT8Layer

/-!
# HATTN-T8 per-layer far budget at 524288 on Qwen2.5-7B-Instruct-1M

`docs/evidence/hattn-decode-tail.md` (per-layer far budget section): `grp-fix2048-r8` at `t = 524288` with the far
budget of chosen layers replaced (`KV_TAIL_LFARB=lo-hi:n`, `lfarb.sh`, `lfarb2.sh`, outputs `g15-t524288-<run>.out`),
instrument `attn_cut_probe.hd128qb`. `B1`: layers 0-6 at 16384; `B2`: layers 0-6 at 65536; `B3`: layers 0-6 and 14-20
at 16384; `C1`: layers 0-6 at 524288 (every page selected). Other layers keep 2048. Mean whole-model KLD and reads/t
in ten-thousandths. Predictions: `PREDICTIONS-T8-LFARB.md`, sha256 `f435a574…`, addendum `88f08eab…`. rustc, LLVM, the
f32 dump kernels and the instrument are outside every statement; one prompt, one model.
-/

namespace Perf

namespace HattnT8LFarb

/-- Mean KLD of `B1`, `B2`, `B3`, `C1`. -/
def kld : List ℕ := [2799, 2091, 2613, 3071]

/-- Mean reads/t of `B1`, `B2`, `B3`, `C1`. -/
def reads : List ℕ := [113, 347, 181, 2534]

/-- `Perf.HattnT8Layer.prefixes` entry for layers 0-6 dense (`E3`) and its reads/t. -/
def e3 : ℕ := HattnT8Layer.prefixes.getD 2 0

def e3Reads : ℕ := 2534

/-- **N1 held.** `B1` is in `[0.15, 0.30]`. -/
theorem n1_held : 1500 ≤ kld.getD 0 0 ∧ kld.getD 0 0 ≤ 3000 := by decide

/-- **N2 held.** `B2 < B1` and `B2 ≥ 0.10`. -/
theorem n2_held : kld.getD 1 0 < kld.getD 0 0 ∧ 1000 ≤ kld.getD 1 0 := by decide

/-- **N3 held.** `B3 < B1`. -/
theorem n3_held : kld.getD 2 0 < kld.getD 0 0 := by decide

/-- **N4 held.** None of `B1`, `B2`, `B3` passes the `0.05` gate. -/
theorem n4_held : ∀ k ∈ kld.take 3, 500 < k := by decide

/-- **N5 held.** `B1` reads under `0.30` of dense. -/
theorem n5_held : reads.getD 0 0 < 3000 := by decide

/-- **C1 held.** Selecting every page in layers 0-6 reproduces the dense-layer run `E3`: same KLD and same reads/t. -/
theorem c1_held : kld.getD 3 0 = e3 ∧ reads.getD 3 0 = e3Reads := by decide

/-- **Finding: exact early layers are not the best early layers.** Both `B1` (16384) and `B2` (65536) are below `C1`
(exact layers 0-6, `0.3071`), and the error falls with the early budget only from `0.2799` to `0.2091`, four times
the gate, at 1/8 of the context. -/
theorem nonmonotone : kld.getD 0 0 < kld.getD 3 0 ∧ kld.getD 1 0 < kld.getD 3 0 ∧ 4 * 500 < kld.getD 1 0 := by
  decide

end HattnT8LFarb

end Perf
