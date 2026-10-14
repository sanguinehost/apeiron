/-
Apeiron — column-cover reduction, quantizer locality and the split FFN.

section "L8 · Exact tensor-parallel layer"; task
. Pure integer mathematics over `ℤ` (Mathlib); no extraction and no Rust
dependency. The module proves the three laws the column/row-paired tensor-parallel layer relies
on (Megatron-LM pairing applied across cores):

* `column_cover_round` — one rounding applied to a sum over an arbitrary block partition equals
  the rounding applied to the slice-partitioned sum, `r (Σ_b acc_b) = r (Σ_i Σ_{b ∈ S_i} acc_b)`,
  with the per-row block form (`column_cover_row`) and the every-output-row matrix form
  (`column_cover_matrix`). A slice `S_i = {b | part b = i}` is the fibre of any `part : Fin n → Fin k`,
  so every partition into slices is covered; alignment is automatic because blocks, not scalar
  positions, are partitioned.
* `quant_locality` / `dequant_locality` — blockwise quantization of a concatenation of 32-aligned
  lists is the concatenation of the blockwise quantizations, `quantChunks (l ++ m) = quantChunks l ++
  quantChunks m` when both lengths are multiples of `32`; the same for `dequantChunks`. The per-block
  operation is `ChartDot.quantBlock` / `ChartDot.dequantBlock`, reused, not restated.
* `split_ffn_exact` — for an elementwise activation `σ`, gate/up row values `Gx`, `Ux`, down matrix
  `D`, a blockwise quantizer `Q = blockQuant φ` and one rounding `r`,
  `r (D · Q(σ(Gx, Ux))) = r (Σ_i D[:, S_i] · Q(σ(G_{S_i}x, U_{S_i}x)))` for every output row, for any
  partition of the hidden blocks. `Gx`, `Ux` are the exact integer row values (`rowVal G x`), the
  values the engine has already rounded each row to; the concrete quantizer (`qbDequant`) is the
  blockwise reading of `ChartDot.dequantBlock`, the same per-block map as statement two.

Trust boundary: this is spec-level integer algebra. It does not model the `i128` container, the
`round_shift_32` assembly, thread scheduling, or rustc/LLVM codegen. The two `#guard` witnesses are
kernel-evaluated concrete instances, not extracted-body evaluation.

The axiom cone is `{propext, Classical.choice, Quot.sound}`; no `sorry`, `axiom`, `opaque`,
`native_decide`, `bv_decide`, `decide`/`rfl` over a large expression, or raised heartbeat bound
appears. `decide` is not used; the witnesses use `#guard` on small closed literals.
-/
import Numeric.Matvec.ChartDot

set_option autoImplicit false
set_option warningAsError true

namespace ColumnCover

/-! ## Statement 1 — column cover under a single rounding -/

/-- **Fibre sum.** Summing the block contributions `acc` over all blocks equals summing first over
the slices `S_i = {b | part b = i}` of any partition `part : Fin n → Fin k`. Every partition into
slices arises this way (label the slices), so the statement ranges over all partitions. -/
theorem sum_fiber_part {n k : ℕ} (part : Fin n → Fin k) (acc : Fin n → ℤ) :
    (∑ b : Fin n, acc b)
      = ∑ i : Fin k, ∑ b ∈ (Finset.univ.filter (fun b : Fin n => part b = i)), acc b := by
  exact (Finset.sum_fiberwise_of_maps_to (s := Finset.univ) (t := Finset.univ)
    (g := part) (fun x _ => Finset.mem_univ _) acc).symm

/-- **Column cover, one rounding.** For any rounding `r : ℤ → ℤ` applied once, the rounded whole sum
equals the rounded slice-partitioned sum. The match is exact: there is no per-slice rounding. -/
theorem column_cover_round {n k : ℕ} (part : Fin n → Fin k) (acc : Fin n → ℤ) (r : ℤ → ℤ) :
    r (∑ b : Fin n, acc b)
      = r (∑ i : Fin k, ∑ b ∈ (Finset.univ.filter (fun b : Fin n => part b = i)), acc b) := by
  rw [sum_fiber_part part acc]

/-- **Column cover, one row.** The blocks are the `nb` width-32 groups of the row `w`; block `b`'s
contribution is the exact partial `Σ_p w b p · x b p`. -/
theorem column_cover_row {nb k : ℕ} (w x : Fin nb → Fin 32 → ℤ)
    (part : Fin nb → Fin k) (r : ℤ → ℤ) :
    r (∑ b : Fin nb, ∑ p : Fin 32, w b p * x b p)
      = r (∑ i : Fin k, ∑ b ∈ (Finset.univ.filter (fun b : Fin nb => part b = i)),
            ∑ p : Fin 32, w b p * x b p) := by
  exact column_cover_round part (fun b => ∑ p : Fin 32, w b p * x b p) r

/-- **Column cover, every output row.** The row form applied to each row `W o` of an `m`-row matrix
`W`, so every output row's single rounding is the slice reduction. -/
theorem column_cover_matrix {m nb k : ℕ} (W : Fin m → Fin nb → Fin 32 → ℤ)
    (x : Fin nb → Fin 32 → ℤ) (part : Fin nb → Fin k) (r : ℤ → ℤ) :
    ∀ o : Fin m,
      r (∑ b : Fin nb, ∑ p : Fin 32, W o b p * x b p)
        = r (∑ i : Fin k, ∑ b ∈ (Finset.univ.filter (fun b : Fin nb => part b = i)),
              ∑ p : Fin 32, W o b p * x b p) := by
  intro o
  exact column_cover_row (W o) x part r

/-! ## Statement 2 — blockwise quantizer locality -/

/-- Split a list into consecutive width-`32` groups; when `32 ∣ l.length` every group has length
exactly `32`. Specification-level: each group is read by `take`/`drop`, so this definition is
`O((l.length/32)²)` list operations, not the `O(l.length)` running loop; it is not on the executing
path. -/
def chunk32 (l : List ℤ) : List (List ℤ) :=
  (List.range ((l.length + 31) / 32)).map (fun i => (l.drop (i * 32)).take 32)

/-- Blockwise activation quantization: `ChartDot.quantBlock` on each width-`32` group, concatenated,
`O(l.length)` times the per-block cost. -/
def quantChunks (l : List ℤ) : List ℤ :=
  ((chunk32 l).map ChartDot.quantBlock).flatten

/-- Blockwise activation dequantization: `ChartDot.dequantBlock` on each width-`32` group,
concatenated. This is the `Q` of the split FFN and of the Q8 activation fast path. -/
def dequantChunks (l : List ℤ) : List ℤ :=
  ((chunk32 l).map ChartDot.dequantBlock).flatten

/-- Chunking a concatenation whose left length is a multiple of `32` splits at that boundary: the
chunk list of `l ++ m` is the chunk list of `l` followed by the chunk list of `m`. (Both lengths are
assumed multiples of `32`; the right hypothesis is only needed for the `m`-side chunk count.) -/
theorem chunk32_append (l m : List ℤ) (hl : l.length % 32 = 0) (hm : m.length % 32 = 0) :
    chunk32 (l ++ m) = chunk32 l ++ chunk32 m := by
  obtain ⟨nl, hnl⟩ : ∃ nl, l.length = 32 * nl := ⟨l.length / 32, by omega⟩
  obtain ⟨nm, hnm⟩ : ∃ nm, m.length = 32 * nm := ⟨m.length / 32, by omega⟩
  have hlen : (l ++ m).length = 32 * (nl + nm) := by rw [List.length_append, hnl, hnm]; ring
  have hcl : (l.length + 31) / 32 = nl := by rw [hnl]; omega
  have hcm : (m.length + 31) / 32 = nm := by rw [hnm]; omega
  have hcall : ((l ++ m).length + 31) / 32 = nl + nm := by rw [hlen]; omega
  unfold chunk32
  rw [hcall, hcl, hcm, List.range_add, List.map_append]
  congr 1
  · apply List.map_congr_left
    intro i hi
    have hi' : i < nl := by simpa using hi
    have hdrop : (l ++ m).drop (i * 32) = l.drop (i * 32) ++ m := by
      apply List.drop_append_of_le_length
      omega
    rw [hdrop]
    apply List.take_append_of_le_length
    rw [List.length_drop]
    omega
  · rw [List.map_map]
    apply List.map_congr_left
    intro j _hj
    dsimp only [Function.comp_apply]
    have h1 : (nl + j) * 32 = l.length + j * 32 := by rw [hnl]; ring
    rw [h1, List.drop_length_add_append]

/-- **Quantizer locality.** Quantizing a concatenation of `32`-aligned lists blockwise equals
concatenating the blockwise quantizations of the pieces. -/
theorem quant_locality (l m : List ℤ) (hl : l.length % 32 = 0) (hm : m.length % 32 = 0) :
    quantChunks (l ++ m) = quantChunks l ++ quantChunks m := by
  unfold quantChunks
  rw [chunk32_append l m hl hm, List.map_append, List.flatten_append]

/-- **Dequantizer locality.** The same for `dequantBlock`. -/
theorem dequant_locality (l m : List ℤ) (hl : l.length % 32 = 0) (hm : m.length % 32 = 0) :
    dequantChunks (l ++ m) = dequantChunks l ++ dequantChunks m := by
  unfold dequantChunks
  rw [chunk32_append l m hl hm, List.map_append, List.flatten_append]

/-! ## Statement 3 — the split FFN -/

/-- The per-block activation dequantizer on a width-`32` block: `ChartDot.scale` of the block,
`ChartDot.qcode` per lane, times the scale. -/
def qbDequant (w : Fin 32 → ℤ) (p : Fin 32) : ℤ :=
  ChartDot.scale (List.ofFn w) * ChartDot.qcode (ChartDot.scale (List.ofFn w)) (w p)

/-- `qbDequant` is exactly `ChartDot.dequantBlock` read off `List.ofFn`; the block quantizer of the
FFN is the same per-block operation as `dequantChunks`. -/
theorem ofFn_qbDequant (w : Fin 32 → ℤ) :
    List.ofFn (qbDequant w) = ChartDot.dequantBlock (List.ofFn w) := by
  unfold qbDequant ChartDot.dequantBlock ChartDot.quantBlock
  rw [List.map_map]
  exact List.ofFn_comp' w
    (fun x => ChartDot.scale (List.ofFn w) * ChartDot.qcode (ChartDot.scale (List.ofFn w)) x)

/-- Apply a per-block operation `φ` blockwise to an `ι`-indexed vector of width-`32` blocks. -/
def blockQuant {ι : Type*} (φ : (Fin 32 → ℤ) → Fin 32 → ℤ) (v : ι → Fin 32 → ℤ) :
    ι → Fin 32 → ℤ :=
  fun b => φ (v b)

/-- The exact gate/up row values `M·x`: row `(b,p)` of projection `M` dotted with the input `x`.
These are the quotient-free integer row values the engine computes and rounds per row before the
elementwise activation. -/
def rowVal {nb e : ℕ} (M : Fin nb → Fin 32 → Fin e → ℤ) (x : Fin e → ℤ) :
    Fin nb → Fin 32 → ℤ :=
  fun b p => ∑ j : Fin e, M b p j * x j

/-- **Split FFN, exact.** Hidden dimension split into `nb` width-`32` blocks; `σ` is the elementwise
activation of the exact gate/up row values; `φ` the per-block quantizer; `r` the single rounding.
The unsplit FFN `r (D · Q(σ(Gx, Ux)))` equals the slice reduction
`r (Σ_i D[:, S_i] · Q(σ(G_{S_i}x, U_{S_i}x)))` for every output row and every partition of the
hidden blocks. `Q = blockQuant φ` is blockwise, so a slice's `Q(σ(G_{S_i}x, U_{S_i}x))` is the
restriction of `Q(σ(Gx, Ux))` to `S_i`: quantizing a slice reads only that slice's blocks. -/
theorem split_ffn_exact {nb k e de : ℕ}
    (σ : ℤ → ℤ → ℤ)
    (G U : Fin nb → Fin 32 → Fin e → ℤ) (x : Fin e → ℤ)
    (D : Fin de → Fin nb → Fin 32 → ℤ)
    (φ : (Fin 32 → ℤ) → Fin 32 → ℤ) (part : Fin nb → Fin k) (r : ℤ → ℤ) (o : Fin de) :
    r (∑ b : Fin nb, ∑ p : Fin 32,
        D o b p * blockQuant φ (fun b p => σ (rowVal G x b p) (rowVal U x b p)) b p)
      = r (∑ i : Fin k, ∑ b ∈ (Finset.univ.filter (fun b : Fin nb => part b = i)),
            ∑ p : Fin 32,
              D o b p * blockQuant φ (fun b p => σ (rowVal G x b p) (rowVal U x b p)) b p) :=
  column_cover_round part
    (fun b => ∑ p : Fin 32,
      D o b p * blockQuant φ (fun b p => σ (rowVal G x b p) (rowVal U x b p)) b p) r

/-- The concrete quantizer instance: `φ = qbDequant`, the FFN's blockwise Q8 activation dequantizer
(the same per-block map as `dequantChunks`). -/
theorem split_ffn_exact_dequant {nb k e de : ℕ}
    (σ : ℤ → ℤ → ℤ)
    (G U : Fin nb → Fin 32 → Fin e → ℤ) (x : Fin e → ℤ)
    (D : Fin de → Fin nb → Fin 32 → ℤ) (part : Fin nb → Fin k) (r : ℤ → ℤ) (o : Fin de) :
    r (∑ b : Fin nb, ∑ p : Fin 32,
        D o b p * qbDequant (fun p' => σ (rowVal G x b p') (rowVal U x b p')) p)
      = r (∑ i : Fin k, ∑ b ∈ (Finset.univ.filter (fun b : Fin nb => part b = i)),
            ∑ p : Fin 32,
              D o b p * qbDequant (fun p' => σ (rowVal G x b p') (rowVal U x b p')) p) :=
  split_ffn_exact σ G U x D qbDequant part r o

/-! ## Joint non-vacuity witness

Hidden dimension `64 = 2` blocks of `32`; `n_embd = 2`; the partition is the two one-block slices.
The activation is `σ g u = g·u`, the dequantizer is `qbDequant`, the rounding is the identity. `G`,
`U`, `D` and `x` are chosen so both sides are equal and non-zero (`#guard` below evaluates the
closed instance; `#guard` fails to elaborate if the equality or the non-zero test is false). -/

namespace Witness

/-- Gate projection: row `(b,p)` has `x₀`-coefficient `b+1`, `x₁`-coefficient `1`. -/
def G : Fin 2 → Fin 32 → Fin 2 → ℤ :=
  fun b _p j => if j = 0 then (b.val : ℤ) + 1 else 1

/-- Up projection: row `(b,p)` has `x₀`-coefficient `1`, `x₁`-coefficient `p+1`. -/
def U : Fin 2 → Fin 32 → Fin 2 → ℤ :=
  fun _b p j => if j = 1 then (p.val : ℤ) + 1 else 1

/-- Input `[1, 2]`. -/
def x2 : Fin 2 → ℤ := ![1, 2]

/-- Down projection: output row `o`, hidden block `b`, lane `p` has weight
`(o+1)·(b+1)·(p mod 3 − 1)`, so the column-cover sum is not trivially zero. -/
def D : Fin 2 → Fin 2 → Fin 32 → ℤ :=
  fun o b p => ((o.val : ℤ) + 1) * ((b.val : ℤ) + 1) * (((p.val % 3 : ℕ) : ℤ) - 1)

/-- `σ g u = g·u`. -/
def σ2 : ℤ → ℤ → ℤ := fun g u => g * u

/-- The identity rounding. -/
def r2 : ℤ → ℤ := fun z => z

/-- The two one-block slices. -/
def part2 : Fin 2 → Fin 2 := fun b => b

/-- The activation at block `b`: `σ (Gx) (Ux)`, reading only block `b`'s rows. -/
def act (b : Fin 2) : Fin 32 → ℤ :=
  fun p => σ2 (rowVal G x2 b p) (rowVal U x2 b p)

/-- The unsplit FFN: one rounding of `D · Q(σ(Gx, Ux))`. -/
def unsplit (o : Fin 2) : ℤ :=
  ∑ b : Fin 2, ∑ p : Fin 32, D o b p * qbDequant (act b) p

/-- The split FFN: the slice reduction of the per-slice `D[:, S_i] · Q(σ(G_{S_i}x, U_{S_i}x))`. -/
def split (o : Fin 2) : ℤ :=
  ∑ i : Fin 2, ∑ b ∈ (Finset.univ.filter (fun b : Fin 2 => part2 b = i)),
    ∑ p : Fin 32, D o b p * qbDequant (act b) p

/-- The instance is a case of `split_ffn_exact_dequant`. -/
theorem split_eq (o : Fin 2) : unsplit o = split o :=
  split_ffn_exact_dequant σ2 G U x2 D part2 r2 o

-- Both sides evaluate equal on the closed instance.
#guard (List.ofFn unsplit) == (List.ofFn split)

-- The value is not trivially zero.
#guard (List.ofFn unsplit) != [0, 0]

-- The concrete output vector, with the identity rounding `r2`.
#guard (List.ofFn unsplit) == [-274, -548]

end Witness

/-! ## Non-vacuity of the quantizer locality on concrete aligned lists -/

namespace WitnessQuant

/-- `32` lanes. -/
def l32 : List ℤ := (List.range 32).map (fun i => (i : ℤ) - 16)

/-- `64` lanes. -/
def m64 : List ℤ := (List.range 64).map (fun i => (i : ℤ) * 3 - 40)

-- Locality holds on the concrete instance, and the pieces are non-empty.
#guard quantChunks (l32 ++ m64) == quantChunks l32 ++ quantChunks m64

-- Dequantizer locality on the same instance.
#guard dequantChunks (l32 ++ m64) == dequantChunks l32 ++ dequantChunks m64

-- The instance is not vacuous: neither side is empty.
#guard (quantChunks (l32 ++ m64)).length == 96

/-- A concrete width-`32` block with negative and positive lanes. -/
def w32 : Fin 32 → ℤ := fun p => (p.val : ℤ) * 5 - 70

-- The FFN's per-block quantizer `qbDequant` is the statement-two blockwise `dequantBlock`.
#guard dequantChunks (List.ofFn w32) == List.ofFn (qbDequant w32)

end WitnessQuant

end ColumnCover
