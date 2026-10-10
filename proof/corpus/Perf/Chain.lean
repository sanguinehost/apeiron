import Perf.Roofline

/-!
# The dependent dispatch chain and its knee

`Perf.Roofline` bounds one kernel by its demand on the binding resource and gives a chain
(`r.depth * m.latency`) only as a depth times a per-step latency; `Perf.Concurrency` amortizes a
serial stall over independent in-flight chains. Neither bounds a **sequence of strictly dependent
dispatches**, where every dispatch pays a floor of its own and the times add, because concurrency is
one. This module adds that axis as a model over `Nat` byte counts and a declared capacity.

A dispatch of demand `W` bytes takes at least `dispLB c L W = max (ceilDiv W c) L`, with `c` the
streaming capacity in bytes per ns and `L` the per-dispatch floor in ns (launch, barrier and the
dependent memory phases). Strictly dependent dispatches add:
`chainLB c L ws = (ws.map (dispLB c L)).sum`. The **knee** is `W* = c * L`: at `W ≤ W*` the ceiling
`ceilDiv W c` is at most `L`, so the dispatch costs its floor regardless of `W`; above it the
dispatch costs `ceilDiv W c`.

## What is proved here and what is not

* **Proved** under the closed axiom cone `{propext, Classical.choice, Quot.sound}` (the same cone
  as `Perf.Roofline`, to be confirmed by the lead's `#audit` lines): `chain_ge_bytes` (the chain never
  beats the byte roofline, and is at least `#dispatches * L`), `knee` (`dispLB = L ↔ W ≤ c * L`),
  `chain_tax` (the exact decomposition `chainLB = Σ ceilDiv + Σ (L - ceilDiv)`), `fuse_le` (fusing two
  adjacent dispatches never raises the bound, and lowers it by at most `L`), `fuse_saves_L` (fusing two
  dispatches whose sum is below the knee saves exactly `L`), `merge_independent` (batching `k`
  independent same-shape dispatches never raises the bound) and `chain_fuse_ratio` (a below-knee chain
  fused in groups of `g` has its bound divided by exactly `g`).
* **Not proved.** No hardware speed follows. `c` and `L` are declared `model` parameters, not
  measurements, and no theorem here produces a byte, a cycle or a microsecond. rustc/LLVM codegen, the
  GPU/NPU drivers, launch and barrier costs, the memory system's real phases and the demand counts of
  the closing examples are outside every statement. The concrete decode numbers below are `model`
  inputs of this module, read from `docs/evidence/perf-predictions.md` P2 provenance and the
  `DECODE-ATTRIB` receipt; they are not re-measured here.

## The shape of "decode is latency-bound", decided by the theorems

**(a) The shape (cite `chain_ge_bytes`, `chain_tax`, `knee`).** A token is a chain of dependent
dispatches, so `chainLB` is a *sum*, not a max: it is the sum over dispatches of
`max (ceilDiv W c) L`. For every dispatch below the knee (`W ≤ c * L`) the cost is the floor `L`
regardless of `W`, so `chainLB ≥ #dispatches * L` (`chain_ge_bytes`) and the gap above the byte
roofline `ceilDiv (ΣW) c` is the tax `Σ (L - ceilDiv W c)` (`chain_tax`). "Decode is latency-bound"
means the byte roofline is not the binding term for the executed body: the *dispatch count* times the
floor is, and each small kernel's bytes contribute at most their own floor.

**(b) Where a byte-reducing move stops paying (cite `knee`).** A byte-reducing move (for example Q8 to
Q5_0) lowers `W`; `dispLB` responds only above the knee `W > W* = c * L`, where `dispLB = ceilDiv W c`
and lowers with `W`. At or below the knee `dispLB = L`, so a byte reduction changes the bound by
nothing. With the model inputs below (`c = 865`, `L = 4000`, `W* = 3_460_000`) a byte-reducing move
pays on `gu`, `dn` and `lm_head`, and is inert on `qkv` and `o`.

**(c) Which move helps a below-knee chain (cite `chain_fuse_ratio`, `fuse_saves_L`, `knee`).** Fewer
dispatches. Below the knee the cost is `L` per dispatch independent of `W`; fusing a below-knee pair
saves exactly `L` (`fuse_saves_L`), and fusing a below-knee chain of `m` dispatches in groups of `g`
(each group still below the knee) divides `chainLB` by exactly `g` (`chain_fuse_ratio`). A byte
reduction on the same below-knee dispatch changes nothing (`knee`). Above the knee the picture
reverses: fusion saves at most `L` per fused pair and no ratio (`fuse_le`), so the move there is a byte
reduction.
-/

namespace Perf

namespace Chain

/-! ## 1 · The dispatch floor and the dependent chain -/

/-- The lower bound of one dispatch: `max (ceilDiv W c) L`, the larger of its byte time and its
per-dispatch floor. `c` is a capacity in bytes per ns and `L` a floor in ns; both are declared
`model` inputs. The only non-O(1) quantity is `ceilDiv`, which is `Nat` division. -/
def dispLB (c L W : Nat) : Nat := max (ceilDiv W c) L

/-- The lower bound of a chain of strictly dependent dispatches: the sum of the per-dispatch floors.
The temporal cost is linear in `ws.length`; each `dispLB` is one `Nat` division. -/
def chainLB (c L : Nat) (ws : List Nat) : Nat := (ws.map (dispLB c L)).sum

/-- The per-dispatch floor's shortfall: `L - ceilDiv W c`, the amount by which the floor exceeds the
byte time. It is zero exactly when the byte time already carries the floor. -/
def dispTax (c L W : Nat) : Nat := L - ceilDiv W c

/-- The chain's tax: the sum of the per-dispatch shortfalls. -/
def chainTax (c L : Nat) (ws : List Nat) : Nat := (ws.map (dispTax c L)).sum

/-! ## 2 · The ceiling arithmetic the chain needs -/

/-- `ceilDiv a c ≤ B` from `a ≤ B * c`, for `c > 0`: the backward direction of the threshold. -/
theorem ceilDiv_le_of_le_mul {a c B : Nat} (hc : 0 < c) (h : a ≤ B * c) : ceilDiv a c ≤ B := by
  unfold ceilDiv
  calc (a + c - 1) / c ≤ (B * c + (c - 1)) / c := by
        apply Nat.div_le_div_right
        have h3 : a + c - 1 ≤ B * c + c - 1 :=
          Nat.sub_le_sub_right (Nat.add_le_add_right h c) 1
        have h4 : B * c + c - 1 = B * c + (c - 1) := by omega
        omega
    _ = B := by
        apply Nat.div_eq_of_lt_le
        · omega
        · rw [Nat.add_mul, Nat.one_mul]; omega

/-- `ceilDiv`'s threshold: `ceilDiv a c ≤ B` exactly when `a ≤ c * B`, for `c > 0`. The
`⌈a/c⌉ ≤ B ↔ a ≤ c·B` adjunction, proved here from `le_ceilDiv_mul`. -/
theorem ceilDiv_le_iff_mul {a c B : Nat} (hc : 0 < c) : ceilDiv a c ≤ B ↔ a ≤ c * B := by
  constructor
  · intro h
    calc a ≤ ceilDiv a c * c := le_ceilDiv_mul hc
      _ ≤ B * c := Nat.mul_le_mul_right c h
      _ = c * B := Nat.mul_comm _ _
  · intro h
    exact ceilDiv_le_of_le_mul hc (by rw [Nat.mul_comm c B] at h; exact h)

/-- For `A = ceilDiv a c`, `A ≥ 1` implies `(A - 1) * c < a`: the byte time strictly exceeds the
largest multiple of `c` below it. -/
theorem ceilDiv_pred_mul_lt {a c : Nat} (hc : 0 < c) (ha : 1 ≤ ceilDiv a c) :
    (ceilDiv a c - 1) * c < a := by
  by_contra h
  rw [not_lt] at h
  have : ceilDiv a c ≤ ceilDiv a c - 1 := ceilDiv_le_of_le_mul hc h
  omega

/-- `k ≤ ceilDiv n c` from `(k - 1) * c < n`, for `c > 0`: the lower threshold of the ceiling. -/
theorem le_ceilDiv_of_lt_mul {n c k : Nat} (hc : 0 < c) (h : (k - 1) * c < n) :
    k ≤ ceilDiv n c := by
  by_contra hk
  rw [not_le] at hk
  have hk' : ceilDiv n c ≤ k - 1 := by omega
  have h2 : n ≤ (k - 1) * c :=
    calc n ≤ ceilDiv n c * c := le_ceilDiv_mul hc
      _ ≤ (k - 1) * c := Nat.mul_le_mul_right c hk'
  omega

/-- `ceilDiv` is subadditive: `ceilDiv (a + b) c ≤ ceilDiv a c + ceilDiv b c`, for `c > 0`. The sum
of the per-dispatch byte times never falls below the byte time of the fused demand. -/
theorem ceilDiv_add_le {c a b : Nat} (hc : 0 < c) :
    ceilDiv (a + b) c ≤ ceilDiv a c + ceilDiv b c := by
  apply ceilDiv_le_of_le_mul hc
  calc a + b ≤ ceilDiv a c * c + ceilDiv b c * c :=
        Nat.add_le_add (le_ceilDiv_mul hc) (le_ceilDiv_mul hc)
    _ = (ceilDiv a c + ceilDiv b c) * c := (Nat.add_mul _ _ _).symm

/-- `ceilDiv`'s subadditivity slack is at most one:
`ceilDiv a c + ceilDiv b c ≤ ceilDiv (a + b) c + 1`, for `c > 0`. -/
theorem ceilDiv_add_le_succ {c a b : Nat} (hc : 0 < c) :
    ceilDiv a c + ceilDiv b c ≤ ceilDiv (a + b) c + 1 := by
  rcases Nat.eq_zero_or_pos (ceilDiv a c) with hA0 | hApos
  · have ha : a = 0 := by
      have := le_ceilDiv_mul (a := a) (b := c) hc
      rw [hA0, Nat.zero_mul] at this
      omega
    subst ha
    simp only [hA0, Nat.zero_add]
    omega
  · rcases Nat.eq_zero_or_pos (ceilDiv b c) with hB0 | hBpos
    · have hb : b = 0 := by
        have := le_ceilDiv_mul (a := b) (b := c) hc
        rw [hB0, Nat.zero_mul] at this
        omega
      subst hb
      simp only [hB0, Nat.add_zero]
      omega
    · have key : ceilDiv a c + ceilDiv b c - 1 ≤ ceilDiv (a + b) c := by
        apply le_ceilDiv_of_lt_mul hc
        have h1 : (ceilDiv a c - 1) * c + (ceilDiv b c - 1) * c < a + b :=
          Nat.add_lt_add (ceilDiv_pred_mul_lt hc hApos) (ceilDiv_pred_mul_lt hc hBpos)
        have h2 : (ceilDiv a c - 1) * c + (ceilDiv b c - 1) * c
            = ((ceilDiv a c + ceilDiv b c - 1) - 1) * c := by
          rw [← Nat.add_mul]
          congr 1
          omega
        rwa [h2] at h1
      omega

/-- `ceilDiv (c * L) c = L` for `c > 0`: the ceiling of a whole number of capacities is exact. -/
theorem ceilDiv_mul_self {c L : Nat} (hc : 0 < c) : ceilDiv (c * L) c = L := by
  have h1 : L * c ≤ L * c + (c - 1) := Nat.le_add_right _ _
  have h2 : L * c + (c - 1) < (L + 1) * c := by
    rw [Nat.add_mul, Nat.one_mul]
    omega
  unfold ceilDiv
  rw [Nat.mul_comm c L]
  rw [show L * c + c - 1 = L * c + (c - 1) by omega]
  exact Nat.div_eq_of_lt_le h1 h2

/-- `ceilDiv 0 c = 0` for every `c`. -/
theorem ceilDiv_zero (c : Nat) : ceilDiv 0 c = 0 := by
  unfold ceilDiv
  rw [Nat.zero_add]
  cases c with
  | zero => rw [Nat.zero_sub, Nat.zero_div]
  | succ k => rw [Nat.succ_sub_one, Nat.div_eq_of_lt (Nat.lt_succ_self k)]

/-- The sum of the per-dispatch ceilings is at least the ceiling of the chain's total demand. -/
theorem ceilDiv_sum_le (c : Nat) (hc : 0 < c) (l : List Nat) :
    ceilDiv l.sum c ≤ (l.map (fun W => ceilDiv W c)).sum := by
  induction l with
  | nil => simp [ceilDiv_zero]
  | cons a t ih =>
      rw [List.sum_cons, List.map_cons, List.sum_cons]
      exact Nat.le_trans (ceilDiv_add_le hc) (Nat.add_le_add_left ih _)

/-! ## 3 · The chain bounds -/

/-- The termwise comparison of two mapped sums. -/
theorem sum_map_le_of_le {l : List Nat} {f g : Nat → Nat} (h : ∀ W ∈ l, f W ≤ g W) :
    (l.map f).sum ≤ (l.map g).sum := by
  induction l with
  | nil => simp
  | cons a t ih =>
      simp only [List.map_cons, List.sum_cons]
      exact Nat.add_le_add (h a (by simp))
        (ih (fun W hW => h W (by simp [hW])))

/-- **The chain never beats the byte roofline, and pays the floor per dispatch.** For `c > 0`:
`ceilDiv (Σ W) c ≤ chainLB c L ws` and `ws.length * L ≤ chainLB c L ws`. The first says the byte time
of the fused demand is a lower bound of the summed per-dispatch byte times; the second says each
dependent dispatch contributes at least its floor. -/
theorem chain_ge_bytes (c L : Nat) (ws : List Nat) (hc : 0 < c) :
    ceilDiv (ws.sum) c ≤ chainLB c L ws ∧ ws.length * L ≤ chainLB c L ws := by
  constructor
  · unfold chainLB
    calc ceilDiv (ws.sum) c ≤ (ws.map (fun W => ceilDiv W c)).sum := ceilDiv_sum_le c hc ws
      _ ≤ (ws.map (dispLB c L)).sum :=
          sum_map_le_of_le (fun W _ => Nat.le_max_left _ _)
  · unfold chainLB
    induction ws with
    | nil => simp
    | cons a t ih =>
        simp only [List.length_cons, List.map_cons, List.sum_cons]
        calc (t.length + 1) * L = t.length * L + L := by ring
          _ ≤ (t.map (dispLB c L)).sum + dispLB c L a :=
              Nat.add_le_add ih (Nat.le_max_right _ _)
          _ = dispLB c L a + (t.map (dispLB c L)).sum := Nat.add_comm _ _

/-- **The knee.** A dispatch's bound is its floor exactly when the demand is at or below the knee
`W* = c * L`: `dispLB c L W = L ↔ W ≤ c * L`, for `c > 0`. Below the knee the dispatch costs `L`
regardless of `W`; above it the bound is `ceilDiv W c`. -/
theorem knee {c L W : Nat} (hc : 0 < c) : dispLB c L W = L ↔ W ≤ c * L := by
  unfold dispLB
  rw [max_eq_right_iff, ceilDiv_le_iff_mul hc]

/-- The lower bound of a below-knee dispatch is exactly its floor. -/
theorem dispLB_of_le_knee {c L W : Nat} (hc : 0 < c) (hW : W ≤ c * L) : dispLB c L W = L :=
  (knee hc).mpr hW

/-- The dispatch floor as a ceiling plus its tax: `dispLB c L W = ceilDiv W c + (L - ceilDiv W c)`
(`max a L = a + (L - a)`). -/
theorem dispLB_eq (c L W : Nat) : dispLB c L W = ceilDiv W c + (L - ceilDiv W c) := by
  unfold dispLB
  rcases le_total (ceilDiv W c) L with h | h
  · rw [max_eq_right h]; omega
  · rw [max_eq_left h]; omega

/-- **The chain tax.** The chain bound is the summed byte time plus the summed shortfalls:
`chainLB c L ws = Σ ceilDiv W c + Σ (L - ceilDiv W c)`. -/
theorem chain_tax (c L : Nat) (ws : List Nat) :
    chainLB c L ws = (ws.map (fun W => ceilDiv W c)).sum
      + (ws.map (fun W => L - ceilDiv W c)).sum := by
  unfold chainLB
  have hm : ws.map (dispLB c L)
      = ws.map (fun W => ceilDiv W c + (L - ceilDiv W c)) := by
    apply List.map_congr_left
    intro W _
    exact dispLB_eq c L W
  rw [hm, List.sum_map_add]

/-- The tax is zero exactly when every dispatch has `L ≤ ceilDiv W c` (the byte time already carries
the floor). -/
theorem chain_tax_zero (c L : Nat) (ws : List Nat) :
    (ws.map (fun W => L - ceilDiv W c)).sum = 0 ↔ ∀ W ∈ ws, L ≤ ceilDiv W c := by
  induction ws with
  | nil => simp
  | cons a t ih =>
      rw [List.map_cons, List.sum_cons, Nat.add_eq_zero_iff, Nat.sub_eq_zero_iff_le, ih,
          List.forall_mem_cons]

/-- A sufficient condition for a zero tax: if every dispatch has `W ≥ c * L` (at or above the knee
`W* = c * L`) the tax vanishes, for `c > 0`. -/
theorem chain_tax_zero_of_knee (c L : Nat) (hc : 0 < c) {ws : List Nat}
    (h : ∀ W ∈ ws, c * L ≤ W) :
    (ws.map (fun W => L - ceilDiv W c)).sum = 0 := by
  rw [chain_tax_zero]
  intro W hW
  have hm : ceilDiv (c * L) c ≤ ceilDiv W c := ceilDiv_mono (h W hW)
  rwa [ceilDiv_mul_self hc] at hm

/-! ## 4 · Fusion of adjacent dependent dispatches -/

/-- The auxiliary bound `max A L + max B L ≤ max C L + L` for `A, B ≤ C`, `A + B ≤ C + 1` and
`1 ≤ L`. It is the case analysis behind `fuse_le`'s second inequality. -/
theorem max_add_max_le_of {A B C L : Nat} (hAC : A ≤ C) (hBC : B ≤ C)
    (h : A + B ≤ C + 1) (hL : 1 ≤ L) :
    max A L + max B L ≤ max C L + L := by
  rcases le_total A L with hA | hA <;> rcases le_total B L with hB | hB
  · rw [max_eq_right hA, max_eq_right hB]
    have : L ≤ max C L := Nat.le_max_right _ _
    omega
  · rw [max_eq_right hA, max_eq_left hB]
    have : A ≤ max C L := le_trans hAC (Nat.le_max_left _ _)
    omega
  · rw [max_eq_left hA, max_eq_right hB]
    have : B ≤ max C L := le_trans hBC (Nat.le_max_left _ _)
    omega
  · rw [max_eq_left hA, max_eq_left hB]
    have : C ≤ max C L := Nat.le_max_left _ _
    omega

/-- **Fusion of two adjacent dependent dispatches.** For `c > 0` and `L ≥ 1`, replacing `a, b` by
one dispatch of demand `a + b` never raises the bound,
`dispLB c L (a + b) ≤ dispLB c L a + dispLB c L b`, and the saving is at most one floor,
`dispLB c L a + dispLB c L b ≤ dispLB c L (a + b) + L`. Above the knee the second bound is the whole
content: fusion changes the bound by at most `L`, no ratio. -/
theorem fuse_le (c L a b : Nat) (hc : 0 < c) (hL : 1 ≤ L) :
    dispLB c L (a + b) ≤ dispLB c L a + dispLB c L b ∧
    dispLB c L a + dispLB c L b ≤ dispLB c L (a + b) + L := by
  constructor
  · apply Nat.max_le_of_le_of_le
    · calc ceilDiv (a + b) c ≤ ceilDiv a c + ceilDiv b c := ceilDiv_add_le hc
        _ ≤ max (ceilDiv a c) L + max (ceilDiv b c) L :=
            Nat.add_le_add (Nat.le_max_left _ _) (Nat.le_max_left _ _)
    · have h1 : L ≤ max (ceilDiv a c) L := Nat.le_max_right _ _
      exact le_trans h1 (le_add_of_nonneg_right (Nat.zero_le _))
  · unfold dispLB
    exact max_add_max_le_of (ceilDiv_mono (by omega)) (ceilDiv_mono (by omega))
      (ceilDiv_add_le_succ hc) hL

/-- **Below the knee, fusion saves exactly one floor.** When `a + b ≤ c * L` both dispatches are
below the knee, so each costs `L` and the fused dispatch costs `L`: the saving is exactly `L`. -/
theorem fuse_saves_L (c L a b : Nat) (hc : 0 < c) (h : a + b ≤ c * L) :
    dispLB c L a + dispLB c L b = dispLB c L (a + b) + L := by
  have ha : dispLB c L a = L := (knee hc).mpr (by omega)
  have hb : dispLB c L b = L := (knee hc).mpr (by omega)
  have hab : dispLB c L (a + b) = L := (knee hc).mpr h
  rw [ha, hb, hab]

/-- **Batching independent same-shape dispatches.** `k ≥ 1` dispatches of demand `W` issued as one
dispatch of demand `k * W` never raises the bound: `dispLB c L (k * W) ≤ k * dispLB c L W`, for
`c > 0`. This is `fuse_le` iterated; it is the qkv/gu batched-GEMV move. -/
theorem merge_independent (c L k W : Nat) (hc : 0 < c) (hk : 1 ≤ k) :
    dispLB c L (k * W) ≤ k * dispLB c L W := by
  unfold dispLB
  apply Nat.max_le_of_le_of_le
  · calc ceilDiv (k * W) c ≤ k * ceilDiv W c := ceilDiv_mul_le hc
      _ ≤ k * max (ceilDiv W c) L := Nat.mul_le_mul_left k (Nat.le_max_left _ _)
  · calc L ≤ max (ceilDiv W c) L := Nat.le_max_right _ _
      _ = 1 * max (ceilDiv W c) L := (Nat.one_mul _).symm
      _ ≤ k * max (ceilDiv W c) L := Nat.mul_le_mul_right _ hk

/-- **Below the knee, batching saves exactly `(k - 1)` floors.** When `k * W ≤ c * L` the batch and
each single dispatch cost `L`, so `k * dispLB c L W = dispLB c L (k * W) + (k - 1) * L`. -/
theorem merge_saves (c L k W : Nat) (hc : 0 < c) (hk : 1 ≤ k) (h : k * W ≤ c * L) :
    k * dispLB c L W = dispLB c L (k * W) + (k - 1) * L := by
  have hW : W ≤ c * L := by
    calc W = 1 * W := (Nat.one_mul W).symm
      _ ≤ k * W := Nat.mul_le_mul_right _ hk
      _ ≤ c * L := h
  rw [dispLB_of_le_knee hc hW, dispLB_of_le_knee hc h]
  have hk' : k = 1 + (k - 1) := by omega
  calc k * L = (1 + (k - 1)) * L := by conv_lhs => rw [hk']
    _ = 1 * L + (k - 1) * L := by rw [Nat.add_mul]
    _ = L + (k - 1) * L := by rw [Nat.one_mul]

/-! ## 5 · The fusion ratio, below and above the knee -/

/-- A constant chain of `n` below-knee dispatches costs `n * L`. -/
theorem chainLB_replicate (c L W n : Nat) (hc : 0 < c) (hW : W ≤ c * L) :
    chainLB c L (List.replicate n W) = n * L := by
  unfold chainLB
  induction n with
  | zero => simp
  | succ m ih =>
      rw [List.replicate_succ, List.map_cons, List.sum_cons, ih, dispLB_of_le_knee hc hW]
      ring

/-- **Below the knee, fusion divides the chain bound by exactly `g`.** A chain of `n * g` identical
below-knee dispatches of demand `W`, fused in groups of `g` into `n` dispatches of demand `g * W`
(each group still below the knee), has its bound divided by exactly `g`:
`chainLB (replicate (n * g) W) = g * chainLB (replicate n (g * W))`. The move is a dispatch-count
reduction, not a byte reduction. -/
theorem chain_fuse_ratio (c L W n g : Nat) (hc : 0 < c) (hW : W ≤ c * L)
    (hgW : g * W ≤ c * L) :
    chainLB c L (List.replicate (n * g) W) = g * chainLB c L (List.replicate n (g * W)) := by
  rw [chainLB_replicate c L W (n * g) hc hW, chainLB_replicate c L (g * W) n hc hgW]
  ring

/-- Fuse the entries of a list in adjacent pairs, leaving an odd final entry alone. -/
def fusePairs : List Nat → List Nat
  | [] => []
  | [a] => [a]
  | a :: b :: t => (a + b) :: fusePairs t

/-- `chainLB` on a cons, its recursion equation. -/
theorem chainLB_cons (c L a : Nat) (ws : List Nat) :
    chainLB c L (a :: ws) = dispLB c L a + chainLB c L ws := by
  simp [chainLB]

/-- **Above the knee, fusion is not a ratio.** Fusing a chain in adjacent pairs saves at most `L` per
fused pair: `chainLB c L ws ≤ chainLB c L (fusePairs ws) + (ws.length / 2) * L`, for `c > 0` and
`L ≥ 1`. Combined with `fuse_le`'s first inequality (fusion never raises the bound), a fused
above-knee pair changes the bound by at most one floor, so the above-knee move is a byte reduction,
not a dispatch-count reduction. -/
theorem chain_fuse_pair_saving (c L : Nat) (hc : 0 < c) (hL : 1 ≤ L) :
    ∀ ws : List Nat,
      chainLB c L ws ≤ chainLB c L (fusePairs ws) + (ws.length / 2) * L
  | [] => by simp [chainLB, fusePairs]
  | [a] => by simp [chainLB, fusePairs]
  | a :: b :: t => by
      have ih := chain_fuse_pair_saving c L hc hL t
      rw [show fusePairs (a :: b :: t) = (a + b) :: fusePairs t from rfl]
      simp only [chainLB_cons]
      have hpair := (fuse_le c L a b hc hL).2
      have hlen : (a :: b :: t).length / 2 = t.length / 2 + 1 := by
        have hl : (a :: b :: t).length = t.length + 1 * 2 := by
          simp [List.length_cons]
        rw [hl]
        exact Nat.add_mul_div_right t.length 1 (by norm_num)
      rw [hlen, Nat.add_mul, Nat.one_mul]
      omega

/-! ## 6 · The concrete decode instance

The capacities and demands below are declared `model` inputs, not measurements: `c = 865` bytes/ns
(streaming read), `L = 4000` ns (the 0.9 MB `o` one-pass small-GEMV floor), and the per-token
dispatch bytes from `docs/evidence/perf-predictions.md` P2 provenance and the `DECODE-ATTRIB`
receipt at `DECODE-ATTRIB-RECEIPT.md`: `qkv = 1_160_000`,
`o = 900_000`, `gu = 9_800_000`, `dn = 3_800_000`, `lm_head = 153_200_000`. Rope, flash attention
and the norms carry `W = 0`, so each costs exactly the floor `L`.
-/

/-- The model capacity, bytes per ns. -/
def decC : Nat := 865

/-- The model per-dispatch floor, ns. -/
def decL : Nat := 4000

/-- `decC = 865 > 0`, the hypothesis every chain theorem carries. -/
theorem decC_pos : 0 < decC := by decide

/-- One decode layer's dependent dispatch pattern: qkv, o, gu, dn. -/
def layerWS : List Nat := [1160000, 900000, 9800000, 3800000]

-- The knee: `c * L = 3_460_000`. At the knee a dispatch still costs its floor; one byte above it
-- costs one byte-time step more. Below the knee the byte term is inert.
#guard decC * decL = 3460000
#guard dispLB decC decL 3460000 = 4000
#guard dispLB decC decL 3460001 = 4001
#guard dispLB decC decL 1 = 4000

-- Classification: qkv 1.16 MB and o 0.9 MB are below the knee (`dispLB = L`); gu 9.8 MB, dn 3.8 MB
-- and lm_head 153.2 MB are above it (`dispLB = ceilDiv W c`).
#guard dispLB decC decL 1160000 = 4000
#guard dispLB decC decL 900000 = 4000
#guard dispLB decC decL 9800000 = 11330
#guard dispLB decC decL 3800000 = 4394
#guard dispLB decC decL 153200000 = 177110

-- `chain_ge_bytes` on the per-layer pattern: the byte roofline is far below the chain bound, and the
-- floor term `4 * L = 16000` is below it too.
#guard ceilDiv (layerWS.sum) decC = 18105
#guard chainLB decC decL layerWS = 23724
#guard chainLB decC decL layerWS ≥ ceilDiv (layerWS.sum) decC
#guard layerWS.length * decL ≤ chainLB decC decL layerWS

-- `chain_tax`: the summed ceilings plus the summed shortfalls equal the chain bound. The tax is
-- `(4000 - 1342) + (4000 - 1041) = 5617`, carried by the two below-knee dispatches.
#guard (layerWS.map (fun W => ceilDiv W decC)).sum = 18107
#guard chainTax decC decL layerWS = 5617
#guard chainLB decC decL layerWS = (layerWS.map (fun W => ceilDiv W decC)).sum
    + chainTax decC decL layerWS
-- Above the knee the tax vanishes; below the knee it is the whole gap.
#guard chainTax decC decL [9800000, 3800000] = 0
#guard chainTax decC decL [1160000, 900000] = 5617

-- `fuse_le` / `fuse_saves_L`: fusing two below-knee dispatches saves exactly `L = 4000`; fusing an
-- above-knee pair saves at most `L`.
#guard dispLB decC decL 1160000 + dispLB decC decL 900000
    = dispLB decC decL 2060000 + decL
#guard dispLB decC decL 9800000 + dispLB decC decL 3800000
    ≤ dispLB decC decL 13600000 + decL
#guard dispLB 2 1 1 + dispLB 2 1 1 = dispLB 2 1 2 + 1

-- `merge_independent` / `merge_saves`: two qkv-size dispatches batched to `2 * W` (still below the
-- knee), and the general batching inequality.
#guard dispLB decC decL (2 * 1160000) ≤ 2 * dispLB decC decL 1160000
#guard 2 * dispLB decC decL 1160000 = dispLB decC decL (2 * 1160000) + 1 * decL

-- `chain_fuse_ratio`: a below-knee chain of 6 dispatches of 100 B, fused in pairs, halves the bound.
#guard chainLB decC decL (List.replicate 6 100)
    = 2 * chainLB decC decL (List.replicate 3 200)
#guard chainLB decC decL (List.replicate 6 100) = 24000

-- `chain_fuse_pair_saving`: an above-knee chain of gu dispatches saves at most `L` per fused pair.
#guard chainLB decC decL [9800000, 9800000]
    ≤ chainLB decC decL (fusePairs [9800000, 9800000]) + (2 / 2) * decL

-- Zero-byte dispatches: rope, flash attention and the norms carry `W = 0`, so each costs the floor.
#guard dispLB decC decL 0 = 4000
#guard chainLB decC decL [0, 0] = 8000

-- The per-token total: 24 layers of the per-layer pattern plus one lm_head.
#guard 24 * chainLB decC decL layerWS + chainLB decC decL [153200000] = 746486

-- Non-vacuity: the theorems instantiated at the concrete model.
example : ceilDiv (layerWS.sum) decC ≤ chainLB decC decL layerWS :=
  (chain_ge_bytes decC decL layerWS decC_pos).1

example : chainLB decC decL layerWS
    = (layerWS.map (fun W => ceilDiv W decC)).sum + chainTax decC decL layerWS :=
  chain_tax decC decL layerWS

example : dispLB decC decL 1160000 = decL :=
  dispLB_of_le_knee decC_pos (by decide)

example : dispLB decC decL 1160000 + dispLB decC decL 900000
    = dispLB decC decL 2060000 + decL :=
  fuse_saves_L decC decL 1160000 900000 decC_pos (by decide)

example : dispLB decC decL (4 * 1160000) ≤ 4 * dispLB decC decL 1160000 :=
  merge_independent decC decL 4 1160000 decC_pos (by decide)

example : chainLB decC decL (List.replicate 6 100)
    = 2 * chainLB decC decL (List.replicate 3 200) :=
  chain_fuse_ratio decC decL 100 3 2 decC_pos (by decide) (by decide)

example : chainLB decC decL [9800000, 9800000]
    ≤ chainLB decC decL (fusePairs [9800000, 9800000]) + (2 / 2) * decL :=
  chain_fuse_pair_saving decC decL decC_pos (by decide) [9800000, 9800000]

end Chain

end Perf
