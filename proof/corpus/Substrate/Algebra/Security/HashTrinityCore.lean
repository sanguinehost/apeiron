import Mathlib.Data.List.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Common
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

-- Pure Lean core (zero Mathlib for the table itself): the hashmap trinity — the
-- dictionary-as-applicative and its one operation `insertWith`, with the four merge faces
-- (Set/Map/Accumulator/Graph) and the four-face parity theorems. This is the NATIVE
-- implementation the execution path stands on. This file also carries the Mathlib-backed `ℝ`
-- estimation instantiation (§ Kalman below) — the parity/application layer, never under the
-- execution path.

/-!
# The hashmap trinity — `insertWith`, the four merges, the four faces (+ the Kalman parity)
@name One Table, Four Jobs, One Knob
@child A lookup table does four different jobs — a set, a dictionary, a counter, a network of links — and the *only* thing you change between them is the little rule for how two values combine.
@scientist One structure (`K → V`) and one operation (`insertWith f`) give four faces (Set/Map/Accumulator/Graph); the merge `f` is the sole freedom — `applicative_is_the_only_freedom` for the hashmap. Applied to `ℝ` estimation, the B/U accumulator face IS the (extended) Kalman filter.
@engineer `insertWith` with `merge_is_the_only_freedom`; the four merges `merge_NB`/`merge_B`/`merge_BU`/`merge_loop` and their face theorems (e.g. `set_member`) — pure Lean, the native execution path. `mean_is_bulk_ratio`/`fusion_is_commutative_monoid`/`kalman_update_from_merge` recover the textbook Kalman gain from one merge.

One structure (the table `K → V`), one operation (`insertWith f`, the merge `f` the only freedom),
four faces (Set/Map/Accumulator/Graph). The four-face theorems below ARE the parity: each merge is
proven to do its job. Native rung of `theory/THE_HASHMAP_PATH.md`.

## What the table warrants

* **Proved (native):** `merge_is_the_only_freedom` — two pointwise-equal merges build the same map,
  by `funext` and case split. That is the extensionality of `insertWith` in `f`. It does NOT say the
  merge is the only degree of freedom of a hash table; the key type, default, and structure are
  fixed by the definitions, not shown minimal.
* **Proved (native):** `set_member`, `map_lookup`, `count_two`, `graph_edge`. Each is one `simp`
  over `insertWith` at a chosen merge, on a single key against a constant map — a worked instance,
  not a characterisation of a Set, Map, Accumulator, or Graph. `count_two` is literally
  `1 + (1 + 0) = 2`. `hashmap_trinity` conjoins those four with `merge_is_the_only_freedom` and adds
  nothing beyond the conjunction.
* **Proved (native):** `get_build_acc` and `build_gauge_invariance` — the substance of the file's
  first half. Reading a built table at `x` is a fold over that key's values, and for a commutative
  associative merge any permutation of the insertion stream builds the SAME function. This is a real
  theorem with a real hypothesis, and the gauge/schedule licence it states is exactly what it says.
* **Proved (Mathlib):** the `ℝ` estimation layer. `mean_is_bulk_ratio` is `rfl`;
  `fusion_is_commutative_monoid` is `add_comm`/`add_assoc` componentwise;
  `kalman_update_from_merge` is `field_simp; ring`. The Kalman gain `τ₂/(τ₁+τ₂)` is recovered as an
  algebraic identity under `τ₁, τ₂, τ₁+τ₂ ≠ 0`.
* **Proved (Mathlib):** the whole BLUP arc — `blup_is_countWeight`, `blup_is_the_accumulator_merge`,
  `countWeight_strict_mono`, `shrinkRisk_completed_square`, `blup_is_the_unique_minimiser`,
  `optimal_weight_is_countWeight`, `deflation_strictly_suboptimal`, `deflation_is_reparametrisation`,
  `countWeight_saturates`, `designEffN_le_inv`, `deflated_weight_ceiling`, and the headline
  `count_weight_is_the_blup`. These are `ℝ` identities and inequalities closed by
  `field_simp`/`ring`/`linarith`/`nlinarith`, and they carry their claims at full width — the risk
  really is a strict parabola, the vertex really is unique, deflation really is strictly worse.
* **Anchor:** that `shrinkRisk` IS the Bayes risk of the one-way random-effects model
  `y_ij = μ + a_i + e_ij`. `shrinkRisk τ2 c w := (w - 1)^2 * τ2 + w^2 * c` is a DEFINITION; no
  expectation, distribution, or model occurs anywhere in the file, so every "posterior", "Bayes
  risk" and "intra-class correlation" word is attached to a polynomial by fiat. Same for
  `blupWeight` and `icc` — named after statistics, defined as ratios.
* **Anchor:** *"the accumulator face IS the (extended) Kalman filter"*.
  `kalman_update_from_merge` recovers the scalar update identity; nothing here has a state
  transition, a covariance, a prediction step, or a nonlinearity, so "extended" in particular is
  unsupported.
* **Anchor:** the `fold`/`castfold` measurements — the deflated form converging to
  `9/17 = 0.529` at `α = 8` against the BLUP's `0.9992` at `n = 10⁴`, commits `c89b715` and
  `ece3a76`. Real evidence, outside the Lean; `deflated_weight_ceiling` proves the ceiling
  `(1+α)/(1+2α)` for every `α`, and that the measured constant is THAT ceiling is the bridge.
* **Anchor:** that `build` is `Song.Table.run` and `insertWith` the shipped one
  operation. The song implementation is the evidence; no statement here mentions it.
* **Open:** the module header understates the file. The four voices announce the
  four faces plus "the Kalman parity"; more than half the declarations are the BLUP/deflation arc,
  which the header does not mention at all and `@engineer` does not name.
-/

namespace Sanguine.HashTrinity

/-- A hashmap **is** the dictionary-as-applicative: a total `K → V`, absence = the default. -/
abbrev HMap (K V : Type) : Type := K → V

/-- Read a key. -/
def get {K V : Type} (m : HMap K V) (k : K) : V := m k

/-- **The one operation.** Update at `k` by merging the new value with the old via `f`. The
    merge `f` (collision resolution) is the only thing that changes between the "types". -/
def insertWith {K V : Type} [DecidableEq K] (f : V → V → V) (k : K) (v : V) (m : HMap K V) :
    HMap K V := fun x => if x = k then f v (m x) else m x

variable {K V : Type} [DecidableEq K]

theorem get_insertWith_self (f : V → V → V) (k : K) (v : V) (m : HMap K V) :
    get (insertWith f k v m) k = f v (m k) := by simp [get, insertWith]

theorem get_insertWith_ne (f : V → V → V) (k : K) (v : V) (m : HMap K V) {x : K} (h : x ≠ k) :
    get (insertWith f k v m) x = get m x := by simp [get, insertWith, h]

/-- **Meaning is the only freedom.** `insertWith` depends on the merge `f` only pointwise —
    two merges agreeing everywhere give the same map. The structure is fixed; `f` is the
    meaning. This is `UniversalApplicativeLift.applicative_is_the_only_freedom`, for the
    hashmap. -/
theorem merge_is_the_only_freedom {f g : V → V → V} (h : ∀ a b, f a b = g a b)
    (k : K) (v : V) (m : HMap K V) : insertWith f k v m = insertWith g k v m := by
  funext x; by_cases hx : x = k <;> simp [insertWith, hx, h]

-- ═══════════════════════════════════════════════════════════════════
-- THE FOUR FACES — one `insertWith`, the trinity of merges (+ the loop)
-- ═══════════════════════════════════════════════════════════════════

/-- **NB (0): the idempotent merge** — presence. Merging is `∨`: once in, in. → **Set**. -/
def merge_NB : Bool → Bool → Bool := (· || ·)
/-- **B (1): replace** — take the new value (last write). → **Map**. -/
def merge_B {A : Type} : A → A → A := fun new _ => new
/-- **B/U (/): a monoid merge** — combine. Here `ℕ, +` (a counter). → **Accumulator**. -/
def merge_BU : Nat → Nat → Nat := (· + ·)
/-- **The 4th: the `/` on the keys** — `V = List K`, merge `++`. → **Graph / adjacency**. -/
def merge_loop {A : Type} : List A → List A → List A := (· ++ ·)

/-- **Set (NB).** Insert marks presence; `get` tests membership. -/
theorem set_member (k : K) :
    get (insertWith merge_NB k true (fun _ => false)) k = true := by
  simp [get, insertWith, merge_NB]

/-- **Map (B).** Insert stores a value; `get` looks it up (last write wins). -/
theorem map_lookup (k : K) (v d : V) :
    get (insertWith merge_B k v (fun _ => d)) k = v := by
  simp [get, insertWith, merge_B]

/-- **Accumulator (B/U).** Inserting `1` twice at a key counts to `2`. -/
theorem count_two (k : K) :
    get (insertWith merge_BU k 1 (insertWith merge_BU k 1 (fun _ => 0))) k = 2 := by
  simp [get, insertWith, merge_BU]

/-- **Graph (the 4th).** Inserting an edge `k → t` records `t` in `k`'s adjacency list. -/
theorem graph_edge (k t : K) :
    get (insertWith merge_loop k [t] (fun _ => [])) k = [t] := by
  simp [get, insertWith, merge_loop]

/-- **The headline — one structure, the merge is the trinity.** The hashmap is fixed (the
    dictionary-as-applicative); the merge `f` is the only freedom (`merge_is_the_only_freedom`);
    and the four standard structures are `insertWith` at the trinity of merges (+ the
    self-referential loop): Set at the idempotent `∨` (NB), Map at replace (B), Accumulator at
    a monoid (`B/U`), Graph at `++` over keys (the `/` on itself). Learn this; the rest is a
    choice of `f`. -/
theorem hashmap_trinity (k t : K) (v d : V) :
    (∀ {f g : V → V → V}, (∀ a b, f a b = g a b) → ∀ (kk : K) (vv : V) (m : HMap K V),
        insertWith f kk vv m = insertWith g kk vv m)
    ∧ get (insertWith merge_NB k true (fun _ => false)) k = true
    ∧ get (insertWith merge_B k v (fun _ => d)) k = v
    ∧ get (insertWith merge_BU k 1 (insertWith merge_BU k 1 (fun _ => 0))) k = 2
    ∧ get (insertWith merge_loop k [t] (fun _ => [])) k = [t] :=
  ⟨fun h kk vv m => merge_is_the_only_freedom h kk vv m,
   set_member k, map_lookup k v d, count_two k, graph_edge k t⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE BUILD IS GAUGE — insertion order is a coordinate choice, never the meaning
-- ═══════════════════════════════════════════════════════════════════
-- The table's meaning is the total function `K → V`. WHERE a key physically sits and IN WHAT
-- ORDER placements happen is gauge: a coordinate choice on the placement bundle that the
-- meaning is invariant under. `merge_is_the_only_freedom` fixed the fiber (the merge);
-- `build_gauge_invariance` below fixes the base (the order): for a commutative-associative
-- merge, ANY permutation of the insertion stream builds the SAME map. This is the license an
-- implementation needs to pick the CHEAP gauge — sort the stream by home address and walk the
-- table monotonically (the write face, `CoordinateReadout.blockTransitions`), shard it across
-- cores, or any other schedule — with the meaning untouched by construction.

/-- **The build**: fold `insertWith` over a stream of `(key, value)` pairs, from the constant
    default. This is `Song.Table.run` — the table's one constructor. -/
def build (f : V → V → V) (d : V) (kvs : List (K × V)) : HMap K V :=
  kvs.foldl (fun m kv => insertWith f kv.1 kv.2 m) (fun _ => d)

/-- Reading a built table at `x` is a fold of the merge over exactly the stream's values AT `x`
    (in stream order) — all other pairs are invisible (`get_insertWith_ne`). The bridge from the
    two-dimensional build to a one-dimensional fold per key. -/
theorem get_build_acc (f : V → V → V) (x : K) :
    ∀ (kvs : List (K × V)) (m : HMap K V),
      get (kvs.foldl (fun m kv => insertWith f kv.1 kv.2 m) m) x
        = ((kvs.filter (fun kv => kv.1 = x)).map Prod.snd).foldl (fun acc v => f v acc) (get m x)
  | [], _ => rfl
  | kv :: kvs, m => by
      rw [List.foldl_cons, get_build_acc f x kvs (insertWith f kv.1 kv.2 m)]
      by_cases h : kv.1 = x
      · rw [List.filter_cons_of_pos (by simpa using h), List.map_cons, List.foldl_cons]
        congr 1
        subst h
        exact get_insertWith_self f kv.1 kv.2 m
      · rw [List.filter_cons_of_neg (by simpa using h)]
        congr 1
        exact get_insertWith_ne f kv.1 kv.2 m (fun hxk => h hxk.symm)

/-- **The build is gauge-invariant (headline).** For a commutative and associative merge, any
    permutation of the insertion stream builds the SAME table — insertion order is a gauge
    choice, not part of the meaning. Per key, the build is a fold of the merge over that key's
    values (`get_build_acc`); a permutation of the stream permutes each key's value list
    (`List.Perm.filter`/`map`); and a fold of a commutative-associative operation is
    permutation-invariant (`List.Perm.foldl_eq`). So the implementation may fix ANY gauge —
    sorted-by-home (the monotone write gauge), sharded-by-core, arrival order — and the
    function it builds is identical. The build-side twin of `Web.preduce_schedule_free`. -/
theorem build_gauge_invariance {f : V → V → V}
    (hcomm : ∀ a b, f a b = f b a) (hassoc : ∀ a b c, f (f a b) c = f a (f b c))
    (d : V) {kvs₁ kvs₂ : List (K × V)} (hperm : kvs₁.Perm kvs₂) :
    build f d kvs₁ = build f d kvs₂ := by
  funext x
  show get (build f d kvs₁) x = get (build f d kvs₂) x
  unfold build
  rw [get_build_acc f x kvs₁ (fun _ => d), get_build_acc f x kvs₂ (fun _ => d)]
  exact ((hperm.filter _).map Prod.snd).foldl_eq'
    (fun a _ b _ z => by
      show f b (f a z) = f a (f b z)
      rw [← hassoc, hcomm b a, hassoc])
    _

-- ═══════════════════════════════════════════════════════════════════
-- THE ACCUMULATOR IS THE KALMAN FILTER — what the B/U merge solves (estimation)
-- ═══════════════════════════════════════════════════════════════════
/-! The B/U **accumulator** face (the monoid merge — `merge_BU` above) is, applied to estimation,
the **(extended) Kalman filter** — orbit determination, GPS/INS fusion (`theory/NAVIGATION_REDERIVATION.md`,
absorbed here from the former `KalmanMerge`). Carry an estimate in **information form** `(η, τ)` =
(information `η = τ·μ`, precision `τ = 1/σ²`); then the **estimate is `μ = η/τ = B/U`** (literally the
bulk ratio — information over precision), and **fusion is the accumulator merge** (componentwise `+`):
a commutative monoid (order-free, batchable — *why* a filter works), with the uninformative prior
`(0,0)` the unit. The textbook update `x̂ = x̄ + K(y−x̄)` (gain `K = τ₂/(τ₁+τ₂)`) is recovered from
the merge. One merge, specialised by the applicative — the hashmap path on estimation. -/

noncomputable section

/-- **The estimate (mean)** of an information-form pair — `μ = η/τ = B/U`, information over precision. -/
def mean (e : ℝ × ℝ) : ℝ := e.1 / e.2

/-- **The estimate is a bulk ratio** `μ = η/τ`. -/
theorem mean_is_bulk_ratio (η τ : ℝ) : mean (η, τ) = η / τ := rfl

/-- **The fusion merge** — componentwise addition in information form (information adds, precision
    adds). The B/U accumulator (`merge_BU`/`merge_loop`'s monoid face) on `ℝ × ℝ`. -/
def fuse (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)

/-- **Fusion is a commutative monoid** — commutative (measurement order irrelevant), associative
    (batch = sequential), with `(0,0)` the unit. *Why* a Kalman filter is order-free: the monoid
    laws, not an accident. -/
theorem fusion_is_commutative_monoid :
    (∀ a b : ℝ × ℝ, fuse a b = fuse b a)
      ∧ (∀ a b c : ℝ × ℝ, fuse (fuse a b) c = fuse a (fuse b c))
      ∧ (∀ a : ℝ × ℝ, fuse (0, 0) a = a ∧ fuse a (0, 0) = a) :=
  ⟨fun a b => by simp only [fuse]; rw [add_comm a.1 b.1, add_comm a.2 b.2],
   fun a b c => by simp only [fuse]; rw [add_assoc a.1 b.1 c.1, add_assoc a.2 b.2 c.2],
   fun a => ⟨by simp [fuse], by simp [fuse]⟩⟩

/-- **The Kalman update IS the merge.** The fused mean equals `x̂ = x̄ + K(y − x̄)` with gain
    `K = τ₂/(τ₁+τ₂)` (variance form `σ₁²/(σ₁²+σ₂²)`): prior `x̄ = η₁/τ₁`, measurement `y = η₂/τ₂`. -/
theorem kalman_update_from_merge (η₁ τ₁ η₂ τ₂ : ℝ)
    (h1 : τ₁ ≠ 0) (h2 : τ₂ ≠ 0) (hs : τ₁ + τ₂ ≠ 0) :
    mean (fuse (η₁, τ₁) (η₂, τ₂))
      = η₁ / τ₁ + (τ₂ / (τ₁ + τ₂)) * (η₂ / τ₂ - η₁ / τ₁) := by
  simp only [mean, fuse]
  field_simp
  ring

-- ═══════════════════════════════════════════════════════════════════
-- THE COUNT-CONFIDENCE WEIGHT IS THE MERGE'S OWN GAIN — `n/(n+α)` IS the BLUP
-- ═══════════════════════════════════════════════════════════════════
/-! The accumulator face read at `α = a cell of a one-way random-effects model`
`y_ij = μ + a_i + e_ij` (`a_i` of variance `τ²`, `e_ij` of variance `σ²`). The prior on the cell
effect carries precision `1/τ²`; `n` observations of the cell carry precision `n/σ²`; the merge
adds them (`fusion_is_commutative_monoid`) and the gain it produces (`kalman_update_from_merge`)
is **exactly** `w = n/(n+α)` at `α = σ²/τ²` — for every `n` and every `α`, no approximation
(`blup_is_countWeight`). So the count-confidence weight is not a heuristic to be corrected: it is
the posterior, and the within-cell correlation one would "correct" for is *generated by* `a_i`,
which is the estimand and not a nuisance.

Two consequences, both refuting a family of proposed fixes without a measurement:

* **any deflation `n ↦ n_eff < n` is strictly suboptimal** — the Bayes risk is a strict
  parabola in the weight with its unique vertex at `n/(n+α)` (`shrinkRisk_completed_square`,
  `blup_is_the_unique_minimiser`), and deflation moves the weight strictly down
  (`countWeight_strict_mono`), hence strictly off the vertex (`deflation_strictly_suboptimal`);
  the design-effect deflation in particular caps the weight at `(1+α)/(1+2α) < 1` forever
  (`deflated_weight_ceiling`) where the BLUP goes to 1 (`countWeight_saturates`);
* **an `n`-independent deflation is a reparametrisation, not a new rule** —
  `nφ/(nφ+α) = n/(n+α/φ)` (`deflation_is_reparametrisation`): the same one-parameter family with
  `α` rescaled, so it still saturates and cannot be the fix either.

Provenance: measured in `fold`'s `castfold` at `α ∈ {0.5, 8, 50}` over `n ∈ [1, 10⁴]` — commits
`c89b715` (the BLUP retraction: `n/(n+α)` reproduces the posterior to every digit, the deflated
form converges to `9/17 = 0.529` at `α = 8` against the BLUP's `0.9992`) and `ece3a76` (the
reparametrisation). The measurement cost hours; the two derivations below cost minutes and cover
every `(n, α)` the measurement sampled and every one it did not. -/

/-- **The count-confidence weight** — the weight a cell mean is given against the global mean,
    `w = n/(n+α)`. One parameter, `α`. -/
def countWeight (α n : ℝ) : ℝ := n / (n + α)

/-- **The BLUP weight** — the posterior weight on the cell mean under the one-way random-effects
    model: data precision `n/σ²` over total posterior precision `1/τ² + n/σ²`. This is
    `kalman_update_from_merge`'s gain `τ₂/(τ₁+τ₂)` at `τ₁ = 1/τ²`, `τ₂ = n/σ²` — the accumulator
    merge, nothing added. -/
def blupWeight (τ2 σ2 n : ℝ) : ℝ := (n / σ2) / (1 / τ2 + n / σ2)

/-- **T1 — the count-confidence weight IS the BLUP.** `n/(n+α)` at `α = σ²/τ²` equals the
    posterior weight, for **every** `n ≥ 0` and every `τ², σ² > 0`. Not asymptotic, not a
    heuristic, not an approximation to be corrected for intra-class correlation: an identity.
    The one-parameter family `w = n/(n+α)` and the BLUP are the same object read twice. -/
theorem blup_is_countWeight {τ2 σ2 n : ℝ} (hτ : 0 < τ2) (hσ : 0 < σ2) (hn : 0 ≤ n) :
    blupWeight τ2 σ2 n = countWeight (σ2 / τ2) n := by
  have h1 : (0:ℝ) < 1 / τ2 := by positivity
  have h2 : (0:ℝ) ≤ n / σ2 := div_nonneg hn (le_of_lt hσ)
  have hA : (0:ℝ) < 1 / τ2 + n / σ2 := by linarith
  have h3 : (0:ℝ) < σ2 / τ2 := div_pos hσ hτ
  have hB : (0:ℝ) < n + σ2 / τ2 := by linarith
  have hσ' : σ2 ≠ 0 := ne_of_gt hσ
  have hτ' : τ2 ≠ 0 := ne_of_gt hτ
  unfold blupWeight countWeight
  rw [div_eq_div_iff (ne_of_gt hA) (ne_of_gt hB)]
  field_simp
  ring

/-- **The BLUP posterior mean IS the accumulator merge.** Fusing the prior `(μ, 1/τ²)` with the
    cell mean `(ȳ, n/σ²)` in information form and reading `B/U` gives
    `μ + w·(ȳ − μ)` with `w = n/(n+α)` exactly. The shrinkage estimator is not bolted onto the
    counter; it is what `fuse` already does (`fusion_is_commutative_monoid`, so it is also
    order-free and shardable). -/
theorem blup_is_the_accumulator_merge {τ2 σ2 n μ ybar : ℝ}
    (hτ : 0 < τ2) (hσ : 0 < σ2) (hn : 0 < n) :
    mean (fuse (μ * (1 / τ2), 1 / τ2) (ybar * (n / σ2), n / σ2))
      = μ + countWeight (σ2 / τ2) n * (ybar - μ) := by
  have hτ1 : (0:ℝ) < 1 / τ2 := by positivity
  have hτ2 : (0:ℝ) < n / σ2 := div_pos hn hσ
  have hs : (0:ℝ) < 1 / τ2 + n / σ2 := by linarith
  rw [kalman_update_from_merge _ _ _ _ (ne_of_gt hτ1) (ne_of_gt hτ2) (ne_of_gt hs),
      ← blup_is_countWeight hτ hσ (le_of_lt hn)]
  unfold blupWeight
  have e1 : μ * (1 / τ2) / (1 / τ2) = μ := by
    field_simp
  have e2 : ybar * (n / σ2) / (n / σ2) = ybar := by
    field_simp
  rw [e1, e2]

/-- **The weight is strictly increasing in the count.** For `α > 0`, more observations means
    strictly more trust — the reason a deflation is never neutral. -/
theorem countWeight_strict_mono {α m n : ℝ} (hα : 0 < α) (hm : 0 ≤ m) (hlt : m < n) :
    countWeight α m < countWeight α n := by
  have h1 : (0:ℝ) < m + α := by linarith
  have h2 : (0:ℝ) < n + α := by linarith
  have h1' : m + α ≠ 0 := ne_of_gt h1
  have h2' : n + α ≠ 0 := ne_of_gt h2
  have key : countWeight α n - countWeight α m = α * (n - m) / ((n + α) * (m + α)) := by
    unfold countWeight
    field_simp
    ring
  have hpos : (0:ℝ) < α * (n - m) / ((n + α) * (m + α)) :=
    div_pos (mul_pos hα (by linarith)) (mul_pos h2 h1)
  rw [← key] at hpos
  linarith

/-- Non-strict form, for bounding a deflated count from above. -/
theorem countWeight_mono {α m n : ℝ} (hα : 0 < α) (hm : 0 ≤ m) (hle : m ≤ n) :
    countWeight α m ≤ countWeight α n := by
  rcases eq_or_lt_of_le hle with h | h
  · rw [h]
  · exact le_of_lt (countWeight_strict_mono hα hm h)

-- ── the risk: a strict parabola whose unique vertex is the BLUP ──

/-- The Bayes risk of the linear read `â = w·(ȳ − μ)` for the cell effect, at prior variance
    `τ²` and cell-mean noise variance `c = σ²/n`: `E[(w·(a+ē) − a)²] = (w−1)²τ² + w²c`. -/
def shrinkRisk (τ2 c w : ℝ) : ℝ := (w - 1) ^ 2 * τ2 + w ^ 2 * c

/-- The risk, completed square: a parabola of positive curvature `τ² + c` with vertex at
    `τ²/(τ²+c)` and floor `τ²c/(τ²+c)`. Every claim about optimality below is this identity. -/
theorem shrinkRisk_completed_square {τ2 c w : ℝ} (h : 0 < τ2 + c) :
    shrinkRisk τ2 c w = (τ2 + c) * (w - τ2 / (τ2 + c)) ^ 2 + τ2 * c / (τ2 + c) := by
  have h' : τ2 + c ≠ 0 := ne_of_gt h
  unfold shrinkRisk
  field_simp
  ring

/-- **The vertex is the unique minimiser** — any other weight is strictly worse, by exactly
    `(τ²+c)·(w − w*)²`. There is no tie and no flat region. -/
theorem blup_is_the_unique_minimiser {τ2 c w : ℝ} (hτ : 0 < τ2) (hc : 0 < c)
    (hw : w ≠ τ2 / (τ2 + c)) :
    shrinkRisk τ2 c (τ2 / (τ2 + c)) < shrinkRisk τ2 c w := by
  have h : (0:ℝ) < τ2 + c := by linarith
  have hne : w - τ2 / (τ2 + c) ≠ 0 := sub_ne_zero.mpr hw
  have hs : (0:ℝ) < (w - τ2 / (τ2 + c)) ^ 2 :=
    lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 hne))
  rw [shrinkRisk_completed_square h, shrinkRisk_completed_square h]
  have hzero : (τ2 + c) * (τ2 / (τ2 + c) - τ2 / (τ2 + c)) ^ 2 = 0 := by ring
  have hpos : (0:ℝ) < (τ2 + c) * (w - τ2 / (τ2 + c)) ^ 2 := mul_pos h hs
  linarith

/-- The risk's vertex, in count coordinates: `τ²/(τ² + σ²/n) = n/(n + σ²/τ²)`. The optimal
    weight *is* the count weight — the same identity as `blup_is_countWeight`, read on the
    risk side. -/
theorem optimal_weight_is_countWeight {τ2 σ2 n : ℝ} (hτ : 0 < τ2) (hσ : 0 < σ2) (hn : 0 < n) :
    τ2 / (τ2 + σ2 / n) = countWeight (σ2 / τ2) n := by
  have hc : (0:ℝ) < σ2 / n := div_pos hσ hn
  have hA : (0:ℝ) < τ2 + σ2 / n := by linarith
  have h3 : (0:ℝ) < σ2 / τ2 := div_pos hσ hτ
  have hB : (0:ℝ) < n + σ2 / τ2 := by linarith
  have hσ' : σ2 ≠ 0 := ne_of_gt hσ
  have hτ' : τ2 ≠ 0 := ne_of_gt hτ
  have hn' : n ≠ 0 := ne_of_gt hn
  unfold countWeight
  rw [div_eq_div_iff (ne_of_gt hA) (ne_of_gt hB)]
  field_simp

/-- **T1's corollary, the one with teeth — any deflation of the count is STRICTLY suboptimal.**
    Replace the honest count `n` by any `m < n` ("effective sample size", a design-effect
    correction, an intra-class-correlation discount) and the resulting weight is strictly below
    the vertex (`countWeight_strict_mono`), hence carries strictly greater Bayes risk under the
    very model the correction appeals to (`blup_is_the_unique_minimiser`). The correction is not
    conservative; it is wrong, and the amount it is wrong by is `(τ²+c)·(w − w*)²`. -/
theorem deflation_strictly_suboptimal {τ2 σ2 n m : ℝ} (hτ : 0 < τ2) (hσ : 0 < σ2)
    (hn : 0 < n) (hm : 0 ≤ m) (hlt : m < n) :
    shrinkRisk τ2 (σ2 / n) (countWeight (σ2 / τ2) n)
      < shrinkRisk τ2 (σ2 / n) (countWeight (σ2 / τ2) m) := by
  have hc : (0:ℝ) < σ2 / n := div_pos hσ hn
  have hα : (0:ℝ) < σ2 / τ2 := div_pos hσ hτ
  have hvertex : τ2 / (τ2 + σ2 / n) = countWeight (σ2 / τ2) n :=
    optimal_weight_is_countWeight hτ hσ hn
  have hne : countWeight (σ2 / τ2) m ≠ τ2 / (τ2 + σ2 / n) := by
    rw [hvertex]
    exact ne_of_lt (countWeight_strict_mono hα hm hlt)
  have := blup_is_the_unique_minimiser (c := σ2 / n) hτ hc hne
  rwa [hvertex] at this

-- ── T2: an `n`-independent deflation only renames `α` ──

/-- **T2 — a count deflation independent of `n` is a reparametrisation, not a new rule.** If the
    deflation factor `φ` does not depend on `n`, then `nφ/(nφ+α) = n/(n + α/φ)`: the "effective
    sample size" `n_eff = nφ` lands back in the SAME one-parameter family with `α` rescaled by
    `1/φ`. Two lines, and they refute every fix of this shape without a measurement — including
    the saturation objection, since the family still saturates (`countWeight_saturates`). -/
theorem deflation_is_reparametrisation {φ α n : ℝ} (hφ : 0 < φ) (hα : 0 < α) (hn : 0 ≤ n) :
    countWeight α (n * φ) = countWeight (α / φ) n := by
  have hnφ : (0:ℝ) ≤ n * φ := mul_nonneg hn (le_of_lt hφ)
  have h1 : (0:ℝ) < n * φ + α := by linarith
  have hq : (0:ℝ) < α / φ := div_pos hα hφ
  have h2 : (0:ℝ) < n + α / φ := by linarith
  have hφ' : φ ≠ 0 := ne_of_gt hφ
  unfold countWeight
  rw [div_eq_div_iff (ne_of_gt h1) (ne_of_gt h2)]
  field_simp

/-- **The weight saturates: `w → 1` as `n → ∞`, for every `α > 0`.** Explicitly: once
    `α ≤ ε·n`, the weight is within `ε` of 1 (and always strictly below it). Combined with
    `deflation_is_reparametrisation`, no `n`-independent deflation can cap the weight — it only
    delays the approach. -/
theorem countWeight_saturates {α ε n : ℝ} (hα : 0 < α) (hε : 0 < ε) (hn0 : 0 < n)
    (hn : α ≤ ε * n) : 1 - ε < countWeight α n ∧ countWeight α n < 1 := by
  have hd : (0:ℝ) < n + α := by linarith
  have hd' : n + α ≠ 0 := ne_of_gt hd
  have hkey : 1 - countWeight α n = α / (n + α) := by
    unfold countWeight
    field_simp
    ring
  have hlo : (0:ℝ) < α / (n + α) := div_pos hα hd
  have hgap : ε - α / (n + α) = (ε * (n + α) - α) / (n + α) := by
    field_simp
  have hnum : (0:ℝ) < ε * (n + α) - α := by nlinarith [mul_pos hε hα]
  have hhi : (0:ℝ) < ε - α / (n + α) := by
    rw [hgap]; exact div_pos hnum hd
  exact ⟨by linarith, by linarith⟩

-- ── the design-effect deflation, priced: a hard ceiling below 1 ──

/-- The intra-class correlation `ρ = τ²/(τ²+σ²)`, in `α = σ²/τ²` coordinates: `ρ = 1/(1+α)`.
    It is generated by `a_i` — the estimand — which is why discounting for it double-counts. -/
def icc (α : ℝ) : ℝ := 1 / (1 + α)

/-- The design-effect "effective sample size" `n_eff = n / (1 + (n−1)ρ)`. -/
def designEffN (ρ n : ℝ) : ℝ := n / (1 + (n - 1) * ρ)

/-- **The design-effect count is capped at `1/ρ` — it does not grow with `n` at all.** This is
    the structural reason the deflated weight converges to a constant instead of to 1. -/
theorem designEffN_le_inv {ρ n : ℝ} (hρ : 0 < ρ) (hρ1 : ρ ≤ 1) (hn : 1 ≤ n) :
    designEffN ρ n ≤ 1 / ρ := by
  have hd : (0:ℝ) < 1 + (n - 1) * ρ := by nlinarith
  have key : 1 / ρ - n / (1 + (n - 1) * ρ) = (1 - ρ) / (ρ * (1 + (n - 1) * ρ)) := by
    rw [div_sub_div _ _ (ne_of_gt hρ) (ne_of_gt hd)]
    congr 1
    ring
  have hnn : (0:ℝ) ≤ (1 - ρ) / (ρ * (1 + (n - 1) * ρ)) :=
    div_nonneg (by linarith) (le_of_lt (mul_pos hρ hd))
  rw [← key] at hnn
  unfold designEffN
  linarith

theorem designEffN_nonneg {ρ n : ℝ} (hρ : 0 < ρ) (hn : 1 ≤ n) : 0 ≤ designEffN ρ n := by
  have hd : (0:ℝ) < 1 + (n - 1) * ρ := by nlinarith
  exact div_nonneg (by linarith) (le_of_lt hd)

/-- **The deflated weight's hard ceiling.** Under the design-effect deflation at the model's own
    intra-class correlation `ρ = 1/(1+α)`, the weight can never exceed `(1+α)/(1+2α)` — a
    constant strictly below 1, for every `n`. At `α = 8` that ceiling is `9/17 ≈ 0.529`, which is
    exactly what `castfold` measured the deflated form converging to (`fold c89b715`) against the
    BLUP's `0.9992` at `n = 10⁴`. The measurement found the number; this finds it for every `α`,
    and finds that it is a ceiling rather than a limit. -/
theorem deflated_weight_ceiling {α n : ℝ} (hα : 0 < α) (hn : 1 ≤ n) :
    countWeight α (designEffN (icc α) n) ≤ (1 + α) / (1 + 2 * α) := by
  have h1α : (0:ℝ) < 1 + α := by linarith
  have hρ : (0:ℝ) < icc α := by unfold icc; positivity
  have hρ1 : icc α ≤ 1 := by
    unfold icc
    rw [div_le_one h1α]
    linarith
  have hinv : 1 / icc α = 1 + α := by
    unfold icc
    rw [one_div_one_div]
  have hcap : countWeight α (designEffN (icc α) n) ≤ countWeight α (1 + α) := by
    refine countWeight_mono hα (designEffN_nonneg hρ hn) ?_
    have := designEffN_le_inv hρ hρ1 hn
    rwa [hinv] at this
  have hval : countWeight α (1 + α) = (1 + α) / (1 + 2 * α) := by
    have h : (1:ℝ) + α + α = 1 + 2 * α := by ring
    unfold countWeight
    rw [h]
  rw [hval] at hcap
  exact hcap

/-- **The headline (T1 + T2).** The count-confidence weight `n/(n+α)` is the BLUP identically
    (`blup_is_countWeight`), it is the unique risk minimiser (`blup_is_the_unique_minimiser`), so
    every deflation `m < n` is strictly worse (`deflation_strictly_suboptimal`); and every
    `n`-independent deflation is only `α ↦ α/φ` (`deflation_is_reparametrisation`), which
    saturates too. There is no correction to make here — the merge already made it. -/
theorem count_weight_is_the_blup {τ2 σ2 n m φ : ℝ} (hτ : 0 < τ2) (hσ : 0 < σ2)
    (hn : 0 < n) (hm : 0 ≤ m) (hlt : m < n) (hφ : 0 < φ) :
    blupWeight τ2 σ2 n = countWeight (σ2 / τ2) n
      ∧ shrinkRisk τ2 (σ2 / n) (countWeight (σ2 / τ2) n)
          < shrinkRisk τ2 (σ2 / n) (countWeight (σ2 / τ2) m)
      ∧ countWeight (σ2 / τ2) (n * φ) = countWeight (σ2 / τ2 / φ) n :=
  ⟨blup_is_countWeight hτ hσ (le_of_lt hn),
   deflation_strictly_suboptimal hτ hσ hn hm hlt,
   deflation_is_reparametrisation hφ (div_pos hσ hτ) (le_of_lt hn)⟩

end

end Sanguine.HashTrinity
