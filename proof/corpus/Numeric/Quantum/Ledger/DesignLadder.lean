/-
Spherical t-design moment ladder in ℤ³ (`APEIRON-QARCH-L4`).

A finite point set `X ⊂ ℤ³` whose points all have squared norm `r²` (`r² = 3` for the tetrahedron
and the cube vertices `(±1,±1,±1)`, `r² = 1` for the octahedron `±eᵢ`) is a *spherical t-design*
for `t ≤ 4` when its moment sums equal `|X|` times the sphere moments at radius `r`
(Delsarte–Goethals–Seidel 1977, the moment form): for a multi-index `α` of degree `d ≤ t`,

  `∑_{x ∈ X} x^α = |X| · r^d · m_α`,

with sphere moments `m_α = 0` for any odd exponent, `m(xᵢ²) = 1/3`, `m(xᵢ⁴) = 1/5`,
`m(xᵢ²xⱼ²) = 1/15` (`i ≠ j`).

`IsDesign X r2 t` below is exactly these equalities for every monomial of degree `1..t` in three
variables, with the target stated in ℚ. Every non-zero target has even degree, so `r^d` is
`r2^(d/2)` and only `r2 = r²` appears; a monomial of odd degree has target `0` (some exponent odd).

Point sets (as explicit `List (ℤ × ℤ × ℤ)`): the tetrahedron `T` (sign triples with an even number
of `−1`), the octahedron `O = {±e₁, ±e₂, ±e₃}`, and the cube `C` (all eight `(±1,±1,±1)`).

Landed here:

* `tetra_design_two`  : `IsDesign T 3 2`;  `tetra_not_design_three` : `¬ IsDesign T 3 3`, by the
  odd-moment witness `∑ xyz = 4 ≠ 0` (`tetra_xyz`).
* `octa_design_three` : `IsDesign O 1 3`;  `octa_not_design_four`  : `¬ IsDesign O 1 4`, by
  `∑ x⁴ = 2 ≠ 6/5`.
* `cube_design_three` : `IsDesign C 3 3`;  `cube_not_design_four`  : `¬ IsDesign C 3 4`.
* `tetra_union_is_cube` : `T ∪ (−T) = C` as finsets: the cube is the antipodal closure of the
  tetrahedron, which is how the odd-moment witness of `tetra_not_design_three` cancels
  (`∑ xyz` over `−T` is `−4`).
* `verdict_l4` : the conjunction of the above.

Anchor, graded `anchor` and not proved here: on the Bloch sphere `T` is the qubit SIC-POVM
(Renes, Blume-Kohout, Scott, Caves, quant-ph/0310075) and `O` is the six stabilizer states, a
state 3-design (Webb, 1510.02769).

Complexity. `sumMono` walks `X` once, O(|X|) time and space; every theorem below is a closed,
bounded evaluation over the 4–8-point lists (`decide` on explicit sums), O(1) at these sizes.
-/
import Mathlib.Tactic

namespace Quantum
namespace DesignLadder

/-- A lattice point of `ℤ³`. -/
abbrev Coord := ℤ × ℤ × ℤ

/-- Multi-index monomial sum `∑_{x ∈ X} x.1^a · x.2.1^b · x.2.2^c` in ℚ, over the list `X` with
    multiplicity. O(|X|) time and space. -/
def sumMono (X : List Coord) (a b c : ℕ) : ℚ :=
  (X.map (fun p => (p.1 : ℚ) ^ a * (p.2.1 : ℚ) ^ b * (p.2.2 : ℚ) ^ c)).sum

/-- `T`: the tetrahedron vertices, the sign triples with an even number of `−1`. -/
def tetrahedron : List Coord :=
  [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]

/-- `O`: the octahedron vertices `±e₁, ±e₂, ±e₃`. -/
def octahedron : List Coord :=
  [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

/-- `C`: the cube vertices, all eight sign triples `(±1,±1,±1)`. -/
def cube : List Coord :=
  [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
   (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]

/-- Antipodal map `x ↦ −x` on the list `X`. -/
def neg (X : List Coord) : List Coord := X.map (fun p => (-p.1, -p.2.1, -p.2.2))

/-- Degree-1 moment equalities: the three odd monomials `x, y, z`, target `0`. -/
def Deg1 (X : List Coord) : Prop :=
  sumMono X 1 0 0 = 0 ∧ sumMono X 0 1 0 = 0 ∧ sumMono X 0 0 1 = 0

/-- Degree-2 moment equalities (6 monomials): `m(xᵢ²) = 1/3`, `m(xᵢxⱼ) = 0` (`i ≠ j`). -/
def Deg2 (X : List Coord) (r2 : ℤ) : Prop :=
  sumMono X 2 0 0 = (X.length : ℚ) * (r2 : ℚ) / 3 ∧
  sumMono X 1 1 0 = 0 ∧
  sumMono X 1 0 1 = 0 ∧
  sumMono X 0 2 0 = (X.length : ℚ) * (r2 : ℚ) / 3 ∧
  sumMono X 0 1 1 = 0 ∧
  sumMono X 0 0 2 = (X.length : ℚ) * (r2 : ℚ) / 3

/-- Degree-3 moment equalities (10 monomials): every degree-3 monomial is odd, target `0`. -/
def Deg3 (X : List Coord) : Prop :=
  sumMono X 3 0 0 = 0 ∧ sumMono X 2 1 0 = 0 ∧ sumMono X 2 0 1 = 0 ∧
  sumMono X 1 2 0 = 0 ∧ sumMono X 1 1 1 = 0 ∧ sumMono X 1 0 2 = 0 ∧
  sumMono X 0 3 0 = 0 ∧ sumMono X 0 2 1 = 0 ∧ sumMono X 0 1 2 = 0 ∧
  sumMono X 0 0 3 = 0

/-- Degree-4 moment equalities (15 monomials): `m(xᵢ⁴) = 1/5`, `m(xᵢ²xⱼ²) = 1/15` (`i ≠ j`), all
    other degree-4 monomials (`2+1+1`, `3+1+0`, `4+0+0` other orderings) target `0`. -/
def Deg4 (X : List Coord) (r2 : ℤ) : Prop :=
  sumMono X 4 0 0 = (X.length : ℚ) * (r2 : ℚ) ^ 2 / 5 ∧
  sumMono X 3 1 0 = 0 ∧
  sumMono X 3 0 1 = 0 ∧
  sumMono X 2 2 0 = (X.length : ℚ) * (r2 : ℚ) ^ 2 / 15 ∧
  sumMono X 2 1 1 = 0 ∧
  sumMono X 2 0 2 = (X.length : ℚ) * (r2 : ℚ) ^ 2 / 15 ∧
  sumMono X 1 3 0 = 0 ∧
  sumMono X 1 2 1 = 0 ∧
  sumMono X 1 1 2 = 0 ∧
  sumMono X 1 0 3 = 0 ∧
  sumMono X 0 4 0 = (X.length : ℚ) * (r2 : ℚ) ^ 2 / 5 ∧
  sumMono X 0 3 1 = 0 ∧
  sumMono X 0 2 2 = (X.length : ℚ) * (r2 : ℚ) ^ 2 / 15 ∧
  sumMono X 0 1 3 = 0 ∧
  sumMono X 0 0 4 = (X.length : ℚ) * (r2 : ℚ) ^ 2 / 5

/-- The moment form for `X` at radius² `r2`, at orders `1..t`: the Delsarte–Goethals–Seidel
    spherical `t`-design condition (`t ≤ 4`; orders above `4` are outside this definition). -/
def IsDesign (X : List Coord) (r2 : ℤ) (t : ℕ) : Prop :=
  (1 ≤ t → Deg1 X) ∧ (2 ≤ t → Deg2 X r2) ∧ (3 ≤ t → Deg3 X) ∧ (4 ≤ t → Deg4 X r2)

/-! ### Tetrahedron: a 2-design, not a 3-design -/

theorem tetra_deg1 : Deg1 tetrahedron := by norm_num [Deg1, sumMono, tetrahedron]

theorem tetra_deg2 : Deg2 tetrahedron 3 := by norm_num [Deg2, sumMono, tetrahedron]

/-- The odd-moment witness: `∑ xyz` over `T` is `4`. -/
theorem tetra_xyz : sumMono tetrahedron 1 1 1 = 4 := by norm_num [sumMono, tetrahedron]

theorem tetra_design_two : IsDesign tetrahedron 3 2 :=
  ⟨fun _ => tetra_deg1, fun _ => tetra_deg2,
   fun h => absurd h (by decide), fun h => absurd h (by decide)⟩

theorem tetra_not_design_three : ¬ IsDesign tetrahedron 3 3 := by
  intro h
  have h3 : Deg3 tetrahedron := h.2.2.1 le_rfl
  obtain ⟨_, _, _, _, hxyz, _, _, _, _, _⟩ := h3
  rw [tetra_xyz] at hxyz
  exact absurd hxyz (by norm_num)

/-! ### Octahedron: a 3-design, not a 4-design -/

theorem octa_deg1 : Deg1 octahedron := by norm_num [Deg1, sumMono, octahedron]

theorem octa_deg2 : Deg2 octahedron 1 := by norm_num [Deg2, sumMono, octahedron]

theorem octa_deg3 : Deg3 octahedron := by norm_num [Deg3, sumMono, octahedron]

/-- The degree-4 witness: `∑ x⁴` over `O` is `2`, against the 4-design target `6/5`. -/
theorem octa_x4 : sumMono octahedron 4 0 0 = 2 := by norm_num [sumMono, octahedron]

theorem octa_design_three : IsDesign octahedron 1 3 :=
  ⟨fun _ => octa_deg1, fun _ => octa_deg2, fun _ => octa_deg3, fun h => absurd h (by decide)⟩

theorem octa_not_design_four : ¬ IsDesign octahedron 1 4 := by
  intro h
  have h4 : Deg4 octahedron 1 := h.2.2.2 le_rfl
  obtain ⟨hx4, _, _, _, _, _, _, _, _, _, _, _, _, _, _⟩ := h4
  rw [octa_x4] at hx4
  exact absurd hx4 (by norm_num [octahedron])

/-! ### Cube: a 3-design, not a 4-design -/

theorem cube_deg1 : Deg1 cube := by norm_num [Deg1, sumMono, cube]

theorem cube_deg2 : Deg2 cube 3 := by norm_num [Deg2, sumMono, cube]

theorem cube_deg3 : Deg3 cube := by norm_num [Deg3, sumMono, cube]

theorem cube_x4 : sumMono cube 4 0 0 = 8 := by norm_num [sumMono, cube]

theorem cube_design_three : IsDesign cube 3 3 :=
  ⟨fun _ => cube_deg1, fun _ => cube_deg2, fun _ => cube_deg3, fun h => absurd h (by decide)⟩

theorem cube_not_design_four : ¬ IsDesign cube 3 4 := by
  intro h
  have h4 : Deg4 cube 3 := h.2.2.2 le_rfl
  obtain ⟨hx4, _, _, _, _, _, _, _, _, _, _, _, _, _, _⟩ := h4
  rw [cube_x4] at hx4
  exact absurd hx4 (by norm_num [cube])

/-! ### Antipodal closure and the unit verdict -/

/-- `T ∪ (−T) = C` as finsets: the cube is the antipodal closure of the tetrahedron. -/
theorem tetra_union_is_cube :
    tetrahedron.toFinset ∪ (neg tetrahedron).toFinset = cube.toFinset := by decide

/-- The conjunction of clauses 1–4: the 2-design ladder opens on `T` and closes at the tesseral
    `C`, with `T ∪ (−T) = C`. -/
theorem verdict_l4 :
    IsDesign tetrahedron 3 2 ∧ ¬ IsDesign tetrahedron 3 3 ∧
    IsDesign octahedron 1 3 ∧ ¬ IsDesign octahedron 1 4 ∧
    IsDesign cube 3 3 ∧ ¬ IsDesign cube 3 4 ∧
    tetrahedron.toFinset ∪ (neg tetrahedron).toFinset = cube.toFinset :=
  ⟨tetra_design_two, tetra_not_design_three, octa_design_three, octa_not_design_four,
   cube_design_three, cube_not_design_four, tetra_union_is_cube⟩

end DesignLadder
end Quantum
