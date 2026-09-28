# Decode Atlas lens corpus: what the two Lean corpora already state

Task `APEIRON-DOM-MAP-T3`. Owner file: this file. Read-only survey of the two Lean corpora; no
Lean build was run.

Corpora and audits:

| corpus | root | audit file(s) | convention for `audited` below |
|---|---|---|---|
| apeiron | `proof/corpus` | `proof/corpus/AxiomAudit.lean` | declaration name occurs in a `#audit` line there |
| sanguine-apeiron | `proof/Apeiron` | `Audit.lean`, `AuditCarrier.lean`, `AuditClosed.lean`, `AuditCombinator.lean`, `AuditConstructive.lean` | name occurs in an `#audit`/`#audit_constructive` line there |

Grade convention (AGENTS §5, and the task's rule): a declaration whose name appears in the
relevant audit file is `proved (audited)`; one that compiles in the corpus but is not listed is
`stated, not audited`. Rust source has no Lean audit and is marked `machine path, no Lean audit`.

`AxiomAudit.lean` audits a selected headline set, not every declaration; absence from it is
absence of an axiom-cone check, not evidence the declaration is false.

Corpus digests (start): `proof/ApeironWord.lean` sha256
`a45996460f1577809337e1b9d647f34c906bc09156e63ef8bd1f8d993078f44a`.

Method: `rg` over both corpora for the object vocabulary, plus full reads of the seven named
modules. In the tables, `path:line` is the declaration's own line. Counts in the matrix are the
declarations recorded in that cell's detail table, not an exhaustive census of the corpus.

---

## The seven mandate files, in ≤5 lines each

**`proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean`** (namespace `Sanguine.SpinorLoop`).
An affine recurrence `S_t = (γ_t·r_t)·S_{t-1} + B_t` is carried as a spinor; `compose` is
associative in the scalar and `Matrix (Fin n)` cases, so prefix-scan equals left-fold. A constant
step's fixed point is exactly `B/(1−A)`, identified with `affineLoopCollapse`'s universal-equation
state. A Cayley step with skew generator is an isometry preserving `Σ‖ψ‖²` (Born density). Golden
section: `φ` is the reflexive fixed point, and the Fibonacci-point golden orbit on the circle has
only the two gap lengths `ρ^{n+1}`, `ρ^n` (three-distance theorem, Fibonacci counts). Aitken Δ² on
three consecutive iterates returns the fixed point exactly; a safe a-posteriori bound needs one
observed step and a known contraction rate. At the time of writing no `Sanguine.SpinorLoop.*`
declaration is in `AxiomAudit.lean`.

**`proof/corpus/Substrate/Ski/IotaTermAlgebra.lean`** (namespace `Sanguine.IotaTermAlgebra`).
`IotaTerm` has three constructors (`K`, `S_routeF`, `S_routeG`) and `state` evaluates by recursive
division at applicative `/`. `fromBulkTower`/`toBulkTower` are mutual inverses and both preserve
`state`. The header states explicitly that the type-polymorphic SKI Iota calculus over generic
juxtaposition is not captured here, and names `GoldbergIotaChains.iota_surface_chains` as the
reduction-surface facts.

**`proof/corpus/Substrate/Ski/StateVectorProjection.lean`** (namespace `Sanguine`).
Models the Church cons-list state vector as `List α` and proves the three projection equations by
`rfl`: `proj_zero_cons` (depth 0 returns the head), `proj_snd_cons` (one cell off exposes the
tail), `proj_succ_cons` (`(h::t).drop (n+1) = t.drop n`). The closed form
`proj_d (cons r_{N-1} … nil) = some r_{N-1-d}` is stated as their composition, not a theorem here.

**`proof/corpus/Nous/Model/FaceContraction.lean`** (namespace `Nous.Model`).
Per real face of `R[x]/(x²−σ)` supplies a metric and a rate with
`d_σ (ringMul σ a x) (ringMul σ a y) ≤ rate a · d_σ x y`: elliptic `σ=−1` (Euclidean magnitude),
hyperbolic `σ=1` (max of the two channel readings), parabolic `σ=0` (weighted `|x₀|c+|x₁|`).
`trajectory_sub` cancels the affine drives; `homRun_bound`, `contraction_orbit` and
`bounded_orbit` bound orbits by `ρ^n` and `B/(1−ρ)`; per-face corollaries close each face.

**`proof/corpus/Nous/Model/FaceRecurrence.lean`** (namespace `Nous.Model`).
The affine step `h ↦ ringMul σ a h + b` over `RingCell R` is the pair `AffineCell σ`; `affComp` is
associative (`affComp_assoc`), `affApply_comp` says applying a composition applies the inner step
then the outer, and `(affId, affComp)` is a `Monoid`. `recur_eq_scan` proves the sequential
recurrence equals applying the single composed step; `scan_split` splits a concatenated scan;
`recur_channel_split` factors the `σ=1` recurrence into two scalar recurrences.

**`proof/corpus/Nous/Model/RopeOrbit.lean`** (namespace `Nous.Model`).
Rotation orbit on the unit circle: `rho θ = (cos θ, sin θ)`, `rhoMul` complex multiplication,
`rho_add` the homomorphism law, `rho_nat` the orbit-power law `ρ(kθ)=ρ(θ)^k`. `rho_sub`/
`rho_relative` and `rotMat_relative` prove `R_mᵀ R_n = R_{n−m}`, so the attention score depends only
on the position difference; `rhoDot_act` proves the rotation is an inner-product isometry.
`rope_orbit` bundles the homomorphism and orbit laws.

**`proof/corpus/Nous/Model/Attention.lean`** (namespace `Nous.Model`).
Channel-form attention over any field: a streaming scan accumulates the running normalizer
`channelDen` and weighted sum `channelNum`, normalized once. `attention_channel_form` is the
headline: on `ChannelFormDomain` the scan, the single normalization, exactness, the factorized
score form `channelWeight_factor`, and bijectivity of `channelPair r` all hold. The parabolic face
(`σ=0`, `r=0`) degenerates to the uniform average on `ParabolicChannelFormDomain`; the elliptic
face over `ℝ` is an excluded population because `r²=σ<0` has no root.

---

## Matrix: object × lens

Cell form: `count — strongest declaration (path:line)`. `no match` means no theorem/def relevant
to that cell was found in either corpus.

| object \ lens | (a) operator | (b) combinator | (c) physics | (d) geometry | (e) machine |
|---|---|---|---|---|---|
| 1 torus / RoPE / orbit | 8 — `rhoDot_act` `proof/corpus/Nous/Model/RopeOrbit.lean:159` | no match | 5 — `golden_three_distance_fib` `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:1107` | 4 — `rho_nat` `proof/corpus/Nous/Model/RopeOrbit.lean:91` | 4 — `rotate_loop_spec` `proof/corpus/Numeric/RopeRotate.lean:433` |
| 2 sphere / RMSNorm / Hadamard | 8 — `hadamardRef_isometry_l2` `proof/corpus/Nous/Model/RotationStage.lean:224` | no match | no match | 3 — `hadamardRef_involutive` `proof/corpus/Nous/Model/RotationStage.lean:262` | 4 — `fwht_fixed_refines_sylvester` `proof/corpus/Numeric/HadamardFixedSpec.lean:559` |
| 3 lattice S·Z³² / quant / residue | 7 — `hasSignature_one_of_gram_eq_mulTranspose` `proof/Apeiron/Lattice/IntegralForm.lean:586` | no match | 6 — `bosonic_twentysix` `proof/Apeiron/Lattice/EvenUnimodular.lean:589` | 3 — `sigPos_add_sigNeg` `proof/Apeiron/Lattice/IntegralForm.lean:308` | 5 — `dequant_q4_k_row_spec` `proof/corpus/Numeric/DequantQ4K.lean:874` |
| 4 Gibbs / softmax / argmax | 5 — `channelWeight_binary_softmax` `proof/corpus/Nous/Model/Weighting.lean:181` | no match | 2 — `cross_ge_entropy` `proof/corpus/Substrate/Algebra/Neural/CountLMFloor.lean:142` | no match | 4 — `pruned_argmax_exact` `proof/corpus/Numeric/PrunedArgmax.lean:92` |
| 5 weight rectangle / matvec | 5 — `matvec_stage_proved` `proof/corpus/Numeric/MatvecBridge.lean:172` | no match | no match | no match | 5 — `matvec_fixed_chunk` `proof/corpus/Numeric/MatvecChunk.lean:345` |
| 6 chain / token fold / recurrence | 5 — `run_state_eq_tokenFold` `proof/corpus/Nous/Model/Model.lean:85` | 3 — `combinatory_completeness` `proof/Apeiron/Combinator/Basic.lean:402` | 5 — `aitken_exact` `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:1240` | 3 — `recur_eq_scan` `proof/corpus/Nous/Model/FaceRecurrence.lean:158` | 2 — `Engine::step_with_kv` `src/engine/step.rs:1128` |
| 7 spinors / double cover / Clifford | 5 — `nmul_is_diagonal` `proof/corpus/Substrate/Algebra/NullCell/SpinorFold.lean:117` | no match | 7 — `cayley_conserves_born_density` `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:595` | 4 — `rotor_compose` `proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean:355` | 1 — `Cell::mul` `src/cell.rs:127` |
| 8 cells R[x]/(x²−σ) | 9 — `trichotomy_cstar_classification` `proof/Apeiron/Trichotomy/Derived/CStar.lean:563` | 3 — `contraction` `proof/Apeiron/Combinator/Linear/Core/Contraction.lean:916` | 7 — `hyperbolic_decomposition` `proof/corpus/Nous/Model/Decomposition.lean:148` | 4 — `act_mul` `proof/Apeiron/Projective/Basic.lean:249` | 3 — `Cell::mul` `src/cell.rs:127` |

Two cells that the dataset's framing suggests but the corpora do not fill, recorded explicitly:
**no real Clifford algebra, `Spin(n)`, covering degree or `Spin(2)→SO(2)` theorem exists in either
corpus**; the double cover appears only as prose (`SpinorFold.lean` header calls it "Next";
`SpinorLoop.lean` names `ConcreteRotor`/`SpinorWinding`, no such declaration exists). **No
combinator-corpus content exists for objects 1–5**; the `Combinator/` corpus is about SKI routes
and linear-logic wiring, which touch objects 6 and 8 only.

---

## Object 1 — torus / RoPE / rotation orbit

### (a) operator / linear-algebraic

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/corpus/Nous/Model/RopeOrbit.lean:81` | `rho_add` | `ρ(θ+φ) = rhoMul (ρ θ) (ρ φ)`: the angle map is a homomorphism to the circle. | proved (audited) |
| `proof/corpus/Nous/Model/RopeOrbit.lean:102` | `rhoMul_assoc` | `rhoMul` is associative. | proved (audited) |
| `proof/corpus/Nous/Model/RopeOrbit.lean:118` | `rho_sub` | `ρ(θ−φ) = rhoMul (ρ θ) (ρ(−φ))`. | proved (audited) |
| `proof/corpus/Nous/Model/RopeOrbit.lean:149` | `rhoAct_add` | rotations compose: `rhoAct θ (rhoAct φ z) = rhoAct (θ+φ) z`. | proved (audited) |
| `proof/corpus/Nous/Model/RopeOrbit.lean:159` | `rhoDot_act` | `rhoDot (rhoAct θ z) (rhoAct θ w) = rhoDot z w`: an inner-product isometry. | proved (audited) |
| `proof/corpus/Nous/Model/RopeOrbit.lean:190` | `rotMat_mul` | `R(θ) R(φ) = R(θ+φ)`. | proved (audited) |
| `proof/corpus/Nous/Model/RopeOrbit.lean:199` | `rotMat_transpose` | `R(θ)ᵀ = R(−θ)`. | proved (audited) |
| `proof/corpus/Nous/Model/RopeOrbit.lean:210` | `rotMat_relative` | `R_mᵀ R_n = R_{n−m}`. | proved (audited) |

### (b) combinator

no match. No `Combinator/` declaration in either corpus refers to a rotation, circle or orbit.

### (c) physics / stationary phase

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:687` | `golden_gap_split` | `ρ^k = ρ^{k+1} + ρ^{k+2}` for `ρ = 1/φ`: one golden gap is the sum of the next two. | stated, not audited |
| `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:719` | `golden_subdivision_preserves_three` | Splitting golden gaps keeps the present exponents inside a window of three. | stated, not audited |
| `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:753` | `golden_tiling` | `fib(n+1)·ρ^n + fib(n)·ρ^{n+1} = 1`: the two gap lengths with Fibonacci multiplicities fill the circle. | stated, not audited |
| `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:858` | `golden_orbit_fract` | `fract(i·ρ)` is a rational-lattice slot plus a perturbation smaller than one cell. | stated, not audited |
| `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:1107` | `golden_three_distance_fib` | Adjacent points of the `fib(n+2)`-point golden orbit differ by exactly `ρ^{n+1}` or `ρ^n`. | stated, not audited |

`Sanguine.SpinorLoop.*` is absent from `AxiomAudit.lean` at this revision; the header itself scopes
the physics reading ("a loop IS a spinor") as an anchor.

### (d) geometry / group action

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/corpus/Nous/Model/RopeOrbit.lean:76` | `rhoPow_succ` | one power unfolds: `rhoPow (k+1) z = rhoMul (rhoPow k z) z`. | stated, not audited |
| `proof/corpus/Nous/Model/RopeOrbit.lean:91` | `rho_nat` | `ρ(k·θ) = ρ(θ)^k`: the position table is the orbit of one point under `rhoMul`. | proved (audited) |
| `proof/corpus/Nous/Model/RopeOrbit.lean:126` | `rho_relative` | rotation by `nθ` after inverse rotation by `mθ` is rotation by `(n−m)θ`. | proved (audited) |
| `proof/corpus/Nous/Model/RopeOrbit.lean:223` | `rope_orbit` | bundles `rho_add` and `rho_nat`. | proved (audited) |

### (e) machine

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `src/engine/rope.rs:29` | `RopeTable` / `rotate` `src/engine/rope.rs:139` | the Rust per-position fixed-point cos/sin table and head rotation. | machine path, no Lean audit |
| `proof/corpus/Numeric/RopeRotate.lean:433` | `rotate_loop_spec` | on the success domain the extracted `rotate_loop` returns `true` and slices denoting the concatenated per-head `rotPrefix` contents. | proved (audited) |
| `proof/corpus/Numeric/RopeHeadContent.lean:369` | `rotate_head_content` | `rotate_head` returns `Ok ()` and a head denoting `rotPrefix x d2 scAt d2`. | proved (audited) |
| `proof/corpus/Numeric/EngineRopeRefinement.lean:399` | `rope_pair_spec_proved` | extracted `rope_rotate_fixed` satisfies `RopePairSpec`. | stated, not audited |

---

## Object 2 — sphere / RMSNorm / orthogonal and Hadamard rotations

### (a) operator

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/corpus/Numeric/Sylvester.lean:61` | `sylvester_succ` | the `2^{k+1}` Sylvester matrix is `fromBlocks` of the `2^k` one and its negation. | proved (audited) |
| `proof/corpus/Numeric/Sylvester.lean:89` | `sylvester_transpose` | `sylvesterH k` is symmetric. | proved (audited) |
| `proof/corpus/Numeric/Sylvester.lean:133` | `sylvester_mul_self` | `sylvesterH k * sylvesterH k = 2^k • 1`: involution up to `2^k`. | proved (audited) |
| `proof/corpus/Numeric/Sylvester.lean:181` | `sylvester_one` | `sylvesterH 1 = !![1,1;1,-1]`. | proved (audited) |
| `proof/corpus/Numeric/Sylvester.lean:211` | `sylvester_mulVec_butterfly` | the matrix-vector product is the radix-2 butterfly on top/bottom halves. | proved (audited) |
| `proof/corpus/Nous/Model/RotationStage.lean:52` | `sylvesterH_abs_entry` | every entry of `sylvesterH k` has absolute value 1. | stated, not audited |
| `proof/corpus/Nous/Model/RotationStage.lean:262` | `hadamardRef_involutive` | `hadamardRef k s` is an involution. | proved (audited) |
| `proof/corpus/Nous/Model/RotationStage.lean:224` | `hadamardRef_isometry_l2` | on sign vectors, `hadamardRef` preserves the `l2` norm. | proved (audited) |

### (b) combinator

no match.

### (c) physics

no match. Neither corpus states a physical or statistical-mechanics property of the Hadamard
transform or of RMSNorm.

### (d) geometry / group action

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/corpus/Nous/Model/RotationStage.lean:95` | `hadamardMat_mul_self` | `hadamardMat k * hadamardMat k = 2^k • 1`. | stated, not audited |
| `proof/corpus/Nous/Model/RotationStage.lean:262` | `hadamardRef_involutive` | the signed transform is an involution. | proved (audited) |
| `proof/corpus/Numeric/Sylvester.lean:133` | `sylvester_mul_self` | the Sylvester matrix is an involution up to the scalar `2^k`. | proved (audited) |

### (e) machine

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `src/engine/hadamard.rs:195` | `fwht_fixed` | the executing fixed-point signed fast Walsh–Hadamard transform. | machine path, no Lean audit |
| `proof/corpus/Numeric/HadamardFixedSpec.lean:340` | `fwht_fixed_spec` | on the checked signs/overflow domain `fwht_fixed` returns `Ok ()` and the rounded reference values. | proved (audited) |
| `proof/corpus/Numeric/HadamardFixedSpec.lean:559` | `fwht_fixed_refines_sylvester` | the same with the block value read through `sylvesterH k` on aligned segments. | proved (audited) |
| `proof/corpus/Numeric/EngineNormRefinement.lean:726` | `rms_norm_refines_proved` | extracted `rms_norm_fixed` refines its real reference. | proved (audited) |

---

## Object 3 — lattice `S·Z³²` / quantization and its residue

### (a) operator / linear-algebraic

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/Apeiron/Lattice/IntegralForm.lean:112` | `IntForm.bil` | the integral bilinear form `xᵀ G y`. | stated, not audited |
| `proof/Apeiron/Lattice/IntegralForm.lean:120` | `bil_eq_double_sum` | the form as the double sum `Σ_i Σ_c x_i G_ic y_c`. | stated, not audited |
| `proof/Apeiron/Lattice/IntegralForm.lean:147` | `IntForm.IsEven` | every vector pairs with itself to an even integer. | stated, not audited |
| `proof/Apeiron/Lattice/IntegralForm.lean:207` | `IntForm.IsUnimodular` | the Gram determinant is a unit of `ℤ`. | stated, not audited |
| `proof/Apeiron/Lattice/IntegralForm.lean:276` | `IntForm.HasSignature` | existence of a real `P` with `Pᵀ Gℝ P = diagonal s`. | proved (audited) |
| `proof/Apeiron/Lattice/IntegralForm.lean:586` | `hasSignature_one_of_gram_eq_mulTranspose` | `Gℝ = B Bᵀ` with `det B ≠ 0` gives the all-positive signature. | proved (audited) |
| `proof/Apeiron/Decider/Residue.lean:83` | `residueRow` | row normalization keeping the first coefficient in `{−1,0,1}` (a different sense of "residue": row divisor modulus, not quantization error). | proved (audited) |

Additional residue rows: `residueModulus` `proof/Apeiron/Decider/Residue.lean:150` (positive common
modulus) and `satRows_residue_iff` `proof/Apeiron/Decider/Residue.lean:186` (satisfaction is
preserved inside a residue class), both proved (audited).

### (b) combinator

no match. The `Combinator/` lattice references are Route/Cell wiring (object 8), not an integral
lattice.

### (c) physics

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/Apeiron/Lattice/EvenUnimodular.lean:145` | `hyperbolicPlane_isEven` | the hyperbolic plane `U` is an even form. | stated, not audited |
| `proof/Apeiron/Lattice/EvenUnimodular.lean:385` | `E8_det` | the E8 Gram determinant is 1. | proved (audited) |
| `proof/Apeiron/Lattice/EvenUnimodular.lean:435` | `evenUnimodular` | the family `E8^e ⊕ U^u`. | stated, not audited |
| `proof/Apeiron/Lattice/EvenUnimodular.lean:486` | `evenUnimodular_hasSignature` | the family has the signature witness `(8e+u, u)`. | proved (audited) |
| `proof/Apeiron/Lattice/EvenUnimodular.lean:550` | `eight_dvd_signature_difference` | 8 divides the signature difference on this family. | proved (audited) |
| `proof/Apeiron/Lattice/EvenUnimodular.lean:589` | `bosonic_twentysix` | `E8³ ⊕ U` is even, unimodular, rank 26 with signature `(25,1)`. | proved (audited) |

### (d) geometry / group action

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/Apeiron/Lattice/IntegralForm.lean:234` | `SignatureSign` | the two finite signs of a signature. | proved (audited) |
| `proof/Apeiron/Lattice/IntegralForm.lean:276` | `HasSignature` | a real congruence diagonalizing the form to the sign vector. | proved (audited) |
| `proof/Apeiron/Lattice/IntegralForm.lean:308` | `sigPos_add_sigNeg` | the two signature counts exhaust the rank. | proved (audited) |

### (e) machine

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `src/engine/dequant.rs:429` | `dequant_q4_k_row` | the executing Q4_K block dequantizer. | machine path, no Lean audit |
| `proof/corpus/Numeric/DequantQ4K.lean:874` | `dequant_q4_k_row_spec` | the extracted Q4_K row dequantizer returns the specified values. | proved (audited) |
| `proof/corpus/Numeric/ChartDot.lean:139` | `qcode_err` | `2·abs(x − S·qcode S x) ≤ S`: the per-element quantization residue bound. | proved (audited) |
| `proof/corpus/Numeric/ChartDot.lean:217` | `chart_factor` | a weight-times-code sum factors into the weighted code sum minus the rounding term. | proved (audited) |
| `proof/corpus/Numeric/ChartDot.lean:299` | `row_err_bound` | the row dot's dequantization residue is bounded by `scale · Σ|w|`. | proved (audited) |

---

## Object 4 — Gibbs measure / softmax / argmax

### (a) operator / linear-algebraic

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/corpus/Nous/Model/TopKRouter.lean:231` | `softmax` | `exp z_i / Σ_j exp z_j`. | stated, not audited |
| `proof/corpus/Nous/Model/TopKRouter.lean:245` | `softmax_sum_eq_one` | the softmax probabilities sum to one. | stated, not audited |
| `proof/corpus/Nous/Model/Weighting.lean:167` | `ratio_eq_exp_log` | `u/(u+l) = exp(log u)/(exp(log u)+exp(log l))` for positive `u,l`. | proved (audited) |
| `proof/corpus/Nous/Model/Weighting.lean:181` | `channelWeight_binary_softmax` | the channel weight of a ring point is the binary softmax of its log channels. | proved (audited) |
| `proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean` / `proof/corpus/Numeric/Accumulator.lean` | `softmax_attention_is_online_spinor` / `softmax_shift_exact` | online softmax is exact under a shared shift. | proved (audited) |

### (b) combinator

no match.

### (c) physics / statistical mechanics

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/corpus/Substrate/Algebra/Neural/CountLMFloor.lean:116` | `kl_nonneg` | the Kullback–Leibler divergence is nonnegative (Gibbs' inequality), under positive probabilities. | stated, not audited |
| `proof/corpus/Substrate/Algebra/Neural/CountLMFloor.lean:142` | `cross_ge_entropy` | cross-entropy is at least entropy. | stated, not audited |

`AxiomAudit.lean` lists only `Sanguine.CountLMFloor.refine_gap_zero_of_kappa_sufficient` from this
module, so the Gibbs-inequality headlines carry no axiom-cone check at this revision.

### (d) geometry

no match.

### (e) machine

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `src/engine/sampler.rs:41` | `argmax_fixed` | the executing fixed-point argmax. | machine path, no Lean audit |
| `proof/corpus/Numeric/PrunedArgmax.lean:92` | `pruned_argmax_exact` | bound-pruned greedy argmax returns the same least maximizer as the unpruned one. | proved (audited) |
| `proof/corpus/Numeric/PrunedArgmax.lean:227` | `word_argmax_fixed_returns_lowest` | the extracted argmax returns the lowest maximizing index. | proved (audited) |
| `proof/corpus/Numeric/EngineSamplerRefinement.lean:158` | `argmax_loop_spec` | the extracted argmax loop returns the specified index. | proved (audited) |

---

## Object 5 — weight rectangle / matvec partitions

### (a) operator / linear-algebraic

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/corpus/Numeric/MatvecBridge.lean:43` | `deqMatrix` | the dequantized matrix as `Fin m → Fin n → ℝ`. | stated, not audited |
| `proof/corpus/Numeric/MatvecBridge.lean:47` | `gridVec` | grid values of a fixed-point vector. | stated, not audited |
| `proof/corpus/Numeric/MatvecBridge.lean:70` | `rowDotReal_eq_matvecRef` | the real row dot equals the reference matvec row. | proved (audited) |
| `proof/corpus/Numeric/MatvecBridge.lean:172` | `matvec_stage_proved` | the built `Stage`'s defect is at most `matvecEps` on its domain. | proved (audited) |
| `proof/corpus/Numeric/MatvecBridgeGrid.lean:207` | `matvec_stage_grid_proved` | the grid-grounded matvec stage defect bound. | proved (audited) |

### (b) combinator

no match.

### (c) physics

no match.

### (d) geometry

no match.

### (e) machine (partition algebra)

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `src/engine/matvec.rs:149` | `matvec_fixed` | the executing row-partitioned fixed-point matrix-vector product. | machine path, no Lean audit |
| `proof/corpus/Numeric/MatvecChunk.lean:321` | `matvec_fixed_rows` | the output is the map of the per-row value. | proved (audited) |
| `proof/corpus/Numeric/MatvecChunk.lean:345` | `matvec_fixed_chunk` | a row-aligned byte split gives output concatenation. | proved (audited) |
| `proof/corpus/Numeric/MatvecChunk.lean:383` | `rowOut_join` | per-row reference over consecutive chunks is the join of the per-chunk references. | proved (audited) |
| `proof/corpus/Numeric/MatvecChunk.lean:443` | `matvec_fixed_chunks` | any list of consecutive chunks joins. | proved (audited) |

---

## Object 6 — autoregressive chain / token fold / recurrence

### (a) operator / fold denotation

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/corpus/Nous/Model/State.lean:82` | `token_step_fold` | the sequential token fold over concatenated lists splits into the product of the folds. | proved (audited) |
| `proof/corpus/Nous/Model/State.lean:101` | `tokenFold_append` | `tokenFold σ (xs ++ ys) = ringMul σ (tokenFold σ xs) (tokenFold σ ys)`. | proved (audited) |
| `proof/corpus/Nous/Model/Model.lean:85` | `run_state_eq_tokenFold` | the streaming state of `run` equals `tokenFold` of the value stream. | proved (audited) |
| `proof/corpus/Nous/Model/Model.lean:98` | `run_memory_eq_memoryFold` | the memory component of `run` equals the sequential `memoryStep` fold. | proved (audited) |
| `proof/corpus/Nous/Model/Model.lean:117` | `run_eq` | `run` is exactly the pair `(tokenFold, memoryFold)`. | proved (audited) |

### (b) combinator

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/Apeiron/Combinator/Basic.lean:199` | `iota` | the single constant `S(S I)(K S))(K K)`. | stated, not audited |
| `proof/Apeiron/Combinator/Basic.lean:208` | `iota_app` | `iota M` reduces to `M S K`. | proved (audited) |
| `proof/Apeiron/Combinator/Basic.lean:402` | `combinatory_completeness` | for any term `e` there is `c` with no free `i`, the same other free variables, and `c x ↝ e[i:=x]`. | proved (audited) |

Iterated-recurrence combinator rows in the same corpus: `fix_unfolds`
`proof/Apeiron/Combinator/Recursion.lean:149` (`fix f ↝ f (fix f)`) and `fix_normalises_for_const`
`proof/Apeiron/Combinator/Recursion.lean:173` (a discarding `f` reaches `K`), both proved
(audited). `iota_is_I`/`iota_is_K`/`iota_is_S` (`proof/Apeiron/Combinator/Basic.lean:226`) are not themselves
in the audit list, but the bundle's members `iota_app` and `iota_generates` are.

### (c) physics / stationary phase

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:190` | `compose_assoc` | the scalar affine-step composition is associative. | stated, not audited |
| `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:219` | `constantStep_fixed_point` | `x = A x + B ↔ x = B/(1−A)` for `A ≠ 1`. | stated, not audited |
| `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:1180` | `iterate_sub_fixed_point` | the error of a constant affine loop is `A^n` times the initial displacement. | stated, not audited |
| `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:1240` | `aitken_exact` | Aitken Δ² on any three consecutive iterates returns `B/(1−A)` exactly. | stated, not audited |
| `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:1299` | `safe_error_bound` | the true error is bounded by one observed step over `1−ρ`, for known `|A| ≤ ρ < 1`. | stated, not audited |

### (d) geometry / monoid action

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/corpus/Nous/Model/FaceRecurrence.lean:65` | `affComp_assoc` | affine-step composition is associative. | proved (audited) |
| `proof/corpus/Nous/Model/FaceRecurrence.lean:158` | `recur_eq_scan` | the sequential recurrence equals applying the single composed step. | proved (audited) |
| `proof/corpus/Nous/Model/FaceRecurrence.lean:179` | `scan_split` | the composed element of `xs ++ ys` is the composition with `ys` outside. | proved (audited) |

### (e) machine

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `src/engine/step.rs:1128` | `Engine::step_with_kv` | the executing per-token forward step against a KV state. | machine path, no Lean audit |
| `proof/corpus/Numeric/DequantRefinement.lean:397` | `block_geom_f16` | a supporting extracted-kernel geometry fact used by the machine step. | proved (audited) |

---

## Object 7 — spinors / double cover / Clifford algebra

### (a) operator / algebra

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/corpus/Substrate/Algebra/NullCell/SpinorFold.lean:106` | `norm_is_product_of_halves` | the null-cell norm is the product of the two channel (half) readings. | proved (audited) |
| `proof/corpus/Substrate/Algebra/NullCell/SpinorFold.lean:117` | `nmul_is_diagonal` | the two-channel multiply is the diagonal of the four-product multiply. | proved (audited) |
| `proof/corpus/Substrate/Algebra/NullCell/SpinorFold.lean:134` | `spinorFold_eq` | the spinor fold equals the pair of channel products. | proved (audited) |
| `proof/corpus/Substrate/Algebra/NullCell/SpinorFold.lean:174` | `nmul_assoc` | the channel multiply is associative. | proved (audited) |
| `proof/corpus/Substrate/Algebra/NullCell/SpinorFold.lean:180` | `rotor_folds_into_weight` | a unit rotor folds into the weight without changing the norm. | proved (audited) |

### (b) combinator

no match.

### (c) quantum mechanics / physics

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:508` | `skew_re_zero` | for skew-adjoint `A`, `re ⟪A s, s⟫ = 0`. | proved (audited) |
| `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:524` | `cayley_isometry` | a Cayley step with skew generator preserves the norm. | proved (audited) |
| `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:549` | `schrodinger_generator_skew` | `i·c·H` is skew-adjoint for Hermitian `H`. | stated, not audited |
| `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:560` | `cayley_factor_unit` | the Cayley factor lies on the unit circle. | stated, not audited |
| `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:595` | `cayley_conserves_born_density` | `Σᵢ‖ψ i‖²` is invariant under one Cayley step. | stated, not audited |
| `proof/Apeiron/Physics/Wick.lean:83` | `wick_flips_the_sign` | `z₀² − σ (z₁ i)² = normForm (−σ) z` over `ℂ`. | proved (audited) |
| `proof/Apeiron/Physics/Wick.lean:111` | `flow_wick` | the elliptic turn is the hyperbolic boost at imaginary time. | proved (audited) |

### (d) geometry / group action

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean:346` | `rotor_isometry` | the plane rotor preserves the inner product. | stated, not audited |
| `proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean:355` | `rotor_compose` | rotors compose: `rotor θ ∘ rotor φ = rotor (θ+φ)`. | stated, not audited |
| `proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean:364` | `rotor_full_turn` | a full turn is the identity. | stated, not audited |
| `proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean:373` | `so2_spinor_recurrence` | the SO(2) step as a spinor recurrence. | stated, not audited |

Double cover / `Spin(n)` / Clifford: **no match**.
`proof/corpus/Substrate/Algebra/NullCell/SpinorFold.lean:58` names the covering-degree
claim as "Next"; `proof/corpus/Substrate/StationaryPhase/SpinorLoop.lean:557` names `ConcreteRotor`
and `SpinorWinding`, neither of which
exists in either corpus. The nearest delivered algebra is the Cayley–Dickson doubling
(`proof/Apeiron/Doubling/CayleyDickson.lean`), which is a doubling construction, not a Clifford
algebra: `doubleNorm_eq_zero_iff_of_neg` `proof/Apeiron/Doubling/CayleyDickson.lean:447` states the
doubled form is definite for `σ < 0`, stated, not audited.

### (e) machine

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `src/cell.rs:127` | `Cell::mul` | the executing word-ring multiplication by `sigma`. | machine path, no Lean audit |

---

## Object 8 — the cells `R[x]/(x²−σ)` and their three faces

### (a) operator / C*-algebra

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/Apeiron/Trichotomy/AnyRing.lean:168` | `ringNormForm_mul` | `N(zw) = N z · N w`: the norm form is multiplicative. | proved (audited) |
| `proof/Apeiron/Trichotomy/AnyRing.lean:191` | `ringMul_assoc` | `ringMul σ` is associative. | stated, not audited |
| `proof/Apeiron/Trichotomy/AnyRing.lean:348` | `ringCayley_normForm` | a Cayley factor with carried inverse has norm form 1. | proved (audited) |
| `proof/Apeiron/Trichotomy/Basic.lean:148` | `mul_assoc` | the real cell multiply is associative. | proved (audited) |
| `proof/Apeiron/Trichotomy/Basic.lean:446` | `normForm_mul` | the real norm form is multiplicative. | proved (audited) |
| `proof/Apeiron/Trichotomy/Basic.lean:925` | `classification` | every real 2-dimensional algebra is linearly isomorphic to `ℝ × ℝ` with the σ-multiplication for `σ ∈ {−1,0,1}`. | proved (audited) |
| `proof/Apeiron/Trichotomy/Basic.lean:1033` | `trichotomy_complete` | the zero-divisor presence is exactly `0 ≤ σ`, matching the classified `σ`. | proved (audited) |
| `proof/Apeiron/Trichotomy/Derived/CStar.lean:107` | `cstar_identity_pos` | on the hyperbolic face `1 + z*z` has norm square `(‖z‖)²` on both channels. | proved (audited) |
| `proof/Apeiron/Trichotomy/Derived/CStar.lean:563` | `trichotomy_cstar_classification` | each face carries its own C*-identity branch: elliptic under `conj`, hyperbolic under `starDiag` (and fails under `conj`), parabolic has a nonzero nilpotent and no faithful C*-norm. | proved (audited) |

### (b) combinator (`σ`-parametric routes)

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/Apeiron/Combinator/Linear/Core/Contraction.lean:916` | `contraction` | `rapp σ (conjRoute wIdx x) (rtensor y y) = rapp σ (rapp σ x y) y`: the contraction route is `x y y`. | proved (audited) |
| `proof/Apeiron/Combinator/Linear/Core/Contraction.lean:373` | `contraction_left` | turns the left route into one path relation over the two loops the two uses run on. | proved (audited) |
| `proof/Apeiron/Combinator/Linear/Core/Contraction.lean:858` | `collapse_step` | the collapse step of the contraction route. | proved (audited) |

### (c) physics / face decomposition

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/corpus/Nous/Model/Decomposition.lean:116` | `channelPair_mul` | the channel pair intertwines `ringMul σ` with componentwise multiplication at a root `r²=σ`. | proved (audited) |
| `proof/corpus/Nous/Model/Decomposition.lean:141` | `channelPair_normForm` | the norm form factors through the channel pair. | proved (audited) |
| `proof/corpus/Nous/Model/Decomposition.lean:148` | `hyperbolic_decomposition` | `channelPair r` is bijective and intertwines multiplication, addition and unit with componentwise operations, factoring the norm form. | proved (audited) |
| `proof/corpus/Nous/Model/Decomposition.lean:260` | `hyperbolic_decomposition_field` | the same over a field under `IsUnit` hypotheses. | proved (audited) |
| `proof/corpus/Nous/Model/FaceContraction.lean:141` | `euclid_contraction` | elliptic `σ=−1` contracts the Euclidean distance by `√(a₀²+a₁²)`. | proved (audited) |
| `proof/corpus/Nous/Model/FaceContraction.lean:204` | `hyper_contraction` | hyperbolic `σ=1` contracts the channel-max distance by the max channel rate. | proved (audited) |
| `proof/corpus/Nous/Model/FaceContraction.lean:261` | `para_contraction` | parabolic `σ=0` contracts the weighted distance by `|a₀|+|a₁|/c`. | proved (audited) |

### (d) geometry / projective action

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `proof/Apeiron/Projective/Basic.lean:235` | `act_one` | the identity matrix acts trivially. | proved (audited) |
| `proof/Apeiron/Projective/Basic.lean:249` | `act_mul` | `act (M*N) q = act M (act N q)`: the `2×2` matrices act as a group on the projective line. | proved (audited) |
| `proof/Apeiron/Trichotomy/Derived/Projection.lean:109` | `proj_idemPlus_halves` | projection by the idempotent `ePlus` reads the two channel halves on `σ>0`. | proved (audited) |
| `proof/Apeiron/Trichotomy/Derived/Projection.lean:147` | `discards_iff_disc_pos` | the parabolic-type projection discards exactly when `0 < σ`. | proved (audited) |

### (e) machine

| path:line | declaration | plain statement | audit |
|---|---|---|---|
| `src/cell.rs:127` | `Cell::mul` | the word-ring multiplication implementing `R[x]/(x²−σ)` at `R = i64`. | machine path, no Lean audit |
| `src/cell.rs:160` | `Cell::norm_form` | the word-ring norm form. | machine path, no Lean audit |
| `src/scalar.rs:28` | `trait Ring` | the ring trait the cell container is generic over. | machine path, no Lean audit |

---

## Residual obligations

- `Sanguine.SpinorLoop.*`, `Sanguine.AttentionSpinor.rotor_*`, `Nous.Model.hadamardMat_*`,
  `Sanguine.CountLMFloor.kl_nonneg/cross_ge_entropy` and the `Lattice`/`Doubling` even/unimodular
  facts not listed above are `stated, not audited`: an axiom-cone check for them is missing.
- No Clifford algebra, `Spin(n)`, covering degree or double-cover theorem exists; the double cover
  is prose. This is a corpus gap for object 7 lens (b) and the covering-degree reading of lens (a).
- The `Decider/Residue` "residue" is a row-divisor modulus, not quantization residue; the
  quantization residue is `ChartDot.qcode_err`/`row_err_bound`.
- Rust machine paths are not refined to Lean at the named functions by any lease; `rotate_*`,
  `fwht_fixed_*`, `rms_norm_refines_proved`, `dequant_*`, `matvec_*`, `argmax_*` are the nearest
  extracted refinements.

## Execution record

- Started `date`: Mon 28 Sep 2026 07:07:12 AEST; final `date`: Mon 28 Sep 2026 07:11:27 AEST
  (elapsed 4 min 15 s of the 120 min budget).
- Gate: `python3 /home/socol/oc-tmp/dom-map-t3/check_paths.py docs/evidence/atlas-lens-corpus.md`
  → `checked=143 failures=0 exit=0`. Control on a scratch copy with one line number moved past its
  file's end → `checked=143 failures=1 exit=1`, diagnostic
  `FAIL proof/corpus/Nous/Model/RopeOrbit.lean at line 99999 (file has 229 lines)`.
- `proof/ApeironWord.lean` sha256 at start and at gate: unchanged,
  `a45996460f1577809337e1b9d647f34c906bc09156e63ef8bd1f8d993078f44a`.
- This file: the gate above ran on the 524-line revision with sha256
  `8ef240f1c3d8eb6f57657db5feeea1a896cdeaaac5d83ac79e3c7da14c606d76`; any later edit supersedes
  that digest, so re-run the gate after editing.
- No Lean build was run (task is read-only `rg`).
