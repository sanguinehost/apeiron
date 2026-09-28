# Study — authorization policy engines

- **Purpose:** prior art for the [policy-engine epic](../epics/policy-engine.md); read before the
  design is committed.
- **Source revision:** `74c037b1cfc79b0d6be6d3907398fe926d78d12d`; working tree dirty (unrelated lanes).
- **Status:** retrieved and read; every row names a local path and revision. Unretrieved rows are
  labelled `anchor`.
- **Clone log:** `/tmp/opencode/policy-study-clones.log`. Clones sit under
  `/home/socol/oc-tmp/study-<name>` (shallow, `--depth 1`); papers sit under
  `/home/socol/oc-tmp/study-papers/`.

## Retrieved artifacts

| Name | Artifact | Revision / digest | License | Local path |
|---|---|---|---|---|
| Cedar (Rust) | `cedar-policy/cedar` | `3b8d672362dc3185a24dead04e094018f0197e09` | Apache-2.0 | `study-cedar` |
| Cedar (Lean spec) | `cedar-policy/cedar-spec` | `017c35c3d1f856a0f8b62a66660ee6628870bbcf` (Lean `v4.34.1`, batteries `v4.34.0`) | Apache-2.0 | `study-cedar-spec/cedar-lean` |
| Cedar RFCs | `cedar-policy/rfcs` | `d51ecf5ec1717609783977a4e45f60a644c52618` | Apache-2.0 | `study-cedar-rfcs` |
| Cedar paper | Cutler et al., *Cedar: A New Language for Expressive, Fast, Safe, and Analyzable Authorization*, Proc. ACM PL 8, OOPSLA1 (2024); arXiv `2403.04651v2` | pdf sha256 prefix `cea61af2106da613` | arXiv | `study-papers/cedar-oopsla.{pdf,txt}` |
| Cedar method paper | Disselkoen et al., *How We Built Cedar: A Verification-Guided Approach*, FSE Companion 2024; arXiv `2407.01688` | `6d5a07ad49be965e` | arXiv | `study-papers/cedar-fse.{pdf,txt}` |
| Oso / Polar | `osohq/oso` (legacy library, deprecated) | `7292df01679c8b3d7c92fbc08754bbb66647c1cc` | Apache-2.0 | `study-oso` |
| Casbin | `casbin/casbin` (Go, v3 module) | `524f3f2dc9baef696d748db491d49b3055d359d1` | Apache-2.0 | `study-casbin` |
| OpenFGA | `openfga/openfga` | `b74685e115e65f52655ed6866818d5a0d985805a` | Apache-2.0 | `study-openfga` |
| SpiceDB | `authzed/spicedb` | `9145a33f9a092d3420be8469cb88f5ac30b16bc2` | Apache-2.0 | `study-spicedb` |
| Zanzibar paper | Pang et al., *Zanzibar: Google's Consistent, Global Authorization System*, USENIX ATC 2019 | `c57af6886593e778` | USENIX open access | `study-papers/zanzibar-atc19.{pdf,txt}` |
| OPA / Rego | `open-policy-agent/opa` | `a6c13a96f0cd11dc7ca6db9296b2af842ffdc0fd` | Apache-2.0 | `study-opa` |
| Ory Keto | `ory/keto` | `f50b4892b31d01d736720c7ec358db9a1bd3e724` | Apache-2.0 | `study-keto` |
| Permit.io OPAL | `permitio/opal` | `5fd47b25cbea164715d8acdaebebd5e4f19670fc` | Apache-2.0 | `study-opal` |
| Permit.io PDP / control plane | hosted product | not retrieved | proprietary | `anchor` |
| telos (consumer shape) | `Ptyktos/telos` (private; `gh`) | `e2aed4ba7c6d7bb4a841dd999d0f5720b951bb5d` | `MIT OR Apache-2.0` (README; no LICENSE file) | `study-telos` |

The brief dated the Cedar paper OOPSLA 2023; the paper states Proc. ACM PL 8, OOPSLA1 (2024)
(`cedar-oopsla.txt:1390-1391`).

## Framings checked

| Framing | Verdict | Evidence |
|---|---|---|
| Cedar has no recursion | **Confirmed, with a caveat.** The language has no user recursion or loops; the entity hierarchy is supplied as data. `inₑ` is a membership test on a precomputed `ancestors` set, so transitive closure is the caller's obligation. | `cedar-lean/Cedar/Spec/Evaluator.lean:46-47`; `Cedar/Spec/Entities.lean:35`; paper `:273-275` ("free of side effects and general loops"); SymCC requires "ancestor relation is the transitive closure of a DAG" (`Cedar/Thm/Verification.lean` header) |
| Polar recurses | **Confirmed; termination is not guaranteed.** `f(x) if f(x);` loads and diverges; resolution is bounded only by `MAX_STACK_SIZE = 10_000` and `DEFAULT_TIMEOUT_MS = 30_000`. | `polar-core/tests/integration_tests.rs:1120-1127`; `polar-core/src/vm.rs:34-35,305-320,892` |
| Zanzibar-style ReBAC recurses | **Confirmed.** Check "may in turn contact other servers … to recursively traverse a hierarchy of group memberships". Implementations bound depth by configuration: OpenFGA 25, SpiceDB 50, Keto 5. Exceeding the bound is an **error**, not a deny. | Zanzibar `:105-111,248-271`; OpenFGA `pkg/server/config/config.go:25-26`, `internal/graph/check.go:416`, `pkg/server/errors/errors.go:21-22`; SpiceDB `pkg/cmd/serve.go:158`; Keto `.schema/config.schema.json:395-401` |
| Casbin matchers are interpreted expressions | **Confirmed.** Matchers are `govaluate` expression strings with user-registered functions; the role manager caps hierarchy depth at 10. | `enforcer.go:932,1096`; `go.mod:5`; `enforcer.go:601-613` |
| Permit.io is administration, not an evaluation core | **Confirmed for OPAL.** OPAL is "an administration layer for Policy Engines such as OPA and AWS' Cedar Agent", pushing policy and data updates. The hosted PDP remains `anchor`. | `study-opal/README.md:37,62` |
| Rego forbids recursion | **Confirmed for rules; reintroduced by built-ins.** The compiler rejects recursive rules (`rego_recursion_error`), but `graph.reachable` and `walk` traverse unbounded data, and `http.send`, `time.now_ns` and `rand.intn` are non-deterministic built-ins. | `v1/ast/compile.go:1321-1344`; `v1/ast/errors.go:57`; `v1/ast/builtins.go:968,1483,2444,2810,2830,2991` |

## What cedar-spec proves, and what it does not

| Claim | Grade in upstream terms | Evidence |
|---|---|---|
| `isAuthorized` = allow iff some satisfied permit and no satisfied forbid | Lean definition | `Cedar/Spec/Authorizer.lean` (`isAuthorized`) |
| `forbid_trumps_permit`, `default_deny`, `allowed_only_if_explicitly_permitted`, `allowed_iff_explicitly_permitted_and_not_denied`, `order_and_dup_independent`, `unchanged_allow_when_add_permit`, `unchanged_deny_when_add_forbid`, `determiningPolicies_of` | Lean theorems; rebuilt locally (spike S4: `lake build Cedar.Thm.Authorization`, 119 jobs, exit 0, Lean `4.34.1`, batteries `f2effa3d`). `forbid_trumps_permit`, `default_deny`, `order_and_dup_independent` and `allowed_iff_explicitly_permitted_and_not_denied` each depend on `[propext, Classical.choice, Quot.sound]`. Receipt: `/home/socol/oc-tmp/policy-spike-s4/report.md` | `Cedar/Thm/Authorization.lean:64-249` (sha256 `c91adbfe…`) |
| Validator soundness (`validation_is_sound`), typechecker soundness | Lean theorems (read, not rebuilt) | `Cedar/Thm/Validation.lean:51`; `Cedar/Thm/Typechecking.lean:43` |
| SymCC encoding sound and complete for strongly well-formed environments | Lean theorems (read, not rebuilt) | `Cedar/Thm/SymbolicCompilation.lean:71-183`; `Cedar/Thm/Verification.lean:46-660` |
| Policy slicing preserves decisions | Lean theorems (read, not rebuilt) | `Cedar/Thm/PolicySlice.lean:47-139` |
| **Rust `cedar-policy` implements the Lean spec** | **Not proved.** Differential random testing (DRT) against the executable Lean model; this is `tested` in local terms. | `cedar-drt/README.md` (fuzz-target table); FSE paper `:29-64`; OOPSLA paper `:670-676` |
| No `sorry` in `Cedar/` | grep, 0 matches | `grep -rn '\bsorry\b' Cedar` |
| `Classical` / `noncomputable` usage | confined to `Cedar/Data/LT.lean` in `Spec`/`Data` | grep |

**Error semantics.** Evaluation errors are skipped: `satisfied` requires `.ok true`, so an erroring
`forbid` counts as absent (`Cedar/Spec/Authorizer.lean`, `satisfied`/`isAuthorized`; Rust
`ErrorHandling::Skip`, `cedar-policy-core/src/authorizer.rs:53-62`). Cedar's own test allows a
request while `forbid … when { context.bad == 2 }` errors on a missing attribute
(`authorizer.rs:236-258`). Cedar relies on validation (`validation_is_sound`) to exclude errors.
The default-deny branch carries an empty `determiningPolicies` set, so a Cedar deny names no rule
when no permit matched.

**Can cedar-spec be the canonical definition?** No:

| Option | Assessment |
|---|---|
| Canonical definition | Rejected. The toolchain is Lean `v4.34.1` + batteries; the bridge pins `4.31.0` + Mathlib `v4.31.0` (`proof/README.md`, "Corpus / toolchain compatibility"). The language carries strings, `like`, decimal/IP/datetime extensions and entity stores, all outside the fixed fragment. The error-skip semantics contradicts the fail-closed law below. |
| Related by a checked mapping | Possible later, on a 1-bit fragment, once a toolchain-compatible cedar-lean revision exists. `anchor` until a Lean theorem relates the two definitions. |
| Differential oracle | **Adopted.** `cedar-policy` as a `tested` oracle on the agreed fragment (no errors, no unknowns, 1-bit rights), never cited as proof. |

## Feature matrix

| Engine | Decision model | Language; recursion | Evaluation complexity (as stated or observed) | Default; forbid | Static analysis | Formal verification (local grade) | Consistency / administration | Storage / scale | License |
|---|---|---|---|---|---|---|---|---|---|
| Cedar | RBAC, ABAC; ReBAC encodable via hierarchy (paper `:1240-1243`) | Cedar; no loops or recursion; hierarchy supplied | "generally linear-time"; set containment "quadratic in the worst case" (paper `:1243-1245`) | default deny; forbid overrides permit; erroring policy skipped | schema validator; SymCC to SMT | Authorizer theorems rebuilt locally (S4, closed cone); still a pointer for any Apeiron claim. Rust linked by DRT: `tested` upstream | none in core; entities supplied per request | in-memory policy set + entity slice | Apache-2.0 |
| Oso (Polar) | RBAC, ReBAC, ABAC | Datalog/Prolog-shaped with unification and recursion; non-terminating rules accepted | SLD-resolution VM; bounded by stack 10,000 and timeout 30 s (`vm.rs:34-35`) | no match = no result (caller treats as deny); no forbid effect | validation pass emitting `ValidationError`/`ValidationWarning` (`polar-core/src/validations.rs`) | none found | library is deprecated; hosted Oso Cloud is `anchor` | host application data via callbacks | Apache-2.0 |
| Casbin | ACL, RBAC (depth ≤ 10), ABAC | PERM `.conf` model; matcher = `govaluate` expression string | linear scan over policy lines × matcher evaluation (`enforcer.go:702`) | selectable effect expression: `DenyOverrideEffect` **allows** when no deny matches; `PriorityEffect` is rule-order dependent (`constant/constants.go:29-33`; `effector/default_effector.go:36-110`) | none beyond parse | none | adapters and watchers per store | storage adapters | Apache-2.0 |
| OpenFGA | ReBAC (Zanzibar) + conditions | model DSL; userset rewrites recurse; CEL conditions | graph resolution; depth ≤ 25, breadth ≤ 10 (`config.go:25-26`); condition cost ≤ 100 (`:67`) | no tuple = not allowed; exclusion (`but not`) | model validation | none | `HIGHER_CONSISTENCY` vs `MINIMIZE_LATENCY` (`pkg/server/check.go:406`) | tuple store (SQL) | Apache-2.0 |
| SpiceDB | ReBAC (Zanzibar) + caveats | schema; recursion; CEL caveats; exclusion `-` | dispatch depth ≤ 50 (`serve.go:158`) | three-valued result: has / no / **conditional** permission when caveat context is missing (`internal/services/v1/permissions.go:185-186`) | schema compiler | none | ZedTokens (`pkg/zedtoken`), Watch API (`internal/services/v1/watch.go:44`) | datastore drivers | Apache-2.0 |
| Zanzibar | ReBAC | namespace configs; userset rewrites (union, intersection, exclusion) | recursive fan-out; Leopard index for deep groups (paper `:103-111,383`) | no path = no access | — | none | zookies against the "new enemy" problem (paper `:155-206`) | trillions of tuples (paper `:112-113`) | paper |
| OPA / Rego | general policy; RBAC/ABAC/ReBAC by data | Datalog-derived; rule recursion rejected; `graph.reachable`/`walk` built-ins | "linear fragment" guidance (Cedar paper `:1274`, ref [37]; OPA doc not retrieved: `anchor`) | undefined = no decision; caller convention | type checker (`v1/ast/compile.go:1920`); `opa check` (`cmd/check.go`) | none | bundles; OPAL for live sync | in-memory data documents | Apache-2.0 |
| Ory Keto | ReBAC (Zanzibar) | OPL namespaces; recursion | read depth default 5, max 65535 (`config.schema.json:395-401`) | no path = not allowed | namespace parser | none | — | SQL | Apache-2.0 |
| Permit.io OPAL | administration only | — | — | — | — | none | policy/data change detection pushed to agents (`README.md:37,62`) | Git/API sources | Apache-2.0 |
| telos (consumer) | permit-default Cedar rules + ZEN tables | Cedar; 110 permit/forbid statements; **337 `like`** string patterns, 103 string equalities, 66 `.contains` (grep census of `telos-policy/policies/cedar/default-permit.cedar`, comments excluded) | per call | deny reason is the fixed string `"cedar policy denies"`; evaluation error → deny (`telos-policy/src/engine.rs:378-388`) | — | none | versions hashed into each decision (`engine.rs:379-380`) | embedded bundle | `MIT OR Apache-2.0` (README) |

## Adoptable

| Mechanism | Source | Adopted as |
|---|---|---|
| Forbid overrides permit; default deny; order and duplicate independence | Cedar `Thm/Authorization.lean` | the canonical laws, re-stated and proved locally over our own definition |
| Hierarchy supplied as data, not computed | Cedar `inₑ` over `ancestors` | ReBAC v1: caller supplies closure bitsets as facts; no in-core reachability |
| Differential testing against an executable reference | Cedar DRT | `tools/policy-oracle/` against `cedar-policy`, grade `tested` |
| Three-valued result on missing context | SpiceDB conditional permission | `unknown` fact values; `ask` verdict; fail-closed forbid |
| Depth bound returns a distinct outcome, not a silent deny | OpenFGA `ErrResolutionDepthExceeded` | any later bounded reachability returns a named `budget` cause |
| Decision versioning (policy hash per decision) | telos `with_versions` | image digest carried in the decision receipt (consumer side) |
| PDP/PEP split; administration outside the evaluator | OPAL | compiler and distribution are trusted, tested; the evaluator is the proved core |

## Rejected

| Mechanism | Why |
|---|---|
| Error-skip (Cedar) | An erroring forbid permits (`authorizer.rs:236-258`). The law here is fail-closed: an unknown forbid applies. |
| Interpreted matcher strings (Casbin), general built-ins (Rego) | Outside the Aeneas fragment; nondeterminism (`rand.intn`, `time.now_ns`, `http.send`) breaks determinism. |
| Unbounded or configured-depth recursion (Polar, Zanzibar family) | Termination is enforced by resource limits, not by the language; no total canonical function. |
| Order-dependent effects (Casbin `PriorityEffect`) | Violates order independence. |
| Allow-by-default effect (Casbin `DenyOverrideEffect`, telos default-permit) | Violates default deny in the core; an application may still write an explicit catch-all permit rule. |
| String patterns in the proved core (telos `like`) | Strings and regex are outside the fragment; classification into bounded enum/bitset facts happens at a tested encoding boundary. |
| cedar-spec as canonical | Toolchain mismatch; language surface outside the fragment; error-skip. |

## Recommendation

Define a small canonical decision object locally, with Cedar's authorizer laws plus two laws Cedar
does not have: **fail-closed unknowns** and **named cause on every deny**. Add **attenuation** as
monotone restriction on a bounded rights order, which covers the Song capability face. Use Cedar
as a `tested` differential oracle on the 1-bit, error-free fragment. Keep ReBAC as supplied
closure facts; defer in-core reachability.
