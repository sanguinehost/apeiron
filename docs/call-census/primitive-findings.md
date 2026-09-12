# Primitive finding — transactional multi-column count update is missing

Scope tag: `primitive-findings`. Read-only product verification. **No product file
was modified.** One high-confidence finding, plus one strictly bounded adjacent
contract boundary.

| | |
|---|---|
| Snapshot | `/tmp/opencode/fold-call-census-20260911` |
| Revision | `4cc14471ae24f3f204fb0b0e91736ef75c909358` (git HEAD; working tree clean) |
| Method | source read + standalone reproduction harness including the real files |
| Product patch | none |
| Probe source | [`probes/unlearn.rs`](probes/unlearn.rs), including the production files below |
| Probe crate (throwaway) | `/tmp/opencode/probe-call-census` |
| `hashtrinity/src/dense.rs` | `sha256 69469f28…b377ad` |
| `modelfold/src/cast/phasefold/table.rs` | `sha256 82ec5d03…a76fa0` |

## 1. Finding (high confidence)

**`Table::unlearn` is not atomic across its three columns. A rejected unlearn can
leave `cnt` already subtracted while `hist` and `sum` are untouched — exactly the
"half-subtracted table" the module doc says cannot happen.**

The production code is `modelfold/src/cast/phasefold/table.rs:164-188`:

```rust
164: pub(crate) fn unlearn(&mut self, o: &Table) {
...
171:     let cell = self.cnt.try_unfuse(&o.cnt);      // commits cnt on Ok
172:     let code = if cell.is_ok() {
173:         self.hist.try_unfuse(&o.hist)            // may fail; hist refuses before writing
174:     } else {
175:         Ok(())
176:     };
177:     assert!(cell.is_ok(), "Table::unlearn: cell count would go negative ...");
182:     assert!(code.is_ok(), "Table::unlearn: code count would go negative ...");
187:     self.sum.unfuse(&o.sum);
188: }
```

`Dense::try_unfuse` (`hashtrinity/src/dense.rs:212-229`) is itself transactional:
it scans the whole column, returns `Err(i)` at the first cell whose difference
leaves the carrier, and only then writes (`dense.rs:202-204`, pinned by
`dense.rs:300-309`). `Table::unlearn` therefore gets the per-column guarantee but
composes three columns with a check-then-commit sequence: `cnt` is validated **and
written** before `hist` is validated. When `cnt` succeeds and `hist` refuses, the
second `assert!` panics with `cnt` already mutated and `sum` never reached.

This contradicts the module's own claim at `table.rs:35-36`:

> `Dense::try_unfuse` also refuses **before** writing, so a rejected unfold cannot
> leave a half-subtracted table behind, which the hand-rolled loop did.

The sentence is true of one `Dense` column and false of the `Table` operation it
is used to justify.

### What primitive is missing

An **all-or-nothing multi-column (whole-record) count update**: validate every
column of the composite before writing any, then commit. `Dense::try_unfuse`
provides this for a single column; nothing provides it for `Table`'s
`(sum, cnt, hist)` tuple. This probe does not establish the failure behavior of
`Table::merge`: its integer additions have their own overflow obligations.

## 2. Reproduction and check result

Controlled counterexample. `a` and `b` are both well-formed single-cell tables
(`cnt = Σ hist`), so no malformed input is involved:

```
a = Table::new(1, 2); a.add(0, 1.0, 0);   // cnt[0]=1, hist=[1,0], sum[0]=1.0
b = Table::new(1, 2); b.add(0, 1.0, 1);   // cnt[0]=1, hist=[0,1], sum[0]=1.0
```

`a.unlearn(&b)`: `cnt` subtracts (`1-1=0`, ok); `hist` would need `0-1` at slot 1
and refuses; the `code` assertion panics. State after catching the panic:

| column | before | after | changed |
|---|---|---|---|
| `cnt[0]` | `1` | `0` | **yes** |
| `hist[0..2]` | `[1, 0]` | `[1, 0]` | no |
| `sum[0]` | `1.0` | `1.0` | no |

The probe also confirms the control: a lone `Dense::<u32>::try_unfuse` that fails
writes nothing.

Command:

```sh
rustc --edition=2021 docs/call-census/probes/unlearn.rs -o /tmp/opencode/unlearn-census-probe
/tmp/opencode/unlearn-census-probe
```

Observed output (exit code `0`):

```text
control Dense::try_unfuse: Err(1), table untouched  [PASS]

before: cnt=1 hist=[1, 0] sum=1
unlearn panicked: true
panic message: Table::unlearn: code count would go negative — nothing to remove (slot 1)
after:  cnt=0 hist=[1, 0] sum=1
cnt_mutated=true hist_unchanged=true sum_unchanged=true message_matches=true

FINDING REPRODUCED: the rejected unlearn left cnt half-subtracted (cnt 1->0)
  while hist stayed [1, 0] and sum stayed 1.0.
```

The harness `#[path]`-includes the real `hashtrinity/src/dense.rs` and
`modelfold/src/cast/phasefold/table.rs`. The only local definition is the `Monoid`
trait plus integer impls (`algebra.rs:423-441` shape); it contains no `Dense` or
`unlearn` logic. No full product build was run.

The retained copy was also compiled directly (a throwaway crate with
`[[bin]] path = "/home/socol/Workspace/apeiron/docs/call-census/probes/unlearn.rs"`)
and reproduced the same result at exit code `0`, so the archived source is the
source that was checked.

**Consequence beyond "partial write":** `add` and `merge` maintain the record
invariant `cnt[i] = Σ_c hist[i·k + c]`. The post-state above has `cnt[0]=0` while
`Σ_c hist[0..2]=1`, so the table is left internally inconsistent, not merely
stale.

**Evidence class:** a controlled counterexample proving the mechanism and the
reachable post-state. It is not a claim about how often production call sites hit
the failing branch.

### Bounded reachability note

`unlearn` is called in product code at `modelfold/src/cast/heads.rs:118-119`
(`DirectFold::fit(...)` then `fold.unlearn(&window)`) and
`modelfold/src/cast/phasefold/fold.rs:713-716`. On the normal path `other` is a
previously-folded sub-table, so `hist ≤ self.hist` elementwise and both columns
succeed. The defect is on the rejected path the `# Panics` doc exists for; it is
observable when a caller catches the panic (e.g. `catch_unwind`) or otherwise
retains the table. With `panic = "abort"` the corrupted state is not observable.

## 3. Similar contract boundary (strictly bounded)

**Same shape, not separately reproduced — `DirectFold::unlearn`,
`modelfold/src/cast/phasefold/fold.rs:711-728`.** It composes four `Table::unlearn`
calls (`l0..l3`) plus `phist.try_unfuse` and `pcnt.checked_sub` sequentially with
no rollback. If `l0.unlearn` succeeds and `l1.unlearn` (or `phist.try_unfuse`)
fails, `l0` is already mutated. This is the same missing whole-record transaction
one level up. Stated from source reading only; no counterexample was built for it,
so it is **not** claimed at the same confidence as §1.

**Documented, not a defect — f64 `sum` exactness.** `Table::unlearn`'s final
`sum.unfuse` is `Group::remove` for `f64`, which always returns `Some(self-other)`
(`dense.rs:82-85`), so it never participates in the refusal. It is not a bit-exact
inverse when the fused additions rounded; this is declared at `table.rs:54-58`,
`dense.rs:87-95` and `fold.rs:704-707` ("the `sum: f64` control is *not* exactly
invertible"). This boundary is a stated limit, not the finding above, and the
integer `cnt`/`hist` columns are the ones that make the atomicity claim.

## 4. Limits of this report

- One snapshot revision, one counterexample; no product source was edited and no
  full `modelfold` build or test suite was run.
- The `Monoid` shim means the harness compiles only `dense.rs` + `table.rs`, not
  the surrounding module graph; `use super::*` in `table.rs` resolves to an empty
  parent, which is sufficient because no code item there is referenced.
- This informs "what should be a primitive but isn't"; it is not a fix, a
  roadmap, or a claim that the scanner census is complete.
