# Hashtrinity Call Census

## Conclusion

Importing Hashtrinity is not evidence that a program's computation is derived from
the trichotomy. The inspected dependencies fall into materially different groups:
API substitutions, incidental maps/caches, essential count/index state, encoding,
parallel execution, and mathematical kernels implemented locally. None acquires
an implementation-refinement proof through the package dependency.

The usage is not generally dead or useless. The unsupported step is upgrading
useful shared code into an algebraic or verification claim.

This review was performed by DeepSeek Flash workers, with tool and evidence
cross-checks by the main assistant. It changes analysis/tooling only. No Fold,
Bedrock, model, application, or roadmap implementation was migrated.

## Scope And Counts

Canonical Fold snapshot: GitLab `twn/RnD/fold`, revision
`4cc14471ae24f3f204fb0b0e91736ef75c909358`. This is **not** the older local
`../fold` checkout. The 21 hosted snapshots have individual full revisions in
[`data/hosted-repos.tsv`](data/hosted-repos.tsv).

| Scope | Rust Files Parsed | Syntactically Attributed Sites | Unattributed Records |
| --- | ---: | ---: | ---: |
| ModelFold | 409 | 1,009 | 7,953 |
| Bedrock | 429 | 1,580 | 11,598 |
| Other 11 Fold subtrees | 586 | 4,059 | 11,248 |
| Selected 21 hosted repositories | 1,389 | 1,311 | 8,053 |
| **Total** | **2,813** | **7,959** | **38,852** |

All files parsed, but parsing is not name resolution or build verification.
These are counts of static source sites, including examples, tests, disabled
features, and potentially orphan files. They are not runtime execution counts.
The hosted cohort is not the entire previously inventoried portfolio.

Two known false attributions (`Rc::clone` mistaken for `Table::clone`) remain in
the raw ledgers for traceability and are flagged in
[`data/reviewed-calls.tsv`](data/reviewed-calls.tsv). After those exclusions,
7,957 attributions remain, **not** 7,957 independently verified calls. An explicit
inventory join separates 839 constructors, 7,111 function/method sites, and 7
unclassified sites within this remaining set. The original scanner kind is
retained beside the joined kind. Constructor use is not a call to an algebraic
kernel.

The 38,852 unattributed records are largely common method names whose receivers
are unknown. They are neither counted as Hashtrinity calls nor certified as
non-Hashtrinity calls. Macro-generated calls and some trait/function-pointer uses
are outside this population entirely. No percentage of "all program computation"
or "vacuous imports" can be computed from these totals.

## Evidence That Changes The Architecture Discussion

### 1. The provider uses an API bundle, not the table

The 2,886 attributed sites under `kleis` are in the absorbed `rauthy-*` provider
packages, not the small core Kleis library. The inspected surface is terms/codecs,
validation/OpenAPI, and legacy crypto/encoding helpers. No table-family site was
attributed there. Calling this a trichotomy-based identity system from its
Hashtrinity imports would be unwarranted.

This subtree is also excluded from the root workspace. Its static source count
must not be described as traffic in the default executable.

### 2. A cache dependency is not a codec derivation

`transcode/src/aac.rs::mdct_twiddles` uses `Table::new`, `get`, and `put` to cache
twiddle factors. The FFT/MDCT computation is local floating-point arithmetic.
The two cache methods need manual receiver attribution through `MutexGuard`;
the automated count alone misses them. This is a concrete example of why both
the purpose review and the unresolved ledger matter.

### 3. Count/index state is genuine reusable work

CountLM, DirectFold, DNA/token counters, and SangDB genuinely store or merge their
working state through table/dense APIs. This is stronger than an incidental cache,
but it establishes collection/accumulator reuse, not a sigma-derived algorithm or
a proved Rust implementation. Overflow, key identity, reordering, persistence,
and composite failure semantics are separate obligations.

ModelFold has 824 attributed collection/addressing expressions. Bedrock has real
SQL, posting/spatial indexes, logs, chunk stores and versioned records beyond its
`Store = Table` alias. These are extraction candidates with substantial existing
behavior to specify; they are not disposable because their table use is conventional.

### 4. Much mathematical work bypasses the alleged primitive

The inspected ridge/Cholesky, quantization, and GPU numeric kernels implement their
operations locally over arrays/buffers. A codec call used to save their output, or
a `par::Pool` used to execute a loop, does not establish the numerical operation's
correspondence to a theorem. Conversely, an inlined specialization could have a
valid correspondence without literally calling a shared algebra function. That
correspondence must be supplied, not inferred either way from call syntax.

### 5. Hosted dependencies can be extremely narrow

The hosted `saga` slice has 32 attributed term/serialization expressions and no
attributed table call. `flexprice-collector` uses `Error::new` plus two legacy
crypto/encoding calls. The selected `hosting-runtime` revision declares a
workspace dependency without a member use found. Other products reach the old
stack only through Bedrock or Transcode. These are not equivalent relationships.

Separately, PII and filesystem-indexing projects carry local std-map/accumulator
implementations with Hashtrinity citations rather than depending on its crate.
The hosted report distinguishes actual calls, passed function values, and type
references in those copies.

## Missing Or Misplaced Primitives

| Candidate | Evidence And Boundary |
| --- | --- |
| Atomic composite count/state update | Reproduced rejection after one column has committed; whole-record atomicity is missing despite a per-column checked API. |
| Explicit numeric accumulator contracts | Count, Gram, merge and unlearn operations recur. Integer overflow, partial subtraction and rounded floating-point state need different contracts. A common trait alone would not prove them. |
| Shared numerical kernels | Repeated Cholesky/SPD and Lloyd/codebook implementations warrant contract/parity comparison before deduplication; matching names are not enough. |
| Safe schema/term encoding | Repeated field accessors and serialization adapters are genuine shared work. The answer to unsupported `Option`/enum/Table fields is an explicit format and round-trip contract, not broad unsafe `Pod` assertions. |
| Database record/index/log core | Actual SangDB algorithms can be separated from HTTP/RESP/product adapters; durability and recovery must not be inferred from map operations. |
| Ordered/configuration structures | Some consumers deliberately retain ordered vectors or standard containers. A collection migration must preserve ordering and equality rather than enforce a crate-name rule. |

OpenAPI generation, billing models, deployment recipes, dashboards, model-specific
experiments, and application policy do not become foundational primitives merely
because they repeatedly call a shared utility.

**Crypto remains intended later.** Only after everything else is mature should
the crypto layer proceed, with full end-to-end formal verification **and**
independent external analysis before adoption. The census inventories existing
crypto-housed calls without promoting or permanently excluding that future work.
Encoding helpers such as `to_hex` are distinguished from cryptographic operations.

## Confirmed Composition Bug

`phasefold::Table::unlearn` subtracts `cnt` successfully and then can reject `hist`.
The retained probe includes the actual production files, with only a small
`Monoid` shim to avoid building the entire product. Starting from two valid
one-cell tables with different categorical observations, rejection leaves:

| Column | Before | After Rejection |
| --- | --- | --- |
| Count | `1` | `0` |
| Histogram | `[1, 0]` | `[1, 0]` |
| Sum | `1.0` | `1.0` |

The count/histogram invariant is broken. A single `Dense::try_unfuse` refuses
without writing; composing those operations sequentially does not preserve that
property. See [the finding and reproduction](primitive-findings.md). This is a
tested counterexample, not an Aeneas proof or a production-frequency claim.

## Files And Reproduction

- [Model programs, calls and candidate primitives](models.md)
- [Bedrock/storage/runtime analysis](bedrock.md)
- [Other Fold consumers and absorbed provider](other-fold.md)
- [Hosted cohort and local reimplementations](hosted.md)
- [Consolidated per-site ledger](data/reviewed-calls.tsv)
- [Machine-checked count/provenance reconciliation](data/census-audit.json)
- [Scanner and limitations](../../tools/call-census/README.md)

Every attributed site has a source revision, file/span, enclosing caller, callee,
classification and source-location label. The separate reports provide purpose
and manually inspected chains, not a claim that every caller's full semantics
has been audited. Raw unknown records remain under `/tmp/opencode/census-v2/`;
the reports record commands to regenerate them from the pinned snapshots.

From the Apeiron root, independently reconcile the four published ledgers:

```sh
python tools/call-census/audit_reports.py --raw-root /tmp/opencode/census-v2
```

Use `--write` to regenerate the consolidated JSON/TSV. This checks unique site
identities, source spans, target symbols, per-run counts, clean snapshot revisions,
and consistent scanner fingerprints. It does not certify syntactic attribution
as compiler resolution.

The remaining step before an exhaustive semantic census is resolving the selected
build/module graph and receiver/trait/macro calls against each consumer's actual
dependency version. Some old sibling dependencies are missing and some hosted
copies diverge. Until those closures are accounted for, these results support
specific extraction findings, not a complete verified call graph.
