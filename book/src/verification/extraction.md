# Extraction

The Lean side does not restate the Rust. It reads the Rust.

1. **Charon** compiles the crate's MIR to an intermediate form (LLBC).
2. **Aeneas** translates that form to Lean definitions, `proof/ApeironWord.lean`.
3. A refinement file in `proof/corpus/` proves those definitions equal to the canonical ones, for example `word_mul_denote`.

`proof/extract/lib.rs` includes the real source files by `#[path]`: the scalar, cell, exchange,
carrier, coordinate and table modules. Its wrappers (`crate::word_mul`, `crate::word_norm`, `crate::word_point_density`
and so on) only fix a concrete type, `i64`, and call the real method. A wrapper that restated the
formula would prove a copy, so the rule is that a wrapper may fix a type and may not compute.

```text
// proof/extract/lib.rs
pub fn word_mul(a: cell::Cell<i64>, sigma: cell::Sigma, b: cell::Cell<i64>) -> cell::Cell<i64> {
    a.mul(sigma, b)
}
```

## The selection

`proof/regen.sh` lists the extraction roots as `--start-from` entries. The call closure of the
roots is what gets translated: the cell and carrier kernels, the exchange kernels, the coordinate
operations and the table's get, insert, build and relocation bodies. The script also prints the
SHA-256 of every extracted source file.

```sh
proof/regen.sh           # extract, audit, build
proof/regen.sh --check   # fail if ApeironWord.lean differs from a fresh extraction
```

The tool revisions are pinned in `proof/README.md`. A run records the versions it used; a pin read
from memory is not a record.

## Drift

A source change that alters a generated definition cannot pass silently. `regen.sh --check`
re-extracts from the current sources and diffs against the committed `ApeironWord.lean`.

## Externals

Some Rust the table reaches has no Lean definition in Aeneas: the SSE2 and AVX2 intrinsics, raw
pointer arithmetic over a slice, `MaybeUninit`, `Vec::set_len`. `proof/patches/aeneas-lean-model.patch`
defines each as an ordinary Lean definition in the backend. Each is a definition and not an
`axiom`, so the audit reports no axiom for them.

| external | modelled as | outside the model |
|---|---|---|
| `_mm_loadu_si128`, `_mm256_loadu_si256` | a read that fails past the window | that LLVM emits the sequence the model describes |
| `ptr.add`, `get_unchecked` | a checked offset that fails out of bounds | that `unsafe` reads stay in bounds; the model turns it into a proof obligation |
| `MaybeUninit` | `Option`; reading `none` fails | none |
| `is_x86_feature_detected!("avx2")` | an unconstrained `Bool`; both branches are kept | whether the machine has AVX2 |

A model that is wrong is a wrong premise, and the axiom audit does not see it. These definitions
are the trusted part of the extraction, listed in [Trust](trust.md).
