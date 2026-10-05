# §6 clippy green — receipt

> (historical; the harness was removed in c9a2d0c) The `examples/reg1`, `examples/smhasher`, `examples/fold` and `examples/support` harnesses, and the comparison dev-dependencies (criterion, hashbrown, foldhash, fxhash, twox-hash, wyhash, ahash, rustc-hash), were removed from the tree in `c9a2d0c`. References below record that harness as it then stood; the named paths are absent and no command naming them runs now.

Row: `apeiron-clippy-rest-clear-BXBcoyaB24BO8jlBMXDO8l` (`APEIRON-CLIPPY-REST`). Prior rows:
`APEIRON-CLIPPY-SHA256`, `APEIRON-CLIPPY-AES-PROBE`.

`cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` **exit 0**, and
`cargo fmt --all -- --check` **exit 0**, on the live tree.

**The count was corrected: the true total was 22, not 87.** `cargo clippy` aborts at the first failing
target, so the 73 (`src/sha256.rs`) + 13 (`src/aes.rs`) + 1 (`src/table/probe.rs`) were the **lib
target's** diagnostics and nothing downstream. The exhaustive count via `--keep-going
--message-format=json`, deduped on code+path+line+col+message: `examples/reg1/attrib.rs`
15 `cast_precision_loss` + 2 `cast_possible_truncation` + 1 `uninlined_format_args`,
`examples/reg1/coverage.rs` 1 `items_after_statements`, `tests/sha256.rs` 3 `useless_vec`. Only 2 units
failed; 41 of 44 declared targets produced artifacts.

**Fixes are real, not suppressed:** `&[b'a'; N]` for `useless_vec`; `REVIEWED_ROOT_SET` hoisted to module
scope; `reps: u64 -> usize` with `parse::<usize>()` at the CLI boundary (checked, no cast);
`src/aes.rs`'s `cast_possible_truncation` **removed** by making the index `u8` running `0..=u8::MAX`;
`too_many_arguments` resolved with a `SpillRoute` struct. **One** item-scoped `reason=` allow, on an
`operand(u64) -> f64` helper carrying its bound (a `StrataKey` is one `u64` tag, so 2^52 resident keys
is 36 PB; `nanos` reaches 2^52 after 142 years).

Control: five mutations, throwaway copy, each firing its named lint at exit 101 and restoring clean;
M2 placed **outside** the allowed helper to prove the allow does not leak.

**Open, stated:** `src/aes.rs` is extraction source (`proof/extract/lib.rs`), so
`proof/regen.sh --check` will report drift until `proof/ApeironWord.lean` is regenerated — the
committed extraction predates this change. `examples/reg1/attrib.rs` is untracked (a cancelled lane's
attribution example, kept and fixed in place). Grade: `tested` (the five gates + five controls);
nothing here is `proved`.
