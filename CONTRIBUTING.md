# Contributing

Contributions are welcome: bug reports, tests, documentation, performance work and proofs.
Report a suspected vulnerability privately, as described in [SECURITY.md](SECURITY.md), not as an
issue.

## Before you start

- Open an issue for anything larger than a fix, so the approach is agreed before the work is done.
- Read the chapter of the [book](book/src/introduction.md) for the area, and `proof/README.md` if
  you touch the Lean.
- The working agreement, including gates and evidence grades, is in [AGENTS.md](AGENTS.md). It
  applies to people and to coding agents alike.

## Build and check

Rust 1.89 or later (`rust-version` in `Cargo.toml`); CI pins a nightly for lint stability.

```sh
cargo build --lib
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
python3 tools/check-claims.py
```

Changes under `proof/`, or to a Rust file the Lean extraction includes, also run
`proof/regen.sh --check` and `cd proof/corpus && lake build`; see `proof/README.md` for the
toolchain. CI runs the same commands.

## A good change

- One concern per pull request, with the tests that cover it.
- A result states its evidence grade (`proved`, `tested`, `measured`) and names the theorem, test or
  command behind it. A claim without one is not accepted.
- A fix comes with a test that fails before it and passes after.
- A speed claim comes with the command, the machine, and the mean over at least three runs with the
  spread.
- Documentation follows the same rules as code: it states what the source does, in the fewest words
  that keep every name and number, and it runs (`mdbook test book -L target/debug/deps`).

## Sign-off

Every commit carries a `Signed-off-by` line certifying the
[Developer Certificate of Origin](https://developercertificate.org/): you wrote the change, or have the
right to submit it under the project licence. Add it with `git commit -s`.

## Licence

apeiron is Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). A contribution is under the same
licence, as Apache-2.0 section 5 provides. Do not submit code you cannot license that way, and do not
copy third-party source into the repository except for benchmark and comparison fixtures, which keep
their own licence and stay outside the shipped library.
