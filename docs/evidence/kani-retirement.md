# Kani retirement evidence

The user retires Kani from development and acceptance. `AGENTS.md` supersedes older Kani
requirements. No Kani command is part of the remaining barrier. The interrupted run is cancelled,
not accepted (`target/handoff-run/final-kani-cancelled.json`). No commit or push is made.

## Source and regression tests

- Delete `src/proofs.rs` and its `src/lib.rs` wiring; remove the alternate geometry from
  `src/table/partition.rs`. Production geometry remains `8`/`256`/`255`.
- `src/table/probe.rs` retains five ordinary concrete regressions for read/locator behavior.
  This is grade `tested`, not equivalent to the retired symbolic population.
- Focused probe tests: exit0, 5 passed. Restored run: 0.05s wall, 0.00s test execution.
- Disposable real-source mutations truncating the locator and read sweeps each exit101 at the
  late-match assertion. Restored source passes. Logs: `target/kret-w1-t1/`.

## Extraction and bridge

`proof/regen.sh` and `proof/regen.sh --check`: exit0, 1708 jobs each. Bounded
`cd proof/corpus && lake build`: exit0, 3482 jobs. Commands use the serialized Lean lane,
8G memory/no swap, one thread and timeout900s. Logs: `target/kret-w2-t1/`.

Generated SHA256: `92c771510c259bc56c9e437e7d3f3108fa9e2796e6ba41ca2cd826f4127ff74e`.
Against the accepted pre-retirement generated artifact, only 94 paired source annotations change.
Stripping those annotations gives identical SHA256
`a628726adaafba9d2c7cc27ab3821008a06f6c74800406aa75c5ff41096141e6`.
The prior geometry mutation control remains attached to identical executable content; it is not rerun.
The checked theorem domains and conditional premises are unchanged.

## Residual obligations

Universal read-path links in `Growable/{Scan,Representation,Search}.lean`,
`SearchDecisionCorrect`, and the allocator adapter remain open. Retiring Kani closes none of them.
Rust tests do not establish canonical refinement. Extraction/bridge checks exclude Rust/LLVM
code generation, allocator/unsafe-container correctness and hardware execution.

## Coverage and integration

`examples/reg1/coverage.{rs,json}` uses schema `apeiron.reg1.coverage.v2` without a Kani axis.
The validator checks the actual 43 extraction selectors. Unselected constructor/insert wrappers
receive no structural grade; retired fixed-shard links receive no current refinement grade.
Current API: selected extraction2/4, refinement0/4, Rust evidence4/4. REG-1: extraction3/8,
refinement0/8, Rust evidence4/8. These are evidence pointers with freshness reported separately;
the census remains open, not REG-1 acceptance.

- `cargo run --release --locked --example reg1 -- --check coverage --controls`: exit0 in the
  integrated checkout. Five corruptions reject: stale digest, failed command called passing,
  missing test witness, retired root called current, and mismatched pin called fresh.
- `python3 tools/check-assurance.py --self-test`: exit0, six rejecting controls.
- `cargo test --workspace --all-features --locked --lib`: exit0, 92 passed, one ignored, 2.89s wall.
- `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings`: exit0.
- `python3 tools/check-assurance.py --check`: exit1, seven existing file/directory size breaches.
  The worker's eighth finding (absent empty `benches/`) was a worktree inventory difference;
  it does not occur in the supplied checkout.

Integration logs: `target/kret-final/`; coverage/self-test receipts: `target/kret-w1-t2/`.
The full workspace test suite passed immediately before retirement (258 passed, one ignored,
541.523s); it is not repeated or claimed as a new post-retirement full-suite receipt.
Production probe bodies and generated executable content are unchanged. The targeted post-change
checks above cover this retirement; the broader implementation epic remains open.

`cargo fmt --all -- --check`: exit0 after two formatting-only corrections in the coverage validator.
Final changed-file digests: `target/kret-final/source-digests.json`; formatting receipt:
`target/kret-final/fmt-restored.log`. No Kani process or source harness remains.
