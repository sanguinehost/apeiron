# Full-profile REG-1 bounded measurement

**Decision: G1 remains open.** The unchanged full-profile gate reached its 600-second
bound and exited **124**, before printing a pair table or acceptance headline. This is
an inconclusive run, not a completed ratio-failure result. Current passing/failing pair
counts, worst families and complete control acceptance are unavailable. No fallback,
threshold change, source edit or second run followed.

## Arc

```text
SCOPE:      docs/evidence/open-area-performance.md and ignored target receipts; frozen-source full REG-1 measurement
REGISTER:   measurement; excludes optimization, proof acceptance and threshold changes
CORPUS:     read-path intent/spec/plan and REG-1 source read; no theorem claim
REFERENCE:  /home/socol/Workspace/worktree-agent-APEIRON-OPEN-W3-PERFORMANCE/examples/reg1/acceptance.rs
GATE:       exact full-profile command below; observed exit 124 at the fixed 600-second bound
GRADE:      measured command duration and emitted controls; pair performance and full acceptance inconclusive
```

Owner/task: Astra, `APEIRON-OPEN-W3-PERFORMANCE`. Base HEAD
`262bb0a94eff2cc7d5655122a954f44d74de96b6`, accepted dirty source. The Arc preceded
the evidence edit. The lead released the exclusive heavy lane after the Rust barrier;
the lane was returned immediately after this command exited.

## Command and host

Cwd: `/home/socol/Workspace/worktree-agent-APEIRON-OPEN-W3-PERFORMANCE`.
The command was wrapped by `/usr/bin/time -v`; stdout/stderr were captured separately.

```sh
flock /tmp/opencode/apeiron-lane.lock timeout 600s \
  env CARGO_TARGET_DIR=/home/socol/Workspace/apeiron/target CARGO_BUILD_JOBS=2 \
  cargo run --release --locked --example reg1 -- \
  --check performance --profile full --accept --controls
```

| Measurement | Result |
|---|---|
| Exit | 124, timeout; no acceptance verdict |
| Wall time, including release build | 600.015 seconds |
| User / system CPU | 597.07 / 3.43 seconds |
| Maximum RSS | 408052 KiB |
| Release build | Fresh compilation from the assigned worktree, 6.07 seconds |
| Host / kernel | leviathan / Linux 7.2.6-1-cachyos, x86_64 |
| CPU / memory | Intel Core i9-14900KF, 32 logical CPUs / 65611464 KiB total |
| Governor / turbo | powersave / intel_pstate no_turbo=0 |
| rustc | 1.97.0-nightly, 17584a181979f04f2aaad867332c22db1caa511a; LLVM 22.1.2 |
| cargo | 1.97.0-nightly, eb94155a9 |
| hashtrinity | 4cc14471ae24f3f204fb0b0e91736ef75c909358 |
| Load averages, start → end | 1.96/1.99/4.60 → 2.15/1.95/3.31 |

No other compiler or benchmark held the lane. Ordinary desktop processes remained;
the lead also reported approximately four CPU-seconds of lightweight Python checking.
This was an exclusive build/benchmark run, not a fully quiescent-host experiment.
No affinity, governor, turbo, priority or profile setting was changed by this task.
Cargo emitted the existing duplicate-key diagnostic from the pinned Fold checkout's
`opsfold/Cargo.toml:61`, then successfully compiled and launched REG-1.

## Protocol and observed output

The inspected harness is unchanged: phase-1 full counts are 1,048,576 lookups and
updates, five warmups and 31 samples. Per-pair registration covers 160 serial cells
and two spatial cells, with three campaigns, two warmups, 15 samples per campaign and
4096 serial operations. The statistic is pooled median elapsed nanoseconds divided
by the incumbent median; acceptance requires every applicable campaign ratio interval
maximum at most **1.0**, with matching identity, address and worker conditions.
These are source-read protocol facts, not completed measurements from this run.

The log identifies `reg1-performance-full-shared-fixed16` and the source/tool identity.
It contains 36 rejecting observations:

| Control | Red observations |
|---|---:|
| placement-only-address | 2 |
| retired-shardfull-receipt | 16 |
| retired-shardfull-digest | 16 |
| fabricated-current-cell | 2 |

There are zero `HEADLINE` records, zero pair tables and no result from the final eleven
performance plus eight acceptance child controls. The registration observations do
not substitute for that unfinished control set. The logs contain no phase timestamps;
no hot function or cause of the elapsed-time increase is established.

The [previous full receipt](perf-ledger.md) records
421 passing and 1029 failing pairs on another source/kernel identity. Those counts and
its worst spill-miss families remain historical. This run supplies neither replacement
counts nor a before/after speed comparison.

## Identity and artifacts

All 51 manifested crate, REG-1, support, workspace and configuration files matched the
main checkout before execution and remained byte-identical afterward. The accepted
baseline manifest also remained unchanged. The document is the only task-owned source
edit. Raw artifacts are ignored local evidence under
`target/open-areas/performance/`; retain them when removing the task worktree.

| Artifact | SHA-256 |
|---|---|
| `source-before.json` | `09cd0fa4e35bf45075077c615870d200b2a4c7bfadd30a461abf68d9af844e6c` |
| `full.stdout.log` | `ec39944fd1062c5d92dabaf351b108b5b61df739a1e92b5fab829316485b1ba5` |
| `full.stderr.log` | `0f4e3d95d011f5bfcb43ba850b8bc453e262622f0aec0388b81c4774c86e4ac5` |
| `time.txt` | `0d585710d715446a3b4f93f8d208060b39b7b149624477db8b21def249ae190e` |
| `run.json` | `15b4b9107ae5c265ed3661bf9ebe1ecbb5d81a56f488da0f84e260ebc850ebd0` |
| `target/open-areas/baseline.json` | `0121b1003cc140a2e1470c490cfd7268e27bba6a0e914297a979cb74144c90f9` |

The complete executing-source manifest follows so identity survives local log cleanup.

| Source | SHA-256 |
|---|---|
| `Cargo.lock` | `043b2a05b7c9b2b25229b96d2a39e50dc3403a840f66c9cf29e855dab4b16c1d` |
| `Cargo.toml` | `ea57123ca44311c14aa7d03447dc376813893aaaf9fc1032043f25625dca12fd` |
| `derive/Cargo.toml` | `8a87e6ef63b1bd6bad1a5139122f053ce5d129aa8bc23f0deadc2072dce2756b` |
| `derive/src/lib.rs` | `36ac84bfa4d2a34d04c8ab11f5b6aa7c1835960947cfd8db9ada10798f815646` |
| `examples/reg1/acceptance/registry_tests.rs` | `d2a0110fdeb3e1f19d717a5bb20f1b6c8d07580ee6e28743ac4cefb30798f80c` |
| `examples/reg1/acceptance/report.rs` | `33904535feebf8a85dcc50e1233b1aa02f26ecdf7f2aefe5c68896cceabfde65` |
| `examples/reg1/acceptance.rs` | `99b263b9b9f607e98401f4abd76d76f3e59a0c3920a1797d7b21a95a12d8a28f` |
| `examples/reg1/adapters/native_matrix.rs` | `9ecec48d78312283a5806afb5844977adfdbaf15cd868bd8ea2d6d9beff0139f` |
| `examples/reg1/adapters/native_strata.rs` | `27fbe7406ba319097796a4bd9c3f04d6f8d716f90ee654adf3608a641c5e4ccd` |
| `examples/reg1/adapters/native_workload.rs` | `410f3f8236bf344ecb6198c1669e9173fd7b93773cf6ed5b1ee32f43a8728283` |
| `examples/reg1/adapters/serial.rs` | `e72c32b2eb707bdf17a84a92327907763f5af0ff2d250cec8a4a72306c5a6c86` |
| `examples/reg1/adapters/spatial.rs` | `a855ac7517509697e8233ae8ea6c2c0c94f8115e1115c96fcceaf06af336e075` |
| `examples/reg1/candidate.rs` | `82c815bd524a4b08dfc50717e17b81dc69a77e3d7eecff4efedd83d72f58e5a9` |
| `examples/reg1/correctness.rs` | `ce97bdd0a5511b338b476275e9499089d74474dc2724bbc1c0b17283a0c4464d` |
| `examples/reg1/coverage.json` | `5b52a88d0f9066192d44245295584d54f04186d4512e9220377d7308ec361e10` |
| `examples/reg1/coverage.rs` | `b7eecb680e839ebe198dd6767955cfa5eadf835378097a693ef20a04059bf917` |
| `examples/reg1/history.rs` | `18f273b36ac8105b46204d774e61da5c8ff4ded9ca3b934fe958a8ed413ca342` |
| `examples/reg1/main.rs` | `16f05253029b2fa43dbe14c216d238db968a98936258437935203eebc33ebdc3` |
| `examples/reg1/matrix.json` | `898787da08f8f4198e29530cc48a10c53459cd016879d9c9e6b7e3a5c7e9b073` |
| `examples/reg1/matrix.rs` | `93c69ffba4a0539045b65276667c1b82528c6818a1f9072e916bd5a7ee5c1982` |
| `examples/reg1/performance.rs` | `ea77fbe2f2cc1248283624f5153eae2669cff4bfeb6efe03c104dc4508c01930` |
| `examples/reg1/strata.rs` | `4c8c0e23bd4b03632fa3fa152a1bbd37c43850ca76d542777482b8fb49c857c5` |
| `examples/reg1/workload.rs` | `98c14c220b2fa92af15dc6f9b05f4763013327d41c3fa2ba8600d7eca1cd44ea` |
| `examples/support/counter.rs` | `d697a9910ebccced7af3b18853b280ce84ea81dc937834895cadca6adcc71468` |
| `examples/support/counter_check.rs` | `656aee69e2c59b1537905176ebd6ab4f27ea2c1c94df441a0117ba5fab5f9184` |
| `fold-abi/Cargo.toml` | `46331e1e9c1fcc464eaad5b1fc03d3424cd4be01a5e8cd6f94cb0bf8e7b80436` |
| `fold-abi/src/lib.rs` | `8209cc660c39ea7706d0120198ab7d5b7c24b7e3e640038a4e5275e4e267d79a` |
| `src/carrier.rs` | `f00d8fcf7194c435b6dcbfc1064c426f912f8f6f5fb14e55002a91cff3dd8dbb` |
| `src/cell.rs` | `d256ce260acc5372af9af60f59d93f389f6a9bf592df05378eb737b39b1a0a45` |
| `src/coord/address.rs` | `a8c5c08164c1c007a72e3089f964cb2cc2167f864f76414e641c2b9e563c70e6` |
| `src/coord/locator.rs` | `996769deb211aeddc0f9ae47de749d4983f54c769ec8ebb952b4a2954525b761` |
| `src/coord/locator_core.rs` | `8e365c413ad33e395f40a19de15dbd128402a6036b570b8304f45cf38d79155b` |
| `src/coord.rs` | `c2756de1ff5eccfa3dceefd984f8db9cae544984e34edc4d1a023e6b382e7dbc` |
| `src/exchange.rs` | `0da6dffc8318cea0179d2e99b20192d63a1b6b7800e36bbd9fedf8f46e3336c6` |
| `src/fold.rs` | `9d41a75673aab4415d81d084389f4070d6e68b59f7e9832a28666be9da2475ad` |
| `src/keystore.rs` | `f7b0c1c966b0c11b3afe45d7621a68b1ca3117f38ca6daf250b657e0f733b925` |
| `src/lib.rs` | `a1dec95f2560335e3cc9ccbf624c297d52f5cf16a86078701430598c833936d8` |
| `src/merge.rs` | `45c4d721552c9a2e7b41fa0dda96fdc17844bb946b68cf24224c8407518d4163` |
| `src/pepper.rs` | `e5f752b4026487a0e03bb3d82cde5fff1bfdbc92ca19fa4fc0a29166fb3f6a3a` |
| `src/scalar.rs` | `9199e955304cb3d54fe973ac5881f219d4c6f71b4be001a0d32a6fb16ffde92e` |
| `src/table/keyrep.rs` | `2b0db4d24967322b0cc95dd16fd68ec7dae734f7502c24c81db2a7ea242ee03c` |
| `src/table/locator.rs` | `5e94080b4c5d8e721a19ca36503c9df8c1139bc0ca10c1d8751f969bacb0b5c8` |
| `src/table/parallel.rs` | `055eaa8894d73fe119fea0273d0758be6d7fd4afb604484fdf69b5d272b59b1a` |
| `src/table/partition.rs` | `60e55bd89e81359ef94dccf2a1ac9338f15a50ceef636f8f43e3da24c9f2eb30` |
| `src/table/prepared.rs` | `cf0c12a9564130b1b6dd7e2d9533a5d6fe4e2196e308da92f82c630beca59e13` |
| `src/table/probe.rs` | `c5aee1a52a999f93db8d70a98243c08c87b3fd7a7368f00b9fddda148706e401` |
| `src/table/resources.rs` | `e6925991d7cf3102d3959322b257a34585f150e33d2add03a54bdbb79e3c61fd` |
| `src/table/scan.rs` | `ec633507a8ad456a7cb11553f559cf1abc08d3d4984a118e145eb5aa1ad91e97` |
| `src/table/state.rs` | `ef89a76e5705d03222efa36b4b372f490173905184bc5cd3ef0260767ada3673` |
| `src/table/tests.rs` | `b6eb15dfd8bfa743d211136285e4421c60b322180611e1b8f44f9f3d00068916` |
| `src/table.rs` | `02ce5b3a9333e8883648b275f38f364c5a461b880cdbadf8f992243c894386c6` |

## Residual and next bounded task

1. First diagnose completion within the existing bound. A one-file harness task may
   add elapsed phase markers outside timed sample regions in
   `examples/reg1/acceptance.rs`, then rerun this exact full command with the same
   counts, controls and 600-second stop. This proposed observability task must be
   dispatched separately; no such edit occurred here. It must retain every registered
   pair and both measurement phases. A timeout remains inconclusive.
2. The smallest existing-plan optimization candidate is exact tag-scan work in
   `src/table/scan.rs::match_tag`, consumed by `src/table/probe.rs::spill_read`, under
   [the serial scan task](../roadmap.md#partition-cut-and-frozen-bridge-maintenance).
   Its existing ownership is `src/table/scan.rs` plus `tests/table.rs`; gate
   `cargo test --locked --lib --test table`, with a throwaway suppressed live lane or
   borrowing/superset mask required to fail readback/mask tests and restore green.
   Dispatch an implementation only after current attribution justifies it; otherwise
   return attribution. The existing SWAR scan and power-of-two summary-index fast path
   already exist, so proposing their introduction again supplies no improvement.
3. Preserve complete late-hit coverage, full key/Sigma identity and the 1.0 interval
   ceiling. Cached summary state, layout or first-empty termination changes require
   the separate Astra/Sol boundary in [programme direction](../direction.md#serial-read-specialization-and-stable-parallel-scatter).

This task proves no theorem and runs no Rust correctness, extraction or Lean gate.
The timeout, source identity and partial control observations are measured; full pair
performance, all final controls and G1 acceptance remain open.
