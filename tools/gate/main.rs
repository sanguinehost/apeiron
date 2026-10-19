//! `tools/gate`: the gate driver, split so that each subcommand is one module and a new one is
//! added without opening a file that already carries three.
//!
//! | module | owns |
//! |---|---|
//! | `core.rs` | the pure admission decision (the library crate) |
//! | `adapter.rs` | the impure boundary: files, hashes, subprocesses, locks |
//! | `cli.rs` | exit codes and the `key=value` verdict block |
//! | `registry.rs` | where the registry lives and which root its rows resolve against |
//! | `report.rs` | the identity report `docs check` prints |
//! | `docs.rs` | `gate docs check`: the invocation and the identity check |
//! | `aggregate.rs` | `gate docs check --controls`: the throwaway copy and the controls |
//! | `runner.rs` | `gate run <row-id>`: one row's gate command under its own budget |
//! | `hold.rs` | `gate hold`: the `HOLD` preflight |
//! | `controls.rs` | the conformance suite (`#[cfg(test]`) |
//!
//! This file owns the dispatch and nothing else; `core.rs` owns the pure decision, so the checking
//! bodies stay extractable through Charon/Aeneas for refinement. The adapter boundary is the trust
//! boundary: it reads the environment and reports what it found, and it decides nothing about
//! admission.

mod adapter;
mod aggregate;
mod cli;
mod coverage;
mod docs;
mod hold;
mod registry;
mod report;
mod runner;
mod structure;

#[cfg(test)]
mod controls;

use std::process::ExitCode;

use docs::USAGE;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("hold") => hold::hold(),
        Some("docs") => match args.get(1).map(String::as_str) {
            Some("check") => docs::docs_command(&args[2..]),
            _ => {
                eprintln!("{USAGE}");
                ExitCode::from(2)
            }
        },
        Some("run") => runner::run_command(&args[1..]),
        Some("structure") => structure::structure_command(&args[1..]),
        Some("coverage") => coverage::coverage_command(&args[1..]),
        Some("roots") => coverage::roots::roots_command(&args[1..]),
        Some(other) => {
            eprintln!(
                "gate: unknown subcommand `{other}`; known: docs, hold, run, structure, coverage, roots"
            );
            ExitCode::from(2)
        }
        None => {
            eprintln!(
                "usage: gate <subcommand>\n  docs check   load the artefact registry, report refusals, and with\n                \
                 --controls run each selected row's control in a throwaway copy\n  \
                 run <id>     execute one registry row's gate command under its budget\n  \
                 hold         refuse while a HOLD marker is in force\n  \
                 structure --check  refuse a tracked layout that breaks docs/contracts/tree.toml\n  \
                 coverage (--check | --report | --baseline)  extraction coverage of src/ fns\n  \
                 roots (--check | --report | --baseline)  kernel routing of src/learn product fns"
            );
            ExitCode::from(2)
        }
    }
}
