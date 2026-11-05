#![forbid(unsafe_code)]
//! `nous-serve`: the CLI wrapper for the `nous` provider library.
//!
//! It is a local, loopback-only OpenAI-compatible server. See `nous/README.md` for the
//! OpenCode and OpenAI-SDK configuration.

use std::process::ExitCode;

fn main() -> ExitCode {
    match nous::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("nous: {e}");
            ExitCode::FAILURE
        }
    }
}
