//! CLI entry point for `call-census`.

use call_census::{run, Config};
use std::path::PathBuf;
use std::process::ExitCode;

fn usage() -> &'static str {
    "call-census — syntax-level Rust call-site census for a target crate's consumers

USAGE:
    call-census --root <PATH> --output <DIR> [OPTIONS]

REQUIRED:
    --root <PATH>          Folder snapshot to census (the consumer tree)
    --output <DIR>         Directory for generated JSON/TSV/summary files

OPTIONS:
    --target-crate <NAME>  Target crate name (repeatable). Default:
                           hashtrinity, hashtrinity_derive,
                           hashtrinity_crypto, hashtrinity_codegen
    --target-type <NAME>   Extra target type name (repeatable)
    --target-method <NAME> Extra target method name (repeatable)
    --target-src <PATH>    Target crate source root to inventory (repeatable);
                           adds its package name(s) and public symbols
    --skip <DIRNAME>       Directory name to skip (repeatable)
    -h, --help             Print this help

NOTES:
    This is an AST census, not a compiler. Read summary.json limitations and the
    `resolution`/`confidence` fields on every record before drawing conclusions.
"
}

struct Args {
    root: Option<PathBuf>,
    output: Option<PathBuf>,
    target_crates: Vec<String>,
    target_types: Vec<String>,
    target_methods: Vec<String>,
    target_src: Vec<PathBuf>,
    skip: Vec<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        root: None,
        output: None,
        target_crates: Vec::new(),
        target_types: Vec::new(),
        target_methods: Vec::new(),
        target_src: Vec::new(),
        skip: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| -> Result<String, String> {
            it.next().ok_or_else(|| format!("{name} requires a value"))
        };
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            "--root" => a.root = Some(PathBuf::from(value("--root")?)),
            "--output" => a.output = Some(PathBuf::from(value("--output")?)),
            "--target-crate" => a.target_crates.push(value("--target-crate")?),
            "--target-type" => a.target_types.push(value("--target-type")?),
            "--target-method" => a.target_methods.push(value("--target-method")?),
            "--target-src" => a.target_src.push(PathBuf::from(value("--target-src")?)),
            "--skip" => a.skip.push(value("--skip")?),
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(a)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\n{}", usage());
            return ExitCode::from(2);
        }
    };
    let (Some(root), Some(output)) = (args.root, args.output) else {
        eprintln!("error: --root and --output are required\n\n{}", usage());
        return ExitCode::from(2);
    };
    let mut config = Config::new(root, output);
    config.target_crates = args.target_crates;
    config.target_types = args.target_types;
    config.target_methods = args.target_methods;
    config.target_src = args.target_src;
    config.extra_skip = args.skip;

    match run(&config) {
        Ok(summary) => {
            println!(
                "call-census: {} files ({} parsed, {} errors), {} call records, {} syntactically attributed",
                summary.files_total,
                summary.files_parsed,
                summary.files_parse_error,
                summary.records_total,
                summary.attributed_calls
            );
            println!(
                "  unattributed candidates {}, ambiguous {}, dynamic {}, other methods {}",
                summary.unattributed_method_candidates,
                summary.ambiguous_methods,
                summary.dynamic_dispatch_methods,
                summary.other_method_calls_total
            );
            println!("  wrote {}", config.output.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("call-census: {e}");
            ExitCode::FAILURE
        }
    }
}
