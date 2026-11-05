//! `tokenize`: count the Qwen BPE tokens in a text file.
//!
//! `tokenize --tokenizer <tokenizer.json> --text <file> [--ids]`
//!
//! The retrieval index is word-based; this binary supplies the exact token count for the
//! context budget and the 1M-token corpus check. It does not write an index.

use std::process::ExitCode;

use apeiron::format::bpe::Bpe;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tokenize: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut tok = String::new();
    let mut text = String::new();
    let mut ids = false;
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{f} needs a value"));
        match f.as_str() {
            "--tokenizer" => tok = v()?,
            "--text" => text = v()?,
            "--ids" => ids = true,
            o => return Err(format!("unknown argument {o:?}")),
        }
    }
    if tok.is_empty() || text.is_empty() {
        return Err("--tokenizer and --text are required".into());
    }
    let bpe = Bpe::from_tokenizer_json(&tok).map_err(|e| format!("{tok}: {e:?}"))?;
    let body = std::fs::read_to_string(&text).map_err(|e| format!("{text}: {e}"))?;
    let out = bpe.encode(&body).map_err(|e| format!("{e:?}"))?;
    println!(
        "{{\"tokens\":{},\"characters\":{},\"bytes\":{}}}",
        out.len(),
        body.chars().count(),
        body.len()
    );
    if ids {
        for id in out {
            println!("{id}");
        }
    }
    Ok(())
}
