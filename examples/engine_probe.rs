//! Probe: load `qwen05b-Q4_K_M.gguf`, print the resolved config, the tensor-directory size and the
//! deterministic checksum of three fixed token embeddings, and assert determinism across two
//! independent loads.
//!
//! Run: `cargo run --release --locked --example engine_probe [path]`
//!
//! The embedding values are Q32.32 `i64` (see `apeiron::engine`): each is the real dequantized
//! weight times `2^32`. The probe asserts only what this slice implements — the loader config, the
//! table size and the embedding checksum — plus two in-band refusals: a token id at the vocabulary
//! boundary and a second load that must reproduce the checksum bit-for-bit.

use std::error::Error;
use std::process::ExitCode;

use apeiron::engine::fixed_vector_checksum;
use apeiron::engine::loader::Engine;

/// Default model, the Qwen2.5-0.5B `Q4_K_M` GGUF named by the task.
const DEFAULT_PATH: &str = "~/models/gguf/qwen05b-Q4_K_M.gguf";
/// Three fixed token ids: `!` at 0, `" Hello"`-adjacent content at 9707, and the EOS id 151643.
const PROBE_TOKENS: [u32; 3] = [0, 9707, 151_643];

fn run(path: &str) -> Result<(), Box<dyn Error>> {
    let first = Engine::open(path)?;
    let second = Engine::open(path)?;

    println!("path          {}", first.path);
    println!("file bytes    {}", first.file_len);
    println!("tensors       {}", first.table().len());
    println!("architecture  {}", first.config.architecture);
    println!("n_layer       {}", first.config.n_layer);
    println!("n_head        {}", first.config.n_head);
    println!("n_head_kv     {}", first.config.n_head_kv);
    println!("head_dim      {}", first.config.head_dim);
    println!("n_embd        {}", first.config.n_embd);
    println!("n_ff          {}", first.config.n_ff);
    println!("vocab_size    {}", first.config.vocab_size);
    println!("ctx_length    {}", first.config.context_length);
    println!("rope_theta    {}", first.config.rope_theta);
    println!("rms_eps       {}", first.config.rms_eps);
    let emb = first.token_embd();
    println!(
        "token_embd    {} {:?} {:?} {:?}",
        emb.quant.name(),
        emb.dims,
        emb.n_elements,
        emb.byte_range
    );

    for &id in &PROBE_TOKENS {
        let va = first.token_embedding(id)?;
        let vb = second.token_embedding(id)?;
        let ca = fixed_vector_checksum(&va);
        let cb = fixed_vector_checksum(&vb);
        println!(
            "token {id:>6}  len {}  checksum {ca:#018x}  head {} {} {}",
            va.len(),
            va[0],
            va[1],
            va[2]
        );
        if ca != cb {
            return Err(
                format!("token {id}: checksum differs across loads {ca:#x} != {cb:#x}").into(),
            );
        }
        if va.len() != first.config.n_embd {
            return Err(format!(
                "token {id}: row length {} != n_embd {}",
                va.len(),
                first.config.n_embd
            )
            .into());
        }
    }

    // In-band refusal control: the id at the vocabulary boundary must be refused.
    let boundary = u32::try_from(first.config.vocab_size)?;
    match first.token_embedding(boundary) {
        Ok(_) => return Err(format!("token {boundary} was accepted past the vocabulary").into()),
        Err(e) => println!("refusal       token {boundary}: {e}"),
    }

    println!(
        "determinism   both loads agree on all {} probe tokens",
        PROBE_TOKENS.len()
    );
    Ok(())
}

fn main() -> ExitCode {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_PATH.to_string());
    match run(&path) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("engine_probe: {e}");
            ExitCode::FAILURE
        }
    }
}
