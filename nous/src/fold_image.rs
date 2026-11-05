//! The reduced `GGUF` image a fold pack serves from: the file's header, the non-linear tensors
//! (embeddings, output norm, `lm_head`) and every block's norms and biases, with the seven fitted
//! linear payloads left as untouched zero pages.  `Engine::from_bytes_with_fold` runs those seven
//! kinds from the pack's faces, so the `Q4_K_M` payloads never need to be read or resident.
//!
//! The image is the file's full length; on Linux the untouched tail is lazily mapped zero pages.

use std::os::unix::fs::FileExt;

use apeiron::engine::loader::Engine;
use apeiron::format::gguf::parse_header;
use apeiron::learn::fold::run::pack::fold_pack_read;

/// The seven fitted GGUF tensor stems, in the pack's order.
const KINDS: [&str; 7] = [
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
];

/// Whether `name` is one of the linear weights the fold faces replace.
fn is_fold_linear(name: &str) -> bool {
    name.strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
        .and_then(|(l, stem)| l.parse::<usize>().ok().map(|_| stem))
        .and_then(|stem| stem.strip_suffix(".weight"))
        .is_some_and(|stem| KINDS.contains(&stem))
}

/// Open `model` with `fold`, reading only the tensors the pack does not carry.
///
/// # Errors
/// A read/parse failure, a tensor without a byte size, or any refusal of
/// [`Engine::from_bytes_with_fold`].
pub fn load(model: &str, fold: &str) -> Result<Engine, String> {
    let planes = fold_pack_read(fold).map_err(|e| format!("fold pack {fold}: {e:?}"))?;
    let file = std::fs::File::open(model).map_err(|e| format!("{model}: {e}"))?;
    let file_len = file.metadata().map_err(|e| e.to_string())?.len();
    let mut want = 16usize << 20;
    let (head, hdr) = loop {
        let n = want.min(usize::try_from(file_len).map_err(|e| e.to_string())?);
        let mut b = vec![0u8; n];
        file.read_exact_at(&mut b, 0).map_err(|e| e.to_string())?;
        match parse_header(&b, file_len) {
            Ok(h) => break (b, h),
            Err(_) if n < usize::try_from(file_len).map_err(|e| e.to_string())? => {
                want *= 2;
            }
            Err(e) => return Err(format!("{model}: {e:?}")),
        }
    };
    let data0 = usize::try_from(hdr.data0).map_err(|e| e.to_string())?;
    let mut image = vec![0u8; usize::try_from(file_len).map_err(|e| e.to_string())?];
    image[..data0].copy_from_slice(&head[..data0]);
    drop(head);
    let mut kept = 0u64;
    for t in &hdr.tensors {
        let name = String::from_utf8_lossy(&t.name);
        if is_fold_linear(&name) {
            continue;
        }
        let n = t
            .n_bytes()
            .map_err(|e| format!("{name}: {e:?}"))?
            .ok_or_else(|| format!("{name}: no byte size"))?;
        let at = data0
            .checked_add(usize::try_from(t.offset).map_err(|e| e.to_string())?)
            .ok_or_else(|| format!("{name}: offset"))?;
        let end = at
            .checked_add(usize::try_from(n).map_err(|e| e.to_string())?)
            .ok_or_else(|| format!("{name}: length"))?;
        file.read_exact_at(
            image
                .get_mut(at..end)
                .ok_or_else(|| format!("{name}: past image"))?,
            u64::try_from(at).map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("{name}: {e}"))?;
        kept = kept.saturating_add(n);
    }
    eprintln!(
        "fold image: kept {:.0} MiB of tensors, {} planes from {fold}",
        kept as f64 / 1048576.0,
        planes.len()
    );
    Engine::from_bytes_with_fold(model, image, &planes).map_err(|e| format!("{model}: {e:?}"))
}
