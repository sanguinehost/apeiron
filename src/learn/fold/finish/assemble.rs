//! The whole-model Q32.32 [`Weights`] from the finishing state: each coded linear's summed plane
//! weight plus its frozen dense addend, the tuned norms and biases, and the frozen embedding and
//! head of [`crate::learn::fold::finish::train`].

#![allow(
    clippy::needless_range_loop,
    reason = "the kind index `k` addresses the planes, the model dimensions and `KIND_NAMES` alike"
)]

use crate::learn::fold::finish::train::{
    kind_dims, Frozen, LayerTunables, Linear, Planes, TrainError, Tunables, KIND_NAMES, N_KINDS,
};
use crate::learn::fold::fit::steps::GROUP;
use crate::learn::grad::model::{Config, LayerWeights, Weights};

/// The summed plane weight of one linear plus its optional frozen dense addend, in Q32.32;
/// `quantized` selects the stored [`CodedPlane::weight`] or the differentiable
/// [`CodedPlane::weight_unquantized`].
///
/// [`CodedPlane::weight`]: crate::learn::fold::finish::plane::CodedPlane::weight
/// [`CodedPlane::weight_unquantized`]: crate::learn::fold::finish::plane::CodedPlane::weight_unquantized
///
/// # Errors
/// [`TrainError::Shape`] on a plane or `extra` width, [`TrainError::Plane`] from a plane,
/// [`TrainError::Range`] on a plane refusal; `O(Σ_p o_p·n·log2(GROUP) + o_total·n)`, the plane
/// tensors pooled.
fn linear_weight(lin: &Linear, quantized: bool) -> Result<Vec<i64>, TrainError> {
    let on = lin.o_total.checked_mul(lin.n).ok_or(TrainError::Overflow)?;
    let mut w = vec![0_i64; on];
    for p in &lin.planes {
        if p.n != lin.n || p.o == 0 || !p.n.is_multiple_of(GROUP) {
            let m = format!("plane {}x{} in linear {}x{}", p.o, p.n, lin.o_total, lin.n);
            return Err(TrainError::Shape(m));
        }
        let pw = if quantized {
            p.weight()
        } else {
            p.weight_unquantized()
        };
        if pw.len() != p.o * p.n {
            return Err(TrainError::Range(format!(
                "plane {}x{} weight refused",
                p.o, p.n
            )));
        }
        p.add_into(&mut w, &pw, lin.o_total)?;
    }
    if let Some(e) = &lin.extra {
        if e.len() != on {
            return Err(TrainError::Shape(format!(
                "extra {} != {}x{}",
                e.len(),
                lin.o_total,
                lin.n
            )));
        }
        for (d, &v) in w.iter_mut().zip(e) {
            *d = d.checked_add(v).ok_or(TrainError::Overflow)?;
        }
    }
    Ok(w)
}

/// The `LayerWeights` of every kind of one layer; `O(Σ_k (o_k·n_k·log2(GROUP) + o_k·n_k))`, the
/// kind loop serial over each weight's pooled plane tensors.
fn assemble_layer(
    cfg: &Config,
    planes: &Planes,
    tun: &LayerTunables,
    l: usize,
    quantized: bool,
) -> Result<LayerWeights, TrainError> {
    let mut weights: Vec<Vec<i64>> = Vec::with_capacity(N_KINDS);
    for k in 0..N_KINDS {
        let lin = planes.linear(l, k)?;
        let (o, n) = kind_dims(cfg, k)?;
        if lin.o_total != o || lin.n != n {
            return Err(TrainError::Shape(format!(
                "layer {l} kind {}: planes {}x{} != model {o}x{n}",
                KIND_NAMES[k], lin.o_total, lin.n
            )));
        }
        weights.push(linear_weight(lin, quantized)?);
    }
    let t = tun.tuned();
    let [wq, wk, wv, wo, w_gate, w_up, w_down] = <[Vec<i64>; N_KINDS]>::try_from(weights)
        .map_err(|_| TrainError::Shape(format!("layer {l} kinds absent")))?;
    Ok(LayerWeights {
        attn_norm: t.ln1,
        wq,
        bq: Some(t.qb),
        wk,
        bk: Some(t.kb),
        wv,
        bv: Some(t.vb),
        wo,
        ffn_norm: t.ln2,
        w_gate,
        w_up,
        w_down,
    })
}

/// Assemble the whole-model Q32.32 [`Weights`]: the stored forward weight of every coded linear
/// plus the tuned norms and biases and the frozen embedding and head.
///
/// # Errors
/// [`TrainError::Shape`] on a plane or tunable shape, [`TrainError::Plane`] from a plane,
/// [`TrainError::Range`] outside the carrier; `O(n_layer·Σ_k o_k·n_k·log2(GROUP) + vocab·n_embd)`,
/// the layer loop serial over the pooled plane tensors.
pub fn assemble(
    cfg: &Config,
    planes: &Planes,
    tun: &Tunables,
    frozen: &Frozen,
) -> Result<Weights, TrainError> {
    assemble_with(cfg, planes, tun, frozen, true)
}

/// As [`assemble`], over the differentiable [`CodedPlane::weight_unquantized`] forward.
///
/// [`CodedPlane::weight_unquantized`]: crate::learn::fold::finish::plane::CodedPlane::weight_unquantized
///
/// # Errors
/// As [`assemble`].
pub fn assemble_unquantized(
    cfg: &Config,
    planes: &Planes,
    tun: &Tunables,
    frozen: &Frozen,
) -> Result<Weights, TrainError> {
    assemble_with(cfg, planes, tun, frozen, false)
}

/// [`assemble`]'s shared body. `O(assemble)`.
fn assemble_with(
    cfg: &Config,
    planes: &Planes,
    tun: &Tunables,
    frozen: &Frozen,
    quantized: bool,
) -> Result<Weights, TrainError> {
    let e = cfg.n_embd;
    if planes.layers.len() != cfg.n_layer || tun.layers.len() != cfg.n_layer {
        return Err(TrainError::Shape(format!(
            "planes {} / tunables {} != n_layer {}",
            planes.layers.len(),
            tun.layers.len(),
            cfg.n_layer
        )));
    }
    let emb = cfg.vocab.checked_mul(e).ok_or(TrainError::Overflow)?;
    if frozen.token_embd.len() != emb || frozen.lm_head.len() != emb {
        return Err(TrainError::Shape(format!(
            "token_embd {} / lm_head {} != vocab·n_embd {emb}",
            frozen.token_embd.len(),
            frozen.lm_head.len()
        )));
    }
    let output_norm = tun.output_norm();
    if output_norm.len() != e {
        return Err(TrainError::Range("output_norm refused".to_string()));
    }
    // The layer loop is structural: each layer's kinds and planes dispatch their own tensor work to
    // the pool, and the pool is not re-entrant (a task that dispatches deadlocks the dispatcher).
    let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.n_layer);
    for l in 0..cfg.n_layer {
        let lt = tun
            .layers
            .get(l)
            .ok_or_else(|| TrainError::Shape(format!("tunable layer {l} absent")))?;
        layers.push(assemble_layer(cfg, planes, lt, l, quantized)?);
    }
    Ok(Weights {
        layers,
        output_norm,
        token_embd: frozen.token_embd.clone(),
        lm_head: frozen.lm_head.clone(),
    })
}
