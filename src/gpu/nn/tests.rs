use super::*;

/// Every module's instruction stream walks (word counts self-consistent).
#[test]
fn modules_walk() {
    let mods: Vec<(&str, Vec<u32>)> = vec![
        ("embed", embed_f16_module(896, 151936, 256)),
        (
            "fusedropekv",
            fused_rope_kv_module(14, 2, 64, 2048, 1e6, 0, 64, false),
        ),
        // the int8-V arms walk too — a plane whose bindings only exist under a flag is a
        // plane no validity test covers unless the flag is one of the cases here.
        (
            "fusedropekvv8",
            fused_rope_kv_module(14, 2, 64, 2048, 1e6, 0, 64, true),
        ),
        (
            "fusedropekvknotv8",
            fused_rope_kv_module(14, 2, 64, 2048, 1e6, 64, 64, true),
        ),
        (
            "flashpartialv8",
            flash_partial_module(14, 2, 64, 2048, 0.125, 8, 64, false, false, true, 0),
        ),
        (
            "flashpartialknotv8",
            flash_partial_module(14, 2, 64, 2048, 0.125, 8, 64, true, false, true, 0),
        ),
        (
            "fusedropekvknot",
            fused_rope_kv_module(14, 2, 64, 2048, 1e6, 64, 64, false),
        ),
        (
            "flashpartial",
            flash_partial_module(14, 2, 64, 2048, 0.125, 8, 64, false, false, false, 0),
        ),
        (
            "flashpartialknot",
            flash_partial_module(14, 2, 64, 2048, 0.125, 8, 64, true, false, false, 0),
        ),
        (
            "thetascan",
            theta_scan_module(14, 2, 64, 2048, 0.125, 64, 8),
        ),
        (
            "thetaselect",
            theta_select_module(14, 2048, 64, 8, 1, 2, 16.635, true, 0, 0),
        ),
        (
            "flashpartialcompact",
            flash_partial_module(14, 2, 64, 2048, 0.125, 8, 64, true, true, false, 0),
        ),
        // the rope-fold flash at the decode bench shape (tmax 512 ⇒ nsplit 7), the default
        // (`DEC_FLASH_V2` unset) merged-prologue / subgroup-max / chunked-pass-3 arm.
        (
            "flashpartialfold",
            flash_partial_fold_module(14, 2, 64, 512, 0.125, 7, 64, 1e6, 8, true, 64),
        ),
        ("flashcombine", flash_combine_module(14, 64, 8)),
        ("silu", silu_mul_module(4864, 256, 4864)),
        ("add", add_inplace_module(896, 256)),
        ("qknorm", qk_norm_module(14, 128, 1e-6, 0)),
        ("rmsnorm", rmsnorm_module(896, 1e-6, 2, None, true)),
        ("rmsaddqseg", rmsnorm_module(896, 1e-6, 2, Some(8), true)),
        ("rmsfused", rmsnorm_module(896, 1e-6, 0, None, false)),
        ("quantvec", quantize_vec_module(896, 2)),
        ("siluq", silu_mul_q_module(4864, 4864, 2)),
        (
            "embchart",
            embed_chart_module(896, 151936, 256, crate::gpu::simt::Carrier::Q4),
        ),
        ("amaxp", argmax_partial_module(151936, 256, 256)),
        ("amaxf", argmax_finish_module(256)),
        (
            "lastrowgather",
            last_row_gather_module(224, 28, 0xf_ffff, 0x1_ffff),
        ),
        ("rmsqsb", rmsnorm_batch_qsb_module(896, 1e-6, true)),
        ("rmsqsb0", rmsnorm_batch_qsb_module(896, 1e-6, false)),
        ("quantsb", quantize_batch_sb_module(4864)),
        (
            "lastrowgathersb",
            last_row_gather_sb_module(224, 4, 0xf_ffff, 0x1_ffff),
        ),
        // The five gated-delta-net modules, at Qwen3.8-27B's own shape. The recurrence is 48 of
        // that model's 64 blocks, so every one of these is on the hot path of three quarters of
        // the network — and none of them was in this walk when the rule kernel landed.
        ("dnrule", deltanet_rule_module(128, 48, 16, (0, 0, 0))),
        (
            "dnrulefused",
            deltanet_rule_module(128, 48, 16, (0, 2048, 4096)),
        ),
        ("dnconv", deltanet_conv_silu_module(10240, 4)),
        ("dnl2", deltanet_l2_heads_module(16, 128, 1e-6, 0)),
        ("dnl2off", deltanet_l2_heads_module(16, 128, 1e-6, 2048)),
        ("dngk", deltanet_gk_module(48)),
        ("dngate", deltanet_gate_norm_module(128, 48, 1e-6)),
        // The gated-attention pair, at Qwen3.5-0.8B's own shape, plus the partial-rope form of
        // the rope kernel (`rope_dims` 64 of a 256-wide head) beside its full-rotation twin.
        ("attnsplit", attn_qgate_split_module(8, 2, 256)),
        ("attngate", attn_gate_apply_module(2048)),
        (
            "ropepartial",
            fused_rope_kv_module(8, 2, 256, 2048, 1e7, 0, 64, false),
        ),
        (
            "ropepartialv8",
            fused_rope_kv_module(8, 2, 256, 2048, 1e7, 0, 64, true),
        ),
        ("copyvec", copy_vec_module(1024, 0)),
    ];
    for (name, m) in mods {
        assert_eq!(m[0], SPV_MAGIC, "{name}");
        let mut i = 5;
        while i < m.len() {
            let wc = (m[i] >> 16) as usize;
            assert!(wc >= 1, "{name}: zero word count at {i}");
            i += wc;
        }
        assert_eq!(i, m.len(), "{name}: instruction stream misaligned");
    }
}
