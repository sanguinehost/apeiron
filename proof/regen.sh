#!/bin/bash
# The five word_count_* roots below.
# Regenerate and check the Aeneas extraction of the Apeiron word kernels.
#
#   proof/regen.sh            regenerate ApeironWord.lean and build the Lean package
#   proof/regen.sh --check    fail if regeneration differs from the committed Lean file
#
# The tools are not vendored. By default they are read from `target/aeneas/`, the layout the pinned
# archive unpacks to. Point `AENEAS_HOME` at another installation (for example Penglai's) to use it
# here; the script links it into `target/aeneas` so the fixed `require` path in `lakefile.lean`
# resolves. Neither the tools nor the generated `.llbc` are source.

set -eu

PROOF_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(CDPATH= cd -- "$PROOF_DIR/.." && pwd)
EXTRACT_DIR="$PROOF_DIR/extract"

AENEAS_HOME=${AENEAS_HOME:-"$REPO_DIR/target/aeneas"}
LINK="$REPO_DIR/target/aeneas"
# `target/` is gitignored, so a fresh clone has no `target/` and the `ln -s` below would fail on an
# absent parent before any tool runs. Create it; a real `target/aeneas` is left untouched.
mkdir -p "$REPO_DIR/target"
# An ambient `AENEAS_HOME` that names no translator (for example a source checkout, not an install)
# is not an installation and would fail the tool check below. Fall back to the repo's `target/aeneas`
# when that is an installation; an explicit AENEAS_HOME that ships `aeneas` is still authoritative.
if [ "$AENEAS_HOME" != "$LINK" ] && [ ! -x "$AENEAS_HOME/aeneas" ] && [ -x "$LINK/aeneas" ]; then
    echo "regen: AENEAS_HOME=$AENEAS_HOME has no aeneas; using $LINK (set AENEAS_HOME to an installation to override)" >&2
    AENEAS_HOME="$LINK"
fi
if [ "$AENEAS_HOME" != "$LINK" ]; then
    # A stale link is removed and recreated. `-e` follows the link, so a self-referential
    # `target/aeneas -> .../target/aeneas` reads as absent and the `ln` below then aborts the run
    # with `File exists` before any tool runs; `-L` tests the link itself. A real directory is left
    # alone: `-e` sees it, so no `ln` is attempted, and `rm -f` on a directory is not this guard's
    # call to make.
    if [ -L "$LINK" ]; then
        rm -f "$LINK"
    fi
    if [ ! -e "$LINK" ]; then
        ln -s "$AENEAS_HOME" "$LINK"
    fi
fi

CHARON="$AENEAS_HOME/charon"
CHARON_DRIVER="$AENEAS_HOME/charon-driver"
AENEAS="$AENEAS_HOME/aeneas"
LEAN_BACKEND="$AENEAS_HOME/backends/lean"

# Charon extracts with its own pinned rustc and needs that rustc's `std` built with full MIR; it
# builds that sysroot with `cargo miri setup` and caches it. The default cache location is one
# fixed path shared by every toolchain, so a `miri setup` run under a different nightly leaves `std`
# compiled by another rustc and the extraction dies with E0514 (`found crate 'std' compiled by an
# incompatible version of rustc`) before any Lean is produced. Name the sysroot after the toolchain
# Charon reports and keep it under `target/`, beside the pinned tools and outside the shared
# `$XDG_CACHE_HOME/miri` every other toolchain on the box writes to. Charon builds it on first run;
# the path is not created here, because an empty directory is not a built sysroot.
CHARON_TOOLCHAIN=$("$CHARON" toolchain-version)
SYSROOT="$REPO_DIR/target/aeneas-sysroot/$CHARON_TOOLCHAIN"
CHARON_MIRI_SYSROOT="$SYSROOT"
MIRI_SYSROOT="$SYSROOT"
export CHARON_MIRI_SYSROOT MIRI_SYSROOT

for f in "$CHARON" "$AENEAS"; do
    [ -f "$f" ] && [ -x "$f" ] || { echo "missing executable tool: $f" >&2; exit 2; }
done
[ -d "$LEAN_BACKEND" ] || { echo "missing Aeneas Lean backend: $LEAN_BACKEND" >&2; exit 2; }

tool_sha256() {
    [ -f "$1" ] && [ -x "$1" ] || { echo "missing executable tool identity: $1" >&2; exit 2; }
    sha256sum "$1"
}

# Charon records `Source:` annotations relative to the working directory. Pin it so a normal run
# and `--check` from any directory produce byte-identical output.
cd "$REPO_DIR"

CRATE=apeiron_word
LLBC="$PROOF_DIR/$CRATE.llbc"
GENERATED=ApeironWord.lean
EXTRACT_TARGET="$REPO_DIR/target/apeiron-extract"

echo "== tools =="
echo "charon:  $("$CHARON" version)"
echo "aeneas:  $("$AENEAS" -version)"
echo "rust:    $CHARON_TOOLCHAIN"
echo "sysroot: $SYSROOT"
echo "lean:    $(cat "$PROOF_DIR/lean-toolchain")"
echo "sha256:"
tool_sha256 "$CHARON"
if [ -e "$CHARON_DRIVER" ]; then
    tool_sha256 "$CHARON_DRIVER"
fi
tool_sha256 "$AENEAS"

# Fail-closed tool identity (added 2026-10-05). The block above PRINTS the tool digests; this
# asserts the selected translator is the pinned one. Before this, an ambient `AENEAS_HOME` (see the
# header) could select a different translator and change the evidence's tool identity with no
# refusal — the script only printed the digest. A repin is deliberate: edit `proof/patches/PINS`.
PINNED_TRANSLATOR_SHA256=$(sed -n 's/^AENEAS_TRANSLATOR_SHA256=//p' "$PROOF_DIR/patches/PINS")
if [ -z "$PINNED_TRANSLATOR_SHA256" ]; then
    echo "tool-identity: FAIL proof/patches/PINS has no AENEAS_TRANSLATOR_SHA256" >&2
    exit 2
fi
SELECTED_TRANSLATOR_SHA256=$(sha256sum "$AENEAS" | cut -d' ' -f1)
CANON_TRANSLATOR_SHA256=$(sed -n 's/^AENEAS_TRANSLATOR_CANON_SHA256=//p' "$PROOF_DIR/patches/PINS")
if [ "$SELECTED_TRANSLATOR_SHA256" != "$PINNED_TRANSLATOR_SHA256" ] && [ "$SELECTED_TRANSLATOR_SHA256" != "$CANON_TRANSLATOR_SHA256" ]; then
    echo "tool-identity: FAIL selected translator sha256=$SELECTED_TRANSLATOR_SHA256 != pinned $PINNED_TRANSLATOR_SHA256 or canonical $CANON_TRANSLATOR_SHA256 (AENEAS_HOME=$AENEAS_HOME)" >&2
    echo "tool-identity: a rebuild that does not reproduce the pinned translator has not reproduced the translator; set AENEAS_HOME to the pinned install, or repin deliberately in proof/patches/PINS" >&2
    exit 1
fi
echo "tool-identity: OK translator $SELECTED_TRANSLATOR_SHA256 == pinned"

echo "== source =="
# Every listed source must exist: a missing live source fails the digest stage instead of being
# skipped, so a stale path (for example a deleted module) cannot pass silently.
# `src/map/table.rs` textually includes `table/{state,locator,prepared}.rs`; `src/map/coord.rs`
# pulls `src/map/coord/address.rs` and `src/map/coord/locator.rs` by `#[path]`, and `src/map/coord/locator.rs`
# pulls `src/map/coord/locator_core.rs`. All are extraction source and carry a digest. Every `#[path]`
# target of `proof/extract/lib.rs` is listed: the engine closure `loader/step/chart/kern/trace`, the
# `mixer.rs` and `hadamard.rs` roots, `moe.rs`, `numeric/fixed_ln.rs` and `crypto.rs` are pulled too, so a change
# to any of them moves the generated Lean and moves this digest list with it. `src/map/table/tests.rs`
# is `#[cfg(test)]` and is not part of the Charon unit; it is deliberately excluded here.
SOURCE_FILES="
$REPO_DIR/src/scalar.rs
$REPO_DIR/src/cell.rs
$REPO_DIR/src/exchange.rs
$REPO_DIR/src/carrier.rs
$REPO_DIR/src/map/table.rs
$REPO_DIR/src/map/table/state.rs
$REPO_DIR/src/map/table/locator.rs
$REPO_DIR/src/map/table/prepared.rs
$REPO_DIR/src/map/table/probe.rs
$REPO_DIR/src/map/coord.rs
$REPO_DIR/src/map/coord/address.rs
$REPO_DIR/src/map/coord/locator.rs
$REPO_DIR/src/map/coord/locator_core.rs
$REPO_DIR/src/map/table/keyrep.rs
$REPO_DIR/src/map/table/partition.rs
$REPO_DIR/src/map/table/resources.rs
$REPO_DIR/src/map/table/parallel.rs
$REPO_DIR/src/numeric/fixed_exp.rs
$REPO_DIR/src/numeric/register.rs
$REPO_DIR/src/numeric/fixed_rsqrt.rs
$REPO_DIR/src/numeric/fixed_trig.rs
$REPO_DIR/src/format/gguf.rs
$REPO_DIR/src/format/quant.rs
$REPO_DIR/src/format/packing.rs
$REPO_DIR/src/engine/mod.rs
$REPO_DIR/src/engine/num/value.rs
$REPO_DIR/src/engine/ops/silu.rs
$REPO_DIR/src/engine/ops/norm.rs
$REPO_DIR/src/engine/ops/rope.rs
$REPO_DIR/src/engine/error.rs
$REPO_DIR/src/engine/num/f32_convert.rs
$REPO_DIR/src/engine/num/dequant.rs
$REPO_DIR/src/engine/ops/matvec.rs
$REPO_DIR/src/engine/attn.rs
$REPO_DIR/src/engine/mixer.rs
$REPO_DIR/src/engine/ops/lm_head.rs
$REPO_DIR/src/engine/ops/sampler.rs
$REPO_DIR/src/engine/ops/hadamard.rs
$REPO_DIR/src/engine/ops/ffn.rs
$REPO_DIR/src/engine/ops/moe.rs
$REPO_DIR/src/engine/ops/fused.rs
$REPO_DIR/src/engine/ops/par.rs
$REPO_DIR/src/engine/kern/pow2.rs
$REPO_DIR/src/gpu/spirv/mod.rs
$REPO_DIR/src/gpu/spirv/check.rs
$REPO_DIR/src/policy/mod.rs
$REPO_DIR/src/policy/decide.rs
$REPO_DIR/src/engine/loader.rs
$REPO_DIR/src/engine/step.rs
$REPO_DIR/src/engine/chart/mod.rs
$REPO_DIR/src/engine/kern/mod.rs
$REPO_DIR/src/engine/trace.rs
$REPO_DIR/src/numeric/fixed_ln.rs
$REPO_DIR/src/crypto/keccak.rs
$REPO_DIR/src/crypto/sha256.rs
$REPO_DIR/src/crypto/chacha.rs
$REPO_DIR/src/crypto/aes.rs
$REPO_DIR/src/map/hashtrinity/mod.rs
$REPO_DIR/src/map/hashtrinity/algebra.rs
$REPO_DIR/src/map/hashtrinity/freq.rs
$REPO_DIR/src/map/hashtrinity/hex.rs
$REPO_DIR/src/map/hashtrinity/base64.rs
$REPO_DIR/src/map/hashtrinity/seed.rs
$REPO_DIR/src/map/hashtrinity/tregex.rs
$REPO_DIR/src/map/hashtrinity/par.rs
$REPO_DIR/src/map/hashtrinity/face.rs
$REPO_DIR/src/map/hashtrinity/crypto.rs
$REPO_DIR/src/learn/gram.rs
$REPO_DIR/src/learn/sigma_delta.rs
$REPO_DIR/src/learn/onebit_bits.rs
$REPO_DIR/src/learn/grad/dense.rs
$REPO_DIR/src/learn/grad/attn.rs
$REPO_DIR/src/learn/grad/head.rs
$REPO_DIR/src/learn/grad/model.rs
$REPO_DIR/src/learn/gguf.rs
$REPO_DIR/src/learn/code/e8.rs
$REPO_DIR/src/learn/solve/q32.rs
$REPO_DIR/src/learn/solve/syrk.rs
$REPO_DIR/src/learn/solve/gemm.rs
$REPO_DIR/src/learn/solve/chol.rs
$REPO_DIR/src/learn/solve/inv.rs
$REPO_DIR/src/learn/solve/eig.rs
$REPO_DIR/src/learn/fold/stats/mod.rs
$REPO_DIR/src/learn/fold/stats/basis.rs
$REPO_DIR/src/learn/fold/stats/budget.rs
$REPO_DIR/src/learn/fold/stats/capture.rs
$REPO_DIR/src/learn/fold/stats/heldout.rs
$REPO_DIR/src/learn/fold/stats/hocapture.rs
$REPO_DIR/src/learn/fold/stats/kldbase.rs
$REPO_DIR/src/learn/fold/stats/lw.rs
$REPO_DIR/src/learn/fold/stats/parts.rs
$REPO_DIR/src/learn/fold/fit/mod.rs
$REPO_DIR/src/learn/fold/fit/corrector.rs
$REPO_DIR/src/learn/fold/fit/eigh.rs
$REPO_DIR/src/learn/fold/fit/fnorm.rs
$REPO_DIR/src/learn/fold/fit/fold_fit.rs
$REPO_DIR/src/learn/fold/fit/gptq_e8.rs
$REPO_DIR/src/learn/fold/fit/gptq_e8_carrier.rs
$REPO_DIR/src/learn/fold/fit/icm.rs
$REPO_DIR/src/learn/fold/fit/nb_refine.rs
$REPO_DIR/src/learn/fold/fit/quant.rs
$REPO_DIR/src/learn/fold/fit/steps.rs
$EXTRACT_DIR/fold_port.rs
$EXTRACT_DIR/learn_port.rs
$EXTRACT_DIR/solve_port.rs
$EXTRACT_DIR/grad_solve_port.rs
$EXTRACT_DIR/hashtrinity_port.rs
$EXTRACT_DIR/ports/format_port.rs
$EXTRACT_DIR/coord_port.rs
$EXTRACT_DIR/ports/fit_port.rs
$EXTRACT_DIR/ports/crypto_port.rs
$EXTRACT_DIR/ports/fold_finish_port.rs
$REPO_DIR/src/learn/fold/finish/mod.rs
$REPO_DIR/src/learn/fold/finish/plane.rs
$REPO_DIR/src/learn/fold/finish/train.rs
$REPO_DIR/src/learn/fold/run/convert.rs
$REPO_DIR/src/learn/fold/run/transfer.rs
$EXTRACT_DIR/ports/fold_layer_port.rs
$EXTRACT_DIR/ports/table_port.rs
$REPO_DIR/src/learn/fold/layer/mod.rs
$REPO_DIR/src/learn/fold/layer/driver.rs
$REPO_DIR/src/learn/fold/layer/tensor.rs
$REPO_DIR/src/learn/fold/layer/resq.rs
$REPO_DIR/src/learn/fold/layer/stage.rs
$REPO_DIR/src/map/hashtrinity/side.rs
$REPO_DIR/src/map/hashtrinity/overflow.rs
$EXTRACT_DIR/Cargo.toml
$EXTRACT_DIR/Cargo.lock
$EXTRACT_DIR/lib.rs
"
for f in $SOURCE_FILES; do
    [ -f "$f" ] || { echo "missing extraction source: $f" >&2; exit 2; }
done
# shellcheck disable=SC2086
sha256sum $SOURCE_FILES

# Serving-path items Aeneas cannot yet translate (bodies stay opaque; signatures extract). One reason
# class per line: simd-intrinsic, f32, thread, ffi, unsafe-ptr, unsupported:<construct>,
# extraction-cost (a body whose translation drives Aeneas past the extraction memory bound; the
# resulting axiom is recorded in `boundary-axioms.txt`).
SERVING_OPAQUE=(
    --opaque 'crate::chart::x86' # simd-twin: AVX2 row kernels, twin = chart::portable
    --opaque 'crate::kern::gemm::x86' # simd-twin: AVX2 GEMM rows, twin = the scalar gemm path
    --opaque 'crate::kern::bitplane::x86' # simd-twin: AVX2 bitplane interval, twin = the scalar bitplane path
    --opaque 'crate::kern::matmat::q5_0_block_kdots_avx2' # simd-twin
    --opaque 'crate::chart::q8_0_row_kcols_avx2' # simd-twin
    --opaque 'crate::chart::_::decode_step_avx2' # simd-twin: twin = decode_step_portable
    --opaque 'crate::mixer::f32_carrier::stage2_gated_norm_f32_avx2_inner' # simd-twin: twin = stage2_gated_norm_f32_scalar
    --opaque 'crate::mixer::conv_f32::range_avx2' # simd-twin: twin = range_scalar
    --opaque 'crate::attn::f32_core::attn_tile8_avx2' # simd-twin: tile of attn_core_f32_strided_scalar
    --opaque 'crate::attn::f32_core::attn_core_f32_strided_avx2_inner' # simd-twin: twin = attn_core_f32_strided_scalar
    --opaque 'crate::mixer::f32_carrier::delta_step_f32_avx2_inner' # simd-twin: twin = delta_step_f32_scalar
    --opaque 'crate::kern::lut::block_inners_avx2' # simd-twin: scalar block inner of the Pq2 LUT row
    --opaque 'crate::kern::lut::tile_accs_avx2' # simd-twin: scalar tile accumulate of the Pq2 LUT
    --opaque 'crate::mixer::f32_carrier::exp_f32_avx2' # simd-twin: twin = exp_f32_scalar
    --opaque 'crate::mixer::f32_carrier::silu_f32_avx2' # simd-twin: twin = silu_f32_scalar
    --opaque 'crate::mixer::f32_carrier::conv_silu_tap4_avx2' # simd-twin: scalar conv_silu tap
    --opaque 'crate::mixer::f32_carrier::l2_norm_sum_sq_avx2_inner' # simd-twin: twin = l2_norm_sum_sq scalar
    --opaque 'crate::mixer::f32_carrier::l2_normalize_f32_in_place_avx2_inner' # simd-twin: scalar l2 normalize
    --opaque 'crate::mixer::f32_carrier::readout8_avx2' # simd-twin: scalar readout
    --opaque 'crate::attn::f32_core::dot_f32_avx2' # simd-twin: scalar dot_f32
    --opaque 'crate::chart::ffn::swiglu_f32_avx2_8_into' # simd-twin: scalar swiglu_f32
    --opaque 'crate::mixer::delta_head_step_fast' # simd-intrinsic
    --opaque 'crate::par::matvec_rows_chunked' # thread
    --opaque 'crate::solve_port::gemm::block_rows_pool' # thread: engine-pool dispatch over disjoint row ranges, src/learn/solve/gemm.rs
    --opaque 'crate::solve_port::syrk::block_rows_pool' # thread: engine-pool dispatch over disjoint row ranges, src/learn/solve/syrk.rs
    --opaque 'crate::solve_port::chol::trailing_update_pool' # thread: engine-pool dispatch over disjoint row ranges, src/learn/solve/chol.rs
    --opaque 'crate::solve_port::inv::invert_columns_pool' # thread: engine-pool dispatch over disjoint column ranges, src/learn/solve/inv.rs
    --opaque 'crate::chart::pool' # thread: std::thread worker pool, dyn Fn jobs, transmute (every item)
    --opaque 'crate::step::prefill_block' # process-flag: APEIRON_PREFILL_COLS
    --opaque 'crate::step::bitplane_enabled' # process-flag: APEIRON_BITPLANE
    --opaque 'crate::step::softmax::exact_scan_enabled' # process-flag: APEIRON_EXACT_SCAN (thread override)
    --opaque 'crate::mixer::exact_scan' # process-flag: APEIRON_EXACT_SCAN
    --opaque 'crate::mixer::carrier_from_env' # process-flag: APEIRON_F32_CARRIER
    --opaque 'crate::mixer::head_slices_setting' # process-flag: APEIRON_HEAD_SLICES
    --opaque 'crate::trace' # instrumentation: counters and APEIRON_TRACE; no effect on computed values
    --opaque 'crate::loader::optional_f64' # load-config: GGUF float metadata (f32/f64/int widening)
    --opaque 'crate::gguf::host_read_prefix' # host-io: file prefix read
    --opaque 'crate::gguf::host_read_at' # host-io: ranged file read
    --opaque 'crate::gguf::host_read_all' # host-io: whole-file read
    --opaque 'crate::par::online_threads' # host-config: online CPU count
)

run_extract() {
    dest_dir=$1
    llbc=$2
    # The reproducible historical configuration. The explicit monomorphizing candidate is
    # withdrawn: it omits `--ops-to-function-calls` and `--index-to-function-calls`, which lower
    # every array/slice index and repeat in the tree, so it rejects on `core::array::repeat` and
    # `TArray`/`TSlice`. The preset translates the current source.
    "$CHARON" cargo \
        --preset aeneas \
        --opaque 'crate::fp::prim' \
        --opaque 'crate::crypto::shake128' \
        --opaque 'crate::crypto::shake256' \
        "${SERVING_OPAQUE[@]}" \
        --dest-file "$llbc" \
        --start-from crate::word_mul \
        --start-from crate::word_norm \
        --start-from crate::word_point_density \
        --start-from crate::word_point_flux \
        --start-from crate::word_conservative_density_step \
        --start-from crate::word_diffusive_flux_half \
        --start-from crate::word_diffusive_line3_step \
        --start-from crate::word_exchange_impulse \
        --start-from crate::word_elastic_exchange \
        --start-from crate::word_accept_derived_impulse \
        --start-from crate::word_i64_codec_encode \
        --start-from crate::word_i64_codec_decode \
        --start-from crate::word_regen_hold \
        --start-from crate::word_regen_matches \
        --start-from crate::word_regen_key_owned \
        --start-from crate::word_regen_address_held \
        --start-from crate::word_stored_table_get_at \
        --start-from crate::word_regen_table_get_at \
        --start-from crate::word_stored_table_build_at \
        --start-from crate::word_regen_table_build_at \
        --start-from crate::word_stored_prepared_shard_get \
        --start-from crate::word_stored_prepared_shard_apply \
        --start-from crate::word_regen_prepared_shard_get \
        --start-from crate::word_regen_prepared_shard_apply \
        --start-from crate::word_stored_apply_get \
        --start-from crate::word_stored_apply_insert \
        --start-from crate::word_regen_apply_get \
        --start-from crate::word_regen_apply_insert \
        --start-from crate::word_schedule_is_complete \
        --start-from crate::word_coord_finish \
        --start-from crate::word_coord_spread \
        --start-from crate::word_coord_unspread \
        --start-from crate::word_coord_id \
        --start-from crate::word_coord_place \
        --start-from crate::word_stored_relocate_entry \
        --start-from crate::word_regen_relocate_entry \
        --start-from crate::word_stored_relocate_all \
        --start-from crate::word_regen_relocate_all \
        --start-from crate::word_stored_apply_pending_and_commit \
        --start-from crate::word_regen_apply_pending_and_commit \
        --start-from crate::word_stored_publish_prepared \
        --start-from crate::word_regen_publish_prepared \
        --start-from crate::word_capacity_bound_holds \
        --start-from crate::word_act_scale \
        --start-from crate::word_f16_split \
        --start-from crate::word_pow2_finish \
        --start-from crate::word_fixed_exp \
        --start-from crate::word_fixed_rsqrt \
        --start-from crate::word_fixed_sin_cos \
        --start-from crate::word_gguf_parse_header \
        --start-from crate::word_q4_k_codes \
        --start-from crate::word_q6_k_codes \
        --start-from crate::word_packing_decode_block \
        --start-from crate::format_port::word_packing_error_from_quant \
        --start-from crate::format_port::word_packing_validate \
        --start-from crate::format_port::word_packing_from_gguf_kind \
        --start-from crate::format_port::word_packing_encode_block \
        --start-from crate::format_port::word_packing_row_decode_fixed \
        --start-from crate::format_port::word_packing_import_row \
        --start-from crate::format_port::word_packing_convert_row \
        --start-from crate::format_port::word_quant_q4_k_scale_mins \
        --start-from crate::format_port::word_quant_ggml_type_name \
        --start-from crate::format_port::word_bpe_hex_val \
        --start-from crate::format_port::word_bpe_contains_nul \
        --start-from crate::format_port::word_bpe_is_nl \
        --start-from crate::format_port::word_bpe_find_sub \
        --start-from crate::coord_port::word_coord_with \
        --start-from crate::coord_port::word_coord_inverse_odd \
        --start-from crate::coord_port::word_coord_bucket \
        --start-from crate::coord_port::word_coord_mix \
        --start-from crate::coord_port::word_coord_pack \
        --start-from crate::coord_port::word_coord_i21_new \
        --start-from crate::coord_port::word_coord_i21_get \
        --start-from crate::coord_port::word_coord_i21_coord \
        --start-from crate::coord_port::word_coord_i21_tuple_packed \
        --start-from crate::coord_port::word_coord_i21_tuple_unpack \
        --start-from crate::coord_port::word_h3_40_new \
        --start-from crate::coord_port::word_h3_40_q \
        --start-from crate::coord_port::word_h3_40_inverse_of_two \
        --start-from crate::coord_port::word_h3_40_r \
        --start-from crate::coord_port::word_h3_40_s \
        --start-from crate::coord_port::word_h3_40_version \
        --start-from crate::coord_port::word_residue_new \
        --start-from crate::coord_port::word_parabolic_dual_scalar \
        --start-from crate::coord_port::word_parabolic_dual_nilpotent \
        --start-from crate::coord_port::word_address_value_face \
        --start-from crate::coord_port::word_address_value_elliptic_word \
        --start-from crate::coord_port::word_address_value_word \
        --start-from crate::coord_port::word_address_value_residue \
        --start-from crate::coord_port::word_address_value_profile \
        --start-from crate::coord_port::word_face_table_address_value \
        --start-from crate::coord_port::word_face_table_address_face \
        --start-from crate::coord_port::word_locator_sigma_dial \
        --start-from crate::coord_port::word_face_index_policy \
        --start-from crate::coord_port::word_word_face_parabolic_dual_u64 \
        --start-from crate::coord_port::word_word_face_hyperbolic_scalar_u64 \
        --start-from crate::coord_port::word_word_face_parabolic_dual_i64 \
        --start-from crate::coord_port::word_word_face_hyperbolic_scalar_i64 \
        --start-from crate::coord_port::word_word_face_elliptic_pack_u64 \
        --start-from crate::coord_port::word_word_face_elliptic_unpack_u64 \
        --start-from crate::coord_port::word_word_face_elliptic_pack_i64 \
        --start-from crate::coord_port::word_word_face_elliptic_unpack_i64 \
        --start-from crate::coord_port::word_sigma_face_domain_sigma \
        --start-from crate::coord_port::word_sigma_face_domain_from_sigma \
        --start-from crate::coord_port::word_sigma_face_domain_encode \
        --start-from crate::coord_port::word_locator_sigma_face_domain_sigma \
        --start-from crate::coord_port::word_locator_sigma_face_domain_from_sigma \
        --start-from crate::coord_port::word_locator_sigma_face_domain_encode \
        --start-from crate::coord_port::word_word_face_coord_u64 \
        --start-from crate::coord_port::word_locator_new \
        --start-from crate::coord_port::word_locator_with_max_reps \
        --start-from crate::coord_port::word_locator_profile \
        --start-from crate::coord_port::word_locator_len \
        --start-from crate::coord_port::word_locator_is_empty \
        --start-from crate::coord_port::word_locator_coverage_established \
        --start-from crate::coord_port::word_locator_residue_node_count \
        --start-from crate::coord_port::word_locator_parabolic_node_count \
        --start-from crate::coord_port::word_locator_identity_node_count \
        --start-from crate::coord_port::word_locator_edge_count \
        --start-from crate::coord_port::word_locator_max_identity_fanout \
        --start-from crate::coord_port::word_locator_invalidate_coverage \
        --start-from crate::coord_port::word_locator_insert \
        --start-from crate::coord_port::word_locator_get \
        --start-from crate::coord_port::word_locator_lookup_absent \
        --start-from crate::coord_port::word_face_index_face \
        --start-from crate::coord_port::word_residue_digits \
        --start-from crate::coord_port::word_coefficient_bits \
        --start-from crate::coord_port::word_covered_scope_index \
        --start-from crate::coord_port::word_covered_scope_face \
        --start-from crate::coord_port::word_covered_scope_residue \
        --start-from crate::coord_port::word_covered_scope_coefficient \
        --start-from crate::coord_port::word_covered_scope_residue_levels \
        --start-from crate::coord_port::word_covered_scope_identity_bytes \
        --start-from crate::coord_port::word_directory_new \
        --start-from crate::coord_port::word_directory_arena_len_residue \
        --start-from crate::coord_port::word_directory_arena_len_parabolic \
        --start-from crate::coord_port::word_directory_max_identity_fanout \
        --start-from crate::coord_port::word_directory_identity_nodes_get \
        --start-from crate::coord_port::word_directory_representative_terminal_residue \
        --start-from crate::coord_port::word_directory_reserve_or_linked_residue \
        --start-from crate::coord_port::word_edge_list_len \
        --start-from crate::coord_port::word_edge_list_windows \
        --start-from crate::coord_port::word_edge_windows_next \
        --start-from crate::coord_port::word_identity_nodes_get \
        --start-from crate::coord_port::word_identity_nodes_iter \
        --start-from crate::coord_port::word_identity_nodes_into_iter \
        --start-from crate::fit_port::word_fit_steps_fwht_words \
        --start-from crate::fit_port::word_fit_steps_had_gram \
        --start-from crate::fit_port::word_fit_steps_compose \
        --start-from crate::fit_port::word_fit_gptq_e8_carrier_feedback_u \
        --start-from crate::fit_port::word_fit_icm_codebook \
        --start-from crate::fit_port::word_fit_icm_nearest_index \
        --start-from crate::fit_port::word_fit_icm_ridged_metric \
        --start-from crate::fit_port::word_fit_icm_plane_w \
        --start-from crate::fit_port::word_fit_icm_j_rows \
        --start-from crate::fit_port::word_fit_icm_e8_icm_step \
        --start-from crate::fit_port::word_fit_icm_r_step2 \
        --start-from crate::fit_port::word_fit_icm_matmul \
        --start-from crate::crypto_port::word_crypto_ct_eq \
        --start-from crate::crypto_port::word_crypto_sha256 \
        --start-from crate::crypto_port::word_crypto_sha512 \
        --start-from crate::crypto_port::word_crypto_sha384 \
        --start-from crate::crypto_port::word_crypto_md5 \
        --start-from crate::crypto_port::word_crypto_blake2b \
        --start-from crate::crypto_port::word_crypto_blake2b512 \
        --start-from crate::crypto_port::word_crypto_ripemd160 \
        --start-from crate::crypto_port::word_crypto_hash160 \
        --start-from crate::crypto_port::word_crypto_hmac_sha256 \
        --start-from crate::crypto_port::word_crypto_hmac_sha512 \
        --start-from crate::crypto_port::word_crypto_hmac_sha384 \
        --start-from crate::crypto_port::word_crypto_pbkdf2_sha256 \
        --start-from crate::crypto_port::word_crypto_hkdf_extract_sha256 \
        --start-from crate::crypto_port::word_crypto_hkdf_expand_sha256 \
        --start-from crate::crypto_port::word_crypto_hkdf_sha256 \
        --start-from crate::crypto_port::word_crypto_hkdf_sha384 \
        --start-from crate::crypto_port::word_crypto_hkdf_sha512 \
        --start-from crate::crypto_port::word_crypto_chacha20_block \
        --start-from crate::crypto_port::word_crypto_ecdsa_p256_verify \
        --start-from crate::crypto_port::word_crypto_p256_pubkey \
        --start-from crate::crypto_port::word_crypto_secp256k1_pubkey \
        --start-from crate::crypto_port::word_crypto_ecdsa_p384_verify \
        --start-from crate::crypto_port::word_crypto_rsa_pkcs1_sha256_verify \
        --start-from crate::crypto_port::word_crypto_bech32_convert_bits \
        --start-from crate::crypto_port::word_crypto2_bn_from_u64 \
        --start-from crate::crypto_port::word_crypto2_bn_is_zero \
        --start-from crate::crypto_port::word_crypto2_bn_is_even \
        --start-from crate::crypto_port::word_crypto2_bn_add \
        --start-from crate::crypto_port::word_crypto2_bn_shr1 \
        --start-from crate::crypto_port::word_crypto2_bn_sub1 \
        --start-from crate::crypto_port::word_crypto2_bn_divmod \
        --start-from crate::crypto_port::word_crypto2_bn_modexp_big \
        --start-from crate::crypto_port::word_crypto2_mod_inverse \
        --start-from crate::crypto_port::word_crypto2_bitlen \
        --start-from crate::crypto_port::word_crypto2_mgf1 \
        --start-from crate::crypto_port::word_crypto2_emsa_pss_encode \
        --start-from crate::crypto_port::word_crypto2_rsa_pkcs1_sign_generic \
        --start-from crate::crypto_port::word_crypto2_rsa_pkcs1_sha384_verify \
        --start-from crate::crypto_port::word_crypto2_rsa_pkcs1_sha512_verify \
        --start-from crate::crypto_port::word_crypto2_rsa_pkcs1_sha256_sign \
        --start-from crate::crypto_port::word_crypto2_rsa_pkcs1_sha384_sign \
        --start-from crate::crypto_port::word_crypto2_rsa_pkcs1_sha512_sign \
        --start-from crate::crypto_port::word_crypto2_rsa_pss_sha256_sign \
        --start-from crate::crypto_port::word_crypto2_rsa_pss_sha256_verify \
        --start-from crate::crypto_port::word_crypto2_rsa_algid \
        --start-from crate::crypto_port::word_crypto2_trim_int \
        --start-from crate::crypto_port::word_crypto2_rsa_to_pkcs8_der \
        --start-from crate::crypto_port::word_crypto2_rsa_from_pkcs8_der \
        --start-from crate::crypto_port::word_crypto2_der_len \
        --start-from crate::crypto_port::word_crypto2_der_tlv \
        --start-from crate::crypto_port::word_crypto2_der_seq \
        --start-from crate::crypto_port::word_crypto2_der_set \
        --start-from crate::crypto_port::word_crypto2_der_int \
        --start-from crate::crypto_port::word_crypto2_der_oid \
        --start-from crate::crypto_port::word_crypto2_der_bitstr \
        --start-from crate::crypto_port::word_crypto2_der_ctx_explicit \
        --start-from crate::crypto_port::word_crypto2_der_algid_ed25519 \
        --start-from crate::crypto_port::word_crypto2_der_name_cn \
        --start-from crate::crypto_port::word_crypto2_der_validity \
        --start-from crate::crypto_port::word_crypto2_der_spki \
        --start-from crate::crypto_port::word_crypto2_x509_der_tlv \
        --start-from crate::crypto_port::word_crypto2_x509_der_children \
        --start-from crate::crypto_port::word_crypto2_age_b64d \
        --start-from crate::crypto_port::word_crypto2_jwt_b64url_decode \
        --start-from crate::fold_finish_port::word_finish_train_linear_new \
        --start-from crate::fold_finish_port::word_finish_train_planes_new \
        --start-from crate::fold_finish_port::word_finish_train_planes_linear \
        --start-from crate::fold_finish_port::word_finish_train_planes_linear_mut \
        --start-from crate::fold_finish_port::word_finish_train_layer_tunables_new \
        --start-from crate::fold_finish_port::word_finish_train_layer_tunables_bias_scale \
        --start-from crate::fold_finish_port::word_finish_train_layer_tunables_tuned \
        --start-from crate::fold_finish_port::word_finish_train_tunables_new \
        --start-from crate::fold_finish_port::word_finish_train_tunables_output_norm \
        --start-from crate::fold_finish_port::word_finish_train_frozen_new \
        --start-from crate::fold_finish_port::word_finish_train_schedule_new \
        --start-from crate::fold_finish_port::word_finish_train_schedule_factor \
        --start-from crate::fold_finish_port::word_finish_train_adam_zeros \
        --start-from crate::fold_finish_port::word_finish_train_adam_len \
        --start-from crate::fold_finish_port::word_finish_train_adam_is_empty \
        --start-from crate::fold_finish_port::word_finish_train_adam_apply \
        --start-from crate::fold_finish_port::word_finish_train_kl_mean \
        --start-from crate::fold_finish_port::word_run_convert_decode_f16 \
        --start-from crate::fold_finish_port::word_run_convert_decode_bf16 \
        --start-from crate::fold_finish_port::word_run_convert_decode_f32 \
        --start-from crate::coord_port::word_identity_node_iter_next \
        --start-from crate::fold_layer_port::word_layer_driver_layer_words \
        --start-from crate::fold_layer_port::word_layer_driver_g_row_from_vec \
        --start-from crate::fold_layer_port::word_layer_driver_g_row_from_matrix \
        --start-from crate::fold_layer_port::word_layer_driver_model_init \
        --start-from crate::fold_layer_port::word_layer_tensor_recon \
        --start-from crate::fold_layer_port::word_layer_tensor_fold_row_scale \
        --start-from crate::fold_layer_port::word_layer_resq_rescale_hier \
        --start-from crate::fold_layer_port::word_layer_resq_rescale_hier_top \
        --start-from crate::hashtrinity_port::word_locate_ref_coord \
        --start-from crate::hashtrinity_port::word_locate_ref_same \
        --start-from crate::hashtrinity_port::word_absorb_ref \
        --start-from crate::hashtrinity_port::word_locate_str_coord \
        --start-from crate::hashtrinity_port::word_locate_str_same \
        --start-from crate::hashtrinity_port::word_absorb_str \
        --start-from crate::hashtrinity_port::word_locate_pair_coord \
        --start-from crate::hashtrinity_port::word_locate_pair_same \
        --start-from crate::hashtrinity_port::word_absorb_pair \
        --start-from crate::hashtrinity_port::word_locate_triple_coord \
        --start-from crate::hashtrinity_port::word_locate_triple_same \
        --start-from crate::hashtrinity_port::word_absorb_triple \
        --start-from crate::hashtrinity_port::word_face_contains_key \
        --start-from crate::hashtrinity_port::word_face_entry_is_occupied \
        --start-from crate::hashtrinity_port::word_face_reserve \
        --start-from crate::hashtrinity_port::word_tregex_match_start \
        --start-from crate::hashtrinity_port::word_tregex_match_end \
        --start-from crate::hashtrinity_port::word_tregex_match_range \
        --start-from crate::hashtrinity_port::word_tregex_is_word \
        --start-from crate::hashtrinity_port::word_tregex_match_segs \
        --start-from crate::table_port::word_table_parallel_phase_absorb \
        --start-from crate::table_port::word_table_parallel_phase_total \
        --start-from crate::table_port::word_table_parallel_reserve_handle_table \
        --start-from crate::table_port::word_table_parallel_reserve_all_shards \
        --start-from crate::table_port::word_table_parallel_reserve_placement_records \
        --start-from crate::table_port::word_table_parallel_schedule_bins_with_regions \
        --start-from crate::table_port::word_table_parallel_reserve_worker_outcomes \
        --start-from crate::table_port::word_table_parallel_distribute_regions \
        --start-from crate::word_silu_fixed \
        --start-from crate::word_rms_norm_fixed \
        --start-from crate::word_rope_rotate_fixed \
        --start-from crate::word_row_dot_fixed \
        --start-from crate::word_matvec_fixed \
        --start-from crate::word_lm_head_fixed \
        --start-from crate::word_kv_state_new \
        --start-from crate::word_kv_append \
        --start-from crate::word_attn_head \
        --start-from crate::word_argmax_fixed \
        --start-from crate::word_fwht_fixed \
        --start-from crate::word_stage_mul \
        --start-from crate::word_route_topk \
        --start-from crate::word_gate_weights \
        --start-from crate::word_check_fragment \
        --start-from crate::word_policy_decide \
        --start-from crate::word_keccakf \
        --start-from crate::word_sha3_256 \
        --start-from crate::word_sha3_512 \
        --start-from crate::word_sha256 \
        --start-from crate::word_sha256d \
        --start-from crate::word_chacha20poly1305_encrypt \
        --start-from crate::word_chacha20poly1305_decrypt \
        --start-from crate::word_aes128_encrypt_block \
        --start-from crate::word_aes128_decrypt_block \
        --start-from crate::word_hashtrinity_new \
        --start-from crate::word_hashtrinity_with_capacity \
        --start-from crate::word_hashtrinity_get \
        --start-from crate::word_hashtrinity_insert \
        --start-from crate::word_coordmap_coordinate_with_capacity \
        --start-from crate::word_coordmap_get \
        --start-from crate::word_coordmap_insert \
        --start-from crate::word_count_home \
        --start-from crate::word_count_admit \
        --start-from crate::word_count_histogram \
        --start-from crate::word_count_offsets \
        --start-from crate::word_count_slot \
        --start-from crate::word_count_aggregate \
        --start-from crate::word_gram_fold_i8 \
        --start-from crate::word_sd_binary_block \
        --start-from crate::word_onebit_total_bits \
        --start-from crate::word_onebit_bits_sign \
        --start-from crate::word_onebit_bits_r \
        --start-from crate::word_onebit_bits_c \
        --start-from crate::word_onebit_bits_d \
        --start-from crate::word_onebit_bits_corr \
        --start-from crate::word_onebit_params \
        --start-from crate::word_onebit_index_bits \
        --start-from crate::learn_port::word_dense_linear_fwd \
        --start-from crate::learn_port::word_dense_linear_bwd_input \
        --start-from crate::learn_port::word_dense_linear_bwd_weight \
        --start-from crate::learn_port::word_dense_rmsnorm_fwd \
        --start-from crate::learn_port::word_dense_rmsnorm_bwd \
        --start-from crate::learn_port::word_dense_swiglu_fwd \
        --start-from crate::learn_port::word_dense_swiglu_bwd \
        --start-from crate::learn_port::word_attn_rope_fwd \
        --start-from crate::learn_port::word_attn_rope_bwd \
        --start-from crate::learn_port::word_attn_attn_fwd \
        --start-from crate::learn_port::word_attn_attn_bwd \
        --start-from crate::learn_port::word_head_log_softmax_row \
        --start-from crate::learn_port::word_head_kl_row \
        --start-from crate::learn_port::word_head_tv_row \
        --start-from crate::learn_port::word_head_head_chunked \
        --start-from crate::learn_port::word_e8_codeword \
        --start-from crate::learn_port::word_e8_nearest \
        --start-from crate::learn_port::word_e8_nearest_scaled \
        --start-from crate::solve_port::word_solve_gemm_nn \
        --start-from crate::solve_port::word_solve_gemm_nt \
        --start-from crate::solve_port::word_solve_wsyrk \
        --start-from crate::solve_port::word_solve_hadamard_into \
        --start-from crate::solve_port::word_solve_chol_solve \
        --start-from crate::solve_port::word_solve_inv_factor_upper \
        --start-from crate::solve_port::word_solve_top_eigh \
        --start-from crate::solve_port::word_solve_q32_ilog2_pos \
        --start-from crate::solve_port::word_solve_q32_rshift_rne \
        --start-from crate::solve_port::word_solve_q32_round_even \
        --start-from crate::solve_port::word_solve_q32_fx_from_i128 \
        --start-from crate::solve_port::word_solve_q32_bfp_from_i128 \
        --start-from crate::solve_port::word_solve_q32_spd_from_q32 \
        --start-from crate::solve_port::word_solve_q32_round_f16 \
        --start-from crate::grad_solve_port::word_learn_gguf_write_header \
        --start-from crate::grad_solve_port::word_learn_gguf_write_file \
        --start-from crate::grad_solve_port::word_learn_gram_accumulator_clear \
        --start-from crate::grad_solve_port::word_learn_gram_accumulate \
        --start-from crate::grad_solve_port::word_onebit_qwen25_05b_nb \
        --start-from crate::grad_solve_port::word_solve_q32_div_rne_pos \
        --start-from crate::grad_solve_port::word_solve_q32_spd_to_q32 \
        --start-from crate::grad_solve_port::word_model_config_q_out \
        --start-from crate::grad_solve_port::word_model_config_kv_out \
        --start-from crate::grad_solve_port::word_model_config_checked \
        --start-from crate::grad_solve_port::word_model_expect \
        --start-from crate::grad_solve_port::word_model_add_flat \
        --start-from crate::grad_solve_port::word_model_col_sum \
        --start-from crate::grad_solve_port::word_model_rmsnorm_rows \
        --start-from crate::grad_solve_port::word_model_rmsnorm_bwd_rows \
        --start-from crate::grad_solve_port::word_model_rope_rows_fwd \
        --start-from crate::grad_solve_port::word_model_rope_rows_bwd \
        --start-from crate::grad_solve_port::word_model_add_bias \
        --start-from crate::grad_solve_port::word_model_embedding_grad \
        --start-from crate::hashtrinity_port::word_seed_new \
        --start-from crate::hashtrinity_port::word_seed_fold \
        --start-from crate::hashtrinity_port::word_seed_unfold \
        --start-from crate::hashtrinity_port::word_seed_home \
        --start-from crate::hashtrinity_port::word_seed_residual \
        --start-from crate::hashtrinity_port::word_seed_regrow_coord \
        --start-from crate::hashtrinity_port::word_seed_regrow \
        --start-from crate::hashtrinity_port::word_seed_seed_of \
        --start-from crate::hashtrinity_port::word_coord_u128 \
        --start-from crate::hashtrinity_port::word_coord_i128 \
        --start-from crate::hashtrinity_port::word_coord_bool \
        --start-from crate::hashtrinity_port::word_coord_bytes \
        --start-from crate::hashtrinity_port::word_table_len \
        --start-from crate::hashtrinity_port::word_table_is_empty \
        --start-from crate::hashtrinity_port::word_table_capacity \
        --start-from crate::hashtrinity_port::word_table_tombstones \
        --start-from crate::hashtrinity_port::word_table_coord_seed \
        --start-from crate::hashtrinity_port::word_table_get_exact \
        --start-from crate::hashtrinity_port::word_freq_get \
        --start-from crate::hashtrinity_port::word_freq_len \
        --start-from crate::hashtrinity_port::word_freq_is_empty \
        --start-from crate::hashtrinity_port::word_freq_capacity \
        --start-from crate::hashtrinity_port::word_freq_overflow_len \
        --start-from crate::hashtrinity_port::word_freq_overflow_height \
        --start-from crate::hashtrinity_port::word_alg_species \
        --start-from crate::hashtrinity_port::word_alg_keep \
        --start-from crate::hashtrinity_port::word_alg_set \
        --start-from crate::hashtrinity_port::word_alg_merge_count \
        --start-from crate::hashtrinity_port::word_alg_merge_link \
        --start-from crate::hashtrinity_port::word_alg_put \
        --start-from crate::hashtrinity_port::word_alg_has \
        --start-from crate::hashtrinity_port::word_alg_table_count \
        --start-from crate::hashtrinity_port::word_alg_fold \
        --start-from crate::hashtrinity_port::word_freq_stream_with_capacity \
        --start-from crate::hashtrinity_port::word_freq_insert_with \
        --start-from crate::hashtrinity_port::word_freq_insert \
        --start-from crate::hashtrinity_port::word_freq_longest_probe \
        --start-from crate::hashtrinity_port::word_freq_primary_examinations \
        --start-from crate::fold_port::word_stats_budget_bytes_ceil \
        --start-from crate::fold_port::word_stats_budget_max_body_bpw \
        --start-from crate::fold_port::word_stats_lw_intensity \
        --start-from crate::fold_port::word_stats_lw_block_intensity \
        --start-from crate::fold_port::word_stats_parts_mul \
        --start-from crate::fold_port::word_stats_parts_sqrt \
        --start-from crate::fold_port::word_stats_parts_sin_cos \
        --start-from crate::fold_port::word_stats_parts_shift \
        --start-from crate::fold_port::word_stats_parts_panel \
        --start-from crate::fold_port::word_stats_parts_rms \
        --start-from crate::fold_port::word_stats_parts_rope_apply \
        --start-from crate::fold_port::word_stats_parts_silu \
        --start-from crate::fold_port::word_stats_parts_silu_mul \
        --start-from crate::fold_port::word_stats_parts_rope_head_dim \
        --start-from crate::fold_port::word_stats_parts_rope_t_max \
        --start-from crate::fold_port::word_fit_lowrank_corrector \
        --start-from crate::fold_port::word_fit_eigen_parts \
        --start-from crate::fold_port::word_fit_eigh \
        --start-from crate::fold_port::word_fit_eigen_gram \
        --start-from crate::fold_port::word_fit_lowrank_corrector_exact \
        --start-from crate::fold_port::word_fit_quant_r_hier \
        --start-from crate::fold_port::word_fit_quant_c_u8 \
        --start-from crate::fold_port::word_fit_f16r \
        --start-from crate::fold_port::word_fit_r_level \
        --start-from crate::fold_port::word_fit_r_hier_reconstruct \
        --start-from crate::fold_port::word_fit_gram_scale \
        --start-from crate::fold_port::word_stats_basis_damped_gram \
        --start-from crate::fold_port::word_stats_basis_identity_basis \
        --start-from crate::fold_port::word_stats_basis_hadamard_basis \
        --start-from crate::fold_port::word_stats_basis_row_energies \
        --start-from crate::fold_port::word_stats_basis_column_energies \
        --start-from crate::fold_port::word_stats_basis_sum_log_column_energies \
        --start-from crate::fold_port::word_stats_basis_distortion_at \
        --start-from crate::fold_port::word_stats_basis_rate_at \
        --start-from crate::fold_port::word_stats_basis_uniform_distortion \
        --start-from crate::fold_port::word_stats_basis_solve_water_fill \
        --start-from crate::fold_port::word_stats_basis_density_side_bits \
        --start-from crate::fold_port::word_stats_basis_density_sigma \
        --start-from crate::fold_port::word_stats_basis_butterfly_side_bits \
        --start-from crate::fold_port::word_stats_hocapture_softmax_row \
        --start-from crate::fold_port::word_stats_hocapture_cross_entropy_sum \
        --start-from crate::fold_port::word_stats_hocapture_logits_grad_ce \
        --start-from crate::fold_port::word_stats_hocapture_sample_targets \
        --start-from crate::fold_port::word_stats_hocapture_kind_stats_zeros \
        --start-from crate::fold_port::word_stats_heldout_score_base_row \
        --start-from crate::word_engine_open \
        --start-from crate::word_engine_prefill_state \
        --start-from crate::word_engine_next_token_state \
        --start-from crate::word_engine_generate_speculative \
        -- \
        --manifest-path "$EXTRACT_DIR/Cargo.toml" \
        --locked \
        --offline \
        --lib \
        --target-dir "$EXTRACT_TARGET"
    # Every Aeneas warning is an error except the classes below. Aeneas's std table predates the
    # pinned rustc's `Iterator::{last,zip}` and `Step::{forward,backward}_overflowing`; a loop over a
    # range reaches those trait declarations although no extracted body calls the missing items.
    # An unknown external definition is not waived here: it surfaces as an `axiom` and `audit`
    # rejects it unless `boundary-axioms.txt` names it.
    ae_log=$(mktemp)
    if ! "$AENEAS" -backend lean -dest "$dest_dir" -no-progress-bar "$llbc" >"$ae_log" 2>&1; then
        cat "$ae_log" >&2
        rm -f "$ae_log"
        exit 1
    fi
    cat "$ae_log"
    unwaived=$(sed 's/\x1b\[[0-9;]*m//g' "$ae_log" | sed -n '/^\[Warn/p' |
        sed "/trait decl 'core::iter::traits::iterator::Iterator', could not find the information for item '\(last\|zip\)'/d" |
        sed "/trait decl 'core::iter::range::Step', could not find the information for item '\(forward\|backward\)_overflowing'/d" |
        sed '/The crate contains extracted external, unknown definitions/d')
    rm -f "$ae_log"
    if [ -n "$unwaived" ]; then
        echo "aeneas: warning outside the waived classes (warnings are errors):" >&2
        echo "$unwaived" >&2
        exit 1
    fi
    # Aeneas may leave blanks at line ends; normalize both regeneration and freshness output.
    sed -i 's/[[:blank:]]*$//' "$dest_dir/$GENERATED"
    # The patched model defines the externals in `strip-modeled-axioms.py`; drop their stubs so the
    # generated bodies resolve to the model. Any other axiom stays for `audit` to reject.
    python3 "$PROOF_DIR/strip-modeled-axioms.py" "$dest_dir/$GENERATED"
}

# Fail-closed audit. `sorry`/`admit` are never admissible and any `axiom` outside
# `boundary-axioms.txt` is an unmodeled external; the four named boundary axioms of the f32
# carrier are the only entries accepted. An entry with no declaration is a failure too, so the
# allowlist is exact in both directions.
audit() {
    python3 "$PROOF_DIR/audit-extraction.py" "$1" "$PROOF_DIR/boundary-axioms.txt"
}

if [ "${1:-}" = "--check" ]; then
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    mkdir -p "$tmp/out"
    run_extract "$tmp/out" "$tmp/$CRATE.llbc"
    audit "$tmp/out/$GENERATED"
    if ! diff -u "$PROOF_DIR/$GENERATED" "$tmp/out/$GENERATED"; then
        echo "stale: $GENERATED does not match regeneration from current source" >&2
        exit 1
    fi
    echo "extraction is fresh"
else
    run_extract "$PROOF_DIR" "$LLBC"
    rm -f "$LLBC"
    audit "$PROOF_DIR/$GENERATED"
fi

cd "$PROOF_DIR"
if [ "${APEIRON_LAKE:-0}" = "1" ]; then timeout 200s lake build; else echo "lake build: SKIPPED (APEIRON_LAKE!=1)"; fi
