#!/bin/sh
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
# `src/table.rs` textually includes `table/{state,locator,prepared}.rs`; `src/coord.rs`
# pulls `src/coord/address.rs` and `src/coord/locator.rs` by `#[path]`, and `src/coord/locator.rs`
# pulls `src/coord/locator_core.rs`. All are extraction source and carry a digest. Every `#[path]`
# target of `proof/extract/lib.rs` is listed: the engine closure `loader/step/chart/kern/trace`, the
# `mixer.rs` and `hadamard.rs` roots, `moe.rs`, `numeric/fixed_ln.rs` and `crypto.rs` are pulled too, so a change
# to any of them moves the generated Lean and moves this digest list with it. `src/table/tests.rs`
# is `#[cfg(test)]` and is not part of the Charon unit; it is deliberately excluded here.
SOURCE_FILES="
$REPO_DIR/src/scalar.rs
$REPO_DIR/src/cell.rs
$REPO_DIR/src/exchange.rs
$REPO_DIR/src/carrier.rs
$REPO_DIR/src/table.rs
$REPO_DIR/src/table/state.rs
$REPO_DIR/src/table/locator.rs
$REPO_DIR/src/table/prepared.rs
$REPO_DIR/src/table/probe.rs
$REPO_DIR/src/coord.rs
$REPO_DIR/src/coord/address.rs
$REPO_DIR/src/coord/locator.rs
$REPO_DIR/src/coord/locator_core.rs
$REPO_DIR/src/table/keyrep.rs
$REPO_DIR/src/table/partition.rs
$REPO_DIR/src/table/resources.rs
$REPO_DIR/src/table/parallel.rs
$REPO_DIR/src/numeric/fixed_exp.rs
$REPO_DIR/src/numeric/register.rs
$REPO_DIR/src/numeric/fixed_rsqrt.rs
$REPO_DIR/src/numeric/fixed_trig.rs
$REPO_DIR/src/format/gguf.rs
$REPO_DIR/src/format/quant.rs
$REPO_DIR/src/format/packing.rs
$REPO_DIR/src/engine/mod.rs
$REPO_DIR/src/engine/value.rs
$REPO_DIR/src/engine/silu.rs
$REPO_DIR/src/engine/norm.rs
$REPO_DIR/src/engine/rope.rs
$REPO_DIR/src/engine/error.rs
$REPO_DIR/src/engine/f32_convert.rs
$REPO_DIR/src/engine/dequant.rs
$REPO_DIR/src/engine/matvec.rs
$REPO_DIR/src/engine/attn.rs
$REPO_DIR/src/engine/mixer.rs
$REPO_DIR/src/engine/lm_head.rs
$REPO_DIR/src/engine/sampler.rs
$REPO_DIR/src/engine/hadamard.rs
$REPO_DIR/src/engine/ffn.rs
$REPO_DIR/src/engine/moe.rs
$REPO_DIR/src/engine/fused.rs
$REPO_DIR/src/engine/par.rs
$REPO_DIR/src/engine/kern/pow2.rs
$REPO_DIR/src/spirv/mod.rs
$REPO_DIR/src/spirv/check.rs
$REPO_DIR/src/policy/mod.rs
$REPO_DIR/src/policy/decide.rs
$REPO_DIR/src/engine/loader.rs
$REPO_DIR/src/engine/step.rs
$REPO_DIR/src/engine/chart/mod.rs
$REPO_DIR/src/engine/kern/mod.rs
$REPO_DIR/src/engine/trace.rs
$REPO_DIR/src/numeric/fixed_ln.rs
$REPO_DIR/src/crypto.rs
$REPO_DIR/src/sha256.rs
$REPO_DIR/src/chacha.rs
$REPO_DIR/src/aes.rs
$REPO_DIR/src/hashtrinity/mod.rs
$REPO_DIR/src/hashtrinity/algebra.rs
$REPO_DIR/src/hashtrinity/freq.rs
$REPO_DIR/src/hashtrinity/hex.rs
$REPO_DIR/src/hashtrinity/base64.rs
$REPO_DIR/src/hashtrinity/seed.rs
$REPO_DIR/src/hashtrinity/tregex.rs
$REPO_DIR/src/hashtrinity/par.rs
$REPO_DIR/src/hashtrinity/face.rs
$REPO_DIR/src/hashtrinity/crypto.rs
$REPO_DIR/src/learn/gram.rs
$REPO_DIR/src/learn/sigma_delta.rs
$REPO_DIR/src/learn/onebit_bits.rs
$REPO_DIR/src/learn/grad/dense.rs
$REPO_DIR/src/learn/grad/attn.rs
$REPO_DIR/src/learn/grad/head.rs
$REPO_DIR/src/learn/code/e8.rs
$EXTRACT_DIR/learn_port.rs
$EXTRACT_DIR/Cargo.toml
$EXTRACT_DIR/Cargo.lock
$EXTRACT_DIR/lib.rs
"
for f in $SOURCE_FILES; do
    [ -f "$f" ] || { echo "missing extraction source: $f" >&2; exit 2; }
done
# shellcheck disable=SC2086
sha256sum $SOURCE_FILES

run_extract() {
    dest_dir=$1
    llbc=$2
    # The reproducible historical configuration. The explicit monomorphizing candidate is
    # withdrawn: it omits `--ops-to-function-calls` and `--index-to-function-calls`, which lower
    # every array/slice index and repeat in the tree, so it rejects on `core::array::repeat` and
    # `TArray`/`TSlice`. The preset translates the current source.
    "$CHARON" cargo \
        --preset aeneas \
        --opaque 'crate::attn::f32_core' \
        --opaque 'crate::mixer::f32_carrier' \
        --opaque 'crate::attn::_::attn_head_f32' \
        --opaque 'crate::attn::alloc_f32_storage' \
        --opaque 'crate::crypto::shake128' \
        --opaque 'crate::crypto::shake256' \
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
        -- \
        --manifest-path "$EXTRACT_DIR/Cargo.toml" \
        --locked \
        --offline \
        --lib \
        --target-dir "$EXTRACT_TARGET"
    "$AENEAS" -backend lean -dest "$dest_dir" -warnings-as-errors -no-progress-bar "$llbc"
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
