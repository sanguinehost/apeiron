#!/bin/sh
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
if [ "$AENEAS_HOME" != "$LINK" ]; then
    if [ ! -e "$LINK" ]; then
        ln -s "$AENEAS_HOME" "$LINK"
    fi
fi

CHARON="$AENEAS_HOME/charon"
CHARON_DRIVER="$AENEAS_HOME/charon-driver"
AENEAS="$AENEAS_HOME/aeneas"
LEAN_BACKEND="$AENEAS_HOME/backends/lean"

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

echo "== tools =="
echo "charon:  $("$CHARON" version)"
echo "aeneas:  $("$AENEAS" -version)"
echo "rust:    $("$CHARON" toolchain-version)"
echo "lean:    $(cat "$PROOF_DIR/lean-toolchain")"
echo "sha256:"
tool_sha256 "$CHARON"
if [ -e "$CHARON_DRIVER" ]; then
    tool_sha256 "$CHARON_DRIVER"
fi
tool_sha256 "$AENEAS"

echo "== source =="
# Every listed source must exist: a missing live source fails the digest stage instead of being
# skipped, so a stale path (for example a deleted module) cannot pass silently.
# `src/table.rs` textually includes `table/{state,locator,prepared}.rs`; `src/coord.rs`
# pulls `src/coord/address.rs` and `src/coord/locator.rs` by `#[path]`, and `src/coord/locator.rs`
# pulls `src/coord/locator_core.rs`. All are extraction source and carry a digest. `src/table/tests.rs`
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
$REPO_DIR/src/table/scan.rs
$REPO_DIR/src/table/parallel.rs
$REPO_DIR/src/numeric/fixed_exp.rs
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
$REPO_DIR/src/engine/lm_head.rs
$REPO_DIR/src/engine/sampler.rs
$REPO_DIR/src/engine/ffn.rs
$REPO_DIR/src/engine/kern/pow2.rs
$REPO_DIR/src/spirv/mod.rs
$REPO_DIR/src/spirv/check.rs
$REPO_DIR/src/policy/mod.rs
$REPO_DIR/src/policy/decide.rs
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
    # The reproducible historical configuration. The explicit
    # monomorphizing candidate is withdrawn: it omits `--ops-to-function-calls` and
    # `--index-to-function-calls`, which lower every array/slice index and repeat in the tree, so
    # it rejects on `core::array::repeat` and `TArray`/`TSlice`. The preset translates the current
    # source; see the anative note.
    "$CHARON" rustc \
        --preset aeneas \
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
        --start-from crate::word_route_topk \
        --start-from crate::word_gate_weights \
        --start-from crate::word_check_fragment \
        --start-from crate::word_policy_decide \
        -- \
        --edition 2021 \
        --crate-type lib \
        --crate-name "$CRATE" \
        "$EXTRACT_DIR/lib.rs"
    "$AENEAS" -backend lean -dest "$dest_dir" -warnings-as-errors -no-progress-bar "$llbc"
    # Aeneas may leave blanks at line ends; normalize both regeneration and freshness output.
    sed -i 's/[[:blank:]]*$//' "$dest_dir/$GENERATED"
    # The patched model defines the externals in `strip-modeled-axioms.py`; drop their stubs so the
    # generated bodies resolve to the model. Any other axiom stays for `audit` to reject.
    python3 "$PROOF_DIR/strip-modeled-axioms.py" "$dest_dir/$GENERATED"
}

audit() {
    file=$1
    if grep -nE '(^|[[:space:]])(axiom|sorry|admit)([[:space:]]|$)' "$file"; then
        echo "unmodeled operation in accepted extraction: $file" >&2
        exit 1
    fi
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
lake build
