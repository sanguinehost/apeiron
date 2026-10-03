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

# Charon extracts with its own pinned rustc and needs that rustc's `std` built with full MIR; it
# builds that sysroot with `cargo miri setup` and caches it. The default cache location is one
# fixed path shared by every toolchain, so a `miri setup` run under a different nightly leaves `std`
# compiled by another rustc and the extraction dies with E0514 (`found crate 'std' compiled by an
# incompatible version of rustc`) before any Lean is produced. Name the sysroot after the toolchain
# Charon reports and keep it under `target/`, beside the pinned tools and outside the shared
# `$XDG_CACHE_HOME/miri` every other toolchain on the box writes to. Charon builds it on first run;
# the path is not created here, because an empty directory is not a built sysroot.
#
# The pinned Charon reads `CHARON_MIRI_SYSROOTS` (the plural name,
# `charon/src/bin/charon-driver/driver.rs:171`) and takes that branch only when the path carries
# `lib/rustlib/<target>`. Exporting the singular `CHARON_MIRI_SYSROOT` alone therefore left the run
# reading `$CHARON_CACHE_DIR/full-mir-sysroot-cache/<toolchain>-<target>`, a cache outside this
# repository, and falling through to a cold `cargo miri setup` when that cache is empty. All three
# names are exported: the plural one is what the pinned driver reads, and the singular one and
# `MIRI_SYSROOT` are read by `cargo miri setup` itself on that cold path.
CHARON_TOOLCHAIN=$("$CHARON" toolchain-version)
SYSROOT="$REPO_DIR/target/aeneas-sysroot/$CHARON_TOOLCHAIN"
CHARON_MIRI_SYSROOTS="$SYSROOT"
CHARON_MIRI_SYSROOT="$SYSROOT"
MIRI_SYSROOT="$SYSROOT"
export CHARON_MIRI_SYSROOTS CHARON_MIRI_SYSROOT MIRI_SYSROOT

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

echo "== source =="
# Every listed source must exist: a missing live source fails the digest stage instead of being
# skipped, so a stale path (for example a deleted module) cannot pass silently.
# `src/table.rs` textually includes `table/{state,locator,prepared}.rs`; `src/coord.rs`
# pulls `src/coord/address.rs` and `src/coord/locator.rs` by `#[path]`, and `src/coord/locator.rs`
# pulls `src/coord/locator_core.rs`. All are extraction source and carry a digest. Every `#[path]`
# target of `proof/extract/lib.rs` is listed: the engine closure `loader/step/chart/kern/trace`, the
# `mixer.rs` and `hadamard.rs` roots, `moe.rs` and `numeric/fixed_ln.rs` are pulled too, so a change
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
    # The reproducible historical configuration. The `APEIRON-ANATIVE-AEN-PIPE` explicit
    # monomorphizing candidate is withdrawn: it omits `--ops-to-function-calls` and
    # `--index-to-function-calls`, which lower every array/slice index and repeat in the tree, so
    # it rejects on `core::array::repeat` and `TArray`/`TSlice`. The preset translates the current
    # source; see `docs/evidence/anative.md`.
    "$CHARON" cargo \
        --preset aeneas \
        --opaque 'crate::attn::f32_core' \
        --opaque 'crate::mixer::f32_carrier' \
        --opaque 'crate::attn::_::attn_head_f32' \
        --opaque 'crate::attn::alloc_f32_storage' \
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
        --start-from crate::word_trace_macs \
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

# `--check` returns two independent verdicts, each printed under its own name: `cone:` for the
# fail-closed audit and `stale:` for the freshness diff. Neither short-circuits the other. Running
# the audit first and exiting on its status made every red `--check` unnameable, because an unlisted
# axiom ended the run before the diff and a stale file with an unlisted axiom, a fresh file with an
# unlisted axiom and a drifted source all printed the same audit rejection. Both verdicts are now
# computed and recorded, and the exit status is their disjunction: green requires both.
if [ "${1:-}" = "--check" ]; then
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    mkdir -p "$tmp/out"
    run_extract "$tmp/out" "$tmp/$CRATE.llbc"
    check_rc=0
    if audit "$tmp/out/$GENERATED"; then
        echo "cone: admitted; no sorry/admit, no unlisted axiom"
    else
        echo "cone: REJECTED; the audit lines above name the sorry/admit or unlisted axiom" >&2
        check_rc=1
    fi
    if diff -u "$PROOF_DIR/$GENERATED" "$tmp/out/$GENERATED"; then :; else
        echo "stale: $GENERATED does not match regeneration from current source" >&2
        check_rc=1
    fi
    [ "$check_rc" -eq 0 ] || exit 1
    echo "extraction is fresh"
else
    run_extract "$PROOF_DIR" "$LLBC"
    rm -f "$LLBC"
    if audit "$PROOF_DIR/$GENERATED"; then
        echo "cone: admitted; no sorry/admit, no unlisted axiom"
    else
        echo "cone: REJECTED; the audit lines above name the sorry/admit or unlisted axiom" >&2
        exit 1
    fi
fi

cd "$PROOF_DIR"
if [ "${APEIRON_LAKE:-0}" = "1" ]; then timeout 200s lake build; else echo "lake build: SKIPPED (APEIRON_LAKE!=1)"; fi
