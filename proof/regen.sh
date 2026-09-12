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
AENEAS="$AENEAS_HOME/aeneas"
LEAN_BACKEND="$AENEAS_HOME/backends/lean"

for f in "$CHARON" "$AENEAS"; do
    [ -x "$f" ] || { echo "missing tool: $f" >&2; exit 2; }
done
[ -d "$LEAN_BACKEND" ] || { echo "missing Aeneas Lean backend: $LEAN_BACKEND" >&2; exit 2; }

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

echo "== source =="
sha256sum "$REPO_DIR/src/scalar.rs" "$REPO_DIR/src/cell.rs" "$REPO_DIR/src/exchange.rs" \
    "$REPO_DIR/src/carrier.rs"

run_extract() {
    dest_dir=$1
    llbc=$2
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
        -- \
        --crate-type lib \
        --crate-name "$CRATE" \
        "$EXTRACT_DIR/lib.rs"
    "$AENEAS" -backend lean -dest "$dest_dir" -warnings-as-errors -no-progress-bar "$llbc"
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
