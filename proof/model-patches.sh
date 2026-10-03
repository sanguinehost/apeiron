#!/bin/sh
# Build the patched Aeneas Lean backend at target/aeneas-model/lean from the installed tree.
#
# The pinned Aeneas release does not model the externals the growable table reaches, and its `Vec`
# carries no capacity. `patches/aeneas-lean-model.patch` adds the model. This script materialises a
# backend tree with that patch applied, which `proof/lakefile.lean` requires.
#
# The same patch declares the bare `F32` name the pinned emitter prints for Rust `f32`
# (`ExtractBase.ml` `float_name`), as an opaque value with the float boundary left to the extracted
# fragment's `axiom`s, plus the `From<u64> for i128` cast the attention fragment reaches; without
# both the committed `ApeironWord.lean` cannot elaborate. They are in this one patch so that this
# script alone reproduces the whole model.
#
# The patch touches eight files under `Aeneas/`: `Std/Vec.lean`,
# `Std/Core/{Cmp,CoreOption,CoreResult}.lean`, `Scalar/{Casts,Ops}.lean`,
# `Scalar/WrappingOps/Shl.lean` and `String.lean`. Enumerate them from the patch itself with
# `rg -o '^\+\+\+ b/(\S+)' -r '$1' proof/patches/aeneas-lean-model.patch | sort -u`.
# It is a Lean-library change, so no Charon/Aeneas OCaml rebuild is involved.
#
#   proof/model-patches.sh
#
# The installed backend defaults to `$AENEAS_HOME/backends/lean`, else `target/aeneas/backends/lean`.

set -eu

REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SRC="${AENEAS_HOME:-$REPO_DIR/target/aeneas}/backends/lean"
DST="$REPO_DIR/target/aeneas-model/lean"
PATCH="$REPO_DIR/proof/patches/aeneas-lean-model.patch"

[ -d "$SRC" ] || { echo "model-patches: no Aeneas Lean backend at $SRC" >&2; exit 2; }
[ -f "$PATCH" ] || { echo "model-patches: missing $PATCH" >&2; exit 2; }
[ -f "$SRC/lakefile.lean" ] || { echo "model-patches: $SRC lacks lakefile.lean (incomplete backend install); point AENEAS_HOME at a complete pinned backend" >&2; exit 2; }

rm -rf "$DST"
mkdir -p "$DST"
tar -C "$SRC" --exclude=.lake -cf - . | tar -C "$DST" -xf -
mkdir -p "$DST/.lake"
ln -sfn "$REPO_DIR/proof/.lake/packages" "$DST/.lake/packages"
(cd "$DST" && patch -p1 --forward < "$PATCH")
echo "model-patches: patched backend at $DST"
