#!/bin/sh
# Reproducibly patch the pinned Charon and Aeneas sources.
#
#   proof/install-patches.sh --check --aeneas-src DIR [--charon-src DIR]
#       Verify each base revision, each patch file's own sha256, and `git apply --check` both patches
#       against the base tree archived from the pinned commit. Writes nothing to the source checkout.
#       Exit 0 only when both patches match their pins and apply to the recorded base.
#
#   proof/install-patches.sh --aeneas-src DIR [--charon-src DIR]
#       Apply both patches to the source checkout. Idempotent: a patch already applied is skipped.
#
# The Charon source defaults to `$AENEAS_SRC/charon`. Aeneas source resolution, in order:
# `--aeneas-src`, `APEIRON_AENEAS_SRC`, `AENEAS_SRC`, `AENEAS_HOME` if it is a git checkout, then
# `target/aeneas` if it is a git checkout. `target/aeneas` normally holds the installed binaries,
# not sources, so a source argument is usually required.
#
# The base pins, the patch-file digests, the expected translator identity and the per-file base
# digests are in `patches/PINS`; the patches are in `patches/`.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PATCH_DIR="$SCRIPT_DIR/patches"
PINS="$PATCH_DIR/PINS"
CHARON_PATCH="$PATCH_DIR/charon-repeat-optional-trait.patch"
AENEAS_PATCH="$PATCH_DIR/aeneas-array-repeat.patch"

die() { echo "install-patches: $*" >&2; exit 1; }
usage() { sed -n '2,16p' "$0"; }

CHECK=0
AENEAS_SRC_ARG=
CHARON_SRC_ARG=
while [ $# -gt 0 ]; do
    case "$1" in
        --check) CHECK=1 ;;
        --aeneas-src) shift; [ $# -gt 0 ] || die "--aeneas-src needs a directory"; AENEAS_SRC_ARG=$1 ;;
        --charon-src) shift; [ $# -gt 0 ] || die "--charon-src needs a directory"; CHARON_SRC_ARG=$1 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
    shift
done

[ -f "$PINS" ] || die "missing pins: $PINS"
[ -f "$CHARON_PATCH" ] || die "missing patch: $CHARON_PATCH"
[ -f "$AENEAS_PATCH" ] || die "missing patch: $AENEAS_PATCH"
command -v git >/dev/null 2>&1 || die "git not found"
command -v tar >/dev/null 2>&1 || die "tar not found"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum not found"

# shellcheck source=patches/PINS
. "$PINS"

AENEAS_SRC=${AENEAS_SRC_ARG:-${APEIRON_AENEAS_SRC:-${AENEAS_SRC:-}}}
if [ -z "${AENEAS_SRC:-}" ]; then
    for cand in "${AENEAS_HOME:-}" "$REPO_DIR/target/aeneas"; do
        [ -n "$cand" ] || continue
        if [ -e "$cand/.git" ]; then AENEAS_SRC=$cand; break; fi
    done
fi
[ -n "${AENEAS_SRC:-}" ] || die "no Aeneas source checkout; pass --aeneas-src DIR or set AENEAS_HOME to a checkout"
[ -e "$AENEAS_SRC/.git" ] || die "not a git checkout: $AENEAS_SRC"

CHARON_SRC=${CHARON_SRC_ARG:-${APEIRON_CHARON_SRC:-${CHARON_SRC:-}}}
if [ -z "${CHARON_SRC:-}" ]; then
    CHARON_SRC="$AENEAS_SRC/charon"
fi
[ -e "$CHARON_SRC/.git" ] || die "not a git checkout: $CHARON_SRC (pass --charon-src DIR)"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

verify_revision() {
    label=$1 src=$2 want_commit=$3 want_tree=$4
    got_commit=$(git -C "$src" rev-parse --verify HEAD) || die "$label: cannot read HEAD in $src"
    got_tree=$(git -C "$src" rev-parse --verify 'HEAD^{tree}') || die "$label: cannot read HEAD tree in $src"
    [ "$got_commit" = "$want_commit" ] || die "$label: base commit mismatch: got $got_commit want $want_commit"
    [ "$got_tree" = "$want_tree" ] || die "$label: base tree mismatch: got $got_tree want $want_tree"
}

verify_base_digests() {
    label=$1 src=$2 list=$3
    printf '%s\n' "$list" > "$TMP/$label.digests"
    while read -r sha path; do
        [ -n "${path:-}" ] || continue
        got=$(git -C "$src" show "HEAD:$path" | sha256sum | cut -d' ' -f1)
        [ "$got" = "$sha" ] || die "$label: base file digest mismatch for $path: got $got want $sha"
    done < "$TMP/$label.digests"
}

verify_patch_digests() {
    label=$1 dir=$2 list=$3
    printf '%s\n' "$list" > "$TMP/$label.patchdigests"
    while read -r sha name; do
        [ -n "${name:-}" ] || continue
        [ -f "$dir/$name" ] || die "$label: pinned patch file missing: $dir/$name"
        got=$(sha256sum "$dir/$name" | cut -d' ' -f1)
        [ "$got" = "$sha" ] || die "$label: patch file digest mismatch for $name: got $got want $sha"
    done < "$TMP/$label.patchdigests"
}

archive_base() {
    label=$1 src=$2
    mkdir -p "$TMP/$label.base"
    git -C "$src" archive --format=tar -o "$TMP/$label.tar" HEAD || die "$label: git archive failed"
    tar -xf "$TMP/$label.tar" -C "$TMP/$label.base" || die "$label: tar extract failed"
}

check_patch() {
    label=$1 patch=$2
    ( cd "$TMP/$label.base" && git apply --check -p1 "$patch" ) \
        || die "$label: patch does not apply to pinned base: $patch"
}

apply_patch() {
    label=$1 src=$2 patch=$3
    if git -C "$src" apply --reverse --check -p1 "$patch" 2>/dev/null; then
        echo "$label: already applied"
    elif git -C "$src" apply --check -p1 "$patch"; then
        git -C "$src" apply -p1 "$patch"
        echo "$label: applied"
    else
        die "$label: patch does not apply cleanly to $src"
    fi
}

verify_revision charon "$CHARON_SRC" "$CHARON_COMMIT" "$CHARON_TREE"
verify_revision aeneas "$AENEAS_SRC" "$AENEAS_COMMIT" "$AENEAS_TREE"
verify_patch_digests charon "$PATCH_DIR" "$CHARON_PATCH_SHA256"
verify_patch_digests aeneas "$PATCH_DIR" "$AENEAS_PATCH_SHA256"
verify_base_digests charon "$CHARON_SRC" "$CHARON_BASE_SHA256"
verify_base_digests aeneas "$AENEAS_SRC" "$AENEAS_BASE_SHA256"

if [ "$CHECK" = 1 ]; then
    archive_base charon "$CHARON_SRC"
    archive_base aeneas "$AENEAS_SRC"
    check_patch charon "$CHARON_PATCH"
    check_patch aeneas "$AENEAS_PATCH"
    echo "charon: $CHARON_PATCH applies to $CHARON_COMMIT"
    echo "aeneas: $AENEAS_PATCH applies to $AENEAS_COMMIT"
    echo "patches apply to pinned base"
else
    apply_patch charon "$CHARON_SRC" "$CHARON_PATCH"
    apply_patch aeneas "$AENEAS_SRC" "$AENEAS_PATCH"
    echo "pinned tool sources patched"
fi
