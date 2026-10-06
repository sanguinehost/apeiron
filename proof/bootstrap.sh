#!/bin/sh
# proof/bootstrap.sh — materialise the pinned Aeneas/Charon toolchain into `target/aeneas`.
#
#   proof/bootstrap.sh [options]            install, build and verify the pinned toolchain
#   proof/bootstrap.sh --check              verify an existing --aeneas-home; build nothing
#
# From a fresh clone this is the route to a runnable `proof/regen.sh --check`:
#
#   1. it fetches the pinned public Aeneas release archive and verifies its sha256;
#   2. it materialises the pinned Aeneas and Charon source revisions from the public upstream
#      repositories (local object databases are used when present), verifies each revision's commit
#      and git tree against `proof/patches/PINS`, and applies the two tool patches;
#   3. it builds `charon`/`charon-driver` (Rust) and `aeneas` (OCaml `dune`) when the pinned
#      products are not already present;
#   4. it strips the symbol/debug metadata and the `.note.gnu.build-id` note from `aeneas`,
#      `charon` and `charon-driver`, installs them into `--aeneas-home` (default `target/aeneas`)
#      with `backends/` and `rust-toolchain`, and refuses unless each stripped digest equals its
#      `PINS` value (`AENEAS_TRANSLATOR_SHA256`, `CHARON_SHA256`, `CHARON_DRIVER_SHA256`);
#   5. it runs `proof/model-patches.sh` so `proof/lakefile.lean`'s `require` resolves.
#
# Path-independent identity. `src/dune` keeps `-g` in every OCaml profile, so the raw `dune build`
# records the compilation directory in debug info, and the linker derives `.note.gnu.build-id` from
# the linked content; the same source therefore gives a different digest at each build root. Before
# installing, `strip -R .note.gnu.build-id` removes the symbol table, the debug sections and that
# note, so the installed digest depends on source and toolchain only and reproduces at any build
# root (measured: three roots under /home/socol/oc-tmp strip to one digest). The pinned
# AENEAS_TRANSLATOR_SHA256 is the digest of that stripped product; the Charon binaries are stripped
# and pinned the same way. The build root is a scratch choice, not an identity.
#
# Pinned artifacts:
#   Aeneas release  https://github.com/AeneasVerif/aeneas/releases/tag/nightly-2026.09.09-505b6ca
#                   aeneas-linux-x86_64.tar.gz  (sha256 pinned below)
#   Aeneas source   https://github.com/AeneasVerif/aeneas.git       @ PINS:AENEAS_COMMIT
#   Charon source   https://github.com/AeneasVerif/charon.git       @ PINS:CHARON_COMMIT
#
# Options:
#   --aeneas-home DIR      install destination (default $REPO/target/aeneas)
#   --build-root DIR       pinned source checkouts and build tree (default
#                          $APEIRON_AENEAS_BUILD_ROOT, else /home/socol/oc-tmp/aeneas-build;
#                          the installed digest is path-independent, see above)
#   --archive-cache DIR    downloaded-archive cache (default $REPO/target/tool-archives)
#   --aeneas-src-repo DIR  local Aeneas git object database (default $APEIRON_AENEAS_SRC_REPO,
#                          else /home/socol/Workspace/aeneas)
#   --charon-src-repo DIR  local Charon git object database (default $APEIRON_CHARON_SRC_REPO,
#                          else $AENEAS_SRC_REPO/charon)
#   --opamroot DIR         opam root holding the pinned OCaml closure (default
#                          $APEIRON_OPAMROOT, else /home/socol/oc-tmp/aeneas-build2/opamroot)
#   --opam-switch NAME     opam switch inside --opamroot (default default)
#   --jobs N               aeneas parallel jobs (default 16)
#   --step-timeout SEC     per-build timeout (default 570)
#   --check                verify an existing --aeneas-home against the pins; build nothing
#   -h | --help            this text
#
# Exit 0 only when the installed translator is the pinned one. A pin mismatch, a missing tool or a
# build step that reaches its timeout is a failure with a named cause.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PINS="$SCRIPT_DIR/patches/PINS"
INSTALL_PATCHES="$SCRIPT_DIR/install-patches.sh"
MODEL_PATCHES="$SCRIPT_DIR/model-patches.sh"
CHARON_PATCH="$SCRIPT_DIR/patches/charon-repeat-optional-trait.patch"
AENEAS_PATCH="$SCRIPT_DIR/patches/aeneas-array-repeat.patch"

# Pinned public artifacts. The archive digest is the sha256 GitHub publishes for the asset; the
# source revisions and patch digests come from proof/patches/PINS.
AENEAS_ARCHIVE_URL="https://github.com/AeneasVerif/aeneas/releases/download/nightly-2026.09.09-505b6ca/aeneas-linux-x86_64.tar.gz"
AENEAS_ARCHIVE_SHA256="d5ba53b1370d30866bcd86de5e34abfbadf09e76b3689a7f70f458cd0c133591"
AENEAS_GIT_URL="https://github.com/AeneasVerif/aeneas.git"
CHARON_GIT_URL="https://github.com/AeneasVerif/charon.git"

AENEAS_HOME=${APEIRON_AENEAS_HOME:-"$REPO_DIR/target/aeneas"}
BUILD_ROOT=${APEIRON_AENEAS_BUILD_ROOT:-/home/socol/oc-tmp/aeneas-build}
ARCHIVE_CACHE=${APEIRON_TOOL_ARCHIVE_CACHE:-"$REPO_DIR/target/tool-archives"}
LOCAL_AENEAS_REPO=${APEIRON_AENEAS_SRC_REPO:-/home/socol/Workspace/aeneas}
LOCAL_CHARON_REPO=${APEIRON_CHARON_SRC_REPO:-}
OPAMROOT_DIR=${APEIRON_OPAMROOT:-/home/socol/oc-tmp/aeneas-build2/opamroot}
OPAM_SWITCH=default
JOBS=16
STEP_TIMEOUT=570
CHECK=0

die() { echo "bootstrap: $*" >&2; exit 1; }
log() { echo "bootstrap: $*"; }
sha256() { sha256sum "$1" | cut -d' ' -f1; }
# strip_identity SRC DEST — copy SRC to DEST with the symbol table, the debug sections and the
# `.note.gnu.build-id` note removed. OCaml records the compilation directory in debug info and the
# linker derives the build-id from the linked content, so an unstripped or build-id-bearing binary
# gives a different digest at every build root; removing both makes the digest path-independent.
strip_identity() {
    rm -f "$2"
    strip -R .note.gnu.build-id -o "$2" "$1" || die "strip -R .note.gnu.build-id failed for $1"
    chmod 755 "$2"
}
usage() { sed -n '2,55p' "$0"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --aeneas-home) shift; [ $# -gt 0 ] || die "--aeneas-home needs a directory"; AENEAS_HOME=$1 ;;
        --build-root) shift; [ $# -gt 0 ] || die "--build-root needs a directory"; BUILD_ROOT=$1 ;;
        --archive-cache) shift; [ $# -gt 0 ] || die "--archive-cache needs a directory"; ARCHIVE_CACHE=$1 ;;
        --aeneas-src-repo) shift; [ $# -gt 0 ] || die "--aeneas-src-repo needs a directory"; LOCAL_AENEAS_REPO=$1 ;;
        --charon-src-repo) shift; [ $# -gt 0 ] || die "--charon-src-repo needs a directory"; LOCAL_CHARON_REPO=$1 ;;
        --opamroot) shift; [ $# -gt 0 ] || die "--opamroot needs a directory"; OPAMROOT_DIR=$1 ;;
        --opam-switch) shift; [ $# -gt 0 ] || die "--opam-switch needs a name"; OPAM_SWITCH=$1 ;;
        --jobs) shift; [ $# -gt 0 ] || die "--jobs needs a count"; JOBS=$1 ;;
        --step-timeout) shift; [ $# -gt 0 ] || die "--step-timeout needs seconds"; STEP_TIMEOUT=$1 ;;
        --check) CHECK=1 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
    shift
done

[ -n "$LOCAL_CHARON_REPO" ] || LOCAL_CHARON_REPO="$LOCAL_AENEAS_REPO/charon"

[ -f "$PINS" ] || die "missing pins: $PINS"
[ -f "$INSTALL_PATCHES" ] || die "missing patch driver: $INSTALL_PATCHES"
[ -f "$MODEL_PATCHES" ] || die "missing model driver: $MODEL_PATCHES"
[ -f "$CHARON_PATCH" ] || die "missing patch: $CHARON_PATCH"
[ -f "$AENEAS_PATCH" ] || die "missing patch: $AENEAS_PATCH"

# shellcheck source=patches/PINS
. "$PINS"
[ -n "${AENEAS_TRANSLATOR_SHA256:-}" ] || die "$PINS has no AENEAS_TRANSLATOR_SHA256"
[ -n "${CHARON_SHA256:-}" ] || die "$PINS has no CHARON_SHA256"
[ -n "${CHARON_DRIVER_SHA256:-}" ] || die "$PINS has no CHARON_DRIVER_SHA256"
[ -n "${AENEAS_COMMIT:-}" ] || die "$PINS has no AENEAS_COMMIT"
[ -n "${CHARON_COMMIT:-}" ] || die "$PINS has no CHARON_COMMIT"
[ -n "${AENEAS_PATCH_SHA256:-}" ] || die "$PINS has no AENEAS_PATCH_SHA256"
[ -n "${CHARON_PATCH_SHA256:-}" ] || die "$PINS has no CHARON_PATCH_SHA256"

AENEAS_PATCH_DIGEST=${AENEAS_PATCH_SHA256%% *}
CHARON_PATCH_DIGEST=${CHARON_PATCH_SHA256%% *}

resolve_path() {
    readlink -m "$1" 2>/dev/null || readlink -f "$1" || die "cannot resolve $1"
}
# Resolve the destination's parent only, so a pre-existing `target/aeneas` symlink cannot redirect
# the install into another tree; the destination itself is a real directory this script owns.
resolve_dest() {
    d=$(dirname "$1"); b=$(basename "$1")
    printf '%s/%s\n' "$(resolve_path "$d")" "$b"
}
AENEAS_HOME=$(resolve_dest "$AENEAS_HOME")
BUILD_ROOT=$(resolve_path "$BUILD_ROOT")
ARCHIVE_CACHE=$(resolve_path "$ARCHIVE_CACHE")
[ "$AENEAS_HOME" != "/" ] || die "refusing --aeneas-home /"
[ "$BUILD_ROOT" != "/" ] || die "refusing --build-root /"

verify_home() {
    [ -f "$AENEAS_HOME/aeneas" ] || die "missing $AENEAS_HOME/aeneas"
    got=$(sha256 "$AENEAS_HOME/aeneas")
    [ "$got" = "$AENEAS_TRANSLATOR_SHA256" ] \
        || die "translator digest $got != pinned $AENEAS_TRANSLATOR_SHA256"
    [ -x "$AENEAS_HOME/charon" ] || die "missing executable $AENEAS_HOME/charon"
    got_c=$(sha256 "$AENEAS_HOME/charon")
    [ "$got_c" = "$CHARON_SHA256" ] \
        || die "charon digest $got_c != pinned $CHARON_SHA256"
    [ -x "$AENEAS_HOME/charon-driver" ] || die "missing executable $AENEAS_HOME/charon-driver"
    got_d=$(sha256 "$AENEAS_HOME/charon-driver")
    [ "$got_d" = "$CHARON_DRIVER_SHA256" ] \
        || die "charon-driver digest $got_d != pinned $CHARON_DRIVER_SHA256"
    [ -d "$AENEAS_HOME/backends/lean" ] || die "missing Aeneas Lean backend at $AENEAS_HOME/backends/lean"
    log "check: aeneas $got == pinned"
    log "check: charon $got_c == pinned"
    log "check: charon-driver $got_d == pinned"
}

if [ "$CHECK" = 1 ]; then
    verify_home
    log "OK (installed toolchain matches the pin)"
    exit 0
fi

command -v sha256sum >/dev/null 2>&1 || die "missing tool: sha256sum"
command -v git >/dev/null 2>&1 || die "missing tool: git"
command -v tar >/dev/null 2>&1 || die "missing tool: tar"
command -v patch >/dev/null 2>&1 || die "missing tool: patch"
command -v strip >/dev/null 2>&1 || die "missing tool: strip (binutils; needed for a path-independent binary identity)"
DL=""
for c in curl wget; do command -v "$c" >/dev/null 2>&1 && { DL=$c; break; }; done
[ -n "$DL" ] || die "missing downloader (curl or wget)"

mkdir -p "$REPO_DIR/target" "$ARCHIVE_CACHE" "$BUILD_ROOT"
ARCHIVE="$ARCHIVE_CACHE/aeneas-linux-x86_64.tar.gz"

# --- 1. pinned release archive ------------------------------------------------------------------
# The archive supplies `backends/`, byte-identical to the pinned source tree's bundle and the
# repository the model patch is taken against. A cached file is verified; a changed pin refuses
# rather than re-fetching.
if [ -f "$ARCHIVE" ]; then
    got=$(sha256 "$ARCHIVE")
    [ "$got" = "$AENEAS_ARCHIVE_SHA256" ] \
        || die "cached archive $ARCHIVE sha256=$got != pinned $AENEAS_ARCHIVE_SHA256 (remove it to refetch, or the pin is wrong)"
    log "archive: cached $ARCHIVE sha256=$got == pinned"
else
    log "archive: fetching $AENEAS_ARCHIVE_URL"
    case "$DL" in
        curl) timeout "$STEP_TIMEOUT" curl -fsSL -o "$ARCHIVE.part" "$AENEAS_ARCHIVE_URL" || die "download failed: $AENEAS_ARCHIVE_URL" ;;
        wget) timeout "$STEP_TIMEOUT" wget -q -O "$ARCHIVE.part" "$AENEAS_ARCHIVE_URL" || die "download failed: $AENEAS_ARCHIVE_URL" ;;
    esac
    got=$(sha256 "$ARCHIVE.part")
    [ "$got" = "$AENEAS_ARCHIVE_SHA256" ] \
        || { rm -f "$ARCHIVE.part"; die "archive sha256=$got != pinned $AENEAS_ARCHIVE_SHA256"; }
    mv "$ARCHIVE.part" "$ARCHIVE"
    log "archive: fetched sha256=$got == pinned"
fi

STAGE="$BUILD_ROOT/archive-stage"
rm -rf "$STAGE"
mkdir -p "$STAGE"
tar -xzf "$ARCHIVE" -C "$STAGE" || die "cannot extract $ARCHIVE"
[ -d "$STAGE/backends/lean" ] || die "archive $ARCHIVE has no backends/lean"

# --- 2. pinned sources, verified against PINS ---------------------------------------------------
# materialise DEST REPO_OR_URL COMMIT LABEL — checkout at COMMIT without mutating any remote.
# `read-tree`+`checkout-index` populate the tree; no `checkout`/`reset` runs.
materialise() {
    dest=$1 origin=$2 commit=$3 label=$4
    if [ -e "$dest/.git" ] && [ "$(git -C "$dest" rev-parse --verify HEAD 2>/dev/null || true)" = "$commit" ]; then
        log "$label: reusing $dest at $commit"
        return
    fi
    case "$dest" in
        "$BUILD_ROOT/aeneas"|"$BUILD_ROOT/charon") : ;;
        *) die "internal: refusing to replace $dest" ;;
    esac
    rm -rf "$dest"
    mkdir -p "$dest"
    if [ -d "$origin/.git" ] && git -C "$origin" cat-file -e "$commit^{commit}" 2>/dev/null; then
        git clone -q --no-checkout --shared "$origin" "$dest" || die "$label: clone from $origin failed"
    else
        case "$origin" in
            http://*|https://*|git@*) : ;;
            *) die "$label: $origin is neither a git object database nor a URL" ;;
        esac
        git init -q "$dest"
        git -C "$dest" fetch -q --depth 1 "$origin" "$commit" \
            || die "$label: fetch $commit from $origin failed"
    fi
    git -C "$dest" update-ref HEAD "$commit" || die "$label: cannot point HEAD at $commit"
    git -C "$dest" read-tree HEAD || die "$label: read-tree failed"
    git -C "$dest" checkout-index -a -f || die "$label: checkout-index failed"
    log "$label: materialised $dest @ $commit"
}

materialise "$BUILD_ROOT/aeneas" "$LOCAL_AENEAS_REPO" "$AENEAS_COMMIT" "aeneas"
materialise "$BUILD_ROOT/charon" "$LOCAL_CHARON_REPO" "$CHARON_COMMIT" "charon"

# src/dune declares `(vendored_dirs charon)`; the aeneas tree carries `src/charon -> ../charon`.
ln -sfn ../charon "$BUILD_ROOT/aeneas/charon"

log "verifying pinned sources against PINS"
"$INSTALL_PATCHES" --check \
    --aeneas-src "$BUILD_ROOT/aeneas" --charon-src "$BUILD_ROOT/charon"
"$INSTALL_PATCHES" \
    --aeneas-src "$BUILD_ROOT/aeneas" --charon-src "$BUILD_ROOT/charon"

# verify_applied SRC WANT LABEL — the whole applied diff must be byte-exactly the pinned patch.
verify_applied() {
    src=$1 want=$2 label=$3
    got=$(git -C "$src" diff | sha256sum | cut -d' ' -f1)
    [ "$got" = "$want" ] || die "$label: applied diff $got != pinned patch $want"
}
verify_applied "$BUILD_ROOT/aeneas" "$AENEAS_PATCH_DIGEST" "aeneas"
verify_applied "$BUILD_ROOT/charon" "$CHARON_PATCH_DIGEST" "charon"

# --- 3. build aeneas (OCaml) --------------------------------------------------------------------
AENEAS_PRODUCT="$BUILD_ROOT/aeneas/src/_build/default/main.exe"
AENEAS_IDENTITY="$BUILD_ROOT/identity/aeneas"
mkdir -p "$BUILD_ROOT/identity"
if [ -f "$AENEAS_PRODUCT" ]; then
    log "aeneas: reusing product $AENEAS_PRODUCT"
else
    command -v opam >/dev/null 2>&1 || die "missing opam (needed to build aeneas)"
    [ -d "$OPAMROOT_DIR" ] || die "missing opam root $OPAMROOT_DIR; set --opamroot to the pinned OCaml closure"
    OPAMROOT="$OPAMROOT_DIR"; export OPAMROOT
    OPAMSWITCH="$OPAM_SWITCH"; export OPAMSWITCH
    eval "$(opam env --root "$OPAMROOT_DIR" --switch="$OPAM_SWITCH" --set-switch --shell=sh)" \
        || die "opam env failed for $OPAMROOT_DIR/$OPAM_SWITCH"
    command -v dune >/dev/null 2>&1 || die "dune not found in opam switch $OPAM_SWITCH"
    log "aeneas: dune build -j $JOBS (timeout ${STEP_TIMEOUT}s) at $BUILD_ROOT/aeneas"
    ( cd "$BUILD_ROOT/aeneas/src" && timeout "$STEP_TIMEOUT" dune build -j "$JOBS" ) \
        || die "aeneas dune build failed or timed out after ${STEP_TIMEOUT}s"
fi
[ -f "$AENEAS_PRODUCT" ] || die "aeneas build produced no $AENEAS_PRODUCT"
strip_identity "$AENEAS_PRODUCT" "$AENEAS_IDENTITY"
built_aeneas=$(sha256 "$AENEAS_IDENTITY")
[ "$built_aeneas" = "$AENEAS_TRANSLATOR_SHA256" ] \
    || die "stripped aeneas $built_aeneas != pinned $AENEAS_TRANSLATOR_SHA256 (build root $BUILD_ROOT)"
log "aeneas: $built_aeneas == pinned (stripped identity)"

# --- 4. build charon / charon-driver (Rust) ------------------------------------------------------
# find_bin NAME — the release binary, first in the crate target dir, then the target directory cargo
# reports (the shared-build shim may redirect it).
find_bin() {
    name=$1
    for c in "$BUILD_ROOT/charon/charon/target/release/$name" "$BUILD_ROOT/charon/target/release/$name"; do
        [ -x "$c" ] && { printf '%s\n' "$c"; return 0; }
    done
    td=$(cd "$BUILD_ROOT/charon/charon" 2>/dev/null && cargo metadata --format-version 1 --no-deps 2>/dev/null \
        | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')
    if [ -n "${td:-}" ] && [ -x "$td/release/$name" ]; then
        printf '%s\n' "$td/release/$name"; return 0
    fi
    return 1
}

charon_ok=0
CHARON_BIN=$(find_bin charon || true)
CHARON_DRIVER=$(find_bin charon-driver || true)
if [ -n "${CHARON_BIN:-}" ] && [ -n "${CHARON_DRIVER:-}" ]; then
    charon_version=$(timeout 30 "$CHARON_BIN" version 2>/dev/null || true)
    case "$charon_version" in *"$CHARON_COMMIT"*) charon_ok=1 ;; esac
fi
if [ "$charon_ok" != 1 ]; then
    command -v cargo >/dev/null 2>&1 || die "missing cargo (needed to build charon)"
    log "charon: cargo build --release --locked (timeout ${STEP_TIMEOUT}s)"
    ( cd "$BUILD_ROOT/charon/charon" && timeout "$STEP_TIMEOUT" cargo build --release --locked ) \
        || die "charon cargo build failed or timed out after ${STEP_TIMEOUT}s"
    CHARON_BIN=$(find_bin charon || true)
    CHARON_DRIVER=$(find_bin charon-driver || true)
fi
[ -n "${CHARON_BIN:-}" ] || die "charon build produced no charon binary"
[ -n "${CHARON_DRIVER:-}" ] || die "charon build produced no charon-driver binary"
charon_version=$(timeout 30 "$CHARON_BIN" version 2>/dev/null || true)
case "$charon_version" in
    *"$CHARON_COMMIT"*) : ;;
    *) die "charon reports '$charon_version', not pinned commit $CHARON_COMMIT" ;;
esac
log "charon: $charon_version"
# The charon build is already path-independent (Rust records crate paths relative to the workspace
# root and the release profile carries no debug sections), but the same strip is applied and pinned
# so all three installed binaries carry one verified identity rule.
CHARON_IDENTITY="$BUILD_ROOT/identity/charon"
CHARON_DRIVER_IDENTITY="$BUILD_ROOT/identity/charon-driver"
strip_identity "$CHARON_BIN" "$CHARON_IDENTITY"
strip_identity "$CHARON_DRIVER" "$CHARON_DRIVER_IDENTITY"
stripped_charon=$(sha256 "$CHARON_IDENTITY")
[ "$stripped_charon" = "$CHARON_SHA256" ] \
    || die "stripped charon $stripped_charon != pinned $CHARON_SHA256 (build root $BUILD_ROOT)"
stripped_driver=$(sha256 "$CHARON_DRIVER_IDENTITY")
[ "$stripped_driver" = "$CHARON_DRIVER_SHA256" ] \
    || die "stripped charon-driver $stripped_driver != pinned $CHARON_DRIVER_SHA256 (build root $BUILD_ROOT)"
log "charon: $stripped_charon == pinned (stripped identity)"
log "charon-driver: $stripped_driver == pinned (stripped identity)"

# --- 5. install ----------------------------------------------------------------------------------
case "$AENEAS_HOME" in
    "$BUILD_ROOT"|"$BUILD_ROOT"/*) die "refusing --aeneas-home $AENEAS_HOME inside --build-root" ;;
esac
[ ! -e "$AENEAS_HOME/.git" ] || die "--aeneas-home $AENEAS_HOME is a git checkout, not an install destination; pass --aeneas-home of an empty or install directory"
# A symlink at the destination would make every `install` below write through it; own the path.
[ -L "$AENEAS_HOME" ] && rm -f "$AENEAS_HOME"
mkdir -p "$AENEAS_HOME"
install -m 755 "$AENEAS_IDENTITY" "$AENEAS_HOME/aeneas"
install -m 755 "$CHARON_IDENTITY" "$AENEAS_HOME/charon"
install -m 755 "$CHARON_DRIVER_IDENTITY" "$AENEAS_HOME/charon-driver"
rm -rf "$AENEAS_HOME/backends"
cp -a "$STAGE/backends" "$AENEAS_HOME/backends"
cp -f "$BUILD_ROOT/charon/charon/rust-toolchain" "$AENEAS_HOME/rust-toolchain"
rm -rf "$STAGE"

verify_home

# --- 6. patched Lean model -----------------------------------------------------------------------
AENEAS_HOME="$AENEAS_HOME" "$MODEL_PATCHES"

log "installed AENEAS_HOME=$AENEAS_HOME"
log "OK (aeneas == pinned $AENEAS_TRANSLATOR_SHA256)"
