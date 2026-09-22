#!/usr/bin/env bash
# Rejecting control for the growable per-entry relocation refinement.
#
#   bash proof/controls/relocation.sh
#   RELOCATION_CONTROL_HEAVY=1 bash proof/controls/relocation.sh
#
# The control never writes to the working tree. It copies only the paths the control needs
# into /tmp/opencode/apeiron-reloc-control.$$ and removes the copy on exit.
#
# DEFAULT (freshness only):
#   STEP 1  in the copy's generated `proof/ApeironWord.lean`, change the flat relocation write
#           `live := true` to `live := false`. `proof/regen.sh --check` re-extracts from the
#           copy's real source and must exit nonzero, with a diff that names the mutated line.
#   STEP 3  restore the write. `proof/regen.sh --check` must be green (`extraction is fresh`).
#
# RELOCATION_CONTROL_HEAVY=1 (adds the semantic mutation, the frame mutation and the corpus build):
#   STEP 2  in the copy's generated `proof/ApeironWord.lean`, change the spill relocation write
#           `table.tag entry_address.place` to `0#u8`. `lake build Growable.Relocation` must exit
#           nonzero with the error inside `Growable/Relocation.lean`; restoring the write must
#           rebuild green.
#   STEP 2c in the copy's `proof/corpus/Growable/Relocation.lean`, change the flat relocation
#           destination write `reserved.resident.slots.set index slot` to `.set 0#usize slot`.
#           `writeFlat` has exactly one non-`rfl` consumer, `writeFlat_frame_flat`; the build must
#           exit nonzero with the error inside that theorem's span, and restoring the index must
#           rebuild green. This reproduces the manual index-zero mutation that validated W6-T1.
#   The heavy and frame-only paths also copy `proof/corpus/.lake/build` (~613 MiB) and therefore
#   need a larger free-space floor.
#
# RELOCATION_CONTROL_FRAME=1 (frame mutation without the heavy spill mutation):
#   runs the default freshness steps and STEP 2c only.
#
# HARDENING (APEIRON-EXTRACT-REP-RELOCATION-CONTROL-T2):
#   * preflight refuses to run on low free space or a detected IO hog, printing SKIPPED;
#   * every step is wrapped in `timeout`; the copy is bounded and never holds the lane lock;
#   * the lane lock is an inherited file descriptor, not a child `flock` process, so no killed
#     runner can orphan a lock holder;
#   * EXIT/INT/TERM trap kills the active process group and stops the transient systemd scope.
#
# Exit 0 only if every counted check behaves as required. Prints a per-step receipt.

set -uo pipefail

PROOF_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REPO_DIR=$(CDPATH= cd -- "$PROOF_DIR/.." && pwd)
GEN="$REPO_DIR/proof/ApeironWord.lean"
LOCK=/tmp/opencode/apeiron-lane.lock
RUN_BASE=/tmp/opencode
RUN="$RUN_BASE/apeiron-reloc-control.$$"
CAP=6G

# Per-step wall bounds, seconds. A timeout is inconclusive, never a red. Overridable for testing.
T_COPY=${RELOCATION_CONTROL_T_COPY:-120}
T_EXTRACT=${RELOCATION_CONTROL_T_EXTRACT:-300}
T_LEAN=${RELOCATION_CONTROL_T_LEAN:-300}
T_MUTATE=${RELOCATION_CONTROL_T_MUTATE:-30}
T_LOCK_WAIT=${RELOCATION_CONTROL_T_LOCK_WAIT:-120}

# Preflight floors. Default copy is source + `proof/.lake/build` (~10 MiB); the heavy and frame-only
# paths add `proof/corpus/.lake/build` (~613 MiB) and extraction scratch.
MIN_FREE_MB=2048
HEAVY=0
FRAME=0
if [ "${RELOCATION_CONTROL_HEAVY:-0}" = "1" ]; then
    HEAVY=1
    FRAME=1
fi
if [ "${RELOCATION_CONTROL_FRAME:-0}" = "1" ]; then
    FRAME=1
fi
if [ "$HEAVY" = 1 ] || [ "$FRAME" = 1 ]; then
    MIN_FREE_MB=4096
fi
NPROC=$(nproc)
IO_HOG_LOAD=$((2 * NPROC))
IO_HOG_DSTATE=$((NPROC / 2))
BTRFS_CLEANER_TICKS=20

PASS=0
FAIL=0
STEP_NO=0
LOCK_HELD=0
RUNNING_PID=""
CUR_UNIT=""

receipt() { printf 'relocation-control: %-26s %-7s %s\n' "$1" "$2" "$3"; }

cleanup() {
    if [ "${APEIRON_RELOC_CONTROL_KEEP:-0}" = "1" ]; then
        echo "relocation-control: kept copy at $RUN"
        return
    fi
    rm -rf -- "$RUN"
}

release_lock() {
    if [ "$LOCK_HELD" = "1" ]; then
        flock -u 9 2>/dev/null || true
        exec 9>&- 2>/dev/null || true
        LOCK_HELD=0
    fi
}

# Kill the active process group only when setsid gave it its own group, never the runner's group.
kill_tree() {
    [ -n "$RUNNING_PID" ] || return 0
    local pgid selfpgid
    pgid=$(ps -o pgid= -p "$RUNNING_PID" 2>/dev/null | tr -d ' ')
    selfpgid=$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ')
    if [ -n "$pgid" ] && [ "$pgid" != "$selfpgid" ]; then
        kill -TERM -- "-$pgid" 2>/dev/null || true
    else
        kill -TERM "$RUNNING_PID" 2>/dev/null || true
    fi
}

on_exit() {
    local rc=$?
    trap - EXIT INT TERM
    kill_tree
    if [ -n "$CUR_UNIT" ]; then
        systemctl --user stop "$CUR_UNIT" >/dev/null 2>&1 || true
    fi
    release_lock
    cleanup
    exit "$rc"
}
trap on_exit EXIT INT TERM

# run_bounded <seconds> <cwd> <command...>  -> child exit status (124 on timeout)
# Runs in its own session so the trap can kill the whole group.
run_bounded() {
    local t=$1 dir=$2
    shift 2
    local rc
    ( cd "$dir" && exec setsid timeout --signal=TERM --kill-after=10 "$t" "$@" ) &
    RUNNING_PID=$!
    wait "$RUNNING_PID"
    rc=$?
    RUNNING_PID=""
    return "$rc"
}

acquire_lock() {
    exec 9>"$LOCK" || return 1
    if flock -w "$T_LOCK_WAIT" 9; then
        LOCK_HELD=1
        return 0
    fi
    exec 9>&- 2>/dev/null || true
    return 1
}

# run_locked <seconds> <cwd> <command...>
# The AGENTS.md §6 lane: inherited lock fd, memory cap, single-threaded, timed. The lock fd is
# closed before exec so the transient scope cannot inherit it.
run_locked() {
    local t=$1 dir=$2
    shift 2
    local rc
    acquire_lock || return 75
    STEP_NO=$((STEP_NO + 1))
    CUR_UNIT="apeiron-reloc-$$-$STEP_NO"
    systemctl --user reset-failed "$CUR_UNIT" >/dev/null 2>&1 || true
    ( cd "$dir" && exec 9>&- && exec setsid systemd-run --user --scope \
        --unit="$CUR_UNIT" --collect \
        -p MemoryMax="$CAP" -p MemorySwapMax=0 -- \
        env LEAN_NUM_THREADS=1 timeout --signal=TERM --kill-after=10 "$t" "$@" ) &
    RUNNING_PID=$!
    wait "$RUNNING_PID"
    rc=$?
    RUNNING_PID=""
    CUR_UNIT=""
    release_lock
    return "$rc"
}

# copy_paths <seconds> <path...>  -> cp -a --parents into $RUN (bounded)
copy_paths() {
    local t=$1
    shift
    run_bounded "$t" "$REPO_DIR" cp -a --reflink=auto --parents -- "$@" "$RUN/"
}

preflight() {
    local missing=0 d
    for d in "$REPO_DIR/proof/.lake/packages" "$REPO_DIR/proof/corpus/.lake/packages" \
             "$REPO_DIR/target/aeneas"; do
        [ -e "$d" ] || { echo "SKIPPED: missing lane $d" >&2; missing=1; }
    done
    [ -f "$GEN" ] || { echo "SKIPPED: missing generated file $GEN" >&2; missing=1; }
    [ -x "$REPO_DIR/proof/regen.sh" ] || { echo "SKIPPED: proof/regen.sh not executable" >&2; missing=1; }
    [ "$missing" = "0" ] || return 2

    local avail_kb need_kb
    need_kb=$((MIN_FREE_MB * 1024))
    avail_kb=$(df -Pk "$RUN_BASE" 2>/dev/null | awk 'NR==2 {print $4}')
    if [ -z "$avail_kb" ] || [ "$avail_kb" -lt "$need_kb" ]; then
        echo "SKIPPED: $RUN_BASE has ${avail_kb:-0} KiB free, need ${need_kb} KiB (${MIN_FREE_MB} MiB)" >&2
        return 2
    fi

    local reason="" load1 dstate
    load1=$(awk '{print $1}' /proc/loadavg)
    dstate=$(ps -eo stat= 2>/dev/null | grep -c '^D' || true)
    if awk -v l="$load1" -v m="$IO_HOG_LOAD" 'BEGIN { exit !(l >= m) }'; then
        reason="load1=$load1 >= $IO_HOG_LOAD (2*nproc)"
    fi
    if [ "${dstate:-0}" -ge "$IO_HOG_DSTATE" ]; then
        reason="${reason:+$reason; }D-state tasks=$dstate >= $IO_HOG_DSTATE"
    fi
    local bpid b1 b2
    bpid=$(pgrep -x btrfs-cleaner 2>/dev/null | head -1)
    if [ -n "$bpid" ]; then
        b1=$(awk '{print $14 + $15}' "/proc/$bpid/stat" 2>/dev/null || echo 0)
        sleep 1
        b2=$(awk '{print $14 + $15}' "/proc/$bpid/stat" 2>/dev/null || echo 0)
        if [ $((b2 - b1)) -ge "$BTRFS_CLEANER_TICKS" ]; then
            reason="${reason:+$reason; }btrfs-cleaner busy ($((b2 - b1)) ticks/s)"
        fi
    fi
    if pgrep -x rsync >/dev/null 2>&1 || pgrep -x dd >/dev/null 2>&1; then
        reason="${reason:+$reason; }heavy IO process (rsync/dd) running"
    fi
    if [ -n "$reason" ]; then
        echo "SKIPPED: heavy IO hog detected: $reason" >&2
        return 2
    fi
    return 0
}

# Mutation bodies. The patterns are unique in the generated file; the assertion is the check.
MUT_FLAT='import sys
p = sys.argv[1]
s = open(p).read()
old = "          live := true,\n          sigma := entry_sigma,"
assert s.count(old) == 1, ("flat relocation write count", s.count(old))
open(p, "w").write(s.replace(old, "          live := false,\n          sigma := entry_sigma,"))
'

MUT_SPILL='import sys
p = sys.argv[1]
s = open(p).read()
old = "      let i \u2190 table.tag entry_address.place\n"
assert s.count(old) == 1, ("spill relocation write count", s.count(old))
open(p, "w").write(s.replace(old, "      let i := 0#u8\n"))
'

# The flat relocation destination write in `proof/corpus/Growable/Relocation.lean`. `writeFlat` is
# consumed only by its definition, the `rfl` `writeFlat_frame_spill`, and the non-trivial
# `writeFlat_frame_flat`; redirecting the destination to index zero therefore falsifies exactly the
# flat frame headline.
MUT_FRAME='import sys
p = sys.argv[1]
s = open(p).read()
old = "      { reserved.resident with slots := reserved.resident.slots.set index slot } }"
assert s.count(old) == 1, ("flat destination write count", s.count(old))
new = "      { reserved.resident with slots := reserved.resident.slots.set 0#usize slot } }"
open(p, "w").write(s.replace(old, new))
'

echo "== setup =="
echo "machine:   $(uname -n) $(uname -m), nproc=$NPROC"
if [ "$HEAVY" = 1 ]; then
    MODE='HEAVY (freshness + semantic mutation + frame mutation)'
elif [ "$FRAME" = 1 ]; then
    MODE='FRAME (freshness + frame mutation)'
else
    MODE='default (freshness only)'
fi
echo "mode:      $MODE"
echo "bound:     MemoryMax=$CAP MemorySwapMax=0 LEAN_NUM_THREADS=1"
echo "timeouts:  copy=${T_COPY}s extract=${T_EXTRACT}s lean=${T_LEAN}s mutate=${T_MUTATE}s lock-wait=${T_LOCK_WAIT}s"
echo "preflight: free>=${MIN_FREE_MB}MiB, load1<${IO_HOG_LOAD}, D-state<${IO_HOG_DSTATE}, btrfs-cleaner idle, no rsync/dd"

PF_RC=0
preflight || PF_RC=$?
if [ "$PF_RC" -ne 0 ]; then
    echo "relocation-control: SKIPPED                    preflight refused (rc=$PF_RC)"
    exit "$PF_RC"
fi

echo "source:    src/table.rs sha256 $(sha256sum "$REPO_DIR/src/table.rs" | awk '{print $1}')"
GEN_SHA_BEFORE=$(sha256sum "$GEN" | awk '{print $1}')
echo "baseline:  proof/ApeironWord.lean sha256 $GEN_SHA_BEFORE"
echo

mkdir -p "$RUN" || { echo "cannot create $RUN" >&2; exit 2; }

# Minimal path set: the extraction sources, the proof package, its build state, the Aeneas link.
COPY_PATHS=(
    Cargo.toml Cargo.lock
    src
    proof/regen.sh proof/extract proof/ApeironWord.lean proof/Refinement.lean
    proof/Denotation.lean proof/lakefile.lean proof/lake-manifest.json proof/lean-toolchain
    proof/.lake/build
)
[ -d "$REPO_DIR/proof/.lake/config" ] && COPY_PATHS+=(proof/.lake/config)
if [ "$HEAVY" = 1 ] || [ "$FRAME" = 1 ]; then
    # Corpus source, excluding its own .lake (copied explicitly below).
    for e in "$REPO_DIR/proof/corpus"/* "$REPO_DIR/proof/corpus"/.[!.]*; do
        [ -e "$e" ] || continue
        b=$(basename -- "$e")
        [ "$b" = ".lake" ] && continue
        COPY_PATHS+=("proof/corpus/$b")
    done
    COPY_PATHS+=(proof/corpus/.lake/build)
    [ -d "$REPO_DIR/proof/corpus/.lake/config" ] && COPY_PATHS+=(proof/corpus/.lake/config)
fi

COPY_RC=0
copy_paths "$T_COPY" "${COPY_PATHS[@]}" || COPY_RC=$?
if [ "$COPY_RC" -ne 0 ]; then
    receipt copy NOT-RUN "bounded cp failed/timeout after ${T_COPY}s (rc=$COPY_RC)"
    exit 3
fi

mkdir -p "$RUN/proof/.lake" "$RUN/target"
ln -s "$REPO_DIR/proof/.lake/packages" "$RUN/proof/.lake/packages"
ln -s "$REPO_DIR/target/aeneas" "$RUN/target/aeneas"
if [ "$HEAVY" = 1 ] || [ "$FRAME" = 1 ]; then
    mkdir -p "$RUN/proof/corpus/.lake"
    ln -s "$REPO_DIR/proof/corpus/.lake/packages" "$RUN/proof/corpus/.lake/packages"
fi

run_bounded "$T_MUTATE" "$RUN" cp -a -- "$RUN/proof/ApeironWord.lean" "$RUN/proof/ApeironWord.lean.orig"
COPY_MB=$(du -sm "$RUN" 2>/dev/null | awk '{print $1}')
receipt copy PASS "bounded cp ${T_COPY}s, $RUN (${COPY_MB:-?} MiB)"
echo

# ---- baseline probe -------------------------------------------------------------
# The unmutated copy decides whether the freshness step can be attributed at all. A stale
# baseline is a SKIP, not a FAIL: a concurrent source lane can move `src/table.rs` ahead of the
# committed `ApeironWord.lean`, and no mutation check can then attribute its own red.
run_locked "$T_EXTRACT" "$RUN" "$RUN/proof/regen.sh" --check > "$RUN/baseline.log" 2>&1
BASE_RC=$?
BASE_STATE=ERROR
if [ "$BASE_RC" -eq 0 ] && grep -q 'extraction is fresh' "$RUN/baseline.log"; then
    BASE_STATE=GREEN
elif [ "$BASE_RC" -eq 1 ] && grep -q 'stale: ApeironWord.lean' "$RUN/baseline.log"; then
    BASE_STATE=STALE
elif [ "$BASE_RC" -eq 124 ]; then
    BASE_STATE=TIMEOUT
elif [ "$BASE_RC" -eq 75 ]; then
    BASE_STATE=LOCKED
fi
case "$BASE_STATE" in
    GREEN)   receipt baseline PASS   "regen.sh --check fresh on the unmutated copy" ;;
    STALE)   receipt baseline SKIP   "unmutated extraction is stale (source lane ahead of ApeironWord.lean)" ;;
    TIMEOUT) receipt baseline FAIL   "timeout ${T_EXTRACT}s (inconclusive)" ;;
    LOCKED)  receipt baseline FAIL   "lane lock busy after ${T_LOCK_WAIT}s" ;;
    *)       receipt baseline FAIL   "extraction error rc=$BASE_RC (mid-edit source?)" ;;
esac

if [ "$BASE_STATE" != "GREEN" ] && [ "$BASE_STATE" != "STALE" ]; then
    receipt RESULT FAIL "baseline not runnable ($BASE_STATE); no mutation check attributable"
    exit 1
fi

# ---- STEP 1: freshness mutation -------------------------------------------------
run_bounded "$T_MUTATE" "$RUN" python3 -c "$MUT_FLAT" "$RUN/proof/ApeironWord.lean"
MUT_RC=$?
if [ "$MUT_RC" -ne 0 ]; then
    receipt step1-freshness-mutation FAIL "mutation edit rc=$MUT_RC (pattern not unique?)"
    FAIL=$((FAIL + 1))
else
    run_locked "$T_EXTRACT" "$RUN" "$RUN/proof/regen.sh" --check > "$RUN/step1.log" 2>&1
    S1_RC=$?
    run_bounded "$T_MUTATE" "$RUN" cp -a -- "$RUN/proof/ApeironWord.lean.orig" "$RUN/proof/ApeironWord.lean"
    if [ "$S1_RC" -eq 124 ]; then
        receipt step1-freshness-mutation FAIL "timeout ${T_EXTRACT}s (inconclusive)"
        FAIL=$((FAIL + 1))
    elif [ "$S1_RC" -eq 75 ]; then
        receipt step1-freshness-mutation FAIL "lane lock busy after ${T_LOCK_WAIT}s"
        FAIL=$((FAIL + 1))
    elif [ "$S1_RC" -eq 0 ]; then
        receipt step1-freshness-mutation FAIL "regen.sh --check exited 0; mutation not detected"
        FAIL=$((FAIL + 1))
    elif grep -q 'stale: ApeironWord.lean' "$RUN/step1.log" \
         && grep -qE '^- *live := false,$' "$RUN/step1.log" \
         && grep -qE '^\+ *live := true,$' "$RUN/step1.log"; then
        receipt step1-freshness-mutation PASS "rc=$S1_RC, diff names the flat write"
        PASS=$((PASS + 1))
    else
        receipt step1-freshness-mutation FAIL "rc=$S1_RC, mutated line absent from the diff"
        FAIL=$((FAIL + 1))
    fi
fi

# ---- STEP 2: semantic mutation (heavy only) -------------------------------------
if [ "$HEAVY" = 1 ]; then
    run_bounded "$T_MUTATE" "$RUN" python3 -c "$MUT_SPILL" "$RUN/proof/ApeironWord.lean"
    MUT_RC=$?
    if [ "$MUT_RC" -ne 0 ]; then
        receipt step2-semantic-mutation FAIL "mutation edit rc=$MUT_RC (pattern not unique?)"
        FAIL=$((FAIL + 1))
    else
        run_locked "$T_LEAN" "$RUN/proof/corpus" lake build Growable.Relocation > "$RUN/step2.log" 2>&1
        S2_RC=$?
        run_bounded "$T_MUTATE" "$RUN" cp -a -- "$RUN/proof/ApeironWord.lean.orig" "$RUN/proof/ApeironWord.lean"
        if [ "$S2_RC" -eq 124 ]; then
            receipt step2-semantic-mutation FAIL "timeout ${T_LEAN}s (inconclusive)"
            FAIL=$((FAIL + 1))
        elif [ "$S2_RC" -ne 0 ] \
             && grep -qE '^error: Growable/Relocation\.lean:[0-9]+' "$RUN/step2.log"; then
            receipt step2-semantic-mutation PASS "rc=$S2_RC, Relocation headlines red"
            PASS=$((PASS + 1))
        else
            receipt step2-semantic-mutation FAIL "rc=$S2_RC, no Relocation error found"
            FAIL=$((FAIL + 1))
        fi
    fi

    run_locked "$T_LEAN" "$RUN/proof/corpus" lake build Growable.Relocation > "$RUN/step2b.log" 2>&1
    S2B_RC=$?
    if [ "$S2B_RC" -eq 0 ]; then
        receipt step2-semantic-restore PASS "restored Relocation build green"
        PASS=$((PASS + 1))
    else
        receipt step2-semantic-restore FAIL "restored Relocation build rc=$S2B_RC"
        FAIL=$((FAIL + 1))
    fi
fi

# ---- STEP 2c: frame mutation (heavy or frame-only) -------------------------------
# Redirect `writeFlat`'s destination to index zero. `writeFlat` is consumed only by its definition,
# the `rfl` `writeFlat_frame_spill`, and the non-trivial `writeFlat_frame_flat`; the error must land
# inside the `writeFlat_frame_flat` span, so the red is attributed to that frame headline alone.
if [ "$FRAME" = 1 ]; then
    RELOC_LEAN="$RUN/proof/corpus/Growable/Relocation.lean"
    FRAME_LO=$(grep -n '^theorem writeFlat_frame_flat' "$RELOC_LEAN" | head -1 | cut -d: -f1)
    FRAME_HI=$(grep -n '^theorem writeFlat_frame_spill' "$RELOC_LEAN" | head -1 | cut -d: -f1)
    run_bounded "$T_MUTATE" "$RUN" cp -a -- "$RELOC_LEAN" "$RELOC_LEAN.orig"
    FRAME_BACKUP_RC=$?
    if [ "$FRAME_BACKUP_RC" -ne 0 ]; then
        receipt step2c-frame-mutation FAIL "backup of Relocation.lean rc=$FRAME_BACKUP_RC"
        FAIL=$((FAIL + 1))
    else
        run_bounded "$T_MUTATE" "$RUN" python3 -c "$MUT_FRAME" "$RELOC_LEAN"
        MUT_RC=$?
        if [ "$MUT_RC" -ne 0 ]; then
            receipt step2c-frame-mutation FAIL "mutation edit rc=$MUT_RC (pattern not unique?)"
            FAIL=$((FAIL + 1))
        else
            run_locked "$T_LEAN" "$RUN/proof/corpus" lake build > "$RUN/step2c.log" 2>&1
            S2C_RC=$?
            ERR_LINE=$(grep -oE '^error: Growable/Relocation\.lean:[0-9]+' "$RUN/step2c.log" \
                | head -1 | grep -oE '[0-9]+$')
            run_bounded "$T_MUTATE" "$RUN" cp -a -- "$RELOC_LEAN.orig" "$RELOC_LEAN"
            if [ "$S2C_RC" -eq 124 ]; then
                receipt step2c-frame-mutation FAIL "timeout ${T_LEAN}s (inconclusive)"
                FAIL=$((FAIL + 1))
            elif [ "$S2C_RC" -ne 0 ] && [ -n "$ERR_LINE" ] \
                 && [ -n "$FRAME_LO" ] && [ -n "$FRAME_HI" ] \
                 && [ "$ERR_LINE" -ge "$FRAME_LO" ] && [ "$ERR_LINE" -lt "$FRAME_HI" ]; then
                receipt step2c-frame-mutation PASS "rc=$S2C_RC, error at Relocation.lean:$ERR_LINE in writeFlat_frame_flat span"
                PASS=$((PASS + 1))
            else
                receipt step2c-frame-mutation FAIL "rc=$S2C_RC, error line ${ERR_LINE:-none} outside writeFlat_frame_flat span [${FRAME_LO:-?},${FRAME_HI:-?})"
                FAIL=$((FAIL + 1))
            fi
        fi
        run_locked "$T_LEAN" "$RUN/proof/corpus" lake build > "$RUN/step2d.log" 2>&1
        S2D_RC=$?
        if [ "$S2D_RC" -eq 0 ]; then
            receipt step2c-frame-restore PASS "restored destination index builds green"
            PASS=$((PASS + 1))
        else
            receipt step2c-frame-restore FAIL "restored build rc=$S2D_RC"
            FAIL=$((FAIL + 1))
        fi
    fi
fi

# ---- STEP 3: restore freshness --------------------------------------------------
SKIPPED=0
if [ "$BASE_STATE" != "GREEN" ]; then
    receipt step3-restore-freshness NOT-RUN "baseline was $BASE_STATE; restored copy cannot be fresh"
    SKIPPED=1
else
    run_locked "$T_EXTRACT" "$RUN" "$RUN/proof/regen.sh" --check > "$RUN/step3.log" 2>&1
    S3_RC=$?
    if [ "$S3_RC" -eq 124 ]; then
        receipt step3-restore-freshness FAIL "timeout ${T_EXTRACT}s (inconclusive)"
        FAIL=$((FAIL + 1))
    elif [ "$S3_RC" -eq 0 ] && grep -q 'extraction is fresh' "$RUN/step3.log"; then
        receipt step3-restore-freshness PASS "regen.sh --check green"
        PASS=$((PASS + 1))
    else
        receipt step3-restore-freshness FAIL "rc=$S3_RC; restore not fresh"
        FAIL=$((FAIL + 1))
    fi
fi

# ---- working-tree guard (informational) -----------------------------------------
GEN_SHA_AFTER=$(sha256sum "$GEN" | awk '{print $1}')
if [ "$GEN_SHA_AFTER" = "$GEN_SHA_BEFORE" ]; then
    receipt working-tree-guard PASS "proof/ApeironWord.lean digest unchanged"
else
    receipt working-tree-guard WARN "proof/ApeironWord.lean changed during the run"
fi

echo
if [ "$FAIL" -ne 0 ]; then
    receipt RESULT FAIL "checks=$((PASS + FAIL)) passed=$PASS failed=$FAIL"
    exit 1
elif [ "$SKIPPED" -ne 0 ]; then
    receipt RESULT SKIP "checks=$PASS passed, freshness attribution skipped (baseline stale)"
    exit 2
else
    receipt RESULT PASS "checks=$((PASS + FAIL)) passed=$PASS failed=0"
    exit 0
fi
