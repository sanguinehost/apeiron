#!/usr/bin/env bash
# Rejecting control for the generated-body refinement of the quantum-register stage.
#
#   bash proof/controls/register.sh
#
# `proof/corpus/Numeric/RegisterRefinement.lean` proves `stage_mul_denote` and its dependents over
# the Aeneas-extracted `register.stage_mul` / `cell.CellI64.checked_mul` generated into
# `proof/ApeironWord.lean` from `src/numeric/register.rs` (`proof/extract/lib.rs`, `word_stage_mul`).
# The statement-side controls (renamed copy of the refinement file, swapped `w` slots, negated `σ`)
# exist. This control mutates the GENERATED BODY and shows the refinement goes red, then restores it
# and shows it goes green.
#
# The control never writes to the working tree. It copies the paths the control needs into
# /home/socol/oc-tmp/apeiron-register-control.$$ (removed on exit; /tmp is a small tmpfs and is not
# used). The canonical corpus, the mathlib/Aeneas package caches and the package links are read-only
# inputs reached by symlink; only the copies' `proof/ApeironWord.lean`, `proof/.lake/build` and
# `proof/corpus/.lake/build` are writable.
#
# STEPS
#   1  copy the control's paths into two throwaway copies, a/ (baseline + mutation A + restore) and
#      b/ (mutation B). Two copies because mutation A and mutation B are alternatives of one file:
#      each must be built from the committed body, and neither may inherit the other's build state.
#   2  BASELINE, copy a/: `lake build Numeric.RegisterRefinement` exits 0.
#   3  MUTATION A, copy a/: swap the pair write-back in `register.stage_block_loop.body`
#      (`Slice.update x lo out.a` / `Slice.update x1 hi out.b` -> `x hi` / `x1 lo`). The build must
#      exit nonzero with its first error inside `Numeric/RegisterRefinement.lean`.
#   4  MUTATION B, copy b/: drop sigma from the first slot of `cell.CellI64.checked_mul`
#      (`I64.checked_mul s self.b` -> `I64.checked_mul 1#i64 self.b`). Same requirement.
#   5  RESTORE, copy a/: put the committed body back and rebuild; the exit must be 0.
#
# Steps 3 and 4 are two independent throwaway copies and are built concurrently, then waited for:
# one build of `Numeric.RegisterRefinement` after a change to `ApeironWord.lean` is a serial chain
# of ~245-260 s (ApeironWord 26 s, `Numeric.Exp.Part2` 186 s, the rest of the `Numeric` closure
# ~50 s; `Numeric.Register` imports that battery through `Numeric.HadamardStage`), so three fresh
# chains would need ~770 s, past the 580 s budget this control declares. Both chains are still full
# fresh builds from mutated sources; nothing is cached, skipped or inherited. No job is left
# running: the script waits for both before printing their receipts.
#
# HARDENING (from `proof/controls/relocation.sh`): every step is wrapped in `timeout`; the copy is
# bounded; the lane lock is an inherited file descriptor rather than a child `flock` process; the
# EXIT/INT/TERM trap kills the active process groups and stops the transient systemd scopes.
#
# Exit 0 only if every step behaves. Prints a per-step receipt with exit status, elapsed seconds and
# first error file:line. Whole-script wall budget 580 s.

set -uo pipefail

PROOF_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REPO_DIR=$(CDPATH= cd -- "$PROOF_DIR/.." && pwd)
GEN="$REPO_DIR/proof/ApeironWord.lean"
CANON="$REPO_DIR/proof/canonical"
AENEAS_MODEL="$REPO_DIR/target/aeneas-model"
LOCK=/tmp/opencode/apeiron-lane.lock
RUN_BASE=/home/socol/oc-tmp
RUN="$RUN_BASE/apeiron-register-control.$$"
CA="$RUN/a"
CB="$RUN/b"
CAP=12G
BUDGET=580
T_COPY=120
T_LEAN_MAX=340
T_MUTATE=30
T_LOCK_WAIT=120
MIN_FREE_MB=4096
STEP_FLOOR=200

NPROC=$(nproc)
IO_HOG_LOAD=$((4 * NPROC))
BUDGET_START=$(date +%s)
DEADLINE=$((BUDGET_START + BUDGET))

PASS=0
FAIL=0
SKIPPED=0
LOCK_HELD=0
RUNNING_PIDS=""
CUR_UNITS=""

receipt() { printf 'register-control: %-22s %-6s %s\n' "$1" "$2" "$3"; }

remaining() {
    local r=$((DEADLINE - $(date +%s)))
    [ "$r" -lt 1 ] && r=1
    printf '%s' "$r"
}

cleanup() {
    if [ "${APEIRON_REGISTER_CONTROL_KEEP:-0}" = "1" ]; then
        echo "register-control: kept copy at $RUN"
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

# Kill the active process groups only when setsid gave them their own group.
kill_tree() {
    local pid pgid selfpgid
    selfpgid=$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ')
    for pid in $RUNNING_PIDS; do
        pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
        if [ -n "$pgid" ] && [ "$pgid" != "$selfpgid" ]; then
            kill -TERM -- "-$pgid" 2>/dev/null || true
        else
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    RUNNING_PIDS=""
}

on_exit() {
    local rc=$?
    trap - EXIT INT TERM
    kill_tree
    local u
    for u in $CUR_UNITS; do
        systemctl --user stop "$u" >/dev/null 2>&1 || true
    done
    release_lock
    cleanup
    exit "$rc"
}
trap on_exit EXIT INT TERM

# run_bounded <seconds> <cwd> <command...> -> child exit status (124 on timeout).
run_bounded() {
    local t=$1 dir=$2 rc
    shift 2
    ( cd "$dir" && exec setsid timeout --signal=TERM --kill-after=10 "$t" "$@" ) &
    RUNNING_PIDS="$!"
    wait "$RUNNING_PIDS"
    rc=$?
    RUNNING_PIDS=""
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

# launch_build <copy> <step>  -> LAUNCH_PID, LAUNCH_T
# The AGENTS.md §6 lane: inherited lock fd, memory cap, single-threaded, timed, own session. The
# lock fd is closed before exec so the transient scope cannot inherit it.
launch_build() {
    local copy=$1 step=$2 t
    t=$(remaining)
    [ "$t" -gt "$T_LEAN_MAX" ] && t=$T_LEAN_MAX
    eval "LAUNCH_T$step=$t"
    CUR_UNITS="$CUR_UNITS apeiron-register-$$-$step"
    (
        cd "$copy/proof/corpus" || exit 96
        exec 9>&-
        exec setsid systemd-run --user --scope -q --collect \
            --unit="apeiron-register-$$-$step" \
            -p MemoryMax="$CAP" -p MemorySwapMax=0 -- \
            env LEAN_NUM_THREADS=1 timeout --signal=TERM --kill-after=10 "$t" \
            lake build Numeric.RegisterRefinement
    ) > "$RUN/step$step.log" 2>&1 &
    LAUNCH_PID=$!
}

# first_error <log> -> "<file>.lean:<line>" of the first Lean error, or empty.
first_error() {
    rg -m1 -N --no-filename -e '^error: ' -e ': error: ' "$1" 2>/dev/null | head -1 \
        | rg -o -N '[A-Za-z0-9_./-]+\.lean:[0-9]+' 2>/dev/null | head -1
}

# copy_control <destination>
copy_control() {
    local dest=$1 e b
    mkdir -p "$dest/proof/.lake" "$dest/proof/corpus/.lake" "$dest/target" || return 1
    run_bounded "$T_COPY" "$REPO_DIR" cp -a --reflink=auto --parents -- \
        proof/ApeironWord.lean proof/Refinement.lean proof/Denotation.lean \
        proof/lakefile.lean proof/lake-manifest.json proof/lean-toolchain \
        proof/.lake/build "$dest/" || return 1
    [ -d "$REPO_DIR/proof/.lake/config" ] \
        && run_bounded "$T_COPY" "$REPO_DIR" cp -a --reflink=auto --parents -- proof/.lake/config "$dest/"
    for e in "$REPO_DIR/proof/corpus"/* "$REPO_DIR/proof/corpus"/.[!.]*; do
        [ -e "$e" ] || continue
        b=$(basename -- "$e")
        [ "$b" = ".lake" ] && continue
        run_bounded "$T_COPY" "$REPO_DIR" cp -a --reflink=auto --parents -- "proof/corpus/$b" "$dest/" || return 1
    done
    run_bounded "$T_COPY" "$REPO_DIR" cp -a --reflink=auto --parents -- \
        proof/corpus/.lake/build "$dest/" || return 1
    [ -d "$REPO_DIR/proof/corpus/.lake/config" ] \
        && run_bounded "$T_COPY" "$REPO_DIR" cp -a --reflink=auto --parents -- proof/corpus/.lake/config "$dest/"
    ln -s "$REPO_DIR/proof/.lake/packages" "$dest/proof/.lake/packages" || return 1
    ln -s "$REPO_DIR/proof/corpus/.lake/packages" "$dest/proof/corpus/.lake/packages" || return 1
    ln -s "$CANON" "$dest/proof/canonical" || return 1
    ln -s "$AENEAS_MODEL" "$dest/target/aeneas-model" || return 1
    return 0
}

# Mutation bodies. Each pattern is unique in the generated file; the assertion is the check. The
# generated file sets `linter.unusedVariables false`, so dropping the `s` operand is type-correct
# and reddens in the refinement file rather than in `ApeironWord.lean`.
MUT_A='import sys
p = sys.argv[1]
s = open(p).read()
old = "        let x1 \u2190 Slice.update x lo out.a\n        let s \u2190 Slice.update x1 hi out.b\n"
assert s.count(old) == 1, ("pair write-back count", s.count(old))
new = "        let x1 \u2190 Slice.update x hi out.a\n        let s \u2190 Slice.update x1 lo out.b\n"
open(p, "w").write(s.replace(old, new))
'

MUT_B='import sys
p = sys.argv[1]
s = open(p).read()
old = "    let o1 \u2190 lift (I64.checked_mul s self.b)\n"
assert s.count(old) == 1, ("sigma-b operand count", s.count(old))
new = "    let o1 \u2190 lift (I64.checked_mul 1#i64 self.b)\n"
open(p, "w").write(s.replace(old, new))
'

preflight() {
    local missing=0 tool
    for tool in lake python3 rg systemd-run flock timeout; do
        command -v "$tool" >/dev/null 2>&1 || { echo "SKIPPED: missing tool $tool" >&2; missing=1; }
    done
    for d in "$REPO_DIR/proof/.lake/packages" "$REPO_DIR/proof/corpus/.lake/packages" \
             "$CANON" "$AENEAS_MODEL"; do
        [ -e "$d" ] || { echo "SKIPPED: missing lane $d" >&2; missing=1; }
    done
    [ -f "$GEN" ] || { echo "SKIPPED: missing generated file $GEN" >&2; missing=1; }
    [ "$missing" = "0" ] || return 2

    local avail_kb need_kb
    need_kb=$((MIN_FREE_MB * 1024))
    avail_kb=$(df -Pk "$RUN_BASE" 2>/dev/null | awk 'NR==2 {print $4}')
    if [ -z "$avail_kb" ] || [ "$avail_kb" -lt "$need_kb" ]; then
        echo "SKIPPED: $RUN_BASE has ${avail_kb:-0} KiB free, need ${need_kb} KiB (${MIN_FREE_MB} MiB)" >&2
        return 2
    fi
    local load1
    load1=$(awk '{print $1}' /proc/loadavg)
    if awk -v l="$load1" -v m="$IO_HOG_LOAD" 'BEGIN { exit !(l >= m) }'; then
        echo "SKIPPED: load1=$load1 >= $IO_HOG_LOAD (4*nproc)" >&2
        return 2
    fi
    if pgrep -x rsync >/dev/null 2>&1 || pgrep -x dd >/dev/null 2>&1; then
        echo "SKIPPED: heavy IO process (rsync/dd) running" >&2
        return 2
    fi
    return 0
}

echo "== setup =="
echo "machine:   $(uname -n) $(uname -m), nproc=$NPROC"
echo "copy:      $RUN (a=baseline+mutationA+restore, b=mutationB)"
echo "bound:     MemoryMax=$CAP MemorySwapMax=0 LEAN_NUM_THREADS=1"
echo "build:     cd <copy>/proof/corpus && lake build Numeric.RegisterRefinement"
echo "budget:    ${BUDGET}s total, per-build cap ${T_LEAN_MAX}s, step floor ${STEP_FLOOR}s, lock-wait ${T_LOCK_WAIT}s"
echo "preflight: free>=${MIN_FREE_MB}MiB, load1<${IO_HOG_LOAD}, no rsync/dd"
echo "source:    src/numeric/register.rs sha256 $(sha256sum "$REPO_DIR/src/numeric/register.rs" | awk '{print $1}')"
GEN_SHA_BEFORE=$(sha256sum "$GEN" | awk '{print $1}')
echo "baseline:  proof/ApeironWord.lean sha256 $GEN_SHA_BEFORE"
echo

PF_RC=0
preflight || PF_RC=$?
if [ "$PF_RC" -ne 0 ]; then
    receipt RESULT SKIP "preflight refused (rc=$PF_RC)"
    exit "$PF_RC"
fi

# ---- STEP 1: copies --------------------------------------------------------------
STEP_START=$(date +%s)
COPY_RC=0
copy_control "$CA" || COPY_RC=$?
if [ "$COPY_RC" -eq 0 ]; then
    copy_control "$CB" || COPY_RC=$?
fi
STEP_ELAPSED=$(( $(date +%s) - STEP_START ))
if [ "$COPY_RC" -ne 0 ]; then
    receipt step1-copy FAIL "exit=$COPY_RC elapsed=${STEP_ELAPSED}s first-error=none bounded cp failed/timeout"
    receipt RESULT FAIL "copy failed (rc=$COPY_RC)"
    exit 3
fi
COPY_MB=$(du -sm "$RUN" 2>/dev/null | awk '{print $1}')
receipt step1-copy PASS "exit=0 elapsed=${STEP_ELAPSED}s first-error=none 2 copies, ${COPY_MB:-?} MiB (reflink)"
echo

# ---- STEP 2: baseline ------------------------------------------------------------
acquire_lock || { receipt step2-baseline FAIL "exit=75 elapsed=0s first-error=none lane lock busy after ${T_LOCK_WAIT}s"; receipt RESULT FAIL "lane lock busy"; exit 75; }
STEP_START=$(date +%s)
launch_build "$CA" 2
RUNNING_PIDS="$LAUNCH_PID"
wait "$LAUNCH_PID"; S2_RC=$?
RUNNING_PIDS=""
STEP_ELAPSED=$(( $(date +%s) - STEP_START ))
S2_ERR=$(first_error "$RUN/step2.log")
if [ "$S2_RC" -eq 0 ]; then
    receipt step2-baseline PASS "exit=0 elapsed=${STEP_ELAPSED}s first-error=none committed generated body builds the refinement"
    PASS=$((PASS + 1))
else
    receipt step2-baseline FAIL "exit=$S2_RC elapsed=${STEP_ELAPSED}s first-error=${S2_ERR:-none} baseline not green; no mutation check is attributable"
    FAIL=$((FAIL + 1))
fi
echo

# ---- STEPS 3+4: the two generated-body mutations, each in its own copy -----------
if [ "$S2_RC" -ne 0 ]; then
    receipt step3-mutation-A NOT-RUN "exit=0 elapsed=0s first-error=none baseline not green"
    receipt step4-mutation-B NOT-RUN "exit=0 elapsed=0s first-error=none baseline not green"
    SKIPPED=1
else
    for copy in "$CA" "$CB"; do
        run_bounded "$T_MUTATE" "$copy" cp -a -- "$copy/proof/ApeironWord.lean" "$copy/proof/ApeironWord.lean.orig"
    done
    MUTA_RC=0
    run_bounded "$T_MUTATE" "$RUN" python3 -c "$MUT_A" "$CA/proof/ApeironWord.lean" || MUTA_RC=$?
    MUTB_RC=0
    run_bounded "$T_MUTATE" "$RUN" python3 -c "$MUT_B" "$CB/proof/ApeironWord.lean" || MUTB_RC=$?

    S3_T0=$(date +%s)
    launch_build "$CA" 3
    PA=$LAUNCH_PID
    RUNNING_PIDS="$PA"
    S4_T0=$(date +%s)
    launch_build "$CB" 4
    PB=$LAUNCH_PID
    RUNNING_PIDS="$PA $PB"

    wait "$PA"; S3_RC=$?
    S3_ELAPSED=$(( $(date +%s) - S3_T0 ))
    S3_ERR=$(first_error "$RUN/step3.log")
    wait "$PB"; S4_RC=$?
    S4_ELAPSED=$(( $(date +%s) - S4_T0 ))
    S4_ERR=$(first_error "$RUN/step4.log")
    RUNNING_PIDS=""

    if [ "$MUTA_RC" -ne 0 ]; then
        receipt step3-mutation-A FAIL "exit=$MUTA_RC elapsed=0s first-error=none mutation edit failed (pattern not unique?)"
        FAIL=$((FAIL + 1))
    elif [ "$S3_RC" -eq 124 ]; then
        receipt step3-mutation-A FAIL "exit=124 elapsed=${S3_ELAPSED}s first-error=${S3_ERR:-none} timeout ${LAUNCH_T3}s (inconclusive)"
        FAIL=$((FAIL + 1))
    elif [ "$S3_RC" -eq 0 ]; then
        receipt step3-mutation-A FAIL "exit=0 elapsed=${S3_ELAPSED}s first-error=none build green under the write-back swap; mutation not detected"
        FAIL=$((FAIL + 1))
    elif [ -z "$S3_ERR" ]; then
        receipt step3-mutation-A FAIL "exit=$S3_RC elapsed=${S3_ELAPSED}s first-error=none no Lean error line in the log"
        FAIL=$((FAIL + 1))
    elif [ "${S3_ERR#*ApeironWord.lean}" != "$S3_ERR" ]; then
        receipt step3-mutation-A FAIL "exit=$S3_RC elapsed=${S3_ELAPSED}s first-error=$S3_ERR error inside the generated file, not the refinement"
        FAIL=$((FAIL + 1))
    elif [ "${S3_ERR#*RegisterRefinement.lean:}" != "$S3_ERR" ]; then
        receipt step3-mutation-A PASS "exit=$S3_RC elapsed=${S3_ELAPSED}s first-error=$S3_ERR write-back swap reddens the refinement"
        PASS=$((PASS + 1))
    else
        receipt step3-mutation-A FAIL "exit=$S3_RC elapsed=${S3_ELAPSED}s first-error=$S3_ERR error outside the refinement file"
        FAIL=$((FAIL + 1))
    fi

    if [ "$MUTB_RC" -ne 0 ]; then
        receipt step4-mutation-B FAIL "exit=$MUTB_RC elapsed=0s first-error=none mutation edit failed (pattern not unique?)"
        FAIL=$((FAIL + 1))
    elif [ "$S4_RC" -eq 124 ]; then
        receipt step4-mutation-B FAIL "exit=124 elapsed=${S4_ELAPSED}s first-error=${S4_ERR:-none} timeout ${LAUNCH_T4}s (inconclusive)"
        FAIL=$((FAIL + 1))
    elif [ "$S4_RC" -eq 0 ]; then
        receipt step4-mutation-B FAIL "exit=0 elapsed=${S4_ELAPSED}s first-error=none build green under the sigma drop; mutation not detected"
        FAIL=$((FAIL + 1))
    elif [ -z "$S4_ERR" ]; then
        receipt step4-mutation-B FAIL "exit=$S4_RC elapsed=${S4_ELAPSED}s first-error=none no Lean error line in the log"
        FAIL=$((FAIL + 1))
    elif [ "${S4_ERR#*ApeironWord.lean}" != "$S4_ERR" ]; then
        receipt step4-mutation-B FAIL "exit=$S4_RC elapsed=${S4_ELAPSED}s first-error=$S4_ERR error inside the generated file, not the refinement"
        FAIL=$((FAIL + 1))
    elif [ "${S4_ERR#*RegisterRefinement.lean:}" != "$S4_ERR" ]; then
        receipt step4-mutation-B PASS "exit=$S4_RC elapsed=${S4_ELAPSED}s first-error=$S4_ERR sigma drop reddens the refinement"
        PASS=$((PASS + 1))
    else
        receipt step4-mutation-B FAIL "exit=$S4_RC elapsed=${S4_ELAPSED}s first-error=$S4_ERR error outside the refinement file"
        FAIL=$((FAIL + 1))
    fi
fi
echo

# ---- STEP 5: restore and rebuild -------------------------------------------------
REST=$(remaining)
if [ "$REST" -lt "$STEP_FLOOR" ]; then
    receipt step5-restore FAIL "exit=0 elapsed=0s first-error=none budget: ${REST}s left < floor ${STEP_FLOOR}s; a fresh chain needs ~260s"
    FAIL=$((FAIL + 1))
else
    run_bounded "$T_MUTATE" "$CA" cp -a -- "$CA/proof/ApeironWord.lean.orig" "$CA/proof/ApeironWord.lean"
    STEP_START=$(date +%s)
    launch_build "$CA" 5
    RUNNING_PIDS="$LAUNCH_PID"
    wait "$LAUNCH_PID"; S5_RC=$?
    RUNNING_PIDS=""
    STEP_ELAPSED=$(( $(date +%s) - STEP_START ))
    S5_ERR=$(first_error "$RUN/step5.log")
    if [ "$S5_RC" -eq 0 ]; then
        receipt step5-restore PASS "exit=0 elapsed=${STEP_ELAPSED}s first-error=none restored body rebuilds green"
        PASS=$((PASS + 1))
    elif [ "$S5_RC" -eq 124 ]; then
        receipt step5-restore FAIL "exit=124 elapsed=${STEP_ELAPSED}s first-error=${S5_ERR:-none} timeout (inconclusive)"
        FAIL=$((FAIL + 1))
    else
        receipt step5-restore FAIL "exit=$S5_RC elapsed=${STEP_ELAPSED}s first-error=${S5_ERR:-none} restored body not green"
        FAIL=$((FAIL + 1))
    fi
fi

release_lock

# ---- working-tree guard ----------------------------------------------------------
GEN_SHA_AFTER=$(sha256sum "$GEN" | awk '{print $1}')
if [ "$GEN_SHA_AFTER" = "$GEN_SHA_BEFORE" ]; then
    receipt working-tree-guard PASS "exit=0 elapsed=0s first-error=none proof/ApeironWord.lean digest unchanged"
else
    receipt working-tree-guard WARN "exit=0 elapsed=0s first-error=none proof/ApeironWord.lean changed during the run"
fi

echo
TOTAL=$(( $(date +%s) - BUDGET_START ))
if [ "$FAIL" -ne 0 ]; then
    receipt RESULT FAIL "checks=$((PASS + FAIL)) passed=$PASS failed=$FAIL total=${TOTAL}s"
    exit 1
elif [ "$SKIPPED" -ne 0 ]; then
    receipt RESULT SKIP "checks=$PASS passed, not attributable total=${TOTAL}s"
    exit 2
elif [ "$TOTAL" -ge "$BUDGET" ]; then
    receipt RESULT FAIL "passed=$PASS but total=${TOTAL}s >= budget ${BUDGET}s"
    exit 1
else
    receipt RESULT PASS "checks=$PASS passed=$PASS failed=0 total=${TOTAL}s of ${BUDGET}s"
    exit 0
fi
