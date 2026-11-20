#!/usr/bin/env bash
# Tenth queue: the Q6_K equality test over all 16 block groups, with a filter that matches.
#
# K5 used `chart::avx2_group_cores_match_the_portable_loop`, but the full test path is
# `engine::chart::tests::avx2_group_cores_match_the_portable_loop`; the filter matched no test and
# cargo reported "0 passed" with exit 0. A filter that selects nothing is not a pass.
set -u

ROOT=/home/lucas/Workspace/apeiron-nq
cd "$ROOT" || exit 1
TS=1794912865
LOGS="bench/cpu/$TS-nq8b-overnight-logs"
DONE="bench/overnight/done"
STATUS="bench/overnight/status.txt"
GUARD=/home/lucas/Workspace/tools/agent-guard

mkdir -p "$DONE" "$LOGS"

exec 9>bench/overnight/.lock
flock 9

log_status() {
    printf '%s\n' "$*" >>"$STATUS"
}

run_job() {
    local id=$1 timeout_s=$2
    shift 2
    local log="$LOGS/$id.log" marker="$DONE/$id" t0 t1 rc
    if [ -e "$marker" ]; then
        log_status "skip $id (done)"
        return 0
    fi
    log_status "start $id epoch=$(date +%s) :: $*"
    t0=$(date +%s)
    timeout "$timeout_s" "$@" >>"$log" 2>&1
    rc=$?
    t1=$(date +%s)
    log_status "end $id rc=$rc wall=$((t1 - t0))s"
    if [ "$rc" -eq 0 ]; then
        touch "$marker"
    fi
    return 0
}

log_status "run10 start epoch=$(date +%s)"

run_job K5b-q6k-test 1800 "$GUARD" run -- cargo test --release --locked --lib --all-features \
    avx2_group_cores_match_the_portable_loop -- --nocapture

# A run that selects no test is not evidence; make that a failure.
if [ -e "$DONE/K5b-q6k-test" ] && ! grep -q "running 1 test" "$LOGS/K5b-q6k-test.log"; then
    log_status "K5b refused: filter selected $(grep -o 'running [0-9]* tests' "$LOGS/K5b-q6k-test.log" | head -1)"
    rm -f "$DONE/K5b-q6k-test"
fi

log_status "run10 done epoch=$(date +%s)"
