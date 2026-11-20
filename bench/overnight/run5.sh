#!/usr/bin/env bash
# Fifth queue: the v4 speed split and the head/trunk attribution, then the equality test for the
# extended Q6_K group coverage. Runs after the Q4/NQ PPL queues release the lock.
#
# The release examples were built at 22:47 with the v4 reader; run2's `--examples` build was the
# wrong command (it also builds the feature-gated GPU examples, which have no `main` without
# `--features gpu`), so this queue does not rebuild them.
set -u

ROOT=/home/lucas/Workspace/apeiron-nq
cd "$ROOT" || exit 1
TS=1794912865
LOGS="bench/cpu/$TS-nq8b-overnight-logs"
DONE="bench/overnight/done"
STATUS="bench/overnight/status.txt"
MODEL=/home/lucas/models/arelath/nq8b/skeleton-q6k-q4kembed.gguf
PACK=/home/lucas/models/arelath/nq8b-late-v4/model.nqp
TOK=/home/lucas/models/arelath/nq8b/tokenizer.json
DATA=/home/lucas/models/wikitext2/wikitext-2-raw/wiki.test.raw
SPLIT="$ROOT/target/release/examples/phase_split"
HEAD="$ROOT/target/release/examples/head_time"
GEN="$ROOT/target/release/examples/generate"
GUARD=/home/lucas/Workspace/tools/agent-guard
SCOPE=(systemd-run --user --scope -q -p MemoryMax=3200M -p MemorySwapMax=0 --)

mkdir -p "$DONE" "$LOGS"

exec 9>bench/overnight/.lock
flock 9

export APEIRON_THREADS=4

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

wait_mem() {
    local need_mb=${1:-2600} waited=0 avail
    while [ "$waited" -lt 1800 ]; do
        avail=$(awk '/MemAvailable/{print int($2 / 1024)}' /proc/meminfo)
        if [ "$avail" -ge "$need_mb" ]; then
            log_status "mem ok avail=${avail}MiB need=${need_mb}MiB"
            return 0
        fi
        log_status "mem wait avail=${avail}MiB need=${need_mb}MiB"
        sleep 60
        waited=$((waited + 60))
    done
    log_status "mem refused avail=${avail}MiB need=${need_mb}MiB after ${waited}s"
    return 1
}

log_status "run5 start epoch=$(date +%s)"

if wait_mem 2600; then
    run_job K1-split 1800 "${SCOPE[@]}" "$SPLIT" --model "$MODEL" --pack "$PACK" --steps 24
    run_job K1j-split-json 1800 "${SCOPE[@]}" "$SPLIT" --model "$MODEL" --pack "$PACK" --steps 24 --json
fi

# K2: the dominant-term gate on the unmodified tree. The JSON prints as two lines, so join them.
if [ -e "$DONE/K1j-split-json" ]; then
    run_job K2-target-gate 1800 bash -c '
        dom=$(grep -o "\"dominant\":\"[^\"]*\"" '"$LOGS"'/K1j-split-json.log | tail -1 | cut -d\" -f4)
        echo "dominant=$dom"
        [ -n "$dom" ] || exit 1
        exec systemd-run --user --scope -q -p MemoryMax=3200M -p MemorySwapMax=0 -- \
            '"$SPLIT"' --model '"$MODEL"' --pack '"$PACK"' --steps 24 --target "$dom"
    '
fi

if wait_mem 2600; then
    run_job K3-head-time 1800 "${SCOPE[@]}" "$HEAD" "$MODEL" "$PACK"
fi

if wait_mem 2600; then
    run_job K4-decode-2k 3600 bash -c '
        p=$(cat /tmp/opencode/nq8b-2k-prompt.txt)
        for i in 1 2 3; do
            echo "== 4 threads run $i =="
            systemd-run --user --scope -q -p MemoryMax=3200M -p MemorySwapMax=0 -- \
                '"$GEN"' --model '"$MODEL"' --pack '"$PACK"' --tokenizer '"$TOK"' \
                --prompt "$p" --n 32 --kernel q8act
        done
        echo "== 1 thread run 1 =="
        systemd-run --user --scope -q -p MemoryMax=3200M -p MemorySwapMax=0 -- \
            env APEIRON_THREADS=1 '"$GEN"' --model '"$MODEL"' --pack '"$PACK"' --tokenizer '"$TOK"' \
            --prompt "$p" --n 32 --kernel q8act
        echo "== 2 threads run 1 =="
        systemd-run --user --scope -q -p MemoryMax=3200M -p MemorySwapMax=0 -- \
            env APEIRON_THREADS=2 '"$GEN"' --model '"$MODEL"' --pack '"$PACK"' --tokenizer '"$TOK"' \
            --prompt "$p" --n 32 --kernel q8act
    '
fi

# K5: the extended Q6_K group coverage, release test target.
run_job K5-q6k-test 1800 "$GUARD" run -- cargo test --release --locked --lib --all-features chart::avx2_group_cores_match_the_portable_loop -- --nocapture

log_status "run5 done epoch=$(date +%s)"
