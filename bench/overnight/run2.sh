#!/usr/bin/env bash
# Second overnight queue: rebuild the release examples with the v4 reader, then measure the
# speed split that the first queue could not open. Waits for the first queue's lock.
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
PPL="$ROOT/target/release/examples/ppl"
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

log_status "run2 start epoch=$(date +%s)"

# R1: every release example, so the v4 reader is in phase_split/head_time/generate.
# The GPU examples have no `main` without `--features gpu`, so name the four CPU examples.
run_job R1-build 900 "$GUARD" run -- cargo build --release --locked \
    --example phase_split --example head_time --example generate --example ppl

if wait_mem 2600; then
    run_job J5b-phase-split 1800 "${SCOPE[@]}" "$SPLIT" --model "$MODEL" --pack "$PACK" --steps 24 --json
fi

# J5c: replay the split with --target set to the dominant term, so the dominant-term gate is
# exercised on the unmodified tree.
if [ -e "$DONE/J5b-phase-split" ]; then
    run_job J5c-target-gate 1800 bash -c '
    j='"$LOGS"'/J5b-phase-split.log
    dom=$(python3 -c "import json,sys; print(json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])[\"dominant\"])" "$j")
    echo "dominant=$dom"
    systemd-run --user --scope -q -p MemoryMax=3200M -p MemorySwapMax=0 -- \
        '"$SPLIT"' --model '"$MODEL"' --pack '"$PACK"' --steps 24 --target "$dom"
'
fi

if wait_mem 2600; then
    run_job J6b-head-time 1800 "${SCOPE[@]}" "$HEAD" "$MODEL" "$PACK"
fi

if wait_mem 2600; then
    run_job J8b-decode-2k 3600 bash -c '
        p=$(cat /tmp/opencode/nq8b-2k-prompt.txt)
        for i in 1 2 3; do
            echo "== run $i =="
            systemd-run --user --scope -q -p MemoryMax=3200M -p MemorySwapMax=0 -- \
                '"$GEN"' --model '"$MODEL"' --pack '"$PACK"' --tokenizer '"$TOK"' \
                --prompt "$p" --n 32 --kernel q8act
        done
    '
fi

# J8c: one 8-thread decode, to see whether SMT helps the shipped pool; four threads stay the product.
if wait_mem 2600; then
    run_job J8c-decode-2k-8t 1200 bash -c '
        p=$(cat /tmp/opencode/nq8b-2k-prompt.txt)
        systemd-run --user --scope -q -p MemoryMax=3200M -p MemorySwapMax=0 -- \
            env APEIRON_THREADS=8 '"$GEN"' --model '"$MODEL"' --pack '"$PACK"' --tokenizer '"$TOK"' \
            --prompt "$p" --n 32 --kernel q8act
    '
fi

# J7c: a clean read probe after the test sweep, 4 and 8 threads, 256 MiB per thread.
run_job J7c-membw-clean 900 bash -c '
    for t in 4 8; do
        echo "== ${t} threads, 256 MiB, 100 passes =="
        '"$GUARD"' run -- taskset -c 0-7 /tmp/membw "$t" 256 100
    done
'

# J4b: the exact PPL, if the first queue skipped it (memory or a failure).
if [ ! -e "$DONE/J4-exact-ppl" ] && wait_mem 2600; then
    run_job J4b-exact-ppl 7200 "${SCOPE[@]}" "$PPL" \
        --model "$MODEL" --pack "$PACK" --tokenizer "$TOK" --data "$DATA" \
        --tokens 2048 --skip 0 --ctx 2048 --head-batch 16 --kernel exact --second-half
fi

log_status "run2 done epoch=$(date +%s)"
