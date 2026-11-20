#!/usr/bin/env bash
# Overnight queue for bench/cpu/1794912865-nq8b-overnight.md.
#
# One heavy job at a time. Builds run through agent-guard; measurements run in a 3.2 GiB
# scope with no swap. A job that has a done/<id> marker is skipped, so a rerun resumes.
# Exit status does not stop the queue: every job is recorded, and the morning agent reads
# status.txt and the logs. A red build/gate is evidence, not a reason to lose the night.
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
if ! flock -n 9; then
    echo "another overnight driver holds the lock" >&2
    exit 1
fi

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

# The 8B pair peaks at ~2.34 GiB; do not start it into a host already under pressure.
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

log_status "driver start epoch=$(date +%s) model=$MODEL pack=$PACK"

# J0: the state of the second-window route, recorded before anything is changed.
run_job J0-q4-probe 300 bash -c '
    df -h /home/lucas | tail -1
    ls -l /home/lucas/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf 2>/dev/null || echo "no full Q4 file"
    vastai show instances 2>&1 | tail -3
'

# J1: the fast half of the contributor gate (fmt, claims, book findings).
run_job J1-fast-gate 900 bash -c "$GUARD run -- cargo fmt --all -- --check && python3 tools/check-claims.py && python3 tools/book-check.py"

# J7: read-only streaming bandwidth on the whole machine, before any model run.
run_job J7a-membw-build 300 "$GUARD" run -- cc -O3 -march=native -pthread -o /tmp/membw tools/membw.c
run_job J7b-membw 900 bash -c '
    for t in 1 2 4 8; do
        echo "== ${t} threads =="
        '"$GUARD"' run -- taskset -c 0-7 /tmp/membw "$t" 128 60
    done
'

# J5: the traced decode split, which names the dominant term and its 1/(1-s) ceiling.
if wait_mem 2600; then
    run_job J5-phase-split 1800 "${SCOPE[@]}" "$SPLIT" --model "$MODEL" --pack "$PACK" --steps 24 --json
fi

# J6: the shipped Q6_K head alone, then the trunk, on the same pair.
if wait_mem 2600; then
    run_job J6-head-time 1800 "${SCOPE[@]}" "$HEAD" "$MODEL" "$PACK"
fi

# J8: the 2k T1 baseline, three runs at four threads.
head -c 8000 "$DATA" >/tmp/opencode/nq8b-2k-prompt.txt
if wait_mem 2600; then
    run_job J8-decode-2k 3600 bash -c '
        p=$(cat /tmp/opencode/nq8b-2k-prompt.txt)
        for i in 1 2 3; do
            echo "== run $i =="
            systemd-run --user --scope -q -p MemoryMax=3200M -p MemorySwapMax=0 -- \
                '"$GEN"' --model '"$MODEL"' --pack '"$PACK"' --tokenizer '"$TOK"' \
                --prompt "$p" --n 32 --kernel q8act
        done
    '
fi

# J2: clippy over the real tree.
run_job J2-clippy 2400 "$GUARD" run -- cargo clippy --workspace --all-features --all-targets --locked -j 1 -- -D warnings

# J3: the filtered workspace sweep. The three glslc targets stay red on this host.
run_job J3-tests 5400 "$GUARD" run -- cargo test --workspace --all-features --locked --no-fail-fast -- --skip gpu --skip spd_inverse_matches_serial

# J4: the exact reader on the conservative pair. Slow; it is the quality oracle for the
# product kernel's 1.499225 ratio.
if wait_mem 2600; then
    run_job J4-exact-ppl 7200 "${SCOPE[@]}" "$PPL" \
        --model "$MODEL" --pack "$PACK" --tokenizer "$TOK" --data "$DATA" \
        --tokens 2048 --skip 0 --ctx 2048 --head-batch 16 --kernel exact --second-half
fi

log_status "queue done epoch=$(date +%s)"
