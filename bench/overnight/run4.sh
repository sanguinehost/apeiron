#!/usr/bin/env bash
# Fourth overnight queue: the Q4 bar on this host, window 1 and window 2, then the same-host
# ratio. Waits for the earlier queues' lock. Requires the completed Q4 file from the download.
#
# The 7.587145 bar was measured on a rented EPYC host. Re-running window 1 here removes the
# cross-host caveat only if this host reproduces it; if it does not, both windows are reported
# as this-host numbers and the window-1 ratio is restated from this host's own Q4 run.
set -u

ROOT=/home/lucas/Workspace/apeiron-nq
cd "$ROOT" || exit 1
TS=1794912865
LOGS="bench/cpu/$TS-nq8b-overnight-logs"
DONE="bench/overnight/done"
STATUS="bench/overnight/status.txt"
MODEL=/home/lucas/models/arelath/nq8b/skeleton-q6k-q4kembed.gguf
PACK=/home/lucas/models/arelath/nq8b-late-v4/model.nqp
Q4=/home/lucas/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf
Q4_SHA=d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785
Q4_BYTES=5027783488
TOK=/home/lucas/models/arelath/nq8b/tokenizer.json
DATA=/home/lucas/models/wikitext2/wikitext-2-raw/wiki.test.raw
PPL="$ROOT/target/release/examples/ppl"
SCOPE=(systemd-run --user --scope -q -p MemoryMax=3600M -p MemorySwapMax=0 --)

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
    local need_mb=${1:-3400} waited=0 avail
    while [ "$waited" -lt 3600 ]; do
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

log_status "run4 start epoch=$(date +%s)"

# R2: the file must be complete and the bar's own sha before any PPL run reads it.
run_job R2-q4-sha 600 bash -c '
    f='"$Q4"'
    b=$(stat -c %s "$f" 2>/dev/null || echo 0)
    echo "bytes=$b expected='"$Q4_BYTES"'"
    [ "$b" = '"$Q4_BYTES"' ] || { echo "size mismatch"; exit 1; }
    s=$(sha256sum "$f" | cut -d" " -f1)
    echo "sha256=$s"
    [ "$s" = '"$Q4_SHA"' ] || { echo "sha mismatch"; exit 1; }
'

if [ -e "$DONE/R2-q4-sha" ] && wait_mem 3400; then
    run_job Q4W1-ppl 10800 "${SCOPE[@]}" "$PPL" \
        --model "$Q4" --tokenizer "$TOK" --data "$DATA" \
        --tokens 2048 --skip 0 --ctx 2048 --head-batch 16 --second-half
fi

if [ -e "$DONE/Q4W1-ppl" ] && wait_mem 3400; then
    run_job Q4W2-ppl 10800 "${SCOPE[@]}" "$PPL" \
        --model "$Q4" --tokenizer "$TOK" --data "$DATA" \
        --tokens 2048 --skip 2048 --ctx 2048 --head-batch 16 --second-half
fi

# NQ window 2 on the same host, if run3 did not reach it.
if [ ! -e "$DONE/W2048-ppl" ] && wait_mem 2600; then
    run_job W2048-ppl 3600 "${SCOPE[@]}" "$PPL" \
        --model "$MODEL" --pack "$PACK" --tokenizer "$TOK" --data "$DATA" \
        --tokens 2048 --skip 2048 --ctx 2048 --head-batch 16 --kernel q8act --second-half
fi

python3 - "$LOGS" "$STATUS" <<'PY'
import re, sys
from pathlib import Path
logs, status = Path(sys.argv[1]), Path(sys.argv[2])

def ppl(name):
    f = logs / f"{name}.log"
    if not f.exists():
        return None
    m = re.findall(r"^ppl ([\d.]+)", f.read_text(), re.M)
    return float(m[-1]) if m else None

q4w1, q4w2 = ppl("Q4W1-ppl"), ppl("Q4W2-ppl")
nqw1, nqw2 = 11.374839, ppl("W2048-ppl")

rows = []
if q4w1:
    rows.append(f"samehost window1 q4={q4w1:.6f} nq={nqw1:.6f} ratio={nqw1/q4w1:.6f} "
                f"bar={1.5*q4w1:.6f} margin={1.5*q4w1-nqw1:+.6f}")
if q4w1 and q4w2:
    rows.append(f"samehost window2 q4={q4w2:.6f} nq={nqw2:.6f} ratio={nqw2/q4w2:.6f} "
                f"bar={1.5*q4w2:.6f} margin={1.5*q4w2-nqw2:+.6f}" if nqw2 else
                f"samehost window2 q4={q4w2:.6f} nq=unmeasured")
if q4w1:
    rows.append(f"host-reproduction q4_window1_here={q4w1:.6f} vs rented={7.587145:.6f} "
                f"delta={q4w1-7.587145:+.6f} ({100*(q4w1/7.587145-1):+.3f}%)")
with status.open("a") as f:
    for r in rows:
        f.write("Q2 " + r + "\n")
PY

log_status "run4 done epoch=$(date +%s)"
