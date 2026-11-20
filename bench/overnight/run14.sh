#!/usr/bin/env bash
# Fourteenth queue: the decisive PPL span, windows 6-25, sampled across the whole test file.
#
# Why this replaces the 0-shot screen for tonight. The per-window ratios on windows 1-3 are 1.499225,
# 1.467112 and 1.646477, and the pooled ratio over those 3,069 scored positions is 1.535661 — above
# the 1.5 bar. The two-window pass was window selection: window 1 is where the Q4 arm's nll is
# highest relative to the pack's. The window-level error of the pooled gap is what decides it, and
# that needs windows, not a 28 h 0-shot screen. If the PPL half fails, Q2 fails and the screen is
# premature; if it passes, the screen is the only open part and runs next.
#
# **Systematic, not contiguous.** The first version used skips 10,240..49,152 (a contiguous 41 k-token
# block). Window 6 (skip 10,240) then measured NQ PPL 5.450666 against 11-17 on the first five: the
# first 10 k tokens are much harder than the next 2 k, so a contiguous block would inherit the start's
# difficulty. The 20 new windows are spaced 11,776 tokens apart from 10,240 to 233,984, covering the
# whole file. The five base windows are a 10 k-token block (4% of the file) but carry 20% of the
# sample weight; the over-weight is bounded by 0.004 nats even if the start's gap is 0.025 above the
# rest, well inside the ±0.025-nat interval the span is built to quote.
#
# Interleaved per skip, so every completed skip adds a *paired* window; a partial run still pools.
# Each job is ~7.5 min (Q4) and ~9.4 min (NQ); 20 skips is ~5.6 h.
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
TOK=/home/lucas/models/arelath/nq8b/tokenizer.json
DATA=/home/lucas/models/wikitext2/wikitext-2-raw/wiki.test.raw
PPL="$ROOT/target/release/examples/ppl"
SCOPE=(systemd-run --user --scope -q -p MemoryMax=3600M -p MemorySwapMax=0 --)

# Windows 6-25: 20 windows across the test file, 11,776 tokens apart.
NEW_SKIPS='10240 22016 33792 45568 57344 69120 80896 92672 104448 116224
128000 139776 151552 163328 175104 186880 198656 210432 222208 233984'

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
    log_status "mem refused after ${waited}s at ${avail}MiB < ${need_mb}MiB"
    return 1
}

log_status "run14 restart epoch=$(date +%s) :: systematic skips"

for skip in $NEW_SKIPS; do
    if wait_mem 3400; then
        run_job "NQ${skip}-ppl" 3600 "${SCOPE[@]}" "$PPL" \
            --model "$MODEL" --pack "$PACK" --tokenizer "$TOK" --data "$DATA" \
            --tokens 2048 --skip "$skip" --ctx 2048 --head-batch 16 --kernel q8act --second-half
    fi
    if wait_mem 3600; then
        run_job "Q4W${skip}-ppl" 10800 "${SCOPE[@]}" "$PPL" \
            --model "$Q4" --tokenizer "$TOK" --data "$DATA" \
            --tokens 2048 --skip "$skip" --ctx 2048 --head-batch 16 --second-half
    fi
done

# Pooled report over every window that has both arms, then the window-level t interval on the gap.
python3 - "$LOGS" "$STATUS" "$NEW_SKIPS" <<'PY'
import math, re, sys
from pathlib import Path

logs, status = Path(sys.argv[1]), Path(sys.argv[2])
DEPTH = Path("bench/cpu/1794896874-nq8b-depth-logs")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
        23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
        30: 2.042}


def nll(path):
    if not path.exists():
        return None
    m = re.findall(r"mean nll ([\d.]+)", path.read_text(), re.M)
    return float(m[-1]) if m else None


NEW = [int(x) for x in sys.argv[3].split()]

# Windows 1-5 predate the naming scheme: 1 is the depth receipt, 2-5 are W/Q4W by skip.
nq_names = {0: DEPTH / "ppl-late-v4-q6k-q4kembed-ctx2048.log",
            2048: logs / "W2048-ppl.log",
            4096: logs / "W4096-ppl.log",
            6144: logs / "W6144-ppl.log",
            8192: logs / "W8192-ppl.log"}
q4_names = {0: logs / "Q4W1-ppl.log",
            2048: logs / "Q4W2-ppl.log",
            4096: logs / "Q4W4096-ppl.log",
            6144: logs / "Q4W6144-ppl.log",
            8192: logs / "Q4W8192-ppl.log"}
for skip in NEW:
    nq_names[skip] = logs / f"NQ{skip}-ppl.log"
    q4_names[skip] = logs / f"Q4W{skip}-ppl.log"

pairs = []
for skip in sorted(nq_names):
    a, b = nll(nq_names[skip]), nll(q4_names[skip])
    if a is not None and b is not None:
        pairs.append((skip, a, b))

with status.open("a") as out:
    if len(pairs) < 2:
        out.write(f"pooled Q2: {len(pairs)} paired windows, not enough\n")
        sys.exit(0)
    gaps = [a - b for _, a, b in pairs]
    mean_gap = sum(gaps) / len(gaps)
    var = sum((g - mean_gap) ** 2 for g in gaps) / (len(gaps) - 1)
    se = math.sqrt(var / len(gaps))
    crit = T975.get(len(gaps) - 1, 1.96)
    half = crit * se
    ratio = math.exp(mean_gap)
    lo, hi = math.exp(mean_gap - half), math.exp(mean_gap + half)
    per = [math.exp(a - b) for _, a, b in pairs]
    out.write(f"pooled Q2 windows={len(pairs)} positions={1023 * len(pairs)} "
              f"gap={mean_gap:.6f} ratio={ratio:.6f} ci95=[{lo:.6f},{hi:.6f}] "
              f"per-window=[{min(per):.6f},{max(per):.6f}] bar=1.5 "
              f"verdict={'over' if lo > 1.5 else 'under' if hi < 1.5 else 'undecided'}\n")
    out.write("pooled skips " + " ".join(str(s) for s, _, _ in pairs) + "\n")
PY

log_status "run14 done epoch=$(date +%s)"
