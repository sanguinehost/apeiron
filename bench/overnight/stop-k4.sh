#!/usr/bin/env bash
# Stop the K4 2k decode job after its three 4-thread runs, so the queued Q2 windows (run7) are not
# blocked behind a 1-thread prefill that duplicates a cheaper measurement.
#
# Why: the 1- and 2-thread *decode* rates are the thread-scaling check, and `phase_split` gives them
# in ~25 s per width with no 30-minute 2k prefill. The 2k prefill rate at 1 thread is a T2 number,
# not a T1 one, and T2's turn is refused by 4x already. Killing the job leaves its log intact and
# lets run5 continue to K5 (the Q6_K equality test).
set -u

ROOT=/home/lucas/Workspace/apeiron-nq
cd "$ROOT" || exit 1
LOG=bench/cpu/1794912865-nq8b-overnight-logs/K4-decode-2k.log
STATUS=bench/overnight/status.txt

log_status() {
    printf '%s\n' "$*" >>"$STATUS"
}

for _ in $(seq 1 240); do
    n=$(grep -c '^decode ' "$LOG" 2>/dev/null || true)
    n=${n:-0}
    if [ "$n" -ge 3 ]; then
        # Every process of the K4 job, widest pattern first. The scope client and the binary are
        # killed by name; the `timeout` wrapper and its `bash -c` by the prompt path in their argv,
        # which no other job names. Killing all of them releases the queue's inherited flock fd.
        pkill -9 -f 'release/examples/generate --model' 2>/dev/null
        pkill -9 -f 'timeout 3600 bash -c' 2>/dev/null
        pkill -9 -f 'nq8b-2k-prompt.txt' 2>/dev/null
        log_status "stop K4 after 3x4t: 1t/2t 2k runs cancelled; thread scaling -> run8 (phase_split)"
        exit 0
    fi
    if ! kill -0 253129 2>/dev/null; then
        exit 0
    fi
    sleep 20
done
log_status "stop-k4 watcher timed out after 80 min without 3 decode lines"
