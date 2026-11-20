#!/usr/bin/env bash
# The v1 0-shot runner (`bench/PROTOCOL.md` §4): six tasks, both arms, shared-prompt scorer.
#
#   QUALITY_BIN=... TOK=... ITEMS=... OUT=... \
#   ARMS="nq=skeleton.gguf:model.nqp q4=Qwen3-8B-Q4_K_M.gguf" \
#   bench/run-screen.sh [limit]
#
# The limit is the per-task item count: `100` selects the even-indexed half of the 200-item file
# (the set the v0 screen scored), `200` or omitted selects the full file. Both arms read the same file.
set -eu
: "${QUALITY_BIN:?}" "${TOK:?}" "${ITEMS:?}" "${OUT:?}" "${ARMS:?}"
LIMIT="${1:-}"
N="${LIMIT:-200}"
TASKS="boolq piqa hellaswag winogrande arc_easy arc_challenge"
mkdir -p "$OUT"

for arm in $ARMS; do
    name=${arm%%=*}
    spec=${arm#*=}
    model=${spec%%:*}
    pack=""
    case "$spec" in *:*) pack=${spec#*:} ;; esac
    mkdir -p "$OUT/$name"
    for task in $TASKS; do
        out="$OUT/$name/$task.jsonl"
        items="$ITEMS/$task-$N.jsonl"
        [ -s "$out" ] && continue
        [ -s "$items" ] || { echo "missing items: $items" >&2; exit 1; }
        args=(--model "$model" --tokenizer "$TOK" --items "$items"
              --out "$out" --head-batch 16)
        [ -z "$pack" ] || args+=(--pack "$pack" --kernel q8act)
        APEIRON_THREADS="${APEIRON_THREADS:-$(nproc)}" "$QUALITY_BIN" "${args[@]}"
    done
    echo "arm $name done"
done
