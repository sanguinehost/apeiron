#!/usr/bin/env bash
# The v1 PPL runner (`bench/PROTOCOL.md` §3). One fixed window list, every arm, full windows.
#
#   PPL_BIN=... DATA=... TOK=... OUT=... \
#   ARMS="nq=/path/skeleton.gguf:/path/model.nqp q4=/path/Qwen3-8B-Q4_K_M.gguf" \
#   bench/run-ppl.sh screen
#
# `screen` is the five-window ladder, `confirm` the seventeen-window list. No `--second-half`; the
# paper's loop and llama.cpp both score all seqlen-1 positions.
set -eu

case "${1:-}" in
    screen)  SKIPS="0 2048 4096 6144 8192" ;;
    confirm) SKIPS="0 2048 4096 6144 8192 10240 22016 33792 45568 57344 69120 80896 92672
                    104448 116224 128000 139776" ;;
    *) echo "usage: $0 screen|confirm" >&2; exit 2 ;;
esac

: "${PPL_BIN:?}" "${DATA:?}" "${TOK:?}" "${OUT:?}" "${ARMS:?}"
THREADS="${APEIRON_THREADS:-$(nproc)}"
mkdir -p "$OUT"

for arm in $ARMS; do
    name=${arm%%=*}
    spec=${arm#*=}
    model=${spec%%:*}
    pack=""
    case "$spec" in *:*) pack=${spec#*:} ;; esac
    [ -s "$model" ] || { echo "missing model: $model" >&2; exit 1; }
    [ -z "$pack" ] || [ -s "$pack" ] || { echo "missing pack: $pack" >&2; exit 1; }
    mkdir -p "$OUT/$name"
    for s in $SKIPS; do
        log="$OUT/$name/$s.log"
        [ -s "$log" ] && grep -q '^ppl ' "$log" && continue
        args=(--model "$model" --tokenizer "$TOK" --data "$DATA"
              --tokens 2048 --skip "$s" --ctx 2048 --head-batch 16)
        [ -z "$pack" ] || args+=(--pack "$pack" --kernel q8act)
        APEIRON_THREADS="$THREADS" "$PPL_BIN" "${args[@]}" >"$log" 2>&1
        grep -q '^ppl ' "$log" || { echo "arm $name skip $s: no ppl line" >&2; exit 1; }
    done
    echo "arm $name done: $(grep -h '^ppl ' "$OUT/$name"/*.log | wc -l) windows"
done
