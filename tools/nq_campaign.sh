#!/usr/bin/env bash
# One NanoQuant campaign, in the reference's order, as one command.
#
#   statistics -> dump -> block loop -> model KD -> pack -> score
#
# The stages are the ignored tests that own them; this driver only orders them, passes the
# calibration paths and refuses a wall the host cannot pay:
#
#   calib_diag_write   the mean input Gram diagonal per linear  (`tests/learn_calib.rs`)
#   calib_onorm_write  the mean squared output gradient per linear (`tests/learn_calib.rs`)
#   nq_dump_write      calib_ids.i32, ranks.json, perm.json, stats/ (`tests/learn_calib.rs`)
#   block_loop         nonfact tune, ADMM, fact tune, finalize, per block (`tests/learn_block_tune.rs`)
#   kd_fit             eight model-KD epochs, then the pack (`tests/learn_kd.rs`)
#   ppl                the served pack on one token stream (`examples/ppl.rs`)
#
# The block loop and the KD stage need a Vulkan device with cooperative matrices; the ADMM inside
# the block loop is `learn::nanoquant::step`, which is the same bits on the device and on the CPU.
# The wall gate is `tools/fit-budget.py`: one outer ADMM iteration over every fitted linear, times
# T. Measure `--iteration-seconds` on the host that will run it (the layer-0 preflight), not on a
# faster one and not from a per-layer figure.
#
# Usage:
#   tools/nq_campaign.sh --model <teacher.gguf> --skeleton <skeleton.gguf> \
#     --base <calib.kld> --cache <stats-dir> --dump <dump-dir> --loop <checkpoint-root> \
#     --tokenizer <tokenizer.json> --data <wiki.test.raw> \
#     --iteration-seconds <s> [--iters 400] [--cap-hours 24] [--stages stats,dump,block,kd,ppl]
#
# Reference defaults: 128 calibration samples of 2048 tokens drawn from wikitext2 **train**
# (`prepare_dataset`, `data_utils.py:206-248`; `--base` must hold 128x2048 tokens), `online`
# clipping, 2048-token attention chunks in the block loop, and the score on every 2048-token
# window of `--data` (`--tokens 0`). Every other flag has the tests' default. `--stages` is
# resumable: a stage whose output exists is skipped by its test (diagonals, o_norm) or by this
# driver (the pack, the score). Each heavy stage runs through `agent-guard` when it is executable
# (`$AGENT_GUARD`, default `~/Workspace/tools/agent-guard`); without it the command runs bare.
set -euo pipefail

cd "$(dirname "$0")/.."

model=""; skeleton=""; base=""; cache=""; dump=""; loop=""; tokenizer=""; data=""
tag="loop"; samples=128; strategy=online; shrink=0.4; ppm=1000000; iters=400; epochs=8
iteration_seconds=""; cap_hours=24; stages="stats,dump,block,kd,ppl"; nctx=2048; ctx=2048
tokens=0; head_batch=16; threads=0

while (($#)); do
  case $1 in
    --model) model=$2; shift 2 ;;
    --skeleton) skeleton=$2; shift 2 ;;
    --base) base=$2; shift 2 ;;
    --cache) cache=$2; shift 2 ;;
    --dump) dump=$2; shift 2 ;;
    --loop) loop=$2; shift 2 ;;
    --tokenizer) tokenizer=$2; shift 2 ;;
    --data) data=$2; shift 2 ;;
    --tag) tag=$2; shift 2 ;;
    --samples) samples=$2; shift 2 ;;
    --strategy) strategy=$2; shift 2 ;;
    --shrink) shrink=$2; shift 2 ;;
    --ppm) ppm=$2; shift 2 ;;
    --iters) iters=$2; shift 2 ;;
    --epochs) epochs=$2; shift 2 ;;
    --iteration-seconds) iteration_seconds=$2; shift 2 ;;
    --cap-hours) cap_hours=$2; shift 2 ;;
    --stages) stages=$2; shift 2 ;;
    --nctx) nctx=$2; shift 2 ;;
    --ctx) ctx=$2; shift 2 ;;
    --tokens) tokens=$2; shift 2 ;;
    --head-batch) head_batch=$2; shift 2 ;;
    --threads) threads=$2; shift 2 ;;
    -h | --help) sed -n '2,34p' "$0"; exit 0 ;;
    *) echo "unknown flag $1" >&2; exit 2 ;;
  esac
done

for v in model base cache dump loop tokenizer data; do
  if [[ -z ${!v} ]]; then
    echo "--$v is required" >&2
    exit 2
  fi
done
if [[ $strategy != online && $strategy != two_phase ]]; then
  echo "--strategy is online|two_phase, got $strategy" >&2
  exit 2
fi
pack=$loop/$tag/pack.nqp
want() { [[ ,$stages, == *,$1,* ]]; }
guard=${AGENT_GUARD:-$HOME/Workspace/tools/agent-guard}
run_guarded() {
  if [[ -x $guard ]]; then "$guard" run -- "$@"; else "$@"; fi
}

# The wall gate before the first linear: one outer iteration over every fitted linear, times T.
if want block; then
  if [[ -z $iteration_seconds ]]; then
    echo "--iteration-seconds is required for the block stage: measure it on this host" >&2
    exit 2
  fi
  python3 tools/fit-budget.py --iteration-seconds "$iteration_seconds" --iters "$iters" \
    --cap-hours "$cap_hours"
fi

stats=(
  "APEIRON_CALIB_MODEL=$model" "APEIRON_CALIB_BASE=$base" "APEIRON_CALIB_CACHE=$cache"
  "APEIRON_NQ_SAMPLES=$samples" "APEIRON_CALIB_STRATEGY=$strategy"
  "APEIRON_CALIB_SHRINK=$shrink"
)
dumpenv=(
  "${stats[@]}" "APEIRON_NQ_DUMP=$dump" "APEIRON_NQ_SAMPLES=$samples" "APEIRON_NQ_PPM=$ppm"
)
loopenv=(
  "APEIRON_NQ_MODEL=$model" "APEIRON_NQ_DUMP=$dump" "APEIRON_NQ_LOOP=$loop"
  "APEIRON_NQ_TAG=$tag" "APEIRON_NQ_BLOCKS=0..28" "APEIRON_NQ_SAMPLES=$samples"
  "APEIRON_NQ_NCTX=$nctx" "APEIRON_NQ_PACK=$pack"
)
kdenv=(
  "APEIRON_NQ_MODEL=$model" "APEIRON_KD_STUDENT=$loop/$tag" "APEIRON_NQ_DUMP=$dump"
  "APEIRON_NQ_SAMPLES=$samples" "APEIRON_KD_EPOCHS=$epochs" "APEIRON_NQ_PACK=$pack"
)
run_test() {
  local test=$1 name=$2; shift 2
  echo "== $name"
  if [[ -x $guard ]]; then
    env "$@" "$guard" run -- cargo test --release --locked --features gpu --test "$test" -- \
      --ignored --exact "$name" --nocapture
  else
    env "$@" cargo test --release --locked --features gpu --test "$test" -- \
      --ignored --exact "$name" --nocapture
  fi
}

if want stats; then
  run_test learn_calib calib_diag_write "${stats[@]}"
  run_test learn_calib calib_onorm_write "${stats[@]}"
fi
if want dump; then
  run_test learn_calib nq_dump_write "${dumpenv[@]}"
fi
if want block; then
  run_test learn_block_tune block_loop "${loopenv[@]}"
fi
if want kd; then
  run_test learn_kd kd_fit "${kdenv[@]}"
fi
if want ppl; then
  if [[ -z $skeleton ]]; then
    echo "--skeleton is required for the score stage: the teacher's directory with the Q8_0 head" >&2
    exit 2
  fi
  ppl=(target/release/examples/ppl --model "$skeleton" --pack "$pack" --tokenizer "$tokenizer"
    --data "$data" --tokens "$tokens" --ctx "$ctx" --head-batch "$head_batch" --kernel q8act)
  if ((threads > 0)); then
    echo "== ppl on $threads threads"
    run_guarded taskset -c "0-$((threads - 1))" env "APEIRON_THREADS=$threads" "${ppl[@]}"
  else
    echo "== ppl"
    run_guarded "${ppl[@]}"
  fi
fi
if [[ -f $pack ]]; then
  echo "== pack $pack $(stat -c%s "$pack") B $(sha256sum "$pack" | cut -d' ' -f1)"
fi
