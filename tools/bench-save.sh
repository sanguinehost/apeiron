#!/usr/bin/env bash
# The one path a measured figure takes from a run to a document.
#   bench-save.sh gpu <model.gguf> <label> [rounds]   interleaved apeiron/llama-bench GPU rounds; writes and registers a receipt
#   bench-save.sh register <file>... [-- summary]     index receipts another tool wrote (engine_profile writes bench/cpu/*)
#   bench-save.sh verify                              fail on an unregistered, edited or unmirrored receipt, or a figure that
#                                                     differs from its receipt or is missing from the document quoting it
# Every writer indexes in docs/evidence/ledger.jsonl (sha256) and copies to $APEIRON_RECEIPT_MIRROR
# (default ~/Backups/apeiron-receipts) at the same relative path. Figures: docs/evidence/figures.json.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
LEDGER=docs/evidence/ledger.jsonl
FIGURES=docs/evidence/figures.json
MIRROR=${APEIRON_RECEIPT_MIRROR:-$HOME/Backups/apeiron-receipts}
LLAMA=${LLAMA_BENCH_CUDA:-$HOME/Workspace/llama.cpp/build-cuda/bin/llama-bench}
APEIRON=tools/gpu-bench/target/release/gpu-bench
sha() { sha256sum "$1" | cut -d' ' -f1; }

register() {
  local summary=$1; shift
  for p in "$@"; do
    local r; r=$(realpath --relative-to="$ROOT" "$p")
    mkdir -p "$MIRROR/$(dirname "$r")"; cp -p "$p" "$MIRROR/$r"
    [ "$(sha "$p")" = "$(sha "$MIRROR/$r")" ] || { echo "mirror copy differs: $r" >&2; exit 1; }
    if ! jq -e --arg p "$r" 'select(.path==$p)' "$LEDGER" >/dev/null 2>&1 || [ -z "$(jq -r --arg p "$r" 'select(.path==$p)|.id' "$LEDGER")" ]; then
      jq -cS -n --arg id "bench-$(basename "${r%.*}")-${r##*.}" --arg p "$r" --arg s "$(sha "$p")" --arg m "$summary" \
        --argjson b "$(stat -c %s "$p")" '{bytes:$b,id:$id,kind:"receipt",path:$p,sha256:$s,summary:$m}' >> "$LEDGER"
      echo "registered $r"
    fi
  done
}

quiet() { # load1 < 2 and GPU util < 10% before every round
  for _ in $(seq 12); do
    local load util
    load=$(cut -d' ' -f1 /proc/loadavg)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    if awk -v l="$load" 'BEGIN{exit !(l<2)}' && [ "$util" -lt 10 ]; then return 0; fi
    sleep 5
  done
  echo "host never went quiet (load1 $load, gpu util $util%)" >&2; exit 1
}

gpu() {
  local model; model=$(realpath "$1"); local label=$2 rounds=${3:-5} epoch; epoch=$(date +%s)
  local d=docs/evidence/receipts/bench; mkdir -p "$d"
  local out=$d/$epoch-$label.json rows=()
  for i in $(seq 0 $((rounds-1))); do
    quiet
    local apn apg lln llg
    apn=$("$APEIRON" "$model" --prefill 512 --reps 3 --carrier q4 | sed -n 's/.*BENCH carrier=.* pp512: mean \([0-9.]*\) tok\/s.*/\1/p')
    lln=$("$LLAMA" -m "$model" -ngl 99 -p 512 -n 0 -r 3 -o json | jq '.[0].avg_ts')
    quiet
    apg=$("$APEIRON" "$model" --gen 128 --reps 3 --carrier q4 | sed -n 's/.*BENCH carrier=.* tg128: mean \([0-9.]*\) tok\/s.*/\1/p')
    llg=$("$LLAMA" -m "$model" -ngl 99 -p 0 -n 128 -r 3 -o json | jq '.[0].avg_ts')
    rows+=("$(jq -cn --argjson i "$i" --argjson a "$apn" --argjson l "$lln" --argjson ag "$apg" --argjson lg "$llg" \
      '{round:$i,apeiron_pp:$a,llama_pp:$l,ratio_pp:($a/$l),apeiron_tg:$ag,llama_tg:$lg,ratio_tg:($ag/$lg)}')")
    echo "round $i: ${rows[-1]}"
  done
  local dirty; dirty=$(git status --porcelain --untracked-files=no | wc -l)
  printf '%s\n' "${rows[@]}" | jq -s \
    --arg label "$label" --arg cmd "tools/bench-save.sh gpu $model $label $rounds" --arg model "$model" --arg msha "$(sha "$model")" \
    --arg git "$(git rev-parse HEAD)" --argjson dirty "$dirty" --arg diff "$(git diff HEAD | sha256sum | cut -d' ' -f1)" \
    --arg abin "$(sha "$APEIRON")" --arg lbin "$(sha "$LLAMA")" --arg lc "$("$LLAMA" -m "$model" -ngl 99 -p 1 -n 0 -r 1 -o json | jq -r '.[0].build_commit')" \
    --arg gpu "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)" --arg cpu "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)" \
    --arg rustc "$(rustc --version)" --argjson epoch "$epoch" '
    def med(k): (map(.[k])|sort) as $v | ($v|length) as $n | if $n%2==1 then $v[($n-1)/2] else ($v[$n/2-1]+$v[$n/2])/2 end;
    def st(k): {median: med(k), min: (map(.[k])|min), max: (map(.[k])|max)};
    {schema:"apeiron.bench-gpu.v2",label:$label,epoch:$epoch,command:$cmd,model:$model,model_sha256:$msha,carrier:"q4",pp:512,tg:128,
     git:$git,dirty_paths:$dirty,diff_sha256:$diff,apeiron_binary_sha256:$abin,llama_binary_sha256:$lbin,llama_commit:$lc,
     gpu:$gpu,cpu:$cpu,rustc:$rustc,gate:"load1<2 and gpu util<10% before every round",
     stat:{apeiron_pp:st("apeiron_pp"),llama_pp:st("llama_pp"),ratio_pp:st("ratio_pp"),apeiron_tg:st("apeiron_tg"),llama_tg:st("llama_tg"),ratio_tg:st("ratio_tg")},
     rounds:.}' > "$out"
  jq -r '"| metric (tok/s) | median | min | max |\n|---|---:|---:|---:|", (.stat|to_entries[]|"| \(.key) | \(.value.median) | \(.value.min) | \(.value.max) |")' "$out" > "${out%.json}.md.tmp"
  { echo "# GPU benchmark \`$label\`"; echo; jq -r '"- command `\(.command)`\n- model `\(.model)` (sha256 `\(.model_sha256)`)\n- git `\(.git)` with \(.dirty_paths) dirty paths; diff sha256 `\(.diff_sha256)`\n- \(.gpu); \(.cpu); \(.rustc)\n- llama.cpp `\(.llama_commit)` (`llama-bench -ngl 99`); apeiron `gpu_decode_bench --carrier q4`\n- \(.rounds|length) interleaved rounds, gate: \(.gate)"' "$out"; echo; cat "${out%.json}.md.tmp"; } > "${out%.json}.md"
  rm "${out%.json}.md.tmp"
  register "GPU $label: apeiron gpu_decode_bench against llama-bench -ngl 99, $rounds interleaved rounds." "$out" "${out%.json}.md"
  cat "${out%.json}.md"
}

verify() {
  local bad=0
  while IFS= read -r p; do
    case $p in *.json|*.md|*.py) ;; *) continue ;; esac
    local reg; reg=$(jq -r --arg p "$p" 'select(.path==$p)|.sha256' "$LEDGER")
    if [ -z "$reg" ]; then echo "unregistered: $p"; bad=1
    elif [ "$reg" != "$(sha "$p")" ]; then echo "edited since registration: $p"; bad=1; fi
    if [ ! -f "$MIRROR/$p" ] || [ "$(sha "$MIRROR/$p")" != "$(sha "$p")" ]; then echo "not mirrored: $p"; bad=1; fi
  done < <(find bench docs/evidence/receipts/bench -type f | sort)
  if [ -f "$FIGURES" ]; then
    while IFS=$'\t' read -r name rec key digits text docs; do
      [ -f "$rec" ] || { echo "figure $name: receipt $rec missing"; bad=1; continue; }
      local v
      case $rec in
        *.md) v=$(grep -m1 -F "| $key |" "$rec" | cut -d'|' -f3 | xargs) ;;
        *) v=$(jq -r ".$key" "$rec") ;;
      esac
      [ -n "$v" ] || { echo "figure $name: no '$key' in $rec"; bad=1; continue; }
      local fmt; fmt=$(LC_ALL=en_US.UTF-8 printf "%'.${digits}f" "$v" 2>/dev/null || printf "%.${digits}f" "$v")
      [ "$fmt" = "$text" ] || { echo "figure $name: receipt says $fmt, figures.json says $text"; bad=1; }
      for d in $docs; do grep -qF -- "$text" "$d" || { echo "figure $name: $text not in $d"; bad=1; }; done
    done < <(jq -r 'to_entries[]|[.key,.value.receipt,.value.key,.value.digits,.value.text,(.value.docs|join(" "))]|@tsv' "$FIGURES")
  fi
  [ $bad = 0 ] && echo "bench-save verify: ok"
  exit $bad
}

case ${1:-} in
  gpu) shift; gpu "$@" ;;
  register) shift; s="engine_profile run: apeiron against llama-bench, paired rounds, quiet-host gated."
    files=(); while [ $# -gt 0 ]; do if [ "$1" = "--" ]; then s=$2; break; fi; files+=("$1"); shift; done
    register "$s" "${files[@]}" ;;
  verify) verify ;;
  *) sed -n 2,9p "$0"; exit 2 ;;
esac
