#!/usr/bin/env bash
# Install the repository gates as git hooks.
#
#   tools/hooks/install.sh              install (refuses to overwrite another path)
#   tools/hooks/install.sh --force      install, replacing an existing hooksPath
#   tools/hooks/install.sh --check      report what is configured, change nothing
#   tools/hooks/install.sh --uninstall  remove core.hooksPath
#
# The hooks live in `.githooks/` (tracked, reviewable); this script only points git at them with
# `core.hooksPath`. It commits nothing and never installs into `.git/hooks`.

set -uo pipefail

repo=$(git rev-parse --show-toplevel 2>/dev/null) || {
  echo "install: not inside a git work tree" >&2
  exit 1
}
cd "$repo" || exit 1

mode=install
case "${1:-}" in
  "") ;;
  --check) mode=check ;;
  --uninstall) mode=uninstall ;;
  --force) mode=force ;;
  *) echo "install: unknown argument '$1' (--check | --uninstall | --force)" >&2; exit 2 ;;
esac

current=$(git config --get core.hooksPath 2>/dev/null || true)
want=.githooks

report() {
  echo "install: repo        $repo"
  echo "install: hooks path  ${current:-<unset>}"
  for hook in pre-commit commit-msg pre-push; do
    if [ -x "$want/$hook" ]; then
      echo "install: $hook  present, executable"
    else
      echo "install: $hook  MISSING or not executable in $want/" >&2
    fi
  done
}

case "$mode" in
  check)
    report
    [ "$current" = "$want" ] && exit 0
    echo "install: core.hooksPath is '${current:-<unset>}', expected '$want'" >&2
    exit 1
    ;;
  uninstall)
    if [ -z "$current" ]; then
      echo "install: core.hooksPath is unset; nothing to remove"
      exit 0
    fi
    git config --unset core.hooksPath
    echo "install: removed core.hooksPath (was '$current')"
    exit 0
    ;;
  install)
    if [ -n "$current" ] && [ "$current" != "$want" ]; then
      echo "install: core.hooksPath is already '$current'; this repository ships '$want'." >&2
      echo "install: pass --force to replace it" >&2
      report
      exit 1
    fi
    ;;
esac

missing=0
for hook in pre-commit commit-msg pre-push; do
  [ -x "$want/$hook" ] || { echo "install: $want/$hook is missing or not executable" >&2; missing=1; }
done
if (( missing )); then
  echo "install: refusing to point git at an incomplete hook directory" >&2
  exit 1
fi

git config core.hooksPath "$want"
current=$(git config --get core.hooksPath)
echo "install: core.hooksPath = $current"
report
cat <<'TXT'
install: tiers enforced
  pre-commit  cargo fmt --all -- --check; python3 tools/check-claims.py
  commit-msg  Signed-off-by line (git commit -s)
  pre-push    clippy -D warnings; cargo test --lib --test table
  CI / barrier  cargo test --workspace --all-features --locked, proof/regen.sh --check,
                cd proof/corpus && lake build
install: `git commit --no-verify` bypasses every hook; CI runs the same gates.
TXT
