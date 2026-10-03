#!/usr/bin/env bash
# Install the repository gates as git hooks.
#
#   tools/hooks/install.sh            install (refuses to overwrite another path)
#   tools/hooks/install.sh --force    install, replacing an existing hooksPath
#   tools/hooks/install.sh --check    report what is configured, change nothing
#   tools/hooks/install.sh --uninstall  remove core.hooksPath
#
# The hooks live in `.githooks/` in the repository (tracked, reviewable); this
# script only points git at them with `core.hooksPath`. It commits nothing, and
# it never installs into `.git/hooks`.

set -uo pipefail

repo=$(git rev-parse --show-toplevel 2>/dev/null) || {
  echo "install: not inside a git work tree" >&2
  exit 1
}
cd "$repo" || exit 1

mode=install
case "${1:-}" in
  "") mode=install ;;
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
  echo "install: hook files  $(ls "$want" 2>/dev/null | tr '\n' ' ')"
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
    if [ "$current" = "$want" ]; then exit 0; fi
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
    echo "install: the gates remain: cargo fmt, check-claims, board reconcile run from the barrier"
    exit 0
    ;;
  install)
    if [ -n "$current" ] && [ "$current" != "$want" ]; then
      echo "install: core.hooksPath is already '$current'; this repository ships '$want'." >&2
      echo "install: refusing to overwrite another hook directory; pass --force to replace it" >&2
      report
      exit 1
    fi
    ;;
esac

missing=0
for hook in pre-commit pre-push; do
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
cat <<'EOF'
install: tiers now enforced
  pre-commit  formatting          cargo fmt --all -- --check
  pre-commit  citation resolution python3 tools/check-claims.py
  pre-commit  board declaration carried by the staged change
  commit-msg  board declaration   python3 tools/hooks/lane.py message \$1
  pre-push    clippy, the named test selection, and the board reconcile (reported)
  barrier     cargo test --workspace --all-features, proof/regen.sh --check,
              cd proof/corpus && lake build   (documented, not wired into a hook)
install: the board reconcile needs OPERATELY_API_KEY in the environment, or
install: APEIRON_HOOKS_BOARD_SOURCE=<schema-1 document> for an offline commit.
install: `git commit --no-verify` bypasses a hook, as it bypasses every hook;
install: the wave barrier is the authority, so a bypassed commit cannot close a wave.
EOF