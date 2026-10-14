#!/usr/bin/env bash
# Checks out Isaac Lab-Arena at the pinned commit, applies the Isaac Lab 3.0 patch and checks that
# Playground's Isaac Sim environment is there. Safe to rerun.
#
#   ./setup.sh           set up
#   ./setup.sh --check   only report
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA="$ROOT/external/IsaacLab-Arena"
PATCH="$ROOT/patches/isaaclab-3.0-d061bf35.patch"
PLAYGROUND_DIR="${PLAYGROUND_DIR:-$ROOT/../Playground}"
PY="$PLAYGROUND_DIR/.venv/bin/python"

CHECK=0
case "${1:-}" in
  "") ;;
  --check) CHECK=1 ;;
  -h | --help) sed -n 2,7p "$0"; exit 0 ;;
  *) echo "unknown flag: $1" >&2; exit 2 ;;
esac

ok() { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s\n' "$1"; FAILED=1; }
FAILED=0

if [ "$CHECK" = 0 ]; then
  git -C "$ROOT" submodule update --init external/IsaacLab-Arena
  if git -C "$ARENA" apply --reverse --check "$PATCH" 2>/dev/null; then
    :
  else
    git -C "$ARENA" apply "$PATCH"
  fi
fi

echo "==> Status"
[ -f "$ARENA/isaaclab_arena/__init__.py" ] && ok "Isaac Lab-Arena $(git -C "$ARENA" rev-parse --short HEAD)" || bad "Isaac Lab-Arena not checked out"
git -C "$ARENA" apply --reverse --check "$PATCH" 2>/dev/null && ok "Isaac Lab 3.0 patch applied" || bad "patch not applied (run ./setup.sh)"
if [ -x "$PY" ]; then
  ok "Playground env $PY"
else
  bad "no Playground env at $PY: clone Playground and run its ./setup.sh, or set PLAYGROUND_DIR"
fi
command -v nvidia-smi >/dev/null && ok "nvidia driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" || bad "nvidia-smi not found"

[ "$FAILED" = 0 ] && echo "Ready: ./run.sh --help" || exit 1
