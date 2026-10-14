#!/usr/bin/env bash
# Evaluates a policy on the SO-101 cube-to-tray environment with Arena's policy runner.
#
#   ./run.sh                                   scripted policy, 10 episodes
#   ./run.sh --num_envs 4 --num_episodes 8     parallel envs
#   POLICY=zero_action ./run.sh                another policy (registered name or module.Class)
#
# Extra flags go to Arena's policy runner. Results land in outputs/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA="$ROOT/external/IsaacLab-Arena"
PLAYGROUND_DIR="${PLAYGROUND_DIR:-$ROOT/../Playground}"
POLICY="${POLICY:-so101_arena.scripted_policy.SO101ScriptedPickPlace}"

[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && { sed -n 2,9p "$0"; exit 0; }

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export PYTHONPATH="$ARENA:$ROOT"
mkdir -p "$ROOT/outputs"
cd "$ARENA"
exec env -u VIRTUAL_ENV "$PLAYGROUND_DIR/.venv/bin/python" isaaclab_arena/evaluation/policy_runner.py \
  --policy_type "$POLICY" --num_episodes 10 --output_base_dir "$ROOT/outputs" \
  --external_environment_class_path so101_arena.env:SO101CubeToTrayEnvironment \
  "$@" so101_cube_to_tray
