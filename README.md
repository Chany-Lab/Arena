# Arena

SO-101 policy evaluation in [Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) on the Isaac Sim 6.2 stack

<br>

### Contents

| Path | Content |
| :--- | :--- |
| `so101_arena/embodiment.py` | `so101_ik`: SO-101 (Isaac-Envs `SO101_HIGH_PD_CFG`), relative-pose IK + binary gripper |
| `so101_arena/env.py` | `so101_cube_to_tray`: desk, cube, tray in the `lift_cube` layout, Arena `PickAndPlaceTask` |
| `so101_arena/scripted_policy.py` | Scripted pick and place from the cube and tray poses |
| `patches/` | Arena fixes for Isaac Lab 3.0 @ `d061bf35` (Franka cfg names, `slice` env ids, cloner API) |
| `external/IsaacLab-Arena` | Arena `main` @ `c3e547b` (v0.3, alpha) |

### Results (RTX 5070 Laptop)

| Policy | Episodes | Success rate | Wall time |
| :--- | :--- | :--- | :--- |
| `zero_action` | 5 | 0.0 | 66 s |
| scripted | 10 | 1.0 | 90 s |
| scripted, `--num_envs 4` | 8 | 1.0 | 27 s |

No randomization yet: these show the loop works, not how robust a policy is.

## Setup Guide

Runs on [Playground](https://github.com/Chany-Lab/Playground)'s environment (Isaac Sim 6.2, Isaac Lab 3.0); nothing is installed into it.

```bash
git clone https://github.com/Chany-Lab/Arena.git && cd Arena   # no --recurse-submodules: Arena's own submodules use SSH URLs
./setup.sh                      # Arena checkout + patch; PLAYGROUND_DIR defaults to ../Playground

./run.sh                        # scripted policy, 10 episodes
./run.sh --num_envs 4 --num_episodes 8
POLICY=zero_action ./run.sh
```

Results go to `outputs/` (per-episode JSONL and an HTML report). Arena pins its own Isaac Lab;
drop the patch once Arena moves to Isaac Lab 3.0 at or after `d061bf35`.
