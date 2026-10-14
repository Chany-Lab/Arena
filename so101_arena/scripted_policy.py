"""Scripted (privileged-state) pick-and-place for the SO-101 cube-to-tray environment.

Reads cube/tray poses from the scene and drives the relative-pose IK action through
approach -> descend -> close -> lift -> carry -> lower -> open -> retreat, keeping the
gripper pointing straight down (its spawn orientation).
"""

from __future__ import annotations

from dataclasses import dataclass

import os

import torch

import isaaclab.utils.math as math_utils

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

TCP_LINK = "gripper_frame_link"
CUBE_HALF = 0.02
# Where a grasped cube's centre sits in the TCP (gripper_frame_link) frame: the closed jaws meet
# ~1.4 cm to the TCP's -x side and the moving jaw presses the cube onto the fixed jaw, so the
# cube centre is ~3.4 cm along -x (measured from the SO-101 USD at rest).
GRASP_IN_TCP = (-0.034, 0.0, 0.0)

# phase: (name, z offset of the TCP target above the reference point, gripper cmd, min steps, max steps)
PHASES = [
    ("approach", 0.08, +1.0, 0, 60),
    ("descend", 0.0, +1.0, 0, 40),
    ("close", 0.0, -1.0, 15, 15),
    ("lift", 0.10, -1.0, 0, 40),
    ("carry", 0.10, -1.0, 0, 80),
    ("lower", 0.03, -1.0, 0, 40),
    ("open", 0.03, +1.0, 15, 15),
    ("retreat", 0.12, +1.0, 1000, 1000),
]
TRAY_PHASES = {"carry", "lower", "open", "retreat"}


def _t(x):
    return x.torch if hasattr(x, "torch") else x


@dataclass
class SO101ScriptedPickPlaceCfg(PolicyCfg):
    max_step_m: float = 0.03
    """Largest TCP translation per control step (m)."""
    reach_tol_m: float = 0.006
    """Position error at which a phase counts as reached (m)."""
    grasp_z_offset_m: float = 0.005
    """TCP height above the cube centre when grasping (m)."""


@register_policy
class SO101ScriptedPickPlace(PolicyBase[SO101ScriptedPickPlaceCfg]):
    name = "so101_scripted_pick_place"

    def __init__(self, config: SO101ScriptedPickPlaceCfg):
        super().__init__(config)
        self.phase = None
        self.count = None
        self.q_des_b = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self.phase is None:
            return
        if env_ids is None:
            self.phase.zero_()
            self.count.zero_()
        else:
            self.phase[env_ids] = 0
            self.count[env_ids] = 0

    def get_action(self, env, observation) -> torch.Tensor:
        u = env.unwrapped
        robot = u.scene["robot"]
        n = u.num_envs
        dev = u.device
        if self.phase is None:
            self.phase = torch.zeros(n, dtype=torch.long, device=dev)
            self.count = torch.zeros(n, dtype=torch.long, device=dev)

        tcp_idx = robot.find_bodies(TCP_LINK)[0][0]
        root_pos = _t(robot.data.root_pos_w)
        root_quat = _t(robot.data.root_quat_w)
        tcp_pos_w = _t(robot.data.body_pos_w)[:, tcp_idx]
        tcp_quat_w = _t(robot.data.body_quat_w)[:, tcp_idx]
        tcp_pos_b, tcp_quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, tcp_pos_w, tcp_quat_w)
        if self.q_des_b is None:
            # Spawn pose points the gripper straight down; hold that orientation throughout.
            self.q_des_b = tcp_quat_b.clone()

        cube_w = _t(u.scene["cube"].data.root_pos_w).clone()
        cube_w[:, 2] += CUBE_HALF + self.config.grasp_z_offset_m  # root is the bottom face
        tray_w = _t(u.scene["tray"].data.root_pos_w).clone()
        tray_w[:, 2] += 0.005 + CUBE_HALF  # floor top + half cube

        z_off = torch.tensor([p[1] for p in PHASES], device=dev)[self.phase]
        grip = torch.tensor([p[2] for p in PHASES], device=dev)[self.phase]
        on_tray = torch.tensor([p[0] in TRAY_PHASES for p in PHASES], device=dev)[self.phase]
        ref_w = torch.where(on_tray.unsqueeze(1), tray_w, cube_w)
        target_w = ref_w.clone()
        target_w[:, 2] += z_off
        # Put the TCP where the cube centre lands between the jaws (horizontal part only).
        grasp_off_w = math_utils.quat_apply(tcp_quat_w, torch.tensor(GRASP_IN_TCP, device=dev).expand(n, 3))
        target_w[:, :2] -= grasp_off_w[:, :2]
        target_b, _ = math_utils.subtract_frame_transforms(root_pos, root_quat, target_w)

        dpos = target_b - tcp_pos_b
        err = dpos.norm(dim=1)
        dpos = dpos * torch.clamp(self.config.max_step_m / (err + 1e-9), max=1.0).unsqueeze(1)
        q_err = math_utils.quat_mul(self.q_des_b.expand(n, 4), math_utils.quat_conjugate(tcp_quat_b))
        drot = torch.clamp(math_utils.axis_angle_from_quat(q_err), -0.1, 0.1)

        if os.environ.get("SO101_TRACE") and int(self.count[0]) % 5 == 0:
            print(f"TRACE phase={PHASES[int(self.phase[0])][0]} n={int(self.count[0])} err={float(err[0]):.4f}"
                  f" tcp_w={[round(v, 3) for v in tcp_pos_w[0].tolist()]} tgt_w={[round(v, 3) for v in target_w[0].tolist()]}"
                  f" cube_w={[round(v, 3) for v in cube_w[0].tolist()]} q={[round(v, 2) for v in _t(robot.data.joint_pos)[0].tolist()]}")
        # Hold still while the jaws close or open, so the arm does not chase a cube being pushed.
        hold = torch.tensor([p[0] in ("close", "open") for p in PHASES], device=dev)[self.phase].unsqueeze(1)
        dpos = torch.where(hold, torch.zeros_like(dpos), dpos)
        drot = torch.where(hold, torch.zeros_like(drot), drot)

        # Advance phases.
        self.count += 1
        mins = torch.tensor([p[3] for p in PHASES], device=dev)[self.phase]
        maxs = torch.tensor([p[4] for p in PHASES], device=dev)[self.phase]
        done = ((err < self.config.reach_tol_m) & (self.count >= mins)) | (self.count >= maxs)
        advance = done & (self.phase < len(PHASES) - 1)
        self.phase = self.phase + advance.long()
        self.count = torch.where(advance, torch.zeros_like(self.count), self.count)

        return torch.cat([dpos, drot, grip.unsqueeze(1)], dim=1)
