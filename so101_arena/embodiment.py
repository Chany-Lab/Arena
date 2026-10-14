"""SO-101 embodiment for Isaac Lab-Arena, built on Isaac-Envs' SO101_HIGH_PD_CFG.

Actions (6): relative end-effector pose (dx, dy, dz, rx, ry, rz) in the robot base frame,
solved by Isaac Lab's damped-least-squares differential IK over the five arm joints, plus a
binary gripper command (>= 0 open, < 0 close). A zero action holds the arm still with the
gripper open, so Arena's ``zero_action`` policy is meaningful.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.contrib.stack.mdp.observations import ee_frame_pos, ee_frame_quat

from isaac_envs.robots import SO101_HIGH_PD_CFG

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TCP_LINK = "gripper_frame_link"
GRIPPER_OPEN = 1.0  # rad (joint range -10..100 deg)
GRIPPER_CLOSED = -0.15


@configclass
class SO101SceneCfg:
    robot: ArticulationCfg | None = None

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/" + TCP_LINK, name="end_effector"),
        ],
    )

    def __post_init__(self):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


@configclass
class SO101IKActionsCfg:
    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        body_name=TCP_LINK,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            # A 5-DOF arm cannot hold a straight-down gripper across the desk (wrist_flex hits its
            # 95 deg limit), so orientation is a soft goal and joints are kept off their limits.
            orientation_weight=0.15,
            joint_limit_avoidance_gain=1.0,
        ),
        scale=1.0,
    )
    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        open_command_expr={"gripper": GRIPPER_OPEN},
        close_command_expr={"gripper": GRIPPER_CLOSED},
    )


@configclass
class SO101ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel)
        eef_pos = ObsTerm(func=ee_frame_pos)
        eef_quat = ObsTerm(func=ee_frame_quat)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class SO101EventCfg:
    reset_joints = EventTerm(
        func=mdp_isaac_lab.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot")},
    )


@register_asset
class SO101IKEmbodiment(EmbodimentBase):
    """SO-101 follower arm with relative-pose differential IK and a binary gripper."""

    name = "so101_ik"
    tags = ["embodiment"]
    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = SO101SceneCfg()
        robot = SO101_HIGH_PD_CFG.copy()
        robot.prim_path = "{ENV_REGEX_NS}/Robot"
        # Start with the gripper open so a zero action (= "open") does not move it.
        robot.init_state.joint_pos = dict(robot.init_state.joint_pos, gripper=GRIPPER_OPEN)
        self.scene_config.robot = robot
        self.action_config = SO101IKActionsCfg()
        self.observation_config = SO101ObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.event_config = SO101EventCfg()

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "ee_frame"

    def get_command_body_name(self) -> str:
        return TCP_LINK
