"""SO-101 cube-to-tray pick-and-place on Isaac-Envs' desk, as an external Arena environment.

Layout follows Isaac-Envs' sims/envs/lift_cube.py: desk top at 0.75 m, cube 28 cm in front of
the robot, tray to its right, robot on the desk facing +Y.

    python isaaclab_arena/evaluation/policy_runner.py --policy_type zero_action --num_episodes 3 \
        --external_environment_class_path so101_arena.env:SO101CubeToTrayEnvironment so101_cube_to_tray
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaac_envs.props import CUBE_CFG, DESK_CFG, TRAY_CFG

import so101_arena.embodiment  # noqa: F401  (registers "so101_ik")
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

DESK_TOP = 0.75
ROBOT_POS = (0.0, -0.2, DESK_TOP)
ROBOT_YAW = math.pi / 2
CUBE_POS = (0.0, 0.08, DESK_TOP)
TRAY_POS = (0.2, 0.0, DESK_TOP)

_TRAY_RIGID_USDA = Path(__file__).with_name("tray_rigid.usda")


def _write_rigid_tray_usd() -> str:
    """Wrap Isaac-Envs' static tray as a kinematic rigid body (Arena's contact filter needs one)."""
    text = f'''#usda 1.0
(
    defaultPrim = "tray"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "tray" (
    prepend references = @{TRAY_CFG.spawn.usd_path}@
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    bool physics:kinematicEnabled = 1
    float physics:mass = 0.3
}}
'''
    if not _TRAY_RIGID_USDA.exists() or _TRAY_RIGID_USDA.read_text() != text:
        _TRAY_RIGID_USDA.write_text(text)
    return str(_TRAY_RIGID_USDA)


def _yaw_quat_xyzw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


class SO101CubeToTrayEnvironment(ExampleEnvironmentBase):

    name: str = "so101_cube_to_tray"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.assets.background import Background
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose

        identity = (0.0, 0.0, 0.0, 1.0)
        desk = Background(
            name="desk",
            usd_path=DESK_CFG.spawn.usd_path,
            object_min_z=DESK_TOP - 0.2,
            initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=identity),
        )
        cube = Object(
            name="cube",
            usd_path=CUBE_CFG.spawn.usd_path,
            object_type=ObjectType.RIGID,
            initial_pose=Pose(position_xyz=CUBE_POS, rotation_xyzw=identity),
        )
        tray = Object(
            name="tray",
            usd_path=_write_rigid_tray_usd(),
            object_type=ObjectType.RIGID,
            initial_pose=Pose(position_xyz=TRAY_POS, rotation_xyzw=identity),
        )
        light = self.asset_registry.get_asset_by_name("light")()

        embodiment = self.asset_registry.get_asset_by_name("so101_ik")(enable_cameras=args_cli.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=ROBOT_POS, rotation_xyzw=_yaw_quat_xyzw(ROBOT_YAW)))

        task = PickAndPlaceTask(
            pick_up_object=cube,
            destination_location=tray,
            background_scene=desk,
            episode_length_s=args_cli.episode_length_s,
        )
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[desk, cube, tray, light]),
            task=task,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--episode_length_s", type=float, default=20.0)  # the scripted policy needs ~12 s
