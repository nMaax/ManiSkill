"""
PushTwoCubes-v1: a deliberately simple two-object extension of PushCube-v1.

The task is PushCube run twice, side by side, with everything that could introduce
multimodality stripped out: the two goal regions are at fixed positions, each cube spawns
in its own narrow lane with only a small positional jitter, and the intended solution order
is always cubeA first, then cubeB. See CUSTOM_ENVS.md for why this does not reuse any of
the PushBlock-v1 machinery (that task is order-agnostic on purpose; this one is not).
"""

from typing import Any, Union

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


@register_env("PushTwoCubes-v1", max_episode_steps=300)
class PushTwoCubesEnv(BaseEnv):
    """
    **Task Description:**
    PushCube with two cubes and two goal regions. The blue cube (A) must be pushed into the
    goal region on the robot's right, the green cube (B) into the goal region on the robot's
    left. Both pushes run along +x, away from the robot, exactly as in PushCube-v1.

    **Randomizations:**
    - each cube's xy position is jittered uniformly by +-SPAWN_JITTER around its own fixed lane
      centre. The lanes are 2 * LANE_Y apart, so a cube never leaves its lane.
    - the two goal regions are at fixed positions and are never randomized.
    - cube orientation is not randomized.

    **Success Conditions:**
    - cubeA is within GOAL_RADIUS of goal_regionA and cubeB is within GOAL_RADIUS of
      goal_regionB (each measured in xy), and both cubes are still flat on the table.
      The pairing is fixed: cubeA only ever counts for goal_regionA.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]

    agent: Union[Panda, Fetch]

    # --- geometry (metres, table frame; the Panda base sits at x = -0.615, +y is the robot's left)
    CUBE_HALF_SIZE = 0.02
    GOAL_RADIUS = 0.08
    # lane centres: cubeA/goalA at y = -LANE_Y (robot's right), cubeB/goalB at y = +LANE_Y
    LANE_Y = 0.16
    SPAWN_CENTER_X = -0.05
    SPAWN_JITTER = 0.03
    TARGET_X = 0.15

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # PushCube's attribute names, kept so code written against PushCube still reads
    @property
    def goal_radius(self):
        return self.GOAL_RADIUS

    @property
    def cube_half_size(self):
        return self.CUBE_HALF_SIZE

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        # aimed at the workspace centroid rather than PushCube's single-cube framing, so both
        # lanes (|y| <= 0.24) and both goal discs fill the 128x128 frame
        pose = sapien_utils.look_at(eye=[0.55, 0, 0.45], target=[0.03, 0, 0.03])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=1.2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        # a symmetric head-on view so neither lane is favoured in the recorded videos
        pose = sapien_utils.look_at([0.62, 0.0, 0.45], [0.03, 0.0, 0.03])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1.1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cubeA = actors.build_cube(
            self.scene,
            half_size=self.CUBE_HALF_SIZE,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cubeA",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, -self.LANE_Y, self.CUBE_HALF_SIZE]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=self.CUBE_HALF_SIZE,
            color=np.array([12, 160, 42, 255]) / 255,
            name="cubeB",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, self.LANE_Y, self.CUBE_HALF_SIZE]),
        )

        self.goal_regionA = actors.build_red_white_target(
            self.scene,
            radius=self.GOAL_RADIUS,
            thickness=1e-5,
            name="goal_regionA",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, -self.LANE_Y, 1e-3]),
        )
        self.goal_regionB = actors.build_red_white_target(
            self.scene,
            radius=self.GOAL_RADIUS,
            thickness=1e-5,
            name="goal_regionB",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, self.LANE_Y, 1e-3]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            for cube, lane_y in ((self.cubeA, -self.LANE_Y), (self.cubeB, self.LANE_Y)):
                xyz = torch.zeros((b, 3))
                jitter = (torch.rand((b, 2)) * 2 - 1) * self.SPAWN_JITTER
                xyz[..., 0] = self.SPAWN_CENTER_X + jitter[..., 0]
                xyz[..., 1] = lane_y + jitter[..., 1]
                xyz[..., 2] = self.CUBE_HALF_SIZE
                cube.set_pose(Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0]))

            # the goal regions are fixed -- they do not follow the cubes as they do in PushCube.
            # rotating 90 degrees about y makes the disc face up.
            for goal, lane_y in (
                (self.goal_regionA, -self.LANE_Y),
                (self.goal_regionB, self.LANE_Y),
            ):
                target_xyz = torch.zeros((b, 3))
                target_xyz[..., 0] = self.TARGET_X
                target_xyz[..., 1] = lane_y
                target_xyz[..., 2] = 1e-3
                goal.set_pose(
                    Pose.create_from_pq(p=target_xyz, q=euler2quat(0, np.pi / 2, 0))
                )

    def _is_placed(self, cube, goal):
        return (
            torch.linalg.norm(cube.pose.p[..., :2] - goal.pose.p[..., :2], axis=1)
            < self.GOAL_RADIUS
        ) & (cube.pose.p[..., 2] < self.CUBE_HALF_SIZE + 5e-3)

    def evaluate(self):
        is_cubeA_placed = self._is_placed(self.cubeA, self.goal_regionA)
        is_cubeB_placed = self._is_placed(self.cubeB, self.goal_regionB)
        return {
            "is_cubeA_placed": is_cubeA_placed,
            "is_cubeB_placed": is_cubeB_placed,
            "success": is_cubeA_placed & is_cubeB_placed,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                goalA_pos=self.goal_regionA.pose.p,
                goalB_pos=self.goal_regionB.pose.p,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def _stage_reward(self, cube, goal):
        """PushCube's dense reward for a single cube/goal pair, in [0, 3]."""
        tcp_push_pose = Pose.create_from_pq(
            p=cube.pose.p
            + torch.tensor([-self.CUBE_HALF_SIZE - 0.005, 0, 0], device=self.device)
        )
        tcp_to_push_pose = tcp_push_pose.p - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)

        reached = tcp_to_push_pose_dist < 0.01
        obj_to_goal_dist = torch.linalg.norm(
            cube.pose.p[..., :2] - goal.pose.p[..., :2], axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward = reward + place_reward * reached

        z_deviation = torch.abs(cube.pose.p[..., 2] - self.CUBE_HALF_SIZE)
        z_reward = 1 - torch.tanh(5 * z_deviation)
        reward = reward + place_reward * z_reward * reached
        return reward

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        # staged so the reward itself encodes the fixed A-then-B order: the cubeB stage only
        # becomes reachable once cubeA is in its goal, and it starts from cubeA's full 3.0.
        reward = self._stage_reward(self.cubeA, self.goal_regionA)
        reward = torch.where(
            info["is_cubeA_placed"],
            3.0 + self._stage_reward(self.cubeB, self.goal_regionB),
            reward,
        )
        reward[info["success"]] = 8.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        max_reward = 8.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
