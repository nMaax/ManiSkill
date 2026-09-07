"""PushCubeHard-v1: a hardened single-cube extension of PushCube-v1.

Features:
- Uniform spawn region: cube spawns anywhere within [CUBE_SPAWN_X_RANGE] x [CUBE_SPAWN_Y_RANGE]
- Unlocked initial z-rotation: cube yaw is randomized uniformly in [-pi, pi]
- Fixed goal region: marked target zone at (TARGET_XY) on the table
- Pre-closed gripper: finger joints initialize at 0.0 on episode reset to eliminate dead demo frames
"""

from typing import Any, Union

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


@register_env("PushCubeHard-v1", max_episode_steps=150)
class PushCubeHardEnv(BaseEnv):
    """
    **Task Description:**
    A hardened single-cube pushing task where the objective is to push a cube with arbitrary
    initial yaw rotation from a uniform spawn region into a fixed target goal area.

    **Randomizations:**
    - the cube's xy position is randomized uniformly in [-0.15, 0.00] x [-0.12, 0.12]
    - the cube's z-rotation (yaw) is randomized uniformly in [-pi, pi]
    - the target goal region is fixed at (0.15, 0.0) on the table

    **Success Conditions:**
    - the cube's xy position is within GOAL_RADIUS (0.08m) of the target's xy position,
      and the cube remains resting on the table.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]

    agent: Union[Panda, Fetch]

    CUBE_HALF_SIZE = 0.02
    GOAL_RADIUS = 0.08
    CUBE_SPAWN_X_RANGE = (-0.15, 0.00)
    CUBE_SPAWN_Y_RANGE = (-0.12, 0.12)
    TARGET_XY = (0.15, 0.0)

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

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
        pose = sapien_utils.look_at(eye=[0.55, 0.0, 0.45], target=[0.03, 0.0, 0.03])
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

        self.obj = actors.build_cube(
            self.scene,
            half_size=self.CUBE_HALF_SIZE,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALF_SIZE]),
        )

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.GOAL_RADIUS,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[self.TARGET_XY[0], self.TARGET_XY[1], 1e-3]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Pre-close fingers so episodes start without dead frames
            qpos = self.agent.robot.get_qpos()
            if self.robot_uids == "panda":
                qpos[:, -2:] = 0.0
            self.agent.reset(qpos)

            # Sample cube XY position uniformly
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = (
                torch.rand(b) * (self.CUBE_SPAWN_X_RANGE[1] - self.CUBE_SPAWN_X_RANGE[0])
                + self.CUBE_SPAWN_X_RANGE[0]
            )
            xyz[:, 1] = (
                torch.rand(b) * (self.CUBE_SPAWN_Y_RANGE[1] - self.CUBE_SPAWN_Y_RANGE[0])
                + self.CUBE_SPAWN_Y_RANGE[0]
            )
            xyz[:, 2] = self.CUBE_HALF_SIZE

            # Sample cube orientation with unlocked z-rotation (yaw)
            q = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=False
            )
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=q))

            # Fixed goal region
            target_xyz = torch.zeros((b, 3))
            target_xyz[:, 0] = self.TARGET_XY[0]
            target_xyz[:, 1] = self.TARGET_XY[1]
            target_xyz[:, 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(
                self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.GOAL_RADIUS
        ) & (self.obj.pose.p[..., 2] < self.CUBE_HALF_SIZE + 5e-3)

        return {
            "success": is_obj_placed,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                goal_pos=self.goal_region.pose.p,
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_region.pose.p - self.obj.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        delta_xy = self.goal_region.pose.p[..., :2] - self.obj.pose.p[..., :2]
        dist_xy = torch.linalg.norm(delta_xy, axis=1, keepdim=True)
        push_dir = delta_xy / torch.clamp(dist_xy, min=1e-6)

        standoff_xy = self.obj.pose.p[..., :2] - push_dir * (self.CUBE_HALF_SIZE + 0.005)
        standoff_z = self.obj.pose.p[..., 2:]
        standoff_p = torch.cat([standoff_xy, standoff_z], dim=-1)

        tcp_to_push_pose = standoff_p - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)

        reached = tcp_to_push_pose_dist < 0.01
        obj_to_goal_dist = torch.linalg.norm(delta_xy, axis=1)
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward = reward + place_reward * reached

        z_deviation = torch.abs(self.obj.pose.p[..., 2] - self.CUBE_HALF_SIZE)
        z_reward = 1 - torch.tanh(5 * z_deviation)
        reward = reward + place_reward * z_reward * reached

        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0
