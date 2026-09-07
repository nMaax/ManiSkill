from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("UnstackCube-v1", max_episode_steps=250)
class UnstackCubeEnv(BaseEnv):
    """
    **Task Description:**
    The red cube starts stacked on top of the green cube. The goal is to pick up the red
    cube and move it off to the side, placing it on the table away from the green cube.

    **Randomizations:**
    - the xy position of the stack (both cubes together) is randomized on top of the table
    - the shared z-axis rotation of the stack is randomized

    **Success Conditions:**
    - the red cube is resting on the table (not on top of the green cube)
    - the red cube is sufficiently far away from the green cube in the xy plane
    - the red cube is static
    - the red cube is not being grasped by the robot (robot must let go of the cube)
    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    CUBE_HALF_SIZE = 0.02
    # minimum center-to-center xy distance between the cubes to count as "unstacked"
    SEPARATION_RADIUS = 0.08
    # if True, the stack's z-axis rotation is fixed (identity) instead of randomized
    LOCK_Z_ROTATION = False
    # cubeB must not drift further than this (in xy) from where it started
    CUBEB_DISTURB_THRESH = 0.01

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([self.CUBE_HALF_SIZE] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene,
            half_size=self.CUBE_HALF_SIZE,
            color=[1, 0, 0, 1],
            name="cubeA",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=self.CUBE_HALF_SIZE,
            color=[0, 1, 0, 1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[1, 0, 0.1]),
        )
        self.cubeB_init_p = torch.zeros((self.num_envs, 3), device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # sample a shared xy position and yaw for the whole stack
            stack_xy = torch.rand((b, 2)) * 0.2 - 0.1
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=self.LOCK_Z_ROTATION,
            )

            # cubeB is the bottom cube, resting on the table
            xyz_B = torch.zeros((b, 3))
            xyz_B[:, :2] = stack_xy
            xyz_B[:, 2] = self.CUBE_HALF_SIZE
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz_B, q=qs))

            # cubeA is the top cube, stacked directly on top of cubeB
            xyz_A = torch.zeros((b, 3))
            xyz_A[:, :2] = stack_xy
            xyz_A[:, 2] = self.CUBE_HALF_SIZE * 3
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz_A, q=qs))

            self.cubeB_init_p[env_idx] = self.cubeB.pose.p[env_idx]

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B

        xy_dist = torch.linalg.norm(offset[..., :2], axis=1)
        is_cubeA_far_from_cubeB = xy_dist >= self.SEPARATION_RADIUS
        is_cubeA_on_table = (
            torch.abs(pos_A[..., 2] - self.cube_half_size[2]) <= 0.005
        )
        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)

        # cubeB must be left (almost) undisturbed: only cubeA should move
        cubeB_disturbance = torch.linalg.norm(
            pos_B[:, :2] - self.cubeB_init_p[:, :2], axis=1
        )
        is_cubeB_disturbed = cubeB_disturbance > self.CUBEB_DISTURB_THRESH

        success = (
            is_cubeA_far_from_cubeB
            * is_cubeA_on_table
            * is_cubeA_static
            * (~is_cubeA_grasped)
            * (~is_cubeB_disturbed)
        )
        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeA_far_from_cubeB": is_cubeA_far_from_cubeB,
            "is_cubeA_on_table": is_cubeA_on_table,
            "is_cubeA_static": is_cubeA_static,
            "cubeB_disturbance": cubeB_disturbance,
            "is_cubeB_disturbed": is_cubeB_disturbed,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))

        # grasp and carry-away reward: push cubeA's xy far from cubeB's xy
        # while keeping it above the table (so it can clear cubeB on the way)
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        xy_dist = torch.linalg.norm(cubeA_pos[:, :2] - cubeB_pos[:, :2], axis=1)
        away_reward = 1 - torch.tanh(5.0 * torch.clamp(self.SEPARATION_RADIUS - xy_dist, min=0))

        is_cubeA_grasped = info["is_cubeA_grasped"]
        reward[is_cubeA_grasped] = (4 + away_reward)[is_cubeA_grasped]

        # ungrasp and static reward, once cubeA is far enough from cubeB and on the table
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )  # NOTE: hard-coded with panda
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~is_cubeA_grasped] = 1.0
        v = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        av = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        is_cubeA_placed = info["is_cubeA_far_from_cubeB"] * info["is_cubeA_on_table"]
        reward[is_cubeA_placed] = (
            6 + (ungrasp_reward + static_reward) / 2.0
        )[is_cubeA_placed]

        reward[info["success"]] = 8

        # penalize disturbing cubeB, regardless of stage
        reward = reward - torch.tanh(10.0 * info["cubeB_disturbance"])
        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
