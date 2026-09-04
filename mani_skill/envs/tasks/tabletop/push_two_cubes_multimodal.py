"""
PushTwoCubesMultimodal-v1: a multimodal extension of PushTwoCubes-v1.

Inherits from PushTwoCubesEnv and extends it to support 4 distinct modalities:
- straight matching (A -> goalA, B -> goalB) vs criss-cross matching (A -> goalB, B -> goalA)
- starting from cubeA vs starting from cubeB
- cubes spawn with unlocked z-rotation (random yaw)

Success requires both cubes to be placed in distinct goal regions (either straight or criss-cross).
Dense rewards adaptively reward whichever pairing the agent is progressing toward.
"""

from typing import Any

import numpy as np
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.push_two_cubes import PushTwoCubesEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array


@register_env("PushTwoCubesMultimodal-v1", max_episode_steps=750)
class PushTwoCubesMultimodalEnv(PushTwoCubesEnv):
    """
    **Task Description:**
    Multimodal extension of PushTwoCubes. Two cubes (cubeA and cubeB) must be pushed into the two
    goal regions (goal_regionA on the robot's right, goal_regionB on the robot's left).
    Unlike PushTwoCubes-v1, either straight pairing (A -> A, B -> B) or criss-cross pairing
    (A -> B, B -> A) is accepted as success. Cubes spawn with unlocked z-rotation (random yaw).

    **Success Conditions:**
    - Straight: cubeA in goal_regionA and cubeB in goal_regionB, OR
    - Criss-cross: cubeA in goal_regionB and cubeB in goal_regionA.
    - Both cubes must remain flat on the table.
    """

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Start with the fist already closed
            qpos = self.agent.robot.get_qpos()
            qpos[..., -2:] = 0.0
            self.agent.reset(qpos)

            # Spawn cubes with unlocked z-rotation (random yaw)
            for cube, lane_y in ((self.cubeA, -self.LANE_Y), (self.cubeB, self.LANE_Y)):
                xyz = torch.zeros((b, 3))
                jitter = (torch.rand((b, 2)) * 2 - 1) * self.SPAWN_JITTER
                xyz[..., 0] = self.SPAWN_CENTER_X + jitter[..., 0]
                xyz[..., 1] = lane_y + jitter[..., 1]
                xyz[..., 2] = self.CUBE_HALF_SIZE
                cube_qs = randomization.random_quaternions(
                    b, lock_x=True, lock_y=True, device=self.device
                )
                cube.set_pose(Pose.create_from_pq(p=xyz, q=cube_qs))

            # The goal regions are fixed
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

    def evaluate(self):
        is_A_at_A = self._is_placed(self.cubeA, self.goal_regionA)
        is_A_at_B = self._is_placed(self.cubeA, self.goal_regionB)
        is_B_at_A = self._is_placed(self.cubeB, self.goal_regionA)
        is_B_at_B = self._is_placed(self.cubeB, self.goal_regionB)

        pairing_straight = is_A_at_A & is_B_at_B
        pairing_cross = is_A_at_B & is_B_at_A

        return {
            "is_cubeA_placed": is_A_at_A | is_A_at_B,
            "is_cubeB_placed": is_B_at_A | is_B_at_B,
            "pairing_straight": pairing_straight,
            "pairing_cross": pairing_cross,
            "success": pairing_straight | pairing_cross,
        }

    def _stage_reward(self, cube, goal):
        """Direction-aware dense reward for pushing `cube` to `goal`, in [0, 3]."""
        delta_xy = goal.pose.p[..., :2] - cube.pose.p[..., :2]
        dist_xy = torch.linalg.norm(delta_xy, axis=1, keepdim=True) + 1e-6
        push_dir = delta_xy / dist_xy

        standoff_xy = cube.pose.p[..., :2] - push_dir * (self.CUBE_HALF_SIZE + 0.005)
        standoff_z = cube.pose.p[..., 2:]
        standoff_p = torch.cat([standoff_xy, standoff_z], dim=-1)

        tcp_to_push_pose = standoff_p - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)

        reached = tcp_to_push_pose_dist < 0.01
        obj_to_goal_dist = torch.linalg.norm(delta_xy, axis=1)
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward = reward + place_reward * reached

        z_deviation = torch.abs(cube.pose.p[..., 2] - self.CUBE_HALF_SIZE)
        z_reward = 1 - torch.tanh(5 * z_deviation)
        reward = reward + place_reward * z_reward * reached
        return reward

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        # Evaluate reward for straight pairing
        rA_straight = self._stage_reward(self.cubeA, self.goal_regionA)
        rB_straight = self._stage_reward(self.cubeB, self.goal_regionB)
        is_A_at_A = self._is_placed(self.cubeA, self.goal_regionA)
        is_B_at_B = self._is_placed(self.cubeB, self.goal_regionB)

        reward_straight = torch.where(
            is_A_at_A,
            3.0 + rB_straight,
            torch.where(
                is_B_at_B,
                3.0 + rA_straight,
                torch.maximum(rA_straight, rB_straight),
            ),
        )

        # Evaluate reward for cross pairing
        rA_cross = self._stage_reward(self.cubeA, self.goal_regionB)
        rB_cross = self._stage_reward(self.cubeB, self.goal_regionA)
        is_A_at_B = self._is_placed(self.cubeA, self.goal_regionB)
        is_B_at_A = self._is_placed(self.cubeB, self.goal_regionA)

        reward_cross = torch.where(
            is_A_at_B,
            3.0 + rB_cross,
            torch.where(
                is_B_at_A,
                3.0 + rA_cross,
                torch.maximum(rA_cross, rB_cross),
            ),
        )

        reward = torch.maximum(reward_straight, reward_cross)
        reward[info["success"]] = 8.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        max_reward = 8.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
