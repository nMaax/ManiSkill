import torch

from mani_skill.envs.tasks.tabletop.stack_cube_clutter import StackCubeClutterEnv
from mani_skill.utils.registration import register_env

_SHAPE_TO_INT = {"cube": 0, "cylinder": 1, "sphere": 2}


@register_env("StackCubeClutterRandomPick-v1", max_episode_steps=250)
class StackCubeClutterRandomPickEnv(StackCubeClutterEnv):
    """
    **Task Description:**
    Same cluttered table as StackCubeClutter-v1, but the pick and target objects are drawn every
    episode from the full pool of objects present on the table -- cubeA, cubeB, and whichever
    clutter objects are active -- instead of always being cubeA and cubeB. Any two distinct
    active objects can be paired, e.g. stacking two clutter cubes on each other while cubeA
    and cubeB sit idle as scenery for that episode.

    **Randomizations:**
    - everything StackCubeClutter-v1 randomizes (cube/clutter positions, clutter count)
    - which two distinct active objects in the pool are drawn as the "pick" and "target"
      objects, resampled every episode independently per parallel environment

    **Success Conditions:**
    - the pick object is on top of the target object (within their combined resting geometry)
    - the pick object is static
    - the pick object is not being grasped by the robot
    """

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        self.pool_objects = [self.cubeA, self.cubeB] + self.clutter_objects
        pool_size = len(self.pool_objects)

        cube_rest_z = self.cube_half_size[2]
        cube_bounding_radius = torch.linalg.norm(self.cube_half_size[:2])
        self.pool_rest_z = torch.tensor(
            [cube_rest_z, cube_rest_z] + [spec.rest_z for spec in self.CLUTTER_SPECS],
            dtype=torch.float32,
            device=self.device,
        )
        self.pool_bounding_radius = torch.tensor(
            [cube_bounding_radius, cube_bounding_radius]
            + [spec.bounding_radius for spec in self.CLUTTER_SPECS],
            dtype=torch.float32,
            device=self.device,
        )
        self.pool_shape_type = torch.tensor(
            [0, 0] + [_SHAPE_TO_INT[spec.shape] for spec in self.CLUTTER_SPECS],
            dtype=torch.long,
            device=self.device,
        )
        cube_half_size = self.cube_half_size[0].item()
        sizes = [[cube_half_size, 0.0], [cube_half_size, 0.0]]
        for spec in self.CLUTTER_SPECS:
            if spec.shape == "cube":
                sizes.append([spec.half_size, 0.0])
            elif spec.shape == "cylinder":
                sizes.append([spec.radius, spec.half_length])
            else:
                sizes.append([spec.radius, 0.0])
        self.pool_size = torch.tensor(sizes, dtype=torch.float32, device=self.device)
        assert len(sizes) == pool_size

        self.pick_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            pool_size = len(self.pool_objects)
            # cubeA/cubeB (first two pool slots) are always active; clutter slots follow
            # self.clutter_active, which StackCubeClutterEnv just populated above.
            pool_active = torch.cat(
                [
                    torch.ones(self.num_envs, 2, dtype=torch.bool, device=self.device),
                    self.clutter_active,
                ],
                dim=1,
            )
            # random keys, -1 for inactive slots so topk never selects them; the two largest
            # keys are both random and (up to float-collision odds) guaranteed distinct
            keys = torch.rand(b, pool_size)
            keys[~pool_active[env_idx]] = -1.0
            top2 = torch.topk(keys, k=2, dim=1).indices
            self.pick_idx[env_idx] = top2[:, 0]
            self.target_idx[env_idx] = top2[:, 1]

    # WARN: without this, replay on another backend silently re-draws
    # them due to a different RNG across backends
    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict["pick_idx"] = self.pick_idx.clone()
        state_dict["target_idx"] = self.target_idx.clone()
        return state_dict

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        super().set_state_dict(state, env_idx)
        if "pick_idx" in state and "target_idx" in state:
            pick_idx = state["pick_idx"].to(device=self.device, dtype=torch.long)
            target_idx = state["target_idx"].to(device=self.device, dtype=torch.long)
            if env_idx is None:
                self.pick_idx[:] = pick_idx
                self.target_idx[:] = target_idx
            else:
                self.pick_idx[env_idx] = pick_idx[env_idx]
                self.target_idx[env_idx] = target_idx[env_idx]

    def _gather_pool_scalar(self, values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """values: (num_envs, pool_size) -> (num_envs,), selecting values[i, idx[i]] per row."""
        return torch.gather(values, 1, idx[:, None]).squeeze(1)

    def _gather_pool_vector(self, values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """values: (num_envs, pool_size, D) -> (num_envs, D), selecting values[i, idx[i], :]."""
        d = values.shape[-1]
        return torch.gather(values, 1, idx[:, None, None].expand(-1, 1, d)).squeeze(1)

    def evaluate(self):
        poses = torch.stack([o.pose.raw_pose for o in self.pool_objects], dim=1)
        linvel = torch.stack([o.linear_velocity for o in self.pool_objects], dim=1)
        angvel = torch.stack([o.angular_velocity for o in self.pool_objects], dim=1)
        # is_static/is_grasping are Actor/Agent methods, not tensors: they can only be called
        # once per object, so the pool is stacked here rather than gathered from a tensor
        is_static = torch.stack(
            [o.is_static(lin_thresh=1e-2, ang_thresh=0.5) for o in self.pool_objects], dim=1
        )
        is_grasped = torch.stack(
            [self.agent.is_grasping(o) for o in self.pool_objects], dim=1
        )

        pick_idx, target_idx = self.pick_idx, self.target_idx
        pick_pose = self._gather_pool_vector(poses, pick_idx)
        target_pose = self._gather_pool_vector(poses, target_idx)
        pick_linvel = self._gather_pool_vector(linvel, pick_idx)
        pick_angvel = self._gather_pool_vector(angvel, pick_idx)
        pick_static = self._gather_pool_scalar(is_static, pick_idx)
        pick_grasped = self._gather_pool_scalar(is_grasped, pick_idx)

        pick_rest_z = self.pool_rest_z[pick_idx]
        target_rest_z = self.pool_rest_z[target_idx]
        target_bounding_radius = self.pool_bounding_radius[target_idx]
        pick_shape_type = self.pool_shape_type[pick_idx]
        target_shape_type = self.pool_shape_type[target_idx]
        pick_size = self.pool_size[pick_idx]
        target_size = self.pool_size[target_idx]

        offset = pick_pose[:, :3] - target_pose[:, :3]
        xy_flag = torch.linalg.norm(offset[:, :2], axis=1) <= target_bounding_radius + 0.005
        z_flag = torch.abs(offset[:, 2] - (pick_rest_z + target_rest_z)) <= 0.005
        is_pick_on_target = torch.logical_and(xy_flag, z_flag)

        success = is_pick_on_target * pick_static * (~pick_grasped)

        return {
            "pick_idx": pick_idx,
            "target_idx": target_idx,
            "pool_poses": poses,
            "pick_pose": pick_pose,
            "target_pose": target_pose,
            "pick_linear_velocity": pick_linvel,
            "pick_angular_velocity": pick_angvel,
            "pick_rest_z": pick_rest_z,
            "target_rest_z": target_rest_z,
            "pick_shape_type": pick_shape_type,
            "target_shape_type": target_shape_type,
            "pick_size": pick_size,
            "target_size": target_size,
            "is_pick_grasped": pick_grasped,
            "is_pick_on_target": is_pick_on_target,
            "is_pick_static": pick_static,
            "success": success.bool(),
        }

    def compute_dense_reward(self, obs, action, info: dict):
        tcp_pos = self.agent.tcp.pose.p
        pick_pos = info["pick_pose"][:, :3]
        target_pos = info["target_pose"][:, :3]

        # reaching reward
        pick_to_tcp_dist = torch.linalg.norm(tcp_pos - pick_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * pick_to_tcp_dist))

        # grasp and place reward
        goal_xyz = torch.hstack(
            [
                target_pos[:, 0:2],
                (target_pos[:, 2] + info["target_rest_z"] + info["pick_rest_z"])[:, None],
            ]
        )
        pick_to_goal_dist = torch.linalg.norm(goal_xyz - pick_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * pick_to_goal_dist)

        reward[info["is_pick_grasped"]] = (4 + place_reward)[info["is_pick_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_pick_grasped = info["is_pick_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~is_pick_grasped] = 1.0

        v = torch.linalg.norm(info["pick_linear_velocity"], axis=1)
        av = torch.linalg.norm(info["pick_angular_velocity"], axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info["is_pick_on_target"]] = (
            6 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_pick_on_target"]]

        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(self, obs, action, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        if "state" in self.obs_mode:
            # cubeA/cubeB aren't special roles here (any pool member can be picked or
            # targeted), so replace the inherited asymmetric cubeA_pose/cubeB_pose +
            # obj_0..obj_{NUM_CLUTTER_SLOTS-1} naming with a uniform obj_0..obj_7 over the
            # whole pool instead of keeping both alongside each other
            obs.pop("cubeA_pose", None)
            obs.pop("cubeB_pose", None)
            for i in range(self.NUM_CLUTTER_SLOTS):
                obs.pop(f"obj_{i}_pose", None)
                obs.pop(f"obj_{i}_active", None)

            pool_size = len(self.pool_objects)
            env_range = torch.arange(self.num_envs, device=self.device)
            pool_active = torch.cat(
                [
                    torch.ones(self.num_envs, 2, dtype=torch.bool, device=self.device),
                    self.clutter_active,
                ],
                dim=1,
            )
            is_pick = torch.zeros(
                self.num_envs, pool_size, dtype=torch.bool, device=self.device
            )
            is_target = torch.zeros(
                self.num_envs, pool_size, dtype=torch.bool, device=self.device
            )
            is_pick[env_range, info["pick_idx"]] = True
            is_target[env_range, info["target_idx"]] = True

            # shape_type/size are per-slot content, not gathered role-only globals: with a
            # permutation/set-invariant encoder, slot index can't be relied on to carry
            # identity, so every token needs its own shape/size, not just the pick/target ones
            for i in range(pool_size):
                obs[f"obj_{i}_pose"] = info["pool_poses"][:, i]
                obs[f"obj_{i}_active"] = pool_active[:, i]
                obs[f"obj_{i}_is_pick"] = is_pick[:, i]
                obs[f"obj_{i}_is_target"] = is_target[:, i]
                obs[f"obj_{i}_shape_type"] = (
                    self.pool_shape_type[i].unsqueeze(0).expand(self.num_envs)
                )
                obs[f"obj_{i}_size"] = self.pool_size[i].unsqueeze(0).expand(self.num_envs, -1)
        return obs
