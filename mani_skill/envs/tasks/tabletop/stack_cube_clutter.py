import math
from dataclasses import dataclass

import sapien
import torch

from mani_skill.envs.tasks.tabletop.configurable_spawn_stack_cube import (
    ConfigurableSpawnStackCubeEnv,
)
from mani_skill.envs.utils import randomization
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@dataclass
class _ClutterSpec:
    shape: str  # "cube" | "cylinder" | "sphere"
    color: list
    half_size: float = 0.0  # cube
    radius: float = 0.0  # cylinder / sphere
    half_length: float = 0.0  # cylinder

    @property
    def bounding_radius(self) -> float:
        """xy circumradius used for non-overlap rejection sampling."""
        if self.shape == "cube":
            return self.half_size * math.sqrt(2)
        return self.radius

    @property
    def rest_z(self) -> float:
        """z of the object's center when resting alone on a z=0 surface."""
        if self.shape == "cube":
            return self.half_size
        if self.shape == "cylinder":
            return self.half_length
        return self.radius


@register_env("StackCubeClutter-v1", max_episode_steps=250)
class StackCubeClutterEnv(ConfigurableSpawnStackCubeEnv):
    """
    **Task Description:**
    Same task as StackCube-v1 (pick up the red cube and stack it on the green cube), but the
    table is cluttered every episode with a random number of extra dynamic distractor cubes
    (assorted sizes/colors) scattered at random, non-overlapping positions. The robot must
    reach, grasp and place around the clutter.

    **Randomizations:**
    - everything StackCube-v1 randomizes (both cubes' xy positions and z-axis rotations)
    - the number of clutter objects present, uniform in ``CLUTTER_COUNT_RANGE`` per episode
    - the position and z-axis rotation of each active clutter object, rejection-sampled so it
      does not collide with cubeA, cubeB, or any other clutter object

    **Success Conditions:**
    identical to StackCube-v1. Clutter objects are scenery/obstacles only and are not tracked
    for success.
    """

    NUM_CLUTTER_SLOTS = 6
    CLUTTER_COUNT_RANGE = (3, 6)  # inclusive
    CLUTTER_X_RANGE = (-0.25, 0.25)
    CLUTTER_Y_RANGE = (-0.35, 0.35)
    # added to each clutter object's own circumradius (and the cubes') for rejection sampling
    CLUTTER_CLEARANCE = 0.02

    # cubes only for now: cylinders/spheres have no flat face for the OBB-based motion
    # planner's antipodal parallel-jaw grasp, so they slip once placed on top of another
    # object -- revisit once the solver (or a friction/geometry-aware grasp strategy) can
    # handle curved surfaces reliably. Sizes/colors still varied, distinct from cubeA
    # (red) / cubeB (green).
    CLUTTER_SPECS = [
        _ClutterSpec(shape="cube", half_size=0.018, color=[0, 0, 1, 1]),  # blue
        _ClutterSpec(shape="cube", half_size=0.020, color=[1, 1, 0, 1]),  # yellow
        _ClutterSpec(shape="cube", half_size=0.022, color=[0.6, 0, 0.8, 1]),  # purple
        _ClutterSpec(shape="cube", half_size=0.025, color=[1, 0.5, 0, 1]),  # orange
        _ClutterSpec(shape="cube", half_size=0.019, color=[0, 1, 1, 1]),  # cyan
        _ClutterSpec(shape="cube", half_size=0.021, color=[0.5, 0.5, 0.5, 1]),  # gray
    ]

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        assert len(self.CLUTTER_SPECS) == self.NUM_CLUTTER_SLOTS

        self.clutter_objects = []
        for i, spec in enumerate(self.CLUTTER_SPECS):
            name = f"clutter_{i}"
            # placeholder pose; _initialize_episode overwrites this every reset
            initial_pose = sapien.Pose(p=[2.0, 2.0 + i * 0.3, 1.0])
            if spec.shape == "cube":
                obj = actors.build_cube(
                    self.scene,
                    half_size=spec.half_size,
                    color=spec.color,
                    name=name,
                    initial_pose=initial_pose,
                )
            elif spec.shape == "cylinder":
                obj = actors.build_cylinder(
                    self.scene,
                    radius=spec.radius,
                    half_length=spec.half_length,
                    color=spec.color,
                    name=name,
                    initial_pose=initial_pose,
                )
            else:
                obj = actors.build_sphere(
                    self.scene,
                    radius=spec.radius,
                    color=spec.color,
                    name=name,
                    initial_pose=initial_pose,
                )
            self.clutter_objects.append(obj)

        self.clutter_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.clutter_active = torch.zeros(
            (self.num_envs, self.NUM_CLUTTER_SLOTS), dtype=torch.bool, device=self.device
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        with torch.device(self.device):
            b = len(env_idx)

            k = torch.randint(
                self.CLUTTER_COUNT_RANGE[0], self.CLUTTER_COUNT_RANGE[1] + 1, (b,)
            )
            self.clutter_count[env_idx] = k
            self.clutter_active[env_idx] = (
                torch.arange(self.NUM_CLUTTER_SLOTS, device=self.device)[None, :]
                < k[:, None]
            )

            # seed a fresh sampler with cubeA/cubeB's just-computed positions so clutter
            # avoids them too, without touching ConfigurableSpawnStackCubeEnv's own sampler
            cube_radius = (
                torch.linalg.norm(self.cube_half_size[:2]) + self.SPAWN_CLEARANCE
            )
            sampler = randomization.UniformPlacementSampler(
                bounds=(
                    [self.CLUTTER_X_RANGE[0], self.CLUTTER_Y_RANGE[0]],
                    [self.CLUTTER_X_RANGE[1], self.CLUTTER_Y_RANGE[1]],
                ),
                batch_size=b,
                device=self.device,
            )
            sampler.fixture_positions = torch.stack(
                [self.cubeA.pose.p[env_idx, :2], self.cubeB.pose.p[env_idx, :2]]
            )
            sampler.fixtures_radii = torch.stack([cube_radius, cube_radius]).reshape(2)

            ground_z = -self.table_scene.table_height
            for i, spec in enumerate(self.CLUTTER_SPECS):
                radius = spec.bounding_radius + self.CLUTTER_CLEARANCE
                xy = sampler.sample(radius, 100, append=True)

                active = i < k
                park_xy = torch.tensor([2.0, 2.0 + i * 0.3])
                final_xy = torch.where(active[:, None], xy, park_xy)
                final_z = torch.where(
                    active,
                    torch.full((b,), spec.rest_z),
                    torch.full((b,), ground_z + spec.rest_z),
                )

                xyz = torch.zeros((b, 3))
                xyz[:, :2] = final_xy
                xyz[:, 2] = final_z
                qs = randomization.random_quaternions(
                    b, lock_x=True, lock_y=True, lock_z=self.LOCK_Z_ROTATION
                )
                self.clutter_objects[i].set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        if "state" in self.obs_mode:
            # redundant with tcp_pose/cubeA_pose/cubeB_pose already in obs (linear
            # combinations of them); dropped here rather than at their source in the
            # untouched stack_cube.py, which every non-clutter variant also inherits from
            obs.pop("tcp_to_cubeA_pos", None)
            obs.pop("tcp_to_cubeB_pos", None)
            obs.pop("cubeA_to_cubeB_pos", None)
            for i in range(self.NUM_CLUTTER_SLOTS):
                obs[f"obj_{i}_pose"] = self.clutter_objects[i].pose.raw_pose
                obs[f"obj_{i}_active"] = self.clutter_active[:, i]
        return obs
