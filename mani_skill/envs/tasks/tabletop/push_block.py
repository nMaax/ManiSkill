from typing import Any, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Panda, PandaStick, XArm6NoGripper
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


class HighFrictionTableSceneBuilder(TableSceneBuilder):
    """TableSceneBuilder, but the table surface gets a high-friction PhysX material
    (static == dynamic == 1.0, as in BESO's PyBullet ``workspace.urdf``), instead of
    SAPIEN's low-friction (0.3/0.3) default."""

    # As in PushT WhiteTableSceneBuilder: call super().build(), then post-process
    # the already-built table.
    def build(self):
        super().build()
        table_material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=self.env.CUBE_FRICTION,
            dynamic_friction=self.env.CUBE_FRICTION,
            restitution=0,
        )
        for part in self.table._objs:
            component = part.find_component_by_type(
                sapien.physx.PhysxRigidDynamicComponent
            )
            for shape in component.collision_shapes:
                shape.set_physical_material(table_material)


# Headroom, not BESO's 300-step budget: plan-and-execute costs more steps per motion than
# BESO's continuous oracle. Worst motion-planned episode over 30 seeds is 345.
@register_env("PushBlock-v1", max_episode_steps=500)
class PushBlockEnv(BaseEnv):
    """
    **Task Description:**
    Reproduction of BESO's (intuitive-robots/beso) PyBullet "block pushing" benchmark:
    push cubeA and cubeB into two distinct target zones. Either cube may end up in
    either target to mark a success.

    The layout is BESO's own, translated into world coordinates: the push runs along +y,
    laterally across the robot's front, and the two targets are separated along x, the
    robot's radial direction. See ``WORKSPACE_CENTER_X`` for why the axes are this way
    round and not the other.

    **Randomizations:**
    - cubeA and cubeB both spawn anywhere in one shared region, kept apart by a
      rejection sample on their LATERAL (x) separation only, as in BESO -- so which
      cube is the near one and which the far one is always well defined
    - targetA/targetB swap sides at random and carry BESO's 5~7.5mm jitter
    - cubeA/cubeB spawn with a random z-rotation (unlocked, matching BESO), so
      reorientation is part of the task's difficulty just as in BESO's oracle
      (which has explicit orient_block_left/orient_block_right correction phases).

    **Success Conditions:**
    - both cubes are within SUCCESS_RADIUS (0.05m) of two distinct target zones, remind that which
      cube ends up in which target does not matter.
    """

    SUPPORTED_ROBOTS = ["panda_stick", "xarm6_nogripper", "panda"]
    agent: Union[PandaStick, XArm6NoGripper, Panda]

    # per-robot base pose
    _ROBOT_BASE_POSE = {
        "panda_stick": sapien.Pose(p=[-0.615, 0, 0]),
        "xarm6_nogripper": sapien.Pose(p=[-0.522, 0, 0]),
        "panda": sapien.Pose(p=[-0.615, 0, 0]),
    }

    CUBE_HALF_SIZE = 0.02
    CUBE_MASS = 0.01

    # BESO's block.urdf uses <contact><inertia_scaling value="3.0"/></contact>, a
    # PyBullet URDF extension that multiplies the block's rotational inertia by 3 while
    # leaving its mass alone
    CUBE_INERTIA_SCALING = 2.0  # BESO default: 3.0
    CUBE_FRICTION = 0.3  # BESO default: 1.0 (workspace.urdf and block.urdf)
    SUCCESS_RADIUS = 0.05
    LOCK_Z_ROTATION = False  # BESO default: False

    # +Y pushes, X separates the targets -- BESO's own layout, not a transposition of it.
    # BESO's base sits at its frame's origin, so swapping its axis names rotates the task
    # 90 degrees about the arm and lands the targets at 99% of the xArm6's reach.
    # Constants below are BESO's, translated by the xarm6 base offset (-0.522).
    WORKSPACE_CENTER_X = -0.122  # BESO's workspace_center_x = 0.4, in front of the base

    CUBE_LATERAL_RANGE = (-0.222, -0.022)  # centre +- 0.10, BESO's RANDOM_X_SHIFT
    CUBE_PUSH_AXIS_RANGE = (-0.35, -0.05)  # -0.20 +- 0.15, BESO's RANDOM_Y_SHIFT

    # Minimum LATERAL separation between the two cubes. BESO's MIN_BLOCK_DIST.
    MIN_CUBE_LATERAL_DIST = 0.10
    NUM_RESET_ATTEMPTS = 100

    TARGET_PUSH_AXIS = 0.20  # BESO's target y
    TARGET_LATERAL_OFFSET = 0.12  # BESO's +-0.12 "add", about WORKSPACE_CENTER_X

    # BESO jitters the targets by +-0.0075 along the push axis and +-0.005 laterally
    TARGET_PUSH_AXIS_JITTER = 0.0075
    TARGET_LATERAL_JITTER = 0.005

    CUBE_HALF_DIAGONAL = CUBE_HALF_SIZE * np.sqrt(2)

    # How far the pusher's contact surface reaches from the TCP along the push axis,
    # measured off the collision meshes (xarm6's link6 flange, panda's closed fingers,
    # panda_stick's cylindrical stick).
    # Solvers size their contact standoff from this. BESO's pin is 0.001.
    PUSHER_RADIUS = {"panda_stick": 0.008, "xarm6_nogripper": 0.052, "panda": 0.026}

    # Where the TCP starts, behind every cube. BESO uses -0.40; a 52mm flange needs more
    # clearance than its 1mm pin or the wrist starts overlapping a cube (6.5% of resets).
    START_LINE_Y = round(
        CUBE_PUSH_AXIS_RANGE[0] - (max(PUSHER_RADIUS.values()) + CUBE_HALF_DIAGONAL + 0.02),
        3,
    )

    # BESO resets its effector to a fixed pose already at pushing height, so its data
    # contains no descent. BESO's [0.3, -0.4], in world coordinates.
    START_TCP_XY = (-0.222, START_LINE_Y)

    # TCP at START_TCP_XY, pushing height, wrist down. Solved offline; regenerate if
    # START_TCP_XY moves.
    _START_QPOS = {
        "panda_stick": np.array(
            [
                -0.73018196,
                0.55598256,
                -0.11993673,
                -2.02965624,
                0.11871456,
                2.57923081,
                -0.14735828,
            ]
        ),
        "xarm6_nogripper": np.array(
            [-0.98269925, 0.90699818, -1.31737271, 0.00001767, 0.41037425, -0.98271620]
        ),
        # last two are the fingers: panda pushes with a closed fist
        "panda": np.array(
            [
                -0.74325794,
                0.66171743,
                -0.10265828,
                -2.00070827,
                0.13534920,
                2.65608579,
                -0.15882563,
                0.0,
                0.0,
            ]
        ),
    }

    # 0.0, not the usual 0.02: BESO's reset is exact, and at pushing height 0.02rad of
    # joint noise drives the wrist through the table.
    def __init__(
        self, *args, robot_uids="panda_stick", robot_init_qpos_noise=0.0, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=240,  # PyBullet's default fixedTimeStep, i.e. BESO's contact dt
            control_freq=10,  # BESO's control frequency
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            ),
        )

    # The workspace is not centred on y=0: cubes spawn right, targets sit left.
    _WORKSPACE_CENTROID = [-0.122, -0.07, 0.05]

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.40, -0.07, 0.60], target=self._WORKSPACE_CENTROID)
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        # BESO's own DEFAULT_CAMERA_POSE (1.0, 0, 0.75), translated by the base offset.
        pose = sapien_utils.look_at([0.478, 0, 0.75], self._WORKSPACE_CENTROID)
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _pusher_links(self):
        """The links that contact the cubes: bare wrist on xarm6, closed fingers on panda,
        or the hand link (carrying the stick collision cylinder) on panda_stick."""
        if self.robot_uids == "panda":
            return [
                self.agent.robot.links_map["panda_leftfinger"],
                self.agent.robot.links_map["panda_rightfinger"],
            ]
        elif self.robot_uids == "panda_stick":
            return [self.agent.robot.links_map["panda_hand"]]
        return [self.agent.tcp]

    def _load_agent(self, options: dict):
        super()._load_agent(options, self._ROBOT_BASE_POSE[self.robot_uids])

        # BESO's suction head declares lateral_friction 1.0 and PyBullet multiplies the
        # two bodies' frictions; PhysX averages, so the pusher's own material matters.
        # Left alone it is SAPIEN's 0.3 (xarm6) or ManiSkill's 2.0 gripper pads (panda).
        material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=self.CUBE_FRICTION,
            dynamic_friction=self.CUBE_FRICTION,
            restitution=0,
        )
        for link in self._pusher_links():
            for obj in link._objs:
                component = obj.entity.find_component_by_type(
                    sapien.physx.PhysxRigidBodyComponent
                )
                for shape in component.collision_shapes:
                    shape.set_physical_material(material)

    def _build_pusher_cube(self, name: str, color):
        """A cube as BESO's block.urdf: 4cm, 10g, friction 1.0."""
        material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=self.CUBE_FRICTION,
            dynamic_friction=self.CUBE_FRICTION,
            restitution=0,
        )
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=[self.CUBE_HALF_SIZE] * 3, material=material
        )
        builder.add_box_visual(
            half_size=[self.CUBE_HALF_SIZE] * 3,
            material=sapien.render.RenderMaterial(base_color=color),
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])

        # NOTE: assigning builder._mass does NOT set the mass. SAPIEN's ActorBuilder only
        # applies _mass/_inertia when _auto_inertial is False, and only
        # set_mass_and_inertia() clears that flag; otherwise mass comes from the
        # collision shape's density (1000 by default), which for this 4cm box is 64g --
        # 6.4x BESO's 10g block, silently, for as long as _mass was being poked directly.
        inertia = self.CUBE_MASS * (2 * self.CUBE_HALF_SIZE) ** 2 / 6
        builder.set_mass_and_inertia(
            mass=self.CUBE_MASS,
            cmass_local_pose=sapien.Pose(),
            inertia=[inertia * self.CUBE_INERTIA_SCALING] * 3,
        )
        return builder.build(name=name)

    def _load_scene(self, options: dict):
        self.table_scene = HighFrictionTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cubeA = self._build_pusher_cube("cubeA", color=[1, 0, 0, 1])
        self.cubeB = self._build_pusher_cube("cubeB", color=[0, 1, 0, 1])

        self.targetA = actors.build_red_white_target(
            self.scene,
            radius=self.SUCCESS_RADIUS,
            thickness=1e-5,
            name="targetA",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
        )
        self.targetB = actors.build_red_white_target(
            self.scene,
            radius=self.SUCCESS_RADIUS,
            thickness=1e-5,
            name="targetB",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 1, 1e-3]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Start at pushing height, as BESO does, so no episode opens with a descent.
            # BaseEnv.reset() syncs controller targets from qpos afterwards.
            qpos = self._START_QPOS[self.robot_uids]
            qpos = (
                self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            if self.robot_uids == "panda":
                qpos[:, -2:] = 0.0
            self.agent.reset(qpos)

            def sample_cube_xy(n):
                xy = torch.zeros((n, 2))
                xy[:, 0] = (
                    torch.rand(n)
                    * (self.CUBE_LATERAL_RANGE[1] - self.CUBE_LATERAL_RANGE[0])
                    + self.CUBE_LATERAL_RANGE[0]
                )
                xy[:, 1] = (
                    torch.rand(n)
                    * (self.CUBE_PUSH_AXIS_RANGE[1] - self.CUBE_PUSH_AXIS_RANGE[0])
                    + self.CUBE_PUSH_AXIS_RANGE[0]
                )
                return xy

            cubeA_xy = sample_cube_xy(b)
            cubeB_xy = sample_cube_xy(b)
            for _ in range(self.NUM_RESET_ATTEMPTS):
                # Lateral only, as BESO does -- never Euclidean, so one cube may sit
                # directly behind the other.
                too_close = (
                    torch.abs(cubeB_xy[:, 0] - cubeA_xy[:, 0])
                    <= self.MIN_CUBE_LATERAL_DIST
                )
                if not too_close.any():
                    break
                # Redraw BOTH cubes, as BESO's outer loop does. The range is 0.20 wide
                # and the gap 0.10, so a cubeA near the middle leaves no valid cubeB at
                # all: resampling only the partner then fails outright, for P = 2/101.
                n = int(too_close.sum())
                cubeA_xy[too_close] = sample_cube_xy(n)
                cubeB_xy[too_close] = sample_cube_xy(n)
            cubeA_xyz = torch.zeros((b, 3))
            cubeA_xyz[:, :2] = cubeA_xy
            cubeA_xyz[:, 2] = self.CUBE_HALF_SIZE
            cubeB_xyz = torch.zeros((b, 3))
            cubeB_xyz[:, :2] = cubeB_xy
            cubeB_xyz[:, 2] = self.CUBE_HALF_SIZE

            flip = (torch.randint(0, 2, (b,)) * 2 - 1).float()
            targetA_xyz = torch.zeros((b, 3))
            targetA_xyz[:, 0] = (
                self.WORKSPACE_CENTER_X
                + flip * self.TARGET_LATERAL_OFFSET
                + (torch.rand(b) * 2 - 1) * self.TARGET_LATERAL_JITTER
            )
            targetA_xyz[:, 1] = (
                self.TARGET_PUSH_AXIS
                + (torch.rand(b) * 2 - 1) * self.TARGET_PUSH_AXIS_JITTER
            )
            targetA_xyz[:, 2] = 1e-3
            targetB_xyz = torch.zeros((b, 3))
            targetB_xyz[:, 0] = (
                self.WORKSPACE_CENTER_X
                - flip * self.TARGET_LATERAL_OFFSET
                + (torch.rand(b) * 2 - 1) * self.TARGET_LATERAL_JITTER
            )
            targetB_xyz[:, 1] = (
                self.TARGET_PUSH_AXIS
                + (torch.rand(b) * 2 - 1) * self.TARGET_PUSH_AXIS_JITTER
            )
            targetB_xyz[:, 2] = 1e-3

            qA = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=self.LOCK_Z_ROTATION
            )
            qB = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=self.LOCK_Z_ROTATION
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=cubeA_xyz, q=qA))
            self.cubeB.set_pose(Pose.create_from_pq(p=cubeB_xyz, q=qB))

            target_q = euler2quat(0, np.pi / 2, 0)
            self.targetA.set_pose(Pose.create_from_pq(p=targetA_xyz, q=target_q))
            self.targetB.set_pose(Pose.create_from_pq(p=targetB_xyz, q=target_q))

    def _cube_target_dists(self):
        cubeA_p = self.cubeA.pose.p[:, :2]
        cubeB_p = self.cubeB.pose.p[:, :2]
        targetA_p = self.targetA.pose.p[:, :2]
        targetB_p = self.targetB.pose.p[:, :2]
        dA_to_A = torch.linalg.norm(cubeA_p - targetA_p, axis=1)
        dA_to_B = torch.linalg.norm(cubeA_p - targetB_p, axis=1)
        dB_to_A = torch.linalg.norm(cubeB_p - targetA_p, axis=1)
        dB_to_B = torch.linalg.norm(cubeB_p - targetB_p, axis=1)
        return dA_to_A, dA_to_B, dB_to_A, dB_to_B

    def _is_on_table(self, cube):
        return torch.abs(cube.pose.p[:, 2] - self.CUBE_HALF_SIZE) < 0.01

    def evaluate(self):
        dA_to_A, dA_to_B, dB_to_A, dB_to_B = self._cube_target_dists()

        pairing_straight = (dA_to_A < self.SUCCESS_RADIUS) & (
            dB_to_B < self.SUCCESS_RADIUS
        )
        pairing_swapped = (dA_to_B < self.SUCCESS_RADIUS) & (
            dB_to_A < self.SUCCESS_RADIUS
        )

        is_A_placed = (dA_to_A < self.SUCCESS_RADIUS) | (dA_to_B < self.SUCCESS_RADIUS)
        is_B_placed = (dB_to_A < self.SUCCESS_RADIUS) | (dB_to_B < self.SUCCESS_RADIUS)

        on_table = self._is_on_table(self.cubeA) & self._is_on_table(self.cubeB)
        success = (pairing_straight | pairing_swapped) & on_table

        return {
            "is_A_placed": is_A_placed,
            "is_B_placed": is_B_placed,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                targetA_pos=self.targetA.pose.p,
                targetB_pos=self.targetB.pose.p,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def _stage_reward(self, cube, is_placed):
        tcp_to_cube = torch.linalg.norm(self.agent.tcp.pose.p - cube.pose.p, axis=1)
        reach = 1 - torch.tanh(5 * tcp_to_cube)

        dA = torch.linalg.norm(cube.pose.p[:, :2] - self.targetA.pose.p[:, :2], axis=1)
        dB = torch.linalg.norm(cube.pose.p[:, :2] - self.targetB.pose.p[:, :2], axis=1)
        target_p = torch.where(
            (dA < dB).unsqueeze(-1), self.targetA.pose.p, self.targetB.pose.p
        )
        cube_to_target = torch.linalg.norm(cube.pose.p[:, :2] - target_p[:, :2], axis=1)
        push = 1 - torch.tanh(5 * cube_to_target)

        r = reach + push
        r[is_placed] = 2.0
        return r

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        reward = self._stage_reward(self.cubeA, info["is_A_placed"])
        reward += self._stage_reward(self.cubeB, info["is_B_placed"])
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0
