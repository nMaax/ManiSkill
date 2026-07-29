import torch

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.structs.pose import Pose


class ConfigurableSpawnStackCubeEnv(StackCubeEnv):
    """Exposes the StackCube cube spawn as class constants. Registers nothing itself.

    Variants override the constants below instead of reimplementing ``_initialize_episode``,
    which keeps ``stack_cube.py`` untouched (StackCube-v1 has published episodes). With the
    defaults this reproduces ``StackCubeEnv._initialize_episode`` exactly, RNG draws included.
    """

    # (min, max) bounds of the rejection sampling region for both cube positions
    CUBE_X_RANGE = (-0.1, 0.1)
    CUBE_Y_RANGE = (-0.2, 0.2)

    # applied to both cubes changing at each episodes, 0 to disable it.
    SHARED_XY_OFFSET = 0.1

    # added to the cube circumradius for rehecting too-close samples, 0 to disable it.
    SPAWN_CLEARANCE = 0.001
    LOCK_Z_ROTATION = False

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            # a branch, not a multiply by zero: skipping the draw keeps the RNG stream of
            # the offset-free variants bit-identical to their pre-refactor behaviour
            if self.SHARED_XY_OFFSET:
                xy = (
                    torch.rand((b, 2)) * (2 * self.SHARED_XY_OFFSET)
                    - self.SHARED_XY_OFFSET
                )
            else:
                xy = torch.zeros((b, 2))

            region = (
                [self.CUBE_X_RANGE[0], self.CUBE_Y_RANGE[0]],
                [self.CUBE_X_RANGE[1], self.CUBE_Y_RANGE[1]],
            )
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = (
                torch.linalg.norm(torch.tensor([0.02, 0.02])) + self.SPAWN_CLEARANCE
            )
            cubeA_xy = xy + sampler.sample(radius, 100)
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=self.LOCK_Z_ROTATION,
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=self.LOCK_Z_ROTATION,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))
