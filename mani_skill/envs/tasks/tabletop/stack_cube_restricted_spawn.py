from mani_skill.envs.tasks.tabletop.configurable_spawn_stack_cube import (
    ConfigurableSpawnStackCubeEnv,
)
from mani_skill.utils.registration import register_env


@register_env("StackCubeRestrictedSpawn-v1", max_episode_steps=250)
class StackCubeRestrictedSpawnEnv(ConfigurableSpawnStackCubeEnv):
    """Same as StackCube-v1, but cubes are spawned within a restricted (x, y) region."""

    CUBE_X_RANGE = (-0.08, 0.08)
    CUBE_Y_RANGE = (-0.08, 0.08)
    SHARED_XY_OFFSET = 0.0
    SPAWN_CLEARANCE = 0.025
    LOCK_Z_ROTATION = False
