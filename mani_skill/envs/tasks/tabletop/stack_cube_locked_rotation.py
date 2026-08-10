from mani_skill.envs.tasks.tabletop.configurable_spawn_stack_cube import (
    ConfigurableSpawnStackCubeEnv,
)
from mani_skill.utils.registration import register_env


@register_env("StackCubeLockedRotation-v1", max_episode_steps=250)
class StackCubeLockedRotationEnv(ConfigurableSpawnStackCubeEnv):
    """StackCube-v1 with cube z-rotation locked within a fully reachable spawn region and clearance around cubes."""

    CUBE_X_RANGE = (-0.20, 0.12)
    CUBE_Y_RANGE = (-0.13, 0.13)
    SHARED_XY_OFFSET = 0.0
    SPAWN_CLEARANCE = 0.015
    LOCK_Z_ROTATION = True
