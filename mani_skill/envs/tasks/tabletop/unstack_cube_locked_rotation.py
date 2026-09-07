from mani_skill.envs.tasks.tabletop.unstack_cube import UnstackCubeEnv
from mani_skill.utils.registration import register_env


@register_env("UnstackCubeLockedRotation-v1", max_episode_steps=250)
class UnstackCubeLockedRotationEnv(UnstackCubeEnv):
    """UnstackCube-v1 with the stack's z-axis rotation locked (identity) instead of randomized."""

    LOCK_Z_ROTATION = True
