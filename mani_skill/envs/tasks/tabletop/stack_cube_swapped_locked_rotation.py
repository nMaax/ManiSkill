from mani_skill.envs.tasks.tabletop.stack_cube_locked_rotation import (
    StackCubeLockedRotationEnv,
)
from mani_skill.envs.tasks.tabletop.stack_cube_swapped import StackCubeSwappedEnv
from mani_skill.utils.registration import register_env


@register_env("StackCubeSwappedLockedRotation-v1", max_episode_steps=250)
class StackCubeSwappedLockedRotationEnv(
    StackCubeSwappedEnv, StackCubeLockedRotationEnv
):
    """StackCubeSwapped-v1's task with StackCubeLockedRotation-v1's spawn."""
