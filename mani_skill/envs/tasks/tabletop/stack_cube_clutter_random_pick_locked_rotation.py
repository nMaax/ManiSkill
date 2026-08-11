from mani_skill.envs.tasks.tabletop.stack_cube_clutter_locked_rotation import (
    StackCubeClutterLockedRotationEnv,
)
from mani_skill.envs.tasks.tabletop.stack_cube_clutter_random_pick import (
    StackCubeClutterRandomPickEnv,
)
from mani_skill.utils.registration import register_env


@register_env("StackCubeClutterRandomPickLockedRotation-v1", max_episode_steps=250)
class StackCubeClutterRandomPickLockedRotationEnv(
    StackCubeClutterRandomPickEnv, StackCubeClutterLockedRotationEnv
):
    """StackCubeClutterRandomPick-v1's task with StackCubeClutterLockedRotation-v1's
    spawn. No body needed: task (evaluate/reward/obs, pool selection) resolves from the
    first parent, spawn constants (LOCK_Z_ROTATION, restricted CUBE_X/Y_RANGE, clearance)
    from the second -- same MRO trick as StackCubeSwappedLockedRotationEnv."""
