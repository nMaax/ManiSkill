from mani_skill.envs.tasks.tabletop.stack_cube_clutter_random_pick import (
    StackCubeClutterRandomPickEnv,
)
from mani_skill.envs.tasks.tabletop.stack_cube_clutter_same_size_locked_rotation import (
    StackCubeClutterSameSizeLockedRotationEnv,
)
from mani_skill.utils.registration import register_env


@register_env("StackCubeClutterRandomPickSameSizeLockedRotation-v1", max_episode_steps=250)
class StackCubeClutterRandomPickSameSizeLockedRotationEnv(
    StackCubeClutterRandomPickEnv, StackCubeClutterSameSizeLockedRotationEnv
):
    """StackCubeClutterRandomPick-v1's task with StackCubeClutterSameSizeLockedRotation-v1's
    uniform object sizes and locked z-rotation. No body needed: task (evaluate/reward/obs,
    pool selection) resolves from the first parent, spawn/geometry constants
    (LOCK_Z_ROTATION, restricted CUBE_X/Y_RANGE, uniform CLUTTER_SPECS) from the second --
    same MRO trick as StackCubeClutterRandomPickLockedRotationEnv."""
