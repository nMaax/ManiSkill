from mani_skill.envs.tasks.tabletop.stack_cube_clutter_locked_rotation import (
    StackCubeClutterLockedRotationEnv,
)
from mani_skill.envs.tasks.tabletop.stack_cube_clutter_same_size import (
    StackCubeClutterSameSizeEnv,
)
from mani_skill.utils.registration import register_env


@register_env("StackCubeClutterSameSizeLockedRotation-v1", max_episode_steps=250)
class StackCubeClutterSameSizeLockedRotationEnv(
    StackCubeClutterSameSizeEnv, StackCubeClutterLockedRotationEnv
):
    """StackCubeClutterSameSize-v1 with StackCubeClutterLockedRotation-v1's locked z-rotation
    and reachable spawn region. No body needed: uniform CLUTTER_SPECS resolves from the first
    parent, LOCK_Z_ROTATION/restricted CUBE_X/Y_RANGE/CLUTTER_X/Y_RANGE from the second --
    same MRO trick as StackCubeClutterRandomPickLockedRotationEnv."""
