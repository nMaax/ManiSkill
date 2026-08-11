from mani_skill.envs.tasks.tabletop.stack_cube_clutter import StackCubeClutterEnv
from mani_skill.utils.registration import register_env


@register_env("StackCubeClutterLockedRotation-v1", max_episode_steps=250)
class StackCubeClutterLockedRotationEnv(StackCubeClutterEnv):
    """StackCubeClutter-v1 with cube (and clutter) z-rotation locked, within the same fully
    reachable spawn region and clearance StackCubeLockedRotation-v1 uses."""

    CUBE_X_RANGE = (-0.20, 0.12)
    CUBE_Y_RANGE = (-0.13, 0.13)
    SHARED_XY_OFFSET = 0.0
    SPAWN_CLEARANCE = 0.015
    LOCK_Z_ROTATION = True
    # StackCubeClutterEnv's own CLUTTER_X/Y_RANGE (-0.25..0.25, -0.35..0.35) reach up to 0.93m
    # from the Panda base -- well past the ~0.82m reachable annulus -- so clutter objects
    # could spawn unreachably even in this "reachable" variant. Reuse the cube box, whose
    # corners were already verified to fall within 0.435-0.746m from the base.
    CLUTTER_X_RANGE = CUBE_X_RANGE
    CLUTTER_Y_RANGE = CUBE_Y_RANGE
