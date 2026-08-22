from mani_skill.envs.tasks.tabletop.stack_cube_clutter import (
    StackCubeClutterEnv,
    _ClutterSpec,
)
from mani_skill.utils.registration import register_env


@register_env("StackCubeClutterSameSize-v1", max_episode_steps=250)
class StackCubeClutterSameSizeEnv(StackCubeClutterEnv):
    """StackCubeClutter-v1 with every clutter object's half_size set to match cubeA/cubeB's
    standard half_size (0.02), instead of the varied 0.018-0.025 range."""

    CLUTTER_SPECS = [
        _ClutterSpec(shape="cube", half_size=0.02, color=[0, 0, 1, 1]),  # blue
        _ClutterSpec(shape="cube", half_size=0.02, color=[1, 1, 0, 1]),  # yellow
        _ClutterSpec(shape="cube", half_size=0.02, color=[0.6, 0, 0.8, 1]),  # purple
        _ClutterSpec(shape="cube", half_size=0.02, color=[1, 0.5, 0, 1]),  # orange
        _ClutterSpec(shape="cube", half_size=0.02, color=[0, 1, 1, 1]),  # cyan
        _ClutterSpec(shape="cube", half_size=0.02, color=[0.5, 0.5, 0.5, 1]),  # gray
    ]
