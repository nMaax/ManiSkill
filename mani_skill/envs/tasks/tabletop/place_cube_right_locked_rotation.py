from mani_skill.envs.tasks.tabletop.place_cube_right import PlaceCubeRightEnv
from mani_skill.utils.registration import register_env


@register_env("PlaceCubeRightLockedRotation-v1", max_episode_steps=250)
class PlaceCubeRightLockedRotationEnv(PlaceCubeRightEnv):
    """PlaceCubeRight-v1 with the cubes spawned at identity yaw. Everything else inherited."""

    LOCK_Z_ROTATION = True
