from mani_skill.envs.tasks.tabletop.place_cube_left import PlaceCubeLeftEnv
from mani_skill.utils.registration import register_env


@register_env("PlaceCubeRight-v1", max_episode_steps=50)
class PlaceCubeRightEnv(PlaceCubeLeftEnv):
    TARGET_Y_OFFSET = -0.16  # -Y is right in SAPIEN
