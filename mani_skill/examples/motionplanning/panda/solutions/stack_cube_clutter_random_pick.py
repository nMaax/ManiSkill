import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import StackCubeClutterRandomPickEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def solve(env: StackCubeClutterRandomPickEnv, seed=None, debug=False, vis=False):
    """Solves StackCubeClutterRandomPick-v1 (and its locked-rotation sibling). Which pool
    object is picked up and which is the stacking target is decided by env.pick_idx/
    env.target_idx, read after reset since both are randomized per episode over the full
    8-object pool, not just cubeA/cubeB. get_actor_obb already tessellates box/cylinder/sphere
    collision shapes generically (mani_skill/utils/geometry/trimesh_utils.py), so the grasp
    computation below is identical to stack_cube.py's -- only which two actors and how tall
    the stack offset is differ, parameterized off the pool instead of hardcoded."""
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    FINGER_LENGTH = 0.025
    env = env.unwrapped

    pick_idx = int(env.pick_idx[0].item())
    target_idx = int(env.target_idx[0].item())
    pick_obj = env.pool_objects[pick_idx]
    target_obj = env.pool_objects[target_idx]
    stack_height = (env.pool_rest_z[pick_idx] + env.pool_rest_z[target_idx]).item()

    obb = get_actor_obb(pick_obj)

    approaching = np.array([0, 0, -1])
    target_closing = (
        env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    )
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # Search a valid pose
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)

    # -------------------------------------------------------------------------- #
    # Stack
    # -------------------------------------------------------------------------- #
    goal_pose = target_obj.pose * sapien.Pose([0, 0, stack_height])
    # remember that all data in ManiSkill is batched and a torch tensor
    offset = (goal_pose.p - pick_obj.pose.p).cpu().numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)

    res = planner.open_gripper()
    planner.close()
    return res
