import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import PlaceCubeRightEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def solve(env: PlaceCubeRightEnv, seed=None, debug=False, vis=False):
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
    obb = get_actor_obb(env.cubeA)

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
    # Place Right
    # -------------------------------------------------------------------------- #
    # 1. Define target position for Cube A (further to the right of Cube B on -Y)
    cubeB_p = env.cubeB.pose.p.cpu().numpy()[0]
    target_p = cubeB_p.copy()
    target_p[1] += env.TARGET_Y_OFFSET  # -Y is right in SAPIEN

    # 2. Calculate the required total world translation to move Cube A to target_p
    current_cubeA_p = env.cubeA.pose.p.cpu().numpy()[0]
    total_offset = target_p - current_cubeA_p

    # 3. Hover: Move horizontally first (keep Z offset at 0)
    hover_offset = total_offset.copy()
    hover_offset[2] = 0.0
    hover_pose = sapien.Pose(lift_pose.p + hover_offset, lift_pose.q)
    planner.move_to_pose_with_screw(hover_pose)

    # 4. Lower: Apply the full offset to go down to table level
    # We add a tiny 2mm Z-clearance so it doesn't aggressively smash into the table
    place_offset = total_offset.copy()
    place_offset[2] += 0.002
    place_pose = sapien.Pose(lift_pose.p + place_offset, lift_pose.q)
    planner.move_to_pose_with_screw(place_pose)

    # 5. Open Gripper
    res = planner.open_gripper()

    # 6. Retreat: Move the arm back up so it doesn't occlude the cubes at the end
    retreat_pose = sapien.Pose([0, 0, 0.1]) * place_pose
    planner.move_to_pose_with_screw(retreat_pose)

    planner.close()
    return res
