from mani_skill.envs.tasks import PushBlockEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    assign_push_pairs,
    push_object_closed_loop,
    push_object_planar_closed_loop,
)
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)


def solve(env: PushBlockEnv, seed=None, debug=False, vis=False):
    """Solves PushBlock-v1 on panda or panda_stick."""
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode

    env_unwrapped = env.unwrapped
    robot_uid = env_unwrapped.robot_uids

    if robot_uid == "panda_stick":
        planner = PandaStickMotionPlanningSolver(
            env,
            debug=debug,
            vis=vis,
            base_pose=env_unwrapped.agent.robot.pose,
            print_env_info=False,
        )
        push_quat = env_unwrapped.agent.tcp.pose.sp.q
        res = None
        for cube, target in assign_push_pairs(
            env_unwrapped.cubeA, env_unwrapped.cubeB, env_unwrapped.targetA, env_unwrapped.targetB
        ):
            other_cube = (
                env_unwrapped.cubeB if cube is env_unwrapped.cubeA else env_unwrapped.cubeA
            )
            res = push_object_planar_closed_loop(
                planner,
                cube,
                target,
                push_quat,
                push_height=env_unwrapped.CUBE_HALF_SIZE,
                contact_clearance=(
                    env_unwrapped.CUBE_HALF_SIZE + env_unwrapped.PUSHER_RADIUS[robot_uid] - 0.003
                ),
                success_radius=env_unwrapped.SUCCESS_RADIUS,
                other_obstacles=[other_cube],
                obs_radius=(
                    env_unwrapped.PUSHER_RADIUS[robot_uid]
                    + env_unwrapped.CUBE_HALF_DIAGONAL
                    + 0.006
                ),
            )
            if res == -1 or bool(res[2]) or bool(res[3]):
                break

        planner.close()
        return res

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env_unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    # closed gripper acts as a pusher
    res = planner.close_gripper()

    push_quat = env_unwrapped.agent.tcp.pose.sp.q

    for cube, target in assign_push_pairs(
        env_unwrapped.cubeA, env_unwrapped.cubeB, env_unwrapped.targetA, env_unwrapped.targetB
    ):
        res = push_object_closed_loop(
            planner,
            cube,
            target,
            push_quat,
            push_height=env_unwrapped.CUBE_HALF_SIZE,
            # Stands the pusher's *surface* ~5mm past the cube's back face. Omit the
            # pusher radius and a 52mm flange is parked inside the cube, and PhysX
            # resolves that by launching it.
            contact_clearance=(
                env_unwrapped.CUBE_HALF_SIZE + env_unwrapped.PUSHER_RADIUS[env_unwrapped.robot_uids] - 0.01
            ),
            success_radius=env_unwrapped.SUCCESS_RADIUS,
        )
        if res == -1 or bool(res[2]) or bool(res[3]):
            break

    planner.close()
    return res
