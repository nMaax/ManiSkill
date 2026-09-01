from mani_skill.envs.tasks import PushBlockEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    assign_push_pairs,
    push_object_closed_loop,
)
from mani_skill.examples.motionplanning.xarm6.motionplanner import (
    XArm6NoGripperMotionPlanningSolver,
)


def solve(env: PushBlockEnv, seed=None, debug=False, vis=False):
    """Solves PushBlock-v1 on xarm6_nogripper."""
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = XArm6NoGripperMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        print_env_info=False,
    )
    env = env.unwrapped

    push_quat = env.agent.tcp.pose.sp.q

    res = None
    for cube, target in assign_push_pairs(
        env.cubeA, env.cubeB, env.targetA, env.targetB
    ):
        res = push_object_closed_loop(
            planner,
            cube,
            target,
            push_quat,
            push_height=env.CUBE_HALF_SIZE,
            # Stands the pusher's *surface* ~5mm past the cube's back face. Omit the
            # pusher radius and a 52mm flange is parked inside the cube, and PhysX
            # resolves that by launching it.
            contact_clearance=(
                env.CUBE_HALF_SIZE + env.PUSHER_RADIUS[env.robot_uids] - 0.01
            ),
            success_radius=env.SUCCESS_RADIUS,
        )
        if res == -1 or bool(res[2]) or bool(res[3]):
            break

    planner.close()
    return res
