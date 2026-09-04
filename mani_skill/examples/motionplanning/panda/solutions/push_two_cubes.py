"""Motion planning solution for PushTwoCubes-v1.

This is solutions/push_cube.py's idea run twice, plus a cross-over between the two pushes. The
order is hardcoded (cubeA then cubeB) on purpose: PushTwoCubes exists to be a single-mode task,
so this solver deliberately does not use assign_push_pairs / push_object_closed_loop from
base_motionplanner/utils.py, which sample the pairing and the order at random.

The closed loop below is deterministic given the state -- it re-measures and corrects, it never
chooses between alternatives -- so it adds precision without adding multimodality.
"""

import numpy as np
import sapien

from mani_skill.envs.tasks import PushTwoCubesEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver

# Distance from the TCP frame origin to the cube centre when the closed fist is in contact,
# measured over 8 seeds as 0.0295 +- 0.0006 m.
CONTACT_OFFSET = 0.0295
# While the fist is pressing the cube, the joint controller settles where its torque balances
# friction rather than on the commanded pose, leaving the TCP a steady-state 0.0060 +- 0.0009 m
# short along the push direction (measured over 30 strokes). refine_steps does not remove this --
# holding the commanded qpos longer converges to 0.0057 and stops -- so the stroke aims through
# it instead. Together these two put the cube centre on the goal centre.
PUSH_LAG = 0.0060
# How far behind the cube the TCP lines up before a stroke (same as PushCube).
PUSH_APPROACH_STANDOFF = 0.05
# Clearance used when lifting over a cube to re-approach it from a different side.
LIFT_HEIGHT = 0.12
# Stop refining once the cube centre is this close to the goal centre.
PUSH_TOLERANCE = 0.005
# Refinement strokes per cube, including the first one.
MAX_PUSH_PASSES = 3
# A refinement stroke skips the lift-and-come-around when the arm is already behind the cube:
# it must be within this distance of the cube centre and this well aligned with the new push
# direction. Both hold whenever a stroke simply stopped short, which is the usual case.
BEHIND_MAX_GAP = 0.08
BEHIND_MIN_COS = 0.9
# A refinement is abandoned if lining up for it would put the TCP further than this from the
# robot base. Legitimate refinements -- a stroke that stopped short, so the arm pushes on in the
# same direction -- line up at most ~0.73 m out. It is overshoot that is expensive: correcting
# it means reaching around to the far side of the cube, which on one measured seed put the
# target 0.81 m out, near the ~0.82 m limit, where the arm crawls through near-singular
# configurations and shoved the cube further off than it started. A cube that overshot is still
# tens of millimetres inside an 80 mm goal, so stopping there costs a little centring and
# nothing else.
REFINE_REACH_LIMIT = 0.76


def _move(planner, env, xy, z, q):
    """Drive the TCP in a straight line to (`xy`, `z`), subdividing if the planner balks.

    Every motion in this solver is a straight line, and none of them may deviate: a stroke has
    to push the cube towards the goal and not sideways, and a transit has to stay on the lane
    it was routed along. So there is deliberately no RRTConnect fallback -- that plans a free
    path, which on one measured seed swept the arm through cubeA and knocked it out of its
    lane. mplib's screw planner does reject the occasional straight line it should accept
    (~1 in 180 here, uncorrelated with distance from the base), so it is retried as collinear
    sub-strokes instead: same path, planned in shorter pieces. Anything a partial attempt
    already executed stays on the line, so the next attempt re-reads where the arm got to.
    """
    end = np.array([xy[0], xy[1], z])
    res = planner.move_to_pose_with_screw(sapien.Pose(p=end, q=q))
    if res != -1:
        return res

    for segments in (2, 4):
        start = env.agent.tcp.pose.sp.p.copy()
        for k in range(1, segments + 1):
            waypoint = start + (end - start) * (k / segments)
            res = planner.move_to_pose_with_screw(sapien.Pose(p=waypoint, q=q))
            if res == -1:
                break
        if res != -1:
            return res
    return -1


def _approach(planner, env, xy, z, lift=True):
    """Move the TCP to (`xy`, `z`), lifting over anything in the way when `lift`.

    With `lift`, this is three straight-line moves -- up, across, down -- so the arm neither
    drags the cube it was just touching nor clips the one it is lining up on. That is needed for
    the cross-over between the two lanes and when re-approaching a cube from a new side between
    refinement strokes. Coming in from the robot's home pose there is nothing to clear, so a
    single direct move is used instead, exactly as push_cube.py does.
    """
    q = env.agent.tcp.pose.sp.q
    cur = env.agent.tcp.pose.sp.p
    if not lift:
        waypoints = [(xy, z)]
    else:
        top_z = max(cur[2], z) + LIFT_HEIGHT
        waypoints = [(cur[:2], top_z), (xy, top_z), (xy, z)]
    res = None
    for wp_xy, wp_z in waypoints:
        res = _move(planner, env, wp_xy, wp_z, q)
        if res == -1:
            return -1
    return res


def _push(planner, env, cube, goal, needs_lift=True):
    """Push `cube` onto the centre of `goal`, refining until it is within PUSH_TOLERANCE.

    Each stroke is aimed along the current cube->goal direction rather than straight down +x,
    so one stroke takes out the cube's lateral spawn offset as well as the distance. The stroke
    stays at the cube's own height: PushCube takes its target straight off the goal disc, whose
    z is 1e-3, which scrapes the fist along the table and makes plan_screw fail outright on a
    measurable fraction of seeds.
    """
    q = env.agent.tcp.pose.sp.q
    res = None
    for i in range(MAX_PUSH_PASSES):
        cube_p = cube.pose.sp.p
        goal_xy = goal.pose.sp.p[:2]
        push_z = cube_p[2]

        error = goal_xy - cube_p[:2]
        distance = np.linalg.norm(error)
        if distance < PUSH_TOLERANCE:
            break
        direction = error / distance

        # a stroke that merely stopped short leaves the arm already behind the cube and
        # pointing the right way, so it can just push on. Lifting and coming around costs
        # three extra moves and is the one motion here that occasionally fails to plan.
        tcp_xy = env.agent.tcp.pose.sp.p[:2]
        to_cube = cube_p[:2] - tcp_xy
        gap = float(np.linalg.norm(to_cube))
        already_behind = (
            i > 0
            and gap < BEHIND_MAX_GAP
            and gap > 1e-6
            and float(np.dot(to_cube / gap, direction)) > BEHIND_MIN_COS
        )

        if not already_behind:
            # line up directly behind the cube on the cube->goal line, so the stroke pushes
            # through the cube's centre and induces as little spin as possible
            reach_xy = cube_p[:2] - direction * PUSH_APPROACH_STANDOFF
            base_xy = env.agent.robot.pose.sp.p[:2]
            if i > 0 and np.linalg.norm(reach_xy - base_xy) > REFINE_REACH_LIMIT:
                break
            res = _approach(planner, env, reach_xy, push_z, lift=needs_lift or i > 0)
            if res == -1:
                return -1

        # stop the TCP one contact offset short of the goal centre, less the lag the arm
        # will not close, which puts the cube centre on it
        stroke_xy = goal_xy - direction * (CONTACT_OFFSET - PUSH_LAG)
        res = _move(planner, env, stroke_xy, push_z, q)
        if res == -1:
            return -1
    return res


def solve(env: PushTwoCubesEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    env = env.unwrapped
    # The env already resets with the fingers closed, so only latch the state the follow_path
    # actions carry -- calling close_gripper() here would step the env 6 times with the arm
    # pinned and write 6 dead zero-action frames into every demo.
    planner.gripper_state = planner.CLOSED

    # cubeA (robot's right) first, then cubeB (robot's left) -- always this order
    # cubeA is approached straight from the home pose with nothing to clear; everything after
    # that starts next to a cube and has to lift over it
    for i, (cube, goal) in enumerate(
        ((env.cubeA, env.goal_regionA), (env.cubeB, env.goal_regionB))
    ):
        res = _push(planner, env, cube, goal, needs_lift=i > 0)
        if res == -1:
            planner.close()
            return -1

    planner.close()
    return res
