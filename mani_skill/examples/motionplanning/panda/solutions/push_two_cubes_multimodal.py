"""Motion planning solution for PushTwoCubesMultimodal-v1.

Upgrades solutions/push_two_cubes.py to support all 4 modalities and unlocked cube z-rotation:
- Mode 0: Straight (A -> goalA, B -> goalB), start with cubeA (25%, vanilla PushTwoCubes)
- Mode 1: Straight (A -> goalA, B -> goalB), start with cubeB (25%)
- Mode 2: Criss-cross (A -> goalB, B -> goalA), start with cubeA (25%)
- Mode 3: Criss-cross (A -> goalB, B -> goalA), start with cubeB (25%)

Key upgrades:
1. Lift-and-descend approach (lift=True on all initial approaches) so the arm never drags
   or presses over the upper face of either cube.
2. Tilt-compensated contact alignment: calculates cube tilt angle relative to the push vector
   and applies a lateral offset along the perpendicular direction to generate restoring torque,
   keeping the cube balanced and moving smoothly.
3. Live slip and early goal detection: terminates the push stroke immediately if the cube
   reaches the goal or slips away from the hand, eliminating laggy overshoots.
4. Robust segment fallback: falls back to collinear waypoint subdivision (_move) if direct
   screw planning balks.
5. Wrapper-safe execution: routes all step() calls through planner.env so wrappers like
   RecordEpisode capture every single frame continuously without skipping.
"""

import random
from typing import Optional

import numpy as np
import sapien
import transforms3d

from mani_skill.envs.tasks import PushTwoCubesMultimodalEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)

# Geometry and contact constants
CUBE_HALF_SIZE = 0.02
CONTACT_OFFSET = 0.0295
PUSH_LAG = 0.0060
PUSH_APPROACH_STANDOFF = 0.05
LIFT_HEIGHT = 0.12
PUSH_TOLERANCE = 0.02
MAX_PUSH_PASSES = 6
REFINE_REACH_LIMIT = 0.80

# Tilt compensation gain (meters of lateral offset per radian of tilt)
TILT_GAIN = 0.020
MAX_LATERAL_OFFSET = 0.010


def _move(planner: PandaArmMotionPlanningSolver, xy, z, q):
    """Drive the TCP in a straight line to (`xy`, `z`), subdividing if the planner balks."""
    end = np.array([xy[0], xy[1], z])
    res = planner.move_to_pose_with_screw(sapien.Pose(p=end, q=q))
    if res != -1:
        return res

    for segments in (2, 4):
        start = planner.base_env.agent.tcp.pose.sp.p.copy()
        for k in range(1, segments + 1):
            waypoint = start + (end - start) * (k / segments)
            res = planner.move_to_pose_with_screw(sapien.Pose(p=waypoint, q=q))
            if res == -1:
                break
        if res != -1:
            return res
    return -1


def _approach(planner: PandaArmMotionPlanningSolver, xy, z, lift=True):
    """Move the TCP to (`xy`, `z`), lifting over anything in the way when `lift`."""
    q = planner.base_env.agent.tcp.pose.sp.q
    cur = planner.base_env.agent.tcp.pose.sp.p
    if not lift:
        waypoints = [(xy, z)]
    else:
        top_z = max(cur[2], z) + LIFT_HEIGHT
        waypoints = [(cur[:2], top_z), (xy, top_z), (xy, z)]
    res = None
    for wp_xy, wp_z in waypoints:
        res = _move(planner, wp_xy, wp_z, q)
        if res == -1:
            return -1
    return res


def get_cube_tilt_angle(cube_quat: np.ndarray, push_dir_2d: np.ndarray) -> float:
    """Calculate the signed tilt angle of the cube's pushing face relative to push_dir.

    Returns an angle in [-pi/4, pi/4]. Positive angle indicates counter-clockwise tilt.
    """
    R = transforms3d.quaternions.quat2mat(cube_quat)
    # The 4 face normals in world 2D:
    normals = [R[:2, 0], -R[:2, 0], R[:2, 1], -R[:2, 1]]
    # Back face has the maximum alignment with -push_dir_2d
    dots = [np.dot(n, -push_dir_2d) for n in normals]
    best_idx = int(np.argmax(dots))
    best_normal = normals[best_idx]

    cos_tilt = np.dot(-best_normal, push_dir_2d)
    sin_tilt = (-best_normal[0]) * push_dir_2d[1] - (-best_normal[1]) * push_dir_2d[0]
    return float(np.arctan2(sin_tilt, cos_tilt))


def _push(planner: PandaArmMotionPlanningSolver, cube, goal):
    """Push `cube` onto `goal` with tilt-compensated contact, live slip detection, and continuous stepping."""
    env_u = planner.base_env
    q = env_u.agent.tcp.pose.sp.q
    res = None
    push_z = CUBE_HALF_SIZE

    for pass_idx in range(MAX_PUSH_PASSES):
        cube_p = cube.pose.sp.p
        cube_q = cube.pose.sp.q
        goal_xy = goal.pose.sp.p[:2]

        err = goal_xy - cube_p[:2]
        dist = np.linalg.norm(err)
        if dist < PUSH_TOLERANCE:
            break
        push_dir = err / dist
        perp_dir = np.array([-push_dir[1], push_dir[0]])

        # Tilt angle in [-pi/4, pi/4]
        tilt = get_cube_tilt_angle(cube_q, push_dir)
        lat_offset = np.clip(
            TILT_GAIN * np.sin(tilt), -MAX_LATERAL_OFFSET, MAX_LATERAL_OFFSET
        )

        # Line up behind the cube along push_dir, offset by lat_offset along perp_dir
        standoff_xy = (
            cube_p[:2]
            - push_dir * (CUBE_HALF_SIZE + PUSH_APPROACH_STANDOFF)
            + perp_dir * lat_offset
        )

        tcp_xy = env_u.agent.tcp.pose.sp.p[:2]
        to_standoff = standoff_xy - tcp_xy
        already_there = (pass_idx > 0) and (np.linalg.norm(to_standoff) < 0.035)

        if not already_there:
            base_xy = env_u.agent.robot.pose.sp.p[:2]
            if pass_idx > 0 and np.linalg.norm(standoff_xy - base_xy) > REFINE_REACH_LIMIT:
                break
            # Always approach with lift=True so we move high and descend vertically behind the cube
            res = _approach(planner, standoff_xy, push_z, lift=True)
            if res == -1:
                return -1

        stroke_xy = (
            goal_xy
            - push_dir * (CONTACT_OFFSET - PUSH_LAG)
            + perp_dir * lat_offset
        )
        stroke_pose = sapien.Pose(p=[stroke_xy[0], stroke_xy[1], push_z], q=q)

        plan_res = planner.move_to_pose_with_screw(stroke_pose, dry_run=True)
        if plan_res == -1 or plan_res["status"] != "Success":
            # Fall back to direct _move with waypoint subdivision
            res = _move(planner, stroke_xy, push_z, q)
            if res == -1:
                return -1
            continue

        # Execute planned path with live slip and early-goal detection
        positions = plan_res["position"]
        n_steps = len(positions)
        for step_i in range(n_steps):
            qpos = positions[step_i]
            if planner.control_mode == "pd_joint_pos_vel":
                qvel = plan_res["velocity"][step_i]
                action = np.hstack([qpos, qvel, planner.gripper_state])
            else:
                action = np.hstack([qpos, planner.gripper_state])

            obs, reward, term, trunc, info = planner.env.step(action)
            planner.elapsed_steps += 1
            if planner.vis:
                planner.base_env.render_human()
            res = (obs, reward, term, trunc, info)

            cur_cube_p = cube.pose.sp.p
            cur_tcp_p = env_u.agent.tcp.pose.sp.p
            cur_dist = np.linalg.norm(goal_xy - cur_cube_p[:2])
            if cur_dist < PUSH_TOLERANCE:
                # Goal reached: stop stroke immediately
                break

            rel = cur_cube_p[:2] - cur_tcp_p[:2]
            along = np.dot(rel, push_dir)
            lateral = np.linalg.norm(rel - along * push_dir)
            # Slip check: if TCP outruns cube along push_dir or cube slips laterally
            if step_i > 5 and (along < -0.015 or lateral > 0.05):
                break

    return res


def select_modality(seed: Optional[int] = None, modality: Optional[int] = None) -> int:
    """Select one of 4 modalities (0, 1, 2, 3) uniformly (25% each).

    - 0: Straight (A->A, B->B), start A
    - 1: Straight (A->A, B->B), start B
    - 2: Criss-cross (A->B, B->A), start A
    - 3: Criss-cross (A->B, B->A), start B
    """
    if modality is not None:
        assert 0 <= modality < 4, f"Modality must be in [0, 3], got {modality}"
        return modality
    if seed is not None:
        rng = np.random.RandomState(seed)
        cross = bool(rng.rand() < 0.5)
        start_with_B = bool(rng.rand() < 0.5)
    else:
        cross = bool(random.random() < 0.5)
        start_with_B = bool(random.random() < 0.5)
    return (2 * int(cross)) + int(start_with_B)


def solve(
    env: PushTwoCubesMultimodalEnv,
    seed: Optional[int] = None,
    debug: bool = False,
    vis: bool = False,
    modality: Optional[int] = None,
):
    """Solve PushTwoCubesMultimodal-v1 across all 4 modalities."""
    env.reset(seed=seed)
    env_u = env.unwrapped
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env_u.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    planner.gripper_state = planner.CLOSED

    selected_mode = select_modality(seed=seed, modality=modality)
    is_cross = selected_mode in (2, 3)
    start_with_B = selected_mode in (1, 3)

    if not is_cross:
        pairA = (env_u.cubeA, env_u.goal_regionA)
        pairB = (env_u.cubeB, env_u.goal_regionB)
    else:
        pairA = (env_u.cubeA, env_u.goal_regionB)
        pairB = (env_u.cubeB, env_u.goal_regionA)

    pairs = [pairB, pairA] if start_with_B else [pairA, pairB]

    for cube, goal in pairs:
        res = _push(planner, cube, goal)
        if res == -1:
            planner.close()
            return -1

    planner.close()
    return res
