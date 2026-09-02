import heapq
import random

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import trimesh
from transforms3d import quaternions
from mani_skill.utils.structs import Actor
from mani_skill.utils import common
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh


def get_actor_obb(actor: Actor, to_world_frame=True, vis=False):
    mesh = get_component_mesh(
        actor._objs[0].find_component_by_type(physx.PhysxRigidDynamicComponent),
        to_world_frame=to_world_frame,
    )
    assert mesh is not None, "can not get actor mesh for {}".format(actor)

    obb: trimesh.primitives.Box = mesh.bounding_box_oriented

    if vis:
        obb.visual.vertex_colors = (255, 0, 0, 10)
        trimesh.Scene([mesh, obb]).show()

    return obb


def compute_grasp_info_by_obb(
    obb: trimesh.primitives.Box,
    approaching=(0, 0, -1),
    target_closing=None,
    depth=0.0,
    ortho=True,
):
    """Compute grasp info given an oriented bounding box.
    The grasp info includes axes to define grasp frame, namely approaching, closing, orthogonal directions and center.

    Args:
        obb: oriented bounding box to grasp
        approaching: direction to approach the object
        target_closing: target closing direction, used to select one of multiple solutions
        depth: displacement from hand to tcp along the approaching vector. Usually finger length.
        ortho: whether to orthogonalize closing  w.r.t. approaching.
    """
    # NOTE(jigu): DO NOT USE `x.extents`, which is inconsistent with `x.primitive.transform`!
    extents = np.array(obb.primitive.extents)
    T = np.array(obb.primitive.transform)

    # Assume normalized
    approaching = np.array(approaching)

    # Find the axis closest to approaching vector
    angles = approaching @ T[:3, :3]  # [3]
    inds0 = np.argsort(np.abs(angles))
    ind0 = inds0[-1]

    # Find the shorter axis as closing vector
    inds1 = np.argsort(extents[inds0[0:-1]])
    ind1 = inds0[0:-1][inds1[0]]
    ind2 = inds0[0:-1][inds1[1]]

    # If sizes are close, choose the one closest to the target closing
    if target_closing is not None and 0.99 < (extents[ind1] / extents[ind2]) < 1.01:
        vec1 = T[:3, ind1]
        vec2 = T[:3, ind2]
        if np.abs(target_closing @ vec1) < np.abs(target_closing @ vec2):
            ind1 = inds0[0:-1][inds1[1]]
            ind2 = inds0[0:-1][inds1[0]]
    closing = T[:3, ind1]

    # Flip if far from target
    if target_closing is not None and target_closing @ closing < 0:
        closing = -closing

    # Reorder extents
    extents = extents[[ind0, ind1, ind2]]

    # Find the origin on the surface
    center = T[:3, 3].copy()
    half_size = extents[0] * 0.5
    center = center + approaching * (-half_size + min(depth, half_size))

    if ortho:
        closing = closing - (approaching @ closing) * approaching
        closing = common.np_normalize_vector(closing)

    grasp_info = dict(
        approaching=approaching, closing=closing, center=center, extents=extents
    )
    return grasp_info


def _push_dir(obj_p, target_p):
    """Compute the normalized 2D direction vector from obj_p to target_p."""
    delta = target_p[:2] - obj_p[:2]
    return delta / np.linalg.norm(delta)


def _episode_over(res):
    """True once the env has ended the episode (terminated or truncated)."""
    if res is None or res == -1:
        return False
    terminated, truncated = res[2], res[3]
    return bool(terminated) or bool(truncated)


def _point_segment_dist(p, a, b):
    """Distance from point p to the segment ab (all 2D)."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _lift_clear(planner, push_quat, retreat_height, res):
    """Move the TCP straight to retreat_height, unless it is already at that height."""
    tcp_p = planner.base_env.agent.tcp.pose.sp.p
    if abs(tcp_p[2] - retreat_height) < 5e-3:
        return res
    return planner.move_to_pose_with_screw(
        sapien.Pose(p=[tcp_p[0], tcp_p[1], retreat_height], q=push_quat)
    )


def assign_push_pairs(cubeA, cubeB, targetA, targetB):
    """Pick which cube goes to which target, plus the order to push them in, for a
    PushBlock-style task where cubes have no spawn-side identity.

    Returns a list of (cube, target) tuples in the order to push them.
    """
    pairs = random.choice(
        [
            [(cubeA, targetA), (cubeB, targetB)],
            [(cubeA, targetB), (cubeB, targetA)],
        ]
    )
    random.shuffle(pairs)
    return pairs


def push_object_closed_loop(
    planner,
    obj,
    target,
    push_quat,
    push_height,
    contact_clearance,
    success_radius,
    standoff=0.05,
    contact_offset=0.005,
    success_margin=0.01,
    max_passes=10,
    max_stroke=0.20,
    seat_tol=0.015,
    settle_steps=0,
    retreat_height=0.15,
):
    """Push obj onto target as one continuos stroke per pass, re-observing between
    passes.

    Args:
        push_height: fixed z coordinate for every commanded pose.
        contact_clearance: xy distance from obj's centre to where the TCP should sit to
            just be touching its back face (typically obj's half-size along the push axis).
        max_passes: cap on re-observe-and-correct rounds per object.
        settle_steps: extra physics steps to hold at the end of a stroke before
            re-reading obj's pose.
        retreat_height: z coordinate the TCP lifts to before transiting to standoff
            or between objects to avoid collisions.
    """
    # First lift: the TCP arrives here either from the robot's rest pose or from the
    # previous cube's push-end pose, and in both cases a direct low transit is the failure
    # mode described under retreat_height.
    res = _lift_clear(planner, push_quat, retreat_height, None)
    if res == -1:
        return res

    for pass_idx in range(max_passes):
        if _episode_over(res):
            break
        obj_p = obj.pose.p[0].cpu().numpy()
        target_p = target.pose.p[0].cpu().numpy()
        remaining = np.linalg.norm(target_p[:2] - obj_p[:2])
        if remaining < success_radius - success_margin:
            break
        push_dir = _push_dir(obj_p, target_p)
        pre_xy = obj_p[:2] - push_dir * (contact_clearance + standoff)

        # A capped stroke ends with the TCP exactly at obj's back face on the push line,
        # so the next stroke can start immediately: no retreat, no re-approach, no backing
        # off by the standoff and running up again. A long push therefore executes as
        # back-to-back strokes with only the settle between them, which is closer to
        # BESO's per-control-step re-observation than one open-loop stroke was.
        tcp_p = planner.base_env.agent.tcp.pose.sp.p
        contact_xy = obj_p[:2] - push_dir * (contact_clearance + contact_offset)
        seated = (
            np.linalg.norm(tcp_p[:2] - contact_xy) < seat_tol
            and abs(tcp_p[2] - push_height) < 0.01
        )

        # A correction pass usually only nudges push_dir, so the TCP can slide straight
        # to the new standoff point at push height. It must NOT when the new standoff
        # point is on the far side of obj (the overshoot case, where push_dir flips):
        # that straight line would plough through the cube it is meant to re-approach.
        if pass_idx > 0 and not seated:
            if (
                _point_segment_dist(obj_p[:2], tcp_p[:2], pre_xy)
                < contact_clearance + contact_offset
            ):
                retreat = _lift_clear(planner, push_quat, retreat_height, res)
                if retreat == -1:
                    break
                res = retreat

        # Standoff ladder. The descent from retreat_height to push_height is a straight
        # screw motion, and deep in the workspace (x approaching the robot's own base) it
        # can be unplannable at the full standoff while being fine a couple of centimetres
        # closer in -- measured: the failures it fixes are all approaches at x < -0.29,
        # where the arm has to fold up against its base. Shrinking the standoff only
        # shortens the free run-up before contact; the stroke itself is unaffected.
        approach = res if seated else -1
        for standoff_frac in () if seated else (1.0, 0.5, 0.0):
            cand_xy = obj_p[:2] - push_dir * (
                contact_clearance + standoff * standoff_frac
            )
            cand = sapien.Pose(p=[cand_xy[0], cand_xy[1], push_height], q=push_quat)
            tcp_p = planner.base_env.agent.tcp.pose.sp.p
            if np.linalg.norm(tcp_p - cand.p) < 5e-3:
                # Already there. Planning to the pose the TCP occupies yields a zero-length
                # path and follow_path then raises UnboundLocalError (same trap as
                # _lift_clear). Reachable on a correction pass, where the TCP sits at the
                # previous stroke's end and the cube barely moved.
                approach = res
                break
            approach = planner.move_to_pose_with_screw(cand)
            if approach != -1:
                break
        if approach == -1:
            break
        res = approach

        # The single stroke. pre_xy and end_xy both lie on the obj->target line, so this
        # is one straight Cartesian translation: standoff of free travel, then sustained
        # contact all the way to the target without ever breaking off.
        # Cap the advance. A full 0.4m straight-line Cartesian path at push height is
        # often unplannable by plan_screw from a given configuration -- 3 of 49 strokes
        # measured -- and each failure used to drop into the chunked fallback, whose
        # stop-and-go action profile is exactly what must NOT contaminate a demo set.
        # Capping the advance keeps every stroke continuous while giving the planner a
        # path length it can actually solve.
        advance = min(remaining, max_stroke)
        end_xy = (
            obj_p[:2]
            + push_dir * advance
            - push_dir * (contact_clearance + contact_offset)
        )
        end_pose = sapien.Pose(p=[end_xy[0], end_xy[1], push_height], q=push_quat)
        stroke = planner.move_to_pose_with_screw(end_pose, refine_steps=settle_steps)
        if _episode_over(stroke):
            return stroke
        if stroke == -1:
            # Out of reach or unplannable from here -- typically after obj overshot and
            # the corrective push would need the TCP on the far side of the target, off
            # the table. Give up on obj rather than on the episode: the other object may
            # still be placeable, and a genuinely-attempted failure is a more honest
            # (and more filterable) trajectory than an aborted one.
            #
            # There used to be a chunked fallback here -- advance in 4cm segments,
            # replanning each -- which recovered these strokes at the cost of a visibly
            # stop-and-go action profile. It was worth +4/30 successes back when a stroke
            # was the whole 0.4m push and failed 3 times in 49. With max_stroke capping
            # the advance it never fires at all (0 failures in 65 strokes; ablating it
            # changes nothing, 23/30 either way), and it is deliberately NOT kept as a
            # safety net: these trajectories feed a diffusion policy, and a rare
            # stuttering demo silently contaminates the action distribution. A clean
            # failure gets dropped by --only-count-success; a contaminated success does
            # not. Restore it only if yield ever matters more than demo purity.
            break
        res = stroke
    # None only if nothing at all was executed: the first lift was skipped as a no-op and
    # the first approach failed to plan. Report that as a motion-planning failure, which
    # is what callers expect and what it is.
    return res if res is not None else -1


def is_path_clear_2d(p1, p2, obstacles, radius):
    """Check if the 2D segment p1 -> p2 is at least radius away from all obstacles."""
    for obs in obstacles:
        if _point_segment_dist(obs, p1, p2) < radius:
            return False
    return True


def plan_2d_waypoints(
    start,
    goal,
    obstacles,
    radius=0.042,
    x_bounds=(-0.35, 0.05),
    y_bounds=(-0.45, 0.25),
):
    """Plan collision-free 2D waypoints at fixed z from start to goal.
    Uses disengagement, local visibility graph, and corridor routing."""
    start = np.asarray(start[:2], dtype=float)
    goal = np.asarray(goal[:2], dtype=float)

    if is_path_clear_2d(start, goal, obstacles, radius):
        return [goal]

    # Check if start is currently close to any obstacle (e.g. just pushed it).
    # If so, disengage first: find direction away from the closest obstacle.
    disengage_pts = []
    curr = start.copy()
    for obs in obstacles:
        d = float(np.linalg.norm(curr - obs))
        if d < radius:
            away = (curr - obs) / (d + 1e-9)
            step = obs + away * (radius + 0.015)
            step[0] = np.clip(step[0], x_bounds[0], x_bounds[1])
            step[1] = np.clip(step[1], y_bounds[0], y_bounds[1])
            disengage_pts.append(step)
            curr = step
            break

    if is_path_clear_2d(curr, goal, obstacles, radius):
        return disengage_pts + [goal]

    nodes = [curr, goal]
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    for obs in obstacles:
        for ang in angles:
            cand = obs + (radius + 0.015) * np.array([np.cos(ang), np.sin(ang)])
            cand[0] = np.clip(cand[0], x_bounds[0], x_bounds[1])
            cand[1] = np.clip(cand[1], y_bounds[0], y_bounds[1])
            if all(np.linalg.norm(cand - o) >= radius for o in obstacles):
                nodes.append(cand)

    # Corridor waypoints
    for y in [-0.42, -0.20, 0.0, 0.22]:
        for x in [x_bounds[0], -0.25, -0.12, 0.0, x_bounds[1]]:
            cand = np.array([x, y])
            if all(np.linalg.norm(cand - o) >= radius for o in obstacles):
                nodes.append(cand)

    n = len(nodes)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if is_path_clear_2d(nodes[i], nodes[j], obstacles, radius):
                d = float(np.linalg.norm(nodes[i] - nodes[j]))
                adj[i].append((j, d))
                adj[j].append((i, d))

    dist = {i: float("inf") for i in range(n)}
    parent = {i: None for i in range(n)}
    dist[0] = 0
    pq = [(0, 0)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == 1:
            break
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                heapq.heappush(pq, (dist[v], v))

    if dist[1] == float("inf"):
        # Fallback: baseline routing
        p1 = np.array([curr[0], -0.42])
        p2 = np.array([goal[0], -0.42])
        return disengage_pts + [p1, p2, goal]

    path = []
    c = 1
    while c is not None:
        path.append(nodes[c])
        c = parent[c]
    path.reverse()

    # Shortcut path
    if len(path) > 2:
        sc = [path[0]]
        i = 0
        while i < len(path) - 1:
            for j in range(len(path) - 1, i, -1):
                if is_path_clear_2d(sc[-1], path[j], obstacles, radius):
                    sc.append(path[j])
                    i = j
                    break
            else:
                i += 1
                sc.append(path[i])
        path = sc

    return disengage_pts + path[1:]


def push_object_planar_closed_loop(
    planner,
    obj,
    target,
    push_quat,
    push_height,
    contact_clearance,
    success_radius,
    other_obstacles=None,
    standoff=0.04,
    success_margin=0.01,
    max_passes=12,
    max_stroke=0.15,
    max_lat_slip=0.022,
    max_lead_slip=0.005,
    settle_steps=0,
    obs_radius=0.042,
):
    """Push obj onto target while keeping the TCP strictly at push_height (no lifting in z).
    Navigates around obstacles in the 2D plane with slip detection to prevent pushing empty air."""
    if other_obstacles is None:
        other_obstacles = []

    res = None
    for pass_idx in range(max_passes):
        if _episode_over(res):
            break
        obj_p = obj.pose.p[0].cpu().numpy()
        target_p = target.pose.p[0].cpu().numpy()
        rem = float(np.linalg.norm(target_p[:2] - obj_p[:2]))
        if rem < success_radius - success_margin:
            break

        push_dir = _push_dir(obj_p, target_p)
        contact_xy = obj_p[:2] - push_dir * contact_clearance
        pre_xy = contact_xy - push_dir * standoff
        tcp_p = planner.base_env.agent.tcp.pose.sp.p

        # Check relative position along and perpendicular to push_dir
        rel = tcp_p[:2] - obj_p[:2]
        proj = float(np.dot(rel, push_dir))
        perp = rel - proj * push_dir
        lat = float(np.linalg.norm(perp))

        # Seated if right behind the cube and aligned laterally
        seated = (
            -(contact_clearance + 0.015) < proj < -(contact_clearance - 0.010)
            and lat < 0.018
            and abs(tcp_p[2] - push_height) < 0.01
        )

        if not seated:
            obstacles = [
                obs.pose.p[0].cpu().numpy()[:2] if hasattr(obs, "pose") else obs[:2]
                for obs in other_obstacles
            ]
            # If moving directly to pre_xy would pass through obj itself, treat obj as obstacle
            if _point_segment_dist(obj_p[:2], tcp_p[:2], pre_xy) < obs_radius:
                obstacles.append(obj_p[:2])

            wps = plan_2d_waypoints(tcp_p, pre_xy, obstacles, radius=obs_radius)
            for wp in wps:
                wp_pose = sapien.Pose(p=[wp[0], wp[1], push_height], q=push_quat)
                step_res = planner.move_to_pose_with_screw(wp_pose)
                if step_res == -1 or _episode_over(step_res):
                    res = step_res
                    break
                res = step_res
            if _episode_over(res):
                break

        advance = min(rem, max_stroke)
        end_xy = (
            obj_p[:2]
            + push_dir * advance
            - push_dir * contact_clearance
        )
        end_pose = sapien.Pose(p=[end_xy[0], end_xy[1], push_height], q=push_quat)

        # Plan screw trajectory and execute step-by-step with slip detection
        plan_res = planner.move_to_pose_with_screw(end_pose, dry_run=True)
        if plan_res == -1 or plan_res.get("status") != "Success":
            stroke = planner.move_to_pose_with_screw(end_pose, refine_steps=settle_steps)
            if stroke == -1 or _episode_over(stroke):
                res = stroke
                break
            res = stroke
        else:
            n_step = plan_res["position"].shape[0]
            for i in range(n_step):
                action = np.hstack([plan_res["position"][i]])
                obs, reward, terminated, truncated, info = planner.env.step(action)
                res = [obs, reward, terminated, truncated, info]
                if terminated or truncated:
                    break
                # Live slip check: terminate early if stick slips off face or leads cube
                tcp_now = planner.base_env.agent.tcp.pose.sp.p
                obj_now = obj.pose.p[0].cpu().numpy()
                rel_now = tcp_now[:2] - obj_now[:2]
                proj_now = float(np.dot(rel_now, push_dir))
                perp_now = rel_now - proj_now * push_dir
                lat_now = float(np.linalg.norm(perp_now))
                if proj_now > max_lead_slip or lat_now > max_lat_slip:
                    break
            if _episode_over(res):
                break

    return res if res is not None else -1
