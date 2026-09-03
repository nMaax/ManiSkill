# Custom environments in this fork

This fork of [ManiSkill](https://github.com/haosulab/ManiSkill) adds a family of tabletop
environments derived from `StackCube-v1`, built for goal-conditioned policy training. Everything
here lives in `mani_skill/envs/tasks/tabletop/`; upstream code is otherwise unmodified.

## The environments

All ranges are metres relative to the table centre. The Panda base sits at `x = -0.615`, so a more
negative `x` is *closer* to the robot. `+y` is to the robot's left.

**Every environment below defaults to Panda except `PushBlock-v1`, which defaults to
`panda_stick`** (`xarm6_nogripper` and `panda` remain available as opt-ins)
— see "How Push Block works" below for why.

| env id | spawn x | spawn y | cube rotation | goal | file |
|---|---|---|---|---|---|
| `StackCube-v1` *(upstream, untouched)* | −0.20 … 0.20 | −0.30 … 0.30 | free | stack A on B | `stack_cube.py` |
| `StackCubeSwapped-v1` | −0.20 … 0.20 | −0.30 … 0.30 | free | stack **B on A** | `stack_cube_swapped.py` |
| `StackCubeRestrictedSpawn-v1` | −0.08 … 0.08 | −0.08 … 0.08 | free | stack A on B | `stack_cube_restricted_spawn.py` |
| `StackCubeLockedRotation-v1` | −0.20 … 0.12 | −0.13 … 0.13 | **locked** | stack A on B | `stack_cube_locked_rotation.py` |
| `StackCubeSwappedLockedRotation-v1` | −0.20 … 0.12 | −0.13 … 0.13 | **locked** | stack **B on A** | `stack_cube_swapped_locked_rotation.py` |
| `StackCubeClutter-v1` | −0.20 … 0.20 | −0.30 … 0.30 | free | stack A on B, **3–6 clutter distractors** | `stack_cube_clutter.py` |
| `StackCubeClutterRandomPick-v1` | −0.20 … 0.20 | −0.30 … 0.30 | free | stack **any pool object on any other**, drawn fresh per episode, 3–6 clutter distractors | `stack_cube_clutter_random_pick.py` |
| `StackCubeClutterLockedRotation-v1` | −0.20 … 0.12 | −0.13 … 0.13 | **locked** | stack A on B, 3–6 clutter distractors | `stack_cube_clutter_locked_rotation.py` |
| `StackCubeClutterRandomPickLockedRotation-v1` | −0.20 … 0.12 | −0.13 … 0.13 | **locked** | stack any pool object on any other (random per episode), 3–6 clutter distractors | `stack_cube_clutter_random_pick_locked_rotation.py` |
| `PlaceCubeLeft-v1` | −0.20 … 0.12 | −0.21 … 0.05 | free | A at B **+0.16 y** | `place_cube_left.py` |
| `PlaceCubeRight-v1` | −0.20 … 0.12 | −0.05 … 0.21 | free | A at B **−0.16 y** | `place_cube_right.py` |
| `PlaceCubeLeftLockedRotation-v1` | −0.20 … 0.12 | −0.21 … 0.05 | **locked** | A at B +0.16 y | `place_cube_left_locked_rotation.py` |
| `PlaceCubeRightLockedRotation-v1` | −0.20 … 0.12 | −0.05 … 0.21 | **locked** | A at B −0.16 y | `place_cube_right_locked_rotation.py` |
| `PlaceSphereRestrictedSpawn-v1` | −0.089 … 0.082 | −0.132 … 0.131 | inherited | sphere in bin | `place_sphere_restricted_spawn.py` |
| `PushBlock-v1` *(push runs along **y**, not x — see below)* | −0.222 … −0.022 (cubes), −0.122 ± 0.12 (targets) | −0.35 … −0.05 (cubes, shared), 0.20 (targets, near-fixed) | free | push A,B into two **distinct** targets, order-agnostic (matches BESO's own success rule) | `push_block.py` |
| `PushTwoCubes-v1` *(a PushCube extension, not a PushBlock variant — see below)* | −0.08 … −0.02 (cubes), 0.15 (targets, fixed) | −0.19 … −0.13 (cubeA) / 0.13 … 0.19 (cubeB), ∓0.16 (targets, fixed) | locked (never randomized) | push A into targetA **and** B into targetB, pairing and order both **fixed** | `push_two_cubes.py` |

**There is exactlu one file per registered environment.**, e.g. `PlaceCubeLeftLockedRotation-v1` is in `place_cube_left_locked_rotation.py`.

Note that `*LockedRotation-v1` environments restrict the spawn *as well as* locking rotation — the name only mentions the more distinctive of the two.

The spawn x/y columns above describe cubeA/cubeB only. `StackCubeClutter*` environments also
scatter 3–6 extra distractor objects across a wider region (`x ∈ [-0.25, 0.25]`,
`y ∈ [-0.35, 0.35]`) that isn't captured by those columns — see "How the clutter variants work"
below.

## Spawn regions

![Spawn regions for every environment](figures/custom_envs_spawn_regions.png)

1500 spawns per environment at seed 0, viewed top-down. Each row is a pair: the left column is a base
task, the right column the variant derived from it. Note the axes are transposed relative to
the usual plot convention: **y (the robot's left) is horizontal, x (away from the robot) is
vertical**, so the layout matches what you see in a render. The black square at the bottom of each
panel is the Panda base at `x = -0.615`.

Reading a panel:

- **red** = cubeA, **green** = cubeB (`obj` / `bin` on the sphere panel), **blue** = the goal, i.e.
  `cubeB + TARGET_Y_OFFSET`, on the `PlaceCube` panels only. Which cube the robot picks up depends
  on the task — cubeA everywhere except the swapped variants, which pick up cubeB
- **solid black box** = the declared spawn region. On the two vanilla-spawn panels a **dotted grey
  box** shows the rejection sampler's own bounds, with the solid box being the wider envelope that
  `SHARED_XY_OFFSET` produces on top of it
- **dashed blue box** = where the goals land
- **red arc** = the ~0.82 m outer limit of the Panda's reach; **orange arc** = 0.736 m, the worst
  case actually observed in the restricted envs

The two vanilla-spawn environments (`StackCube-v1`, `StackCubeSwapped-v1`) are the only ones that
cross the reach arc, with ~0.17% of spawns outside it — which is what caps motion planning on
`StackCube-v1` at ~0.975 rather than 1.0. Every restricted variant sits entirely inside.

On the `PlaceCube` panels the solid and dashed boxes together span a symmetric ±0.21 m in y, with
neither crossing the reach arc: that is the `GOAL_Y_LIMIT` construction described below. Left and
Right are exact mirrors.

Several panels are deliberately indistinguishable from each other, because the figure only shows
position: `LOCK_Z_ROTATION` changes yaw alone, and `StackCubeSwappedLockedRotation-v1` differs from
`StackCubeLockedRotation-v1` only in which cube is picked up. Those pairs are in fact bit-identical
in spawn at a given seed — the difference is in the task, not the layout.

## How the spawn variants work

`ConfigurableSpawnStackCubeEnv` (`configurable_spawn_stack_cube.py`) is a non-registered base class
that exposes `StackCubeEnv`'s cube spawn as five class constants:

| constant | meaning | vanilla value |
|---|---|---|
| `CUBE_X_RANGE`, `CUBE_Y_RANGE` | rejection-sampling bounds for both cubes | `(-0.1, 0.1)`, `(-0.2, 0.2)` |
| `SHARED_XY_OFFSET` | per-episode shift applied to both cubes, leaving their separation unchanged; `0` disables it | `0.1` |
| `SPAWN_CLEARANCE` | added to the cube circumradius when rejection sampling, i.e. how far apart the two cubes must land | `0.001` |
| `LOCK_Z_ROTATION` | spawn cubes at identity yaw instead of a random one | `False` |

At the defaults it reproduces `StackCubeEnv._initialize_episode` exactly, RNG draws included.
Variants override constants; none of them reimplement the spawn.

**One exception to the single-inheritance pattern.** `StackCubeSwappedLockedRotation-v1` combines two
otherwise orthogonal parents:

```python
class StackCubeSwappedLockedRotationEnv(StackCubeSwappedEnv, StackCubeLockedRotationEnv):
    ...  # no body
```

`StackCubeSwappedEnv` owns the *task* (`evaluate`, `compute_dense_reward`) and descends straight from
`StackCubeEnv`; `StackCubeLockedRotationEnv` owns the *spawn* (`_initialize_episode` via the base,
plus the four constants). The MRO is
`Swapped → LockedRotation → ConfigurableSpawn → StackCube`, so each concern resolves to the parent
that defines it and there is nothing to write in the body. Neither parent is modified, and its
spawn is bit-identical to `StackCubeLockedRotation-v1` at the same seed.

The alternative — reparenting `StackCubeSwappedEnv` onto the configurable base and re-declaring the
four constants — would duplicate numbers that can then drift apart. If you add a third axis of
variation, prefer extending this pattern over copying constants.

**`stack_cube.py` must stay byte-identical to upstream.** `StackCube-v1` has published episodes on
the ManiSkill demo server that must remain reproducible. The base class exists precisely so that
parameterising the spawn does not require touching it.

## How the clutter variants work

`StackCubeClutterEnv` (`stack_cube_clutter.py`) extends `ConfigurableSpawnStackCubeEnv` with a
cluttered table: every episode, a random number of extra dynamic distractor objects (cubes in
sizes/colors distinct from cubeA/cubeB) are scattered at random, non-overlapping positions. In
`StackCubeClutter-v1` itself the robot never grasps them — they exist purely to make reaching and
placing harder; the `StackCubeClutterRandomPick*` variants (below) can draw any of them as the
actual pick/target.

**Cubes only, for now.** `CLUTTER_SPECS` originally cycled cube/cylinder/sphere shapes, but
cylinders and spheres have no flat face for the OBB-based motion planner's antipodal
parallel-jaw grasp (`compute_grasp_info_by_obb` in
`mani_skill/examples/motionplanning/base_motionplanner/utils.py`) — they slip once placed on top
of another object. Restricted to cubes repo-wide (every `StackCubeClutter*` env shares
`CLUTTER_SPECS` from this base class) until the solver — or a friction/geometry-aware grasp
strategy — can handle curved surfaces reliably.

**Fixed slots, variable count.** ManiSkill's batched/GPU sim has no per-parallel-env "delete
actor" mechanism, so `StackCubeClutterEnv` builds a fixed `NUM_CLUTTER_SLOTS = 6` actor slots at
scene-load time. Each episode draws a per-environment-index active count
`k ~ Uniform{3, ..., 6}` (`CLUTTER_COUNT_RANGE`), and slot `i` is "active" iff `i < k`. Inactive
slots are **parked**: moved far outside the table's collision footprint (e.g. `x = 2.0`, a
distinct `y` per slot) resting on the ground plane, rather than hidden — there is no per-index
visibility toggle for actors with collision shapes.

> Parking gotcha: `TableSceneBuilder` builds the table as **one solid collision box** from the
> tabletop surface down to the ground, not four legs with an open gap underneath. Parking
> "under the table" would embed the object in solid geometry. Park *outside* the table's xy
> footprint instead, at `z = -table_height + object_half_extent` so it rests without
> interpenetration.

**Clutter reuses cubeA/cubeB's already-placed positions without touching the shared base
class.** `StackCubeClutterEnv._initialize_episode` calls `super()._initialize_episode(...)`
first (placing cubeA/cubeB exactly as `ConfigurableSpawnStackCubeEnv` always has — this is what
keeps every existing registered env byte-identical), then builds a **fresh**
`UniformPlacementSampler` over a wider region (`CLUTTER_X_RANGE`, `CLUTTER_Y_RANGE`) and
manually seeds its `fixture_positions`/`fixtures_radii` from cubeA's and cubeB's just-computed
poses before sampling clutter. Clutter therefore avoids both cubes and every earlier clutter
slot for free, and `configurable_spawn_stack_cube.py`/`stack_cube.py` need zero changes.

**Clutter rotation follows `LOCK_Z_ROTATION` too.** `LOCK_Z_ROTATION = True` locks every object
on the table to identity yaw, not just cubeA/cubeB.

**`StackCubeClutterLockedRotationEnv`** overrides `StackCubeLockedRotationEnv`'s four spawn
constants for cubeA/cubeB (`CUBE_X/Y_RANGE`, `SPAWN_CLEARANCE`, `LOCK_Z_ROTATION`), **and**
`CLUTTER_X_RANGE`/`CLUTTER_Y_RANGE`. The latter two were missing for a while: the base class's
own clutter bounds (`x ∈ [-0.25, 0.25]`, `y ∈ [-0.35, 0.35]`) reach up to **0.93 m** from the
Panda base — well outside the ~0.32–0.82 m reachable annulus — so even in this "reachable"
variant clutter could spawn somewhere the arm physically can't reach (confirmed by a real solver
run: the arm extended fully and still couldn't grasp a clutter object). Fixed by reusing
`CUBE_X_RANGE`/`CUBE_Y_RANGE` for clutter too — their corners were already verified to fall
within 0.435–0.746 m of the base, comfortably inside the annulus with margin.

## How the random pick/target variant works

Every existing swap in this suite (`StackCubeSwapped-v1`) is a *static* choice baked in at
registration time, and the original random-pick attempt only randomized a coin flip between
cubeA/cubeB while treating every clutter object as permanent, non-interactive scenery. That's
too narrow: clutter objects are just ordinary pickable shapes, so the pick and target should be
allowed to come from anywhere in the pool. `StackCubeClutterRandomPickEnv`
(`stack_cube_clutter_random_pick.py`) draws **both** the pick and target objects, every episode,
from the full pool of up to 8 candidates — cubeA, cubeB, and up to 6 clutter objects — instead of
being fixed to the two cubes.

**Pool selection.** `_load_scene` builds `self.pool_objects = [cubeA, cubeB, *clutter_objects]`
plus three episode-invariant length-8 tensors describing each slot's stacking geometry —
`pool_rest_z` (center height above the table when resting alone, reusing `_ClutterSpec.rest_z`
from `stack_cube_clutter.py`, with cubeA/cubeB using `cube_half_size` directly), `pool_bounding_radius`
(xy footprint, reusing `_ClutterSpec.bounding_radius`), and `pool_shape_type`/`pool_size` (for
obs, see below). `_initialize_episode` then draws two **distinct** indices per environment from
whichever pool slots are active that episode (cubeA/cubeB always; clutter slots per
`self.clutter_active`) via a random-key top-k: assign every active slot a `torch.rand` key
(inactive slots get `-1` so they can never win), then `torch.topk(keys, k=2)` — this is both
uniformly random and distinctness-guaranteed (up to float-collision odds), and there's no
existing sampling utility in `mani_skill/envs/utils/randomization/` for "distinct indices from a
variable-size active set", so this is bespoke.

**Stack-then-gather, not `torch.where`.** The old A/B-only version computed both cubeA's and
cubeB's `is_static`/`is_grasping`/pose every step and `torch.where`-selected between them, since
those are `Actor`/`Agent` methods that can't be vectorized across objects directly. With 8
candidates that pattern is generalized to **stack all 8, then `torch.gather`** by `pick_idx`/
`target_idx` — still one `is_static`/`is_grasping` call per pool object (8 calls, a fixed
constant), but a single gather instead of an N-way `torch.where` cascade.
(`Actor.merge`, used by `pick_clutter_ycb.py` to combine genuinely distinct per-env-built
objects into one batched Actor, was considered and ruled out: `cubeA`/`cubeB`/each clutter slot
here are already shared batched Actors across all parallel envs, not per-env objects, so
re-deriving a merged per-env Actor from them isn't a proven operation in this codebase.)

**Success geometry generalizes `StackCubeEnv`'s fixed-cube-size formula.** The original xy/z
stacking checks assumed both objects were the same-size cube (`cube_half_size`). Generalized:
the z-gap between pick and target centers when pick rests directly on target's top surface is
`pick_rest_z + target_rest_z` (each object's own rest height above its center — this collapses
to the original `cube_half_size[2] * 2` when both are same-size cubes), and the xy tolerance is
`target_bounding_radius + 0.005` (pick's center must land within target's own footprint, plus the
original's small slack).

**Observation naming is pool-wide, not A/B-special, and carries no role- or slot-based
shortcuts.** Unlike `StackCubeClutter-v1` (where cubeA/cubeB keep their own `cubeA_pose`/
`cubeB_pose` keys, distinct from the clutter slots' `obj_0`..`obj_5`, because A/B are genuinely
fixed roles in that task), here cubeA/cubeB are just 2 of 8 interchangeable pool members, so
`_get_obs_extra` replaces the inherited asymmetric keys with a uniform `obj_0`..`obj_7` (pool
index 0/1 = cubeA/cubeB, 2-7 = the clutter slots), each carrying `_pose`/`_active`/`_is_pick`/
`_is_target`/`_shape_type`/`_size` — every field a policy could need, on every token, uniformly.
`active` only means "spawned on the table this episode"; it says nothing about role — `is_pick`/
`is_target` answers that separately, and (since `pick_idx`/`target_idx` are drawn via
`topk` over i.i.d. random keys, i.e. a uniformly random permutation of the active slots) neither
flag is biased toward any particular slot index across episodes.

There is deliberately **no** separate `pick_pose`/`target_pose`/`pick_shape_type`/etc. — an
earlier version of this env exposed those as additional always-present keys alongside the
per-slot ones, but for a permutation/set-invariant encoder that's a shortcut in disguise: the
policy could just always read the fixed key `pick_pose` instead of learning to attend over the
token set using each token's own `is_pick` flag, which defeats the point of randomizing
`pick_idx`/`target_idx` across slots in the first place. Keeping shape/size per-slot rather than
gathered-for-role-only is the same argument: slot index can't be relied on as an implicit
identity signal once you can no longer assume a fixed token order, so every token carries its
own content instead of a global summarizing just two of them.

**Two more `_get_obs_extra` keys were dropped, repo-wide in the clutter family.**
`tcp_to_cubeA_pos`/`tcp_to_cubeB_pos`/`cubeA_to_cubeB_pos` (inherited from `stack_cube.py`) are
pure linear combinations of `tcp_pose`/`cubeA_pose`/`cubeB_pose`, already in obs — redundant.
`StackCubeClutterEnv._get_obs_extra` pops them (propagating to every clutter/random-pick
subclass) rather than editing them out of `stack_cube.py` itself, which stays untouched per the
convention at the top of this document — plain `StackCube-v1`/`Swapped`/etc. keep them.

**Colors don't track role.** Every object keeps its fixed identity color regardless of whether
it's drawn as pick, target, or neither this episode — color denotes identity, not role, exactly
as `StackCubeSwapped-v1` already does; this is deliberate, not an oversight.

**Motion-planning solver reuses the same OBB grasp approach as every other solver in this
suite.** `get_actor_obb` (`mani_skill/utils/geometry/trimesh_utils.py`) tessellates whichever
PhysX collision primitive an actor has — box, cylinder, or sphere — into a mesh and takes its
oriented bounding box; `actors.build_cube`/`build_cylinder`/`build_sphere` (used for every pool
member, cubeA/cubeB included) all attach exactly those primitive collision shapes. So no
shape-specific grasp logic was needed: `solveStackCubeClutterRandomPick` is `stack_cube.py`'s
solver with `env.pick_idx`/`env.target_idx` (read after reset, since both are randomized) used
to select which two of the 8 pool objects to grasp/stack, instead of hardcoding cubeA/cubeB —
see "Motion planning" below.

**`StackCubeClutterRandomPickLockedRotationEnv`** combines this task with
`StackCubeClutterLockedRotationEnv`'s spawn via the same multiple-inheritance MRO trick as
`StackCubeSwappedLockedRotationEnv`:

```python
class StackCubeClutterRandomPickLockedRotationEnv(
    StackCubeClutterRandomPickEnv, StackCubeClutterLockedRotationEnv
):
    ...  # no body
```

The MRO resolves task overrides (`evaluate`, rewards, obs, pool selection) from the first parent
and spawn constants (`LOCK_Z_ROTATION`, restricted `CUBE_X/Y_RANGE`, clearance) from the second,
exactly as documented above for `StackCubeSwappedLockedRotationEnv`.

## How Push Block works

`push_block.py`'s `PushBlockEnv` is a reproduction of BESO's (`intuitive-robots/beso`) PyBullet
"block pushing" benchmark — 2 cubes, 2 flat target zones. There is only one registered variant: an
earlier `PushBlockStraight-v1` (forcing cubeA→targetA/cubeB→targetB specifically) was removed once
it became clear BESO's own environment has no counterpart to it at all — BESO's success condition
never distinguishes which target a given block is "supposed" to reach (see below), so a
fixed-pairing variant wasn't reproducing anything from BESO, just adding surface area.

**Robot defaults to `panda_stick`, the Push-T robot equipped with an 8mm cylindrical stick.**
This closely reproduces BESO's thin contact pin morphology. `xarm6_nogripper` (BESO's arm family
with bare wrist) and `panda` (closed-fist gripper) remain supported via `robot_uids="xarm6_nogripper"`
and `robot_uids="panda"`. `_ROBOT_BASE_POSE` supplies `_load_agent`'s per-robot base pose (`-0.615` for
panda/panda_stick, `-0.522` for xarm6) — no other task-side code is robot-specific: `evaluate()`,
`_get_obs_extra`, reward shaping, and spawn logic only ever touch `self.agent.tcp.pose`.

**Cube/target spawn identity (which slot each object occupies) matches BESO, and is not tied to a
spatial side.** BESO's `block`/`block2` always occupy the same fixed slots in its flattened
observation vector — that's real identity, and matches our `cubeA`/`cubeB` dict keys. But that slot
never implies a spatial side: both cubes spawn at a fully random x/y drawn from the *same* shared
region, so which one ends up on the left is pure chance each episode. The earlier scheme that
pinned cubeA/targetA to the right and cubeB/targetB to the left (added for the since-removed
`PushBlockStraight-v1`) is gone.

**The two cubes are kept apart LATERALLY, not by Euclidean distance.** BESO's rejection test is
`np.linalg.norm(block_translation[0] - avoid[0])` (`block_pushing_multimodal.py:186`) — index `[0]`
is the axis its two targets are separated along, so the constraint is one-dimensional and
`MIN_BLOCK_DIST = 0.1` is a *lateral* gap, and `MIN_CUBE_LATERAL_DIST = 0.10` reproduces it exactly
on the `x` axis. This matters more than it looks: it is what keeps "near cube"/"far cube" always
well defined, and therefore what makes the four `(cube, target)` goal modes below meaningful. A
Euclidean test would happily place one cube directly behind the other.

**The rejection loop must redraw *both* cubes, not just the second one.** With BESO's real numbers
the lateral range is 0.20 wide and the required gap is 0.10, so a cubeA landing near the middle of
the range leaves **no** valid position for cubeB anywhere: holding cubeA fixed and resampling only
its partner can never succeed, and the loop falls through with an invalid pair. Analytically that is
`P = 2/101 ≈ 2%` of episodes, and 2.15% was measured before the fix. BESO's outer loop redraws both
blocks for exactly this reason. Verified 0 violations in 2000 resets afterwards. The pre-rotation
ranges (0.28 wide, 0.08 gap) had enough slack to hide the same bug entirely.

**Target jitter follows BESO's axes.** ±0.0075 along the push axis, ±0.005 laterally
(`TARGET_PUSH_AXIS_JITTER`/`TARGET_LATERAL_JITTER`, from BESO's `0.05 * RANDOM_Y_SHIFT` and
`0.05 * RANDOM_X_SHIFT`). The two were previously assigned to the opposite axes.

### Dimensional audit against BESO

Measured, not eyeballed: BESO's reset replicated analytically, ours sampled from 1500 real
`env.reset()` calls per robot. All figures in metres, as `mean [min, max]`.

| quantity | BESO | ours | verdict |
|---|---|---|---|
| cube edge / mass / friction | 0.04 / 0.010 / 1.0 | 0.04 / 0.010 / 1.0 | **match** |
| rotational inertia | 3x uniform-cube | 3x uniform-cube | **match** (`CUBE_INERTIA_SCALING`) |
| goal tolerance | 0.05 | 0.05 | **match** |
| control frequency | 10 Hz | 10 Hz | **match** |
| physics timestep | 1/240 (PyBullet default) | 1/240 (`sim_freq=240`) | **match** |
| cube–cube lateral gap | 0.125 [0.100, 0.199] | 0.124 [0.100, 0.194] | **match** |
| target–target separation | 0.240 [0.230, 0.250] | 0.240 [0.230, 0.250] | **match** |
| push travel to nearest target | 0.400 [0.243, 0.557] | 0.405 [0.245, 0.562] | **match** |
| cube reach radius from base | 0.447 [0.304, 0.610] | 0.455 [0.307, 0.606] | **match** (xarm6) |
| target reach radius from base | 0.344 / 0.557 | 0.336 / 0.564 | **match** (xarm6) |

**The task frame is BESO's, and it is rotated 90° from where a naive port puts it.** This is the
single change that made every row above line up, and it is worth being explicit about because the
mistake is easy and silent.

BESO's docstrings invite you to read its `x`/`y` as our `y`/`x`. You cannot: **BESO's robot base is
at the origin of that frame**, so swapping the axis names rotates the task about the arm. BESO
pushes along `+y`, *laterally across the robot's front*, with the two targets separated along `x`,
the robot's radial direction. Four independent confirmations in its source:

- the xArm is loaded at `[0, 0, 0]` with no rotation;
- `WORKSPACE_BOUNDS = ((0.15, -0.5), (0.7, 0.5))` — `x` forward, `y` symmetric and lateral;
- `INITIAL_JOINT_POSITIONS[0] = −0.9255 rad = −53.02°` agrees to a tenth of a degree with
  `atan2(−0.4, 0.3) = −53.13°`, the bearing of its own effector reset point;
- `workspace.urdf` scales `plane.obj` by `0.0167` in x and `0.0333` in y — a mat twice as wide
  laterally as it is deep, matching the 0.24 × 0.55 object region.

What that buys, concretely. A BESO push traces a shallow arc at near-constant radius
(0.447 → 0.40 → 0.447), driven almost entirely by joint1 sweeping ~53°, with the shoulder and elbow
holding a fixed extension. Pushing *radially* instead — the naive port — sends the cube from
r = 0.222 to r = 0.642 while joint1 never moves: the arm starts folded against its own base and
finishes at `sqrt(0.65² + 0.247²) = 0.695` from the shoulder, **~99% of the xArm6's 0.70 m reach**,
precisely where it must place a cube within 5 cm.

That is why the old notes concluded BESO's 0.24 m target separation and 0.40 m travel were "blocked
by reach". They were only unreachable in the rotated frame; in BESO's own frame they sit comfortably
mid-envelope, which is why BESO could choose those numbers in the first place. Everything the old
audit listed as a structural gap — narrower travel, targets inside the cube spread, a tighter
corridor between goal zones — closed with the rotation, at no cost.

Anchoring: `WORKSPACE_CENTER_X = -0.122` is BESO's `workspace_center_x = 0.4` translated by the
xarm6 base offset (`-0.522 + 0.4`). Panda's base is at `-0.615` and sees the same table coordinates,
so it reaches ~9 cm further — verified still well inside its envelope (worst pose r = 0.727).

**The episode starts at pushing height, as BESO's does.** BESO pins
`target_effector_translation[-1] = effector_height` on every control step, and its reset teleports
the effector to a fixed pose already at that height, so its data contains no descent anywhere. Here
the xarm6 TCP would otherwise start at z = 0.283 and panda's at 0.182, opening every episode with a
26 cm descent that has no BESO counterpart — and that descent is exactly the motion the old notes
flagged as unplannable below x = −0.36. `_initialize_episode` now overrides the table scene's rest
keyframe with `_START_QPOS`, a per-robot IK solution placing the TCP at `START_TCP_XY` at pushing
height, pointing straight down.

`robot_init_qpos_noise` defaults to **0.0** here rather than this fork's usual 0.02: BESO's reset is
exact, and at pushing height ±0.02 rad of joint noise is enough to drive the wrist a centimetre or
two through the table. The argument is still accepted if you want the jitter.

`START_LINE_Y = -0.45`, not BESO's own `-0.40`. BESO's start sits exactly 0.05 behind its nearest
possible block, which is fine for a 1 mm pin but not for a 52 mm flange: at `-0.40`, 6.5% of resets
begin with the wrist overlapping a cube. The line is derived as
`CUBE_PUSH_AXIS_RANGE[0] − (max pusher radius + cube half-diagonal + margin)` and moves back towards
BESO's value when a thin pusher lands.

**Contact height matches BESO; pusher *shape* does not.** Measured world-z extents with the TCP at
`push_height = CUBE_HALF_SIZE = 0.02`, against a cube spanning z = 0.000..0.040 with its CoM at
0.020:

| pusher | z extent | contact band on the cube | band midpoint vs CoM |
|---|---|---|---|
| BESO suction tip (pin, r = 0.001, 0.029 below the flange at `EFFECTOR_HEIGHT = 0.06`) | 0.017 .. 0.045 | 0.017 .. 0.040 | +0.0085 |
| `xarm6_nogripper` `link6` flange | 0.018 .. 0.048 | 0.018 .. 0.040 | +0.009 |
| `panda` closed fingers | 0.012 .. 0.067 | 0.012 .. 0.040 | +0.006 |

All three push above the centre of mass by a comparable margin, which is BESO's design, not an
error. The footprint is another matter. `PUSHER_RADIUS` records how far each pusher's contact
surface reaches from the TCP along the push axis, **measured off the collision mesh** rather than
guessed: **0.0516** for xarm6's flange (0.0374 the other way — it is asymmetric, but the wrist's yaw
is pinned by the fixed downward push orientation, so only the forward figure ever touches a cube)
and **0.026** for panda's closed fingers, against BESO's **0.001**.

Two consequences follow from that number, and both were live bugs:

1. **The solver must stand the pusher's radius off the cube** — see the motion planning section.
2. **A planar detour around a cube needs `pusher_radius + cube half-diagonal` ≈ 0.080 of clearance
   for xarm6, against BESO's 0.100 minimum cube separation.** Passing *between* two cubes therefore
   needs 0.160 of gap and has 0.100 — it does not fit, and the push line itself clears the other
   cube by only 0.020. BESO's pin needs 0.029 against the same 0.100, three times the margin. This
   is why the solver still lifts over cubes rather than routing around them in-plane: fully planar
   transits are blocked on a thin pusher, not on the routing logic.

**Friction is set on the pusher too, not just the cubes and table.** BESO's suction head and tip
both declare `lateral_friction 1.0`, and PyBullet *multiplies* the two bodies' frictions
(1.0 × 1.0 = 1.0). PhysX *averages*, and the pusher would otherwise carry SAPIEN's 0.3 default on
xarm6's `link6` or ManiSkill's 2.0 gripper pads on panda — 0.65 and 1.5 against a μ = 1.0 cube.
`_load_agent` post-processes the pusher links' collision shapes, the same "build, then fix up the
material" pattern `HighFrictionTableSceneBuilder` uses for the table.

**Rotation is unlocked, matching BESO.** BESO's oracle has explicit `orient_block_left`/
`orient_block_right` correction phases — i.e. cube z-rotation is *not* locked there, and
reorientation mid-push is part of the task's core difficulty. `PushBlockEnv.LOCK_Z_ROTATION = False`
here to match.

**Cube mass must go through `set_mass_and_inertia`, not `builder._mass`.** SAPIEN's `ActorBuilder`
only applies `_mass`/`_inertia` when `_auto_inertial` is `False`, and *only* `set_mass_and_inertia()`
clears that flag. Assigning `builder._mass` directly is silently ignored: the mass then comes from
the collision shape's density (1000 by default), which for this 4cm box is **64g — 6.4x BESO's 10g
block**. This was live for as long as `_mass` was being poked directly, and it is the single
largest physical divergence found in the audit below.

**Rotational inertia is scaled 3x, matching BESO.** `block.urdf` carries
`<contact><inertia_scaling value="3.0"/></contact>`, a PyBullet URDF extension that multiplies the
block's computed inertia by 3 while leaving its mass alone — BESO's blocks are deliberately three
times harder to tip or spin than a uniform 10g cube. `CUBE_INERTIA_SCALING = 3.0` reproduces this;
set it to `1.0` for textbook-correct rigid-body inertia. Measured effect of the mass fix plus this
scaling, over 20 seeds, tracking the worst tilt of either cube away from flat during an episode:

| | episodes containing a >45° roll | mean worst tilt | p90 worst tilt |
|---|---|---|---|
| 64g / 1x inertia, xarm6 | 10/20 | 86° | 179° (full face-over-face flip) |
| 10g / 3x inertia, xarm6 | 4/20 | 42° | 106° |
| 64g / 1x inertia, panda | 16/20 | 77° | 93° |
| 10g / 3x inertia, panda | 10/20 | 51° | 93° |

The p90 column matches what is visible in rendered rollouts: xarm6 was flipping cubes a full 180°
face-over-face, panda mostly tipping them onto an edge and letting them settle back.

**Physical parameters are otherwise matched to BESO's PyBullet config, not ManiSkill's own
`PushCube-v1` defaults.** Cube mass 10g, static/dynamic friction 1.0 on *both* the cubes and the table surface
(`HighFrictionTableSceneBuilder`, same file — mirrors `push_t.py`'s `WhiteTableSceneBuilder`
"`super().build()`, then post-process" pattern, but sets a friction-1.0 `PhysxMaterial` on the
table's already-built collision shape instead of re-texturing it), and a 0.05m success radius
(BESO's own number — `PushCube-v1` itself uses 0.1m). `actors.build_cube` has no density/material
kwargs at all, so cube construction bypasses it entirely in favor of `push_t.py`'s
`builder._mass` + `PhysxMaterial` pattern.

**Control frequency and physics timestep are both matched to BESO.** `_default_sim_config` sets
`control_freq=10` (ManiSkill's default is 20) and `sim_freq=240`, giving 24 substeps per control
step — exactly PyBullet's default `fixedTimeStep` of 1/240, where `sim_freq=100` gave contact
substeps 2.4x coarser than the reference.

`max_episode_steps=500` is **not** comparable to BESO's `eval_n_steps` of 300. BESO's oracle emits
one command per control step and pushes continuously; this suite is plan-and-execute, so every
stroke carries its own accel/decel ramp and the same motion costs more steps. It is headroom rather
than a budget: the motion planner's worst episode over 30 seeds is 345 steps.

**`evaluate()` matches BESO's own success rule exactly: order-agnostic.** Success holds if either
valid distinct pairing is reached (A→targetA & B→targetB, OR A→targetB & B→targetA) — mirroring
BESO's `_get_reward`, which only checks "both blocks in *some* target, not the same one," never
which specific target a given block reached. `_stage_reward`'s dense reward picks whichever target
is currently closer per cube (it doesn't need to match `evaluate()`'s combinatorics, only to point
downhill).

## Design notes

**Why the spawn regions shrank.** Vanilla `StackCube-v1` spawns cubes in corners near the edge of the
Panda's reach envelope, where motion planning and clumsy policies both struggle. The usable annulus
is roughly 0.32–0.82 m from the base. `x ∈ [-0.20, 0.12]` spans its depth, centred on the midpoint,
rather than sitting against the outer edge; `y` is the <1 sigma core of `StackCube-v1`'s own spawn
distribution, so spawns stay in-distribution with the parent task.

**Why clearance is raised.** At vanilla's 0.0586 m minimum centre-to-centre distance the fully open
gripper (each finger's outer edge ~0.05 m from the grasped cube's centre) overlaps the other cube.
The variants use 0.015–0.025 m.

**The PlaceCube y band is goal-aware.** `PlaceCubeLeftEnv.CUBE_Y_RANGE` is a `@property`, not a
constant: it derives cubeB's band from `TARGET_Y_OFFSET` so that cubeB *and* its goal straddle
`y = 0` within `GOAL_Y_LIMIT = 0.21`. With a fixed `+0.16` offset and a centred band the goal would
span `[+0.03, +0.29]` — skewed to one side and partly out of reach. `PlaceCubeRight` mirrors
automatically by flipping the offset's sign.

> Subclassing gotcha: do not give a subclass a plain `CUBE_Y_RANGE` tuple. It wins on MRO and
> silently replaces the derivation with a constant, reintroducing unreachable goals.

**PlaceCube success includes a disturbance check.** The goal is defined relative to cubeB's *live*
pose, so knocking cubeB would drag the goal region along with the cube and let a clumsy episode
count as a success. Success therefore also requires cubeB to have moved less than
`CUBEB_DISTURB_THRESH = 0.01` m from its spawn. The dense reward is unaffected — `is_placed` stays
purely geometric.

## How Push Two Cubes works

`push_two_cubes.py`'s `PushTwoCubesEnv` is `PushCube-v1` run twice, side by side. It exists to be a
deliberately **single-mode** task: the intended use is to train a policy on this one narrow task and
then probe whether zero-shot capacity can be recovered, so anything that would let a policy learn a
distribution over solutions has been removed.

| | cubeA | cubeB |
|---|---|---|
| colour | blue | green |
| lane | `y = -0.16` (robot's right) | `y = +0.16` (robot's left) |
| spawn | `(-0.05, -0.16)` ± 0.03 in x and y | `(-0.05, +0.16)` ± 0.03 in x and y |
| target | fixed at `(0.15, -0.16)` | fixed at `(0.15, +0.16)` |
| push order | first | second |

`GOAL_RADIUS = 0.08`, `CUBE_HALF_SIZE = 0.02`, `max_episode_steps = 300`. The push runs along `+x`
(away from the robot) exactly as in PushCube, with a nominal `0.20 m` stroke — the same distance
PushCube gets from `0.1 + goal_radius` at its defaults.

What is deliberately *not* randomized, and why:

- **Targets never move.** PushCube places its goal at `cube_xy + [0.1 + goal_radius, 0]`, i.e. the
  goal follows the cube. Here both goals are at fixed absolute positions and only the cubes jitter.
- **The pairing is fixed.** `evaluate()` scores cubeA against `goal_regionA` only. Contrast
  `PushBlock-v1`, which accepts either pairing.
- **The order is fixed.** The dense reward is staged: `reward = stage(A)`, and only once
  `is_cubeA_placed` does it become `3.0 + stage(B)`, with `8.0` on success (`stage()` is PushCube's
  own reward for one cube, in `[0, 3]`). So B is not worth attempting until A is done.
- **Cube orientation is not randomized** — spawn quaternion is always identity.

The lanes are `0.32` apart against a `2 * GOAL_RADIUS = 0.16` goal diameter, so the goals never
overlap, a cube can never sit in the wrong goal, and no rejection sampling is needed at spawn. The
final TCP push pose sits `≈0.70 m` from the Panda base, well inside the ~0.82 m reach arc.

**This does not reuse any PushBlock machinery.** `push_object_closed_loop` and especially
`assign_push_pairs` in `base_motionplanner/utils.py` sample the pairing and the order at random,
which is precisely the multimodality this task is built to exclude.

## Motion planning

Solvers live in `mani_skill/examples/motionplanning/panda/solutions/`, and are mapped to env ids in
`run.py`'s `MP_SOLUTIONS`.

Spawn-only variants reuse their parent's solver, since the solver derives the grasp from cubeA's
oriented bounding box and is indifferent to where the cube spawned or how it is rotated. One solver,
`place_cube.py`, covers all four `PlaceCube` ids: it reads the placement side off
`env.TARGET_Y_OFFSET`. Only the swapped task needs a solver of its own, because it grasps cubeB and
targets cubeA; both `StackCubeSwapped*` ids share it.

`StackCubeClutter-v1` and `StackCubeClutterLockedRotation-v1` still stack cubeA on cubeB, just
with extra distractors on the table, so both reuse `solveStackCube` unchanged — clutter objects
sit well below the solver's lift height, and rotation-lock never affects grasp planning either.

`stack_cube_clutter_random_pick.py` provides one solver, `solveStackCubeClutterRandomPick`, for
both `StackCubeClutterRandomPick-v1` and `StackCubeClutterRandomPickLockedRotation-v1`. Unlike
the deleted binary `pick_is_cubeA` solver this replaces, it doesn't need a runtime branch between
two hardcoded actors — it just indexes `env.pool_objects` with `env.pick_idx[0]`/
`env.target_idx[0]` (read after `env.reset(seed=...)`) to get the pick/target `Actor`s, and reads
`env.pool_rest_z[pick_idx] + env.pool_rest_z[target_idx]` for the stack height offset instead of
the fixed `cube_half_size[2] * 2` `stack_cube.py` uses. `get_actor_obb` handles the rest
generically regardless of which shape got drawn (see the note above).

**`PushTwoCubes-v1` has its own solver, `panda/solutions/push_two_cubes.py`**, which pushes cubeA
then cubeB — hardcoded, never `assign_push_pairs`. It aims for the goal *centre* rather than just
somewhere inside the goal, since a sloppy oracle makes for sloppy demonstrations. Four things it
does that `push_cube.py` does not:

- **The stroke stays at the cube's height.** `push_cube.py` takes its target straight off the goal
  disc, whose `z` is `1e-3`, so the closed fist scrapes the table and `plan_screw` rejects the
  motion outright on ~10% of seeds.
- **Each stroke is aimed along the current cube→goal direction**, not straight down `+x`, so one
  stroke takes out the cube's lateral spawn offset as well as the distance.
- **The stroke stops `CONTACT_OFFSET - PUSH_LAG` short of the goal centre.** `CONTACT_OFFSET`
  (0.0295 m) is the fist-to-cube-centre distance at contact. `PUSH_LAG` (0.0060 m) is a
  steady-state tracking offset: while the fist presses the cube, the joint controller settles
  where its torque balances friction rather than on the commanded pose. This is *not* settling
  lag — `refine_steps` converges to 0.0057 and stops — so the stroke aims through it. Both
  constants were measured, and both are tight (±0.0006 and ±0.0009 over 30 strokes).
- **A closed loop re-measures and corrects**, up to `MAX_PUSH_PASSES`. With the two constants
  above the first stroke normally lands inside `PUSH_TOLERANCE` (5 mm) and the loop exits, so this
  is a safety net rather than a routine second stroke. It is deterministic given the state — it
  corrects, it never chooses between alternatives — so it adds precision without adding
  multimodality.

Two guards keep the loop from being worse than the error it fixes:

- **Do not refine below ~5 mm.** At a 1.5 mm tolerance the corrections become sub-millimetre
  strokes, which are ill-conditioned: measured over 25 seeds it produced IK failures, 8/25 aborted
  plans, and a *worse* mean error (10.9 mm) than not refining at all.
- **Abandon a refinement that would reach too far** (`REFINE_REACH_LIMIT`, 0.76 m). Correcting a
  stroke that stopped short lines the arm up at most ~0.73 m out. Correcting an *overshoot* means
  reaching around to the far side of the cube; on one seed that put the target 0.81 m out, near
  the ~0.82 m limit, where the arm crawled through near-singular configurations (single moves of
  123 and 319 steps) and shoved the cube further off than it started.

**Every motion is a straight line, and there is no RRTConnect fallback.** A free path may not
deviate here: a stroke has to push the cube at the goal and not sideways. On one measured seed the
fallback swept the arm through cubeA and knocked it out of its lane. `plan_screw` does reject the
occasional straight line it should accept (~1 in 180, and *uncorrelated* with distance from the
base: failed targets averaged 0.651 m against 0.649 m for successful ones), so `_move` retries it
as collinear sub-strokes — same path, planned in shorter pieces.

The cross-over between lanes is a lift of `LIFT_HEIGHT`, a translate, and a descent, so the arm
neither drags the cube it just pushed nor clips the next one. cubeA is approached straight from the
home pose, where there is nothing to clear.

Measured over 200 seeds: **100% success, 0 failed motion plans**, cube-to-goal-centre error 1.7 mm
mean / 3.1 mm p95 / 5.6 mm max against an 80 mm goal radius, episode length 216 avg / 261 max
(hence the 300-step limit). The earlier open-loop version, which reused PushCube's fixed standoff,
sat at 48 mm mean error.

Note: `figures/custom_envs_spawn_regions.png` has no generator script committed anywhere in this
repo's history; it is not being regenerated/extended for the clutter variants.

Generate demonstrations with:

```bash
python -m mani_skill.examples.motionplanning.panda.run \
    -e PlaceCubeLeftLockedRotation-v1 -n 100 --only-count-success --save-video
```

**Push Block has two solvers, one per robot family, both sharing the same core algorithm.**
Following this fork's own convention (motion planning is entirely separate infrastructure per
robot — `motionplanning/panda/` vs `motionplanning/xarm6/`, each with its own `run.py`/
`MP_SOLUTIONS`), `panda/solutions/push_block.py` and `xarm6/solutions/push_block.py` each provide
a `solve()` for `PushBlock-v1`. Neither uses a fixed cubeA→targetA assignment: `assign_push_pairs`
samples one of the two distinct pairings and one of the two push orders uniformly (see below). The xarm6 solver drops
all gripper handling entirely (no `close_gripper()` call) since `xarm6_nogripper` has no gripper to
manage — simpler than the Panda solver, which keeps the gripper closed the whole episode as a
blunt "fist" pusher (Panda has no bare-wrist option).

**The actual push algorithm is robot-agnostic**, factored into `push_object_closed_loop` in
`base_motionplanner/utils.py` (alongside `get_actor_obb`/`compute_grasp_info_by_obb`, this repo's
other robot-agnostic solver helpers) — it only calls `planner.move_to_pose_with_screw`, so it works
with any `BaseMotionPlanningSolver` subclass regardless of gripper.

**Each push is ONE continuous stroke, because that is what BESO does.** BESO's `push_block` phase
commands the effector toward `xy_touchingblock` (`block_centre - dir * 0.01`, i.e. 1cm *inside* the
4cm block) every control step, velocity-capped at 0.35 m/s, and never lets go until the block
reaches the target — it never pauses, lifts, or retracts mid-push
(`oracles/oriented_push_oracle.py::_get_push_block`). So a pass here is a single
`move_to_pose_with_screw` from a standoff point behind the cube straight through to where the TCP
must end up for the cube's centre to sit on the target: one trapezoidal velocity profile over the
whole travel, one acceleration and one deceleration. The loop is still closed — the cube's live
pose is re-read between passes and a corrective stroke is run if it drifted or rotated off the
line — but in practice the first cube takes exactly one stroke.

The earlier implementation advanced in 4cm chunks with `refine_steps=10` held between them, which
at 10Hz control is a full second of standing still roughly every 4cm of travel. That is the
"push a little, wait, push again" stutter visible in rendered rollouts, and it is not BESO
behaviour. It is gone entirely — see the stroke cap below.

**`max_stroke` caps the advance per stroke at 0.20m.** A full-length straight Cartesian path at
push height is not always plannable by `plan_screw` from a given configuration, and a stroke that
fails to plan is worse than two that succeed. A capped stroke ends with the TCP exactly at the
cube's back face on the push line, so the next one starts immediately — no retreat, no re-approach
— which makes a long push a run of back-to-back continuous strokes with only a settle between them.

**Standing the pusher's own radius off the cube is not optional.** `contact_clearance` is
`CUBE_HALF_SIZE + PUSHER_RADIUS[robot] - 0.01`, so the pusher's *surface* — not the TCP — ends up
~5mm past the cube's back face. Using `CUBE_HALF_SIZE` alone parks xarm6's 52mm flange 25mm from the
cube's centre, i.e. ~45mm inside it, and PhysX resolves that penetration by launching the cube. That
was the real cause of the cube tumbling previously blamed on pusher shape: measured **7/20 -> 19/20**
on xarm6 from this one term.

**Five failure modes surfaced empirically, all applying to both arms:**
- **The wrist riding over the cube instead of pushing it.** mplib's `plan_screw` sometimes returns
  `Success` for a Cartesian path it did not complete, leaving the TCP up to ~2cm short at
  particular reaches — reproducibly, and unaffected by how long the arm is given to settle, so it
  is the plan that is short, not the tracking. Against a 4cm cube, starting a stroke 2cm high means
  contacting only the top edge: the cube tips, the wrist rides over it, and a one-stroke push
  degenerates into four passes each creeping ~4cm. `plan_qpos_to_pose` (RRTConnect) *does* land
  sub-millimetre and looks like the obvious fix, but it is a joint-space planner run here with
  `use_point_cloud=False`: it knows nothing about the cubes and sweeps the arm straight through an
  already-placed one, and its exit configuration often defeats the straight-line screw plan for the
  stroke that follows. Measured over 30 seeds: **23/30 with screw alone, 13/30 with the RRTConnect
  seat.** Screw-only, plus re-observing, wins — there is deliberately no RRT fallback in the
  approach any more.
- **Ploughing through the cube on a correction pass.** When a cube overshoots, the push direction
  flips and the new standoff point lands on the *far* side of it; a straight transit to that point
  would go through the cube it is meant to re-approach. `_point_segment_dist` tests for exactly
  this and inserts a lift when it triggers.
- **Long-transit reachability failures on the 6-DOF arm.** Screw planning is a local,
  straight-line-in-jointspace method: transiting directly from wherever cubeA's push left the TCP
  to cubeB's approach pose could fail on `xarm6_nogripper` (less redundant than Panda's 7-DOF) if
  the current joint configuration needed to "unwind" through a large angle, even though the target
  pose itself was perfectly reachable. Fixed by retreating straight up to `retreat_height` (0.15m)
  *before* transiting — mirroring BESO's own oracle, which explicitly retreats between pushing its
  two blocks. `_lift_clear` normalises the transit height in *both* directions (xarm6's rest pose
  sits ~13cm above it, and coming down to it first is worth ~4 successes in 30), and skips only the
  degenerate case where the TCP is already at that exact height — a zero-length screw plan makes
  `follow_path` fall off the end of its loop and raise `UnboundLocalError`, which panda hits
  because its rest pose sits at z=0.1495.

- **Unplannable approach silently skipping a cube.** The descent to push height can be unplannable
  deep in the workspace while being fine a couple of centimetres closer in. The approach now walks a
  standoff ladder (full 0.05, half, then zero) before giving up; shrinking the standoff only
  shortens the free run-up, never the stroke. Zero-length rungs are skipped explicitly — planning to
  the pose the TCP already occupies yields an empty path and the same `follow_path`
  `UnboundLocalError` as `_lift_clear`, reachable on a correction pass where the cube barely moved.
**Stop stepping once the episode is over.** `_episode_over` checks the `terminated`/`truncated` flags
that `follow_path` returns and aborts the remaining passes. Without it the solver kept planning past
`max_episode_steps` — the `TimeLimit` wrapper stops counting but `env.step` still runs — so runs were
reaching 4026 steps for a 350-step episode, and worse, **success was being recorded for goals reached
after the time limit had already fired**. Any success rate measured without this guard is inflated.

**Mode coverage, not travel distance, decides the cube→target assignment.** BESO's oracle picks
uniformly among all four `((first_block, first_target), (second_block, second_target))` tuples
(`oracles/multimodal_push_oracle.py::_choose_goal_order`) — 2 assignments × 2 orders. Those four
*are* the benchmark: the env latches them as `task_idx = 2 * block_idx + target_idx` into
`all_completions`, the demo dataset labels every trajectory with a 4-dim one-hot over them
(`onehot_goals.pth`), and the reported score is `len(all_completions) / 2`. It is tempting to
instead pick whichever pairing has the shorter total travel — but for a 2×2 Euclidean assignment
the min-cost matching is provably the non-crossing one, so that rule can *never* emit a
criss-crossing demonstration and silently halves the dataset's mode coverage, leaving two of the
four goals unreachable for a goal-conditioned policy. `assign_push_pairs` therefore samples the
assignment and the order uniformly, with plain `random.choice`/`random.shuffle` rather than the
env's seeded RNG — mode selection is deliberately not tied to reproducibility in BESO either.

Verified success rates (`PushBlock-v1`, `sim_backend=cpu`, BESO geometry, honest step accounting at
the registered 500-step limit): **30/30 on panda_stick** (mean 224 steps with smooth slip cutoff, down from 268 steps; 100% planar, strictly $z=0.02$m),
**29/30 on xarm6_nogripper** (mean 243 steps, max 345), and **18/20 on panda** (mean 254 steps, max 357).

**Planar transits and smooth continuous pushing on `panda_stick`.** While `xarm6_nogripper` (52mm flange) and `panda`
(closed fingers) lift to `retreat_height = 0.15m` to clear cubes during transit, `panda_stick`'s 8mm
radius reduces the required clearance to ~0.042m, easily fitting through the $\ge 0.10$m cube gap.
The `panda_stick` solver uses `push_object_planar_closed_loop`, which navigates around cubes in the
2D plane and routes via the rear baseline corridor ($y \le -0.42$) without ever leaving $z = 0.02$m.
Furthermore, the solver features **live slip detection**: during stroke execution, if contact drifts
laterally past the cube face or the stick starts outrunning the cube, the stroke terminates early immediately,
preventing the robot from pushing empty air and eliminating long retreat loops around the cube.
This matches BESO's and Push-T's smooth, controlled planar demonstrations.

```bash
# PushBlock-v1 defaults to panda_stick and runs the strictly planar solver:
python -m mani_skill.examples.motionplanning.panda.run \
    -e PushBlock-v1 -n 100 --only-count-success --save-video

# Opt-in robots:
python -m mani_skill.examples.motionplanning.panda.run \
    -e PushBlock-v1 --robot_uids panda -n 100 --only-count-success --save-video

python -m mani_skill.examples.motionplanning.xarm6.run \
    -e PushBlock-v1 --robot_uids xarm6_nogripper -n 100 --only-count-success --save-video
```
