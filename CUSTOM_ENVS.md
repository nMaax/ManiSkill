# Custom environments in this fork

This fork of [ManiSkill](https://github.com/haosulab/ManiSkill) adds a family of tabletop
environments derived from `StackCube-v1`, built for goal-conditioned policy training. Everything
here lives in `mani_skill/envs/tasks/tabletop/`; upstream code is otherwise unmodified.

## The environments

All ranges are metres relative to the table centre. The Panda base sits at `x = -0.615`, so a more
negative `x` is *closer* to the robot. `+y` is to the robot's left.

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

Note: `figures/custom_envs_spawn_regions.png` has no generator script committed anywhere in this
repo's history; it is not being regenerated/extended for the clutter variants.

Generate demonstrations with:

```bash
python -m mani_skill.examples.motionplanning.panda.run \
    -e PlaceCubeLeftLockedRotation-v1 -n 100 --only-count-success --save-video
```
