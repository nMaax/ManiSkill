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
| `PlaceCubeLeft-v1` | −0.20 … 0.12 | −0.21 … 0.05 | free | A at B **+0.16 y** | `place_cube_left.py` |
| `PlaceCubeRight-v1` | −0.20 … 0.12 | −0.05 … 0.21 | free | A at B **−0.16 y** | `place_cube_right.py` |
| `PlaceCubeLeftLockedRotation-v1` | −0.20 … 0.12 | −0.21 … 0.05 | **locked** | A at B +0.16 y | `place_cube_left_locked_rotation.py` |
| `PlaceCubeRightLockedRotation-v1` | −0.20 … 0.12 | −0.05 … 0.21 | **locked** | A at B −0.16 y | `place_cube_right_locked_rotation.py` |
| `PlaceSphereRestrictedSpawn-v1` | −0.089 … 0.082 | −0.132 … 0.131 | inherited | sphere in bin | `place_sphere_restricted_spawn.py` |

**There is exactlu one file per registered environment.**, e.g. `PlaceCubeLeftLockedRotation-v1` is in `place_cube_left_locked_rotation.py`.

Note that `*LockedRotation-v1` environments restrict the spawn *as well as* locking rotation — the name only mentions the more distinctive of the two.

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

Generate demonstrations with:

```bash
python -m mani_skill.examples.motionplanning.panda.run \
    -e PlaceCubeLeftLockedRotation-v1 -n 100 --only-count-success --save-video
```
