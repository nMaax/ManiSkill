import torch
from mani_skill.envs.tasks.tabletop.place_sphere import PlaceSphereEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("PlaceSphereRestrictedSpawn-v1", max_episode_steps=250)
class PlaceSphereRestrictedSpawnEnv(PlaceSphereEnv):
    """PlaceSphere-v1 with the sphere and bin spawned in a restricted region.

    The region is the union of the per-object <1 sigma cores of StackCube-v1's spawn; both
    objects are sampled from it, so they share one region rather than having one each.
    """

    SPAWN_X_RANGE = (-0.08909, 0.08211)
    SPAWN_Y_RANGE = (-0.13197, 0.13146)
    # separation between the two sampled positions, calibrated for gripper clearance --
    # not a geometric property of either object
    SPAWN_CLEARANCE = 0.025

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        with torch.device(self.device):
            b = len(env_idx)

            radius = (
                torch.linalg.norm(torch.tensor([0.02, 0.02])) + self.SPAWN_CLEARANCE
            )

            region = (
                [self.SPAWN_X_RANGE[0], self.SPAWN_Y_RANGE[0]],
                [self.SPAWN_X_RANGE[1], self.SPAWN_Y_RANGE[1]],
            )

            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )

            sphere_xy = sampler.sample(radius, 100)
            bin_xy = sampler.sample(radius, 100, verbose=False)

            current_obj_pose = self.obj.pose
            new_obj_p = current_obj_pose.p[env_idx].clone()
            new_obj_p[:, :2] = sphere_xy
            new_obj_q = current_obj_pose.q[env_idx]
            self.obj.set_pose(Pose.create_from_pq(p=new_obj_p, q=new_obj_q))  # type: ignore

            current_bin_pose = self.bin.pose
            new_bin_p = current_bin_pose.p[env_idx].clone()
            new_bin_p[:, :2] = bin_xy
            new_bin_q = current_bin_pose.q[env_idx]
            self.bin.set_pose(Pose.create_from_pq(p=new_bin_p, q=new_bin_q))  # type: ignore
