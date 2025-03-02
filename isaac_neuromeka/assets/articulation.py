from collections.abc import Sequence

import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.utils.math import matrix_from_quat


class FiniteArticulation(Articulation):
    def _create_buffers(self):
        super()._create_buffers()
        self._prevprev_joint_pos = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._prev_joint_pos = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._prevprev_finite_joint_vel = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._prev_finite_joint_vel = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._finite_joint_vel = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._op_state = torch.zeros(self.num_instances, 1, device=self.device)  # used for demo

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | slice | None = None,
    ):
        super().write_joint_state_to_sim(position, velocity, joint_ids, env_ids)
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)
        self._prevprev_joint_pos[env_ids, joint_ids] = position
        self._prev_joint_pos[env_ids, joint_ids] = position
        self._prevprev_finite_joint_vel[env_ids, joint_ids] = velocity
        self._prev_finite_joint_vel[env_ids, joint_ids] = velocity
        self._finite_joint_vel[env_ids, joint_ids] = velocity

    def update(self, dt: float):
        self._prevprev_joint_pos[:] = self._prev_joint_pos[:]
        self._prev_joint_pos[:] = self._data.joint_pos[:]
        self._prevprev_finite_joint_vel[:] = self._prev_finite_joint_vel[:]
        self._prev_finite_joint_vel[:] = self._finite_joint_vel[:]
        super().update(dt)
        self._finite_joint_vel[:] = (self._data.joint_pos - self._prev_joint_pos) / dt