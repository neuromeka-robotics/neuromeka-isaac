import numpy as np
import torch
from collections.abc import Sequence
from isaaclab.envs.mdp.actions import JointAction, actions_cfg
from isaaclab.envs import ManagerBasedEnv

class CustomJointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        else:
            self._offset = torch.zeros_like(self._asset.data.default_joint_pos[:, self._joint_ids])  # TODO (remove)

    def apply_actions(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        target = self.processed_actions[env_ids, :]

        # if self._joint_ids is not None and self._joint_ids != slice(None):
        #         env_ids = env_ids.unsqueeze(-1)
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids, env_ids=env_ids)

    # def reset(self, env_ids: Sequence[int] | None = None) -> None:
    #     if env_ids is None:
    #         env_ids = slice(None)
    #     target = self._offset[env_ids, :]

    #     if self._joint_ids is not None and self._joint_ids != slice(None):
    #         env_ids = env_ids.unsqueeze(-1)
    #     self._asset.set_joint_position_target(target, joint_ids=self._joint_ids, env_ids=env_ids)  # TODO: fix required
