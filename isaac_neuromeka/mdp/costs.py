from __future__ import annotations

import pdb
from typing import TYPE_CHECKING

import torch
from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import (
    combine_frame_transforms,
    quat_error_magnitude,
    quat_mul,
)

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
from isaac_neuromeka.assets.articulation import FiniteArticulation


def joint_vel_cost_01(env: ManagerBasedRLEnv, 
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                    soft_limit_ratio: float = 0.99) -> torch.Tensor:
    """
    Joint velocity cost function based on the soft joint velocity limits.
    Indicator function to penalize joint velocities exceeding the soft limits.
    """
    
    # extract the used quantities (to enable type-hinting)
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    
    joint_vel = asset._finite_joint_vel[:, asset_cfg.joint_ids]
    joint_vel_lim = asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids] * soft_limit_ratio
    
    violation = torch.abs(joint_vel) - joint_vel_lim
    
    
    return torch.any(torch.gt(violation, torch.zeros_like(violation)), dim=1).int()


def joint_vel_cost_relu(env: ManagerBasedRLEnv, 
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                        soft_limit_ratio: float = 0.99) -> torch.Tensor:
    """
    Joint velocity cost function based on the soft joint velocity limits.
    Indicator function to penalize joint velocities exceeding the soft limits.
    """
    
    # extract the used quantities (to enable type-hinting)
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    
    joint_vel = asset._finite_joint_vel[:, asset_cfg.joint_ids]
    joint_vel_lim = asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids] * soft_limit_ratio
    
    violation = torch.abs(joint_vel) - joint_vel_lim
    
    return torch.sum(torch.relu(violation), dim=1)


# End-effector speed limit (1m/s). TODO: variable speed limit
def ee_speed_cost_relu(env: ManagerBasedRLEnv, 
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                        speed_limit: float = 1.0) -> torch.Tensor:
    
    # extract the used quantities (to enable type-hinting)
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    
    ee_speed = torch.abs(asset.data.body_state_w[:, asset_cfg.body_ids[0], 7:10])

    violation = torch.abs(ee_speed) - speed_limit
    
    return torch.sum(torch.relu(violation), dim=1)