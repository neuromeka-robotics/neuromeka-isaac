# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pdb
from typing import TYPE_CHECKING

import torch
from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import RLTaskEnv
from isaac_neuromeka.assets.articulation import FiniteArticulation
from isaac_neuromeka.assets.objects import RigidObject_w_FullPCL


def object_position_in_robot_root_frame(
    env: RLTaskEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def full_object_pointcloud(
    env: RLTaskEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    object: RigidObject_w_FullPCL = env.scene[object_cfg.name]
    # #############
    # import matplotlib.pyplot as plt
    # import numpy as np
    # pcl_np = object.full_pcl[2]
    # pcl_np = pcl_np.cpu().numpy()
    # x = pcl_np[:, 0]
    # y = pcl_np[:, 1]
    # z = pcl_np[:, 2]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, c='b', marker='.')
    # plt.show()
    # plt.close()
    # #############
    return torch.reshape(object.full_pcl, (object.num_instances, -1))
