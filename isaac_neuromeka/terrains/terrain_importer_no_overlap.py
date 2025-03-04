# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import trimesh
from typing import TYPE_CHECKING

import warp
from pxr import UsdGeom


import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.warp import convert_to_warp_mesh

from isaaclab.terrains.terrain_importer import TerrainImporter

from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.terrains.utils import create_prim_from_mesh

if TYPE_CHECKING:
    from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg


class TerrainImporterNoOverlap(TerrainImporter):
    r"""A class to handle terrain meshes and import them into the simulator.

    We assume that a terrain mesh comprises of sub-terrains that are arranged in a grid with
    rows ``num_rows`` and columns ``num_cols``. The terrain origins are the positions of the sub-terrains
    where the robot should be spawned.

    Based on the configuration, the terrain importer handles computing the environment origins from the sub-terrain
    origins. In a typical setup, the number of sub-terrains (:math:`num\_rows \times num\_cols`) is smaller than
    the number of environments (:math:`num\_envs`). In this case, the environment origins are computed by
    sampling the sub-terrain origins.

    If a curriculum is used, it is possible to update the environment origins to terrain origins that correspond
    to a harder difficulty. This is done by calling :func:`update_terrain_levels`. The idea comes from game-based
    curriculum. For example, in a game, the player starts with easy levels and progresses to harder levels.
    """

    meshes: dict[str, trimesh.Trimesh]
    """A dictionary containing the names of the meshes and their keys."""
    warp_meshes: dict[str, warp.Mesh]
    """A dictionary containing the names of the warp meshes and their keys."""
    terrain_origins: torch.Tensor | None
    """The origins of the sub-terrains in the added terrain mesh. Shape is (num_rows, num_cols, 3).

    If None, then it is assumed no sub-terrains exist. The environment origins are computed in a grid.
    """
    env_origins: torch.Tensor
    """The origins of the environments. Shape is (num_envs, 3)."""

    def __init__(self, cfg: TerrainImporterCfg):
        """Initialize the terrain importer.

        Args:
            cfg: The configuration for the terrain importer.

        Raises:
            ValueError: If input terrain type is not supported.
            ValueError: If terrain type is 'generator' and no configuration provided for ``terrain_generator``.
            ValueError: If terrain type is 'usd' and no configuration provided for ``usd_path``.
            ValueError: If terrain type is 'usd' or 'plane' and no configuration provided for ``env_spacing``.
        """
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        # create a dict of meshes
        self.meshes = dict()
        self.warp_meshes = dict()
        self.env_origins = None
        self.terrain_origins = None
        # private variables
        self._terrain_flat_patches = dict()


        # auto-import the terrain based on the config
        if self.cfg.terrain_type == "generator":
            # check config is provided
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            # generate the terrain
            terrain_generator = TerrainGenerator(cfg=self.cfg.terrain_generator, device=self.device)
            self.import_mesh("terrain", terrain_generator.terrain_mesh)
            # configure the terrain origins based on the terrain generator

            self.configure_env_origins(terrain_generator.terrain_origins)
            # refer to the flat patches
            self._terrain_flat_patches = terrain_generator.flat_patches
        else:
            raise ValueError(f"Terrain type '{self.cfg.terrain_type}' not available.")
        
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)


    """
    Operations - Origins.
    """

    def configure_env_origins(self, origins: np.ndarray | None = None):
        """Configure the origins of the environments based on the added terrain.

        Args:
            origins: The origins of the sub-terrains. Shape is (num_rows, num_cols, 3).
        """
        # decide whether to compute origins in a grid or based on curriculum
        if origins is not None:
            # convert to numpy
            if isinstance(origins, np.ndarray):
                origins = torch.from_numpy(origins)
            # store the origins
            self.terrain_origins = origins.to(self.device, dtype=torch.float)
            # compute environment origins
            self.env_origins = self._compute_env_origins_in_order(self.cfg.num_envs, self.terrain_origins)
        else:
            self.terrain_origins = None
            # check if env spacing is valid
            if self.cfg.env_spacing is None:
                raise ValueError("Environment spacing must be specified for configuring grid-like origins.")
            # compute environment origins
            self.env_origins = self._compute_env_origins_grid(self.cfg.num_envs, self.cfg.env_spacing)

    def _compute_env_origins_in_order(self, num_envs: int, origins: torch.Tensor) -> torch.Tensor:
        
        """Compute the origins of the environments defined by the sub-terrains origins."""
        # extract number of rows and cols
        num_rows, num_cols = origins.shape[:2]
        
        # create tensor based on number of environments
        env_origins = torch.zeros(num_envs, 3, device=self.device)
        
        # compute the origins in order
        for i in range(num_envs):
            row = i // num_cols
            col = i % num_cols
            env_origins[i] = origins[row, col]
        return env_origins

    def _compute_env_origins_random(self, num_envs: int, origins: torch.Tensor) -> torch.Tensor:
        """Compute the origins of the environments defined by the sub-terrains origins."""
        # extract number of rows and cols
        num_rows, num_cols = origins.shape[:2]
        
        origins_flat = origins.view(-1, 3)
        num_terrains = origins_flat.shape[0]
        
        # throw if num_terrains < num_envs
        if num_terrains < num_envs:
            raise ValueError(f"Number of terrains ({num_terrains}) is less than number of environments ({num_envs}).")
        
        # Sample index randomly with no replacement
        terrain_indicies = torch.randperm(num_terrains, device=self.device)[:num_envs]
        
        # back to row and col
        row_indices = terrain_indicies // num_cols
        col_indices = terrain_indicies % num_cols
        

        # create tensor based on number of environments
        env_origins = torch.zeros(num_envs, 3, device=self.device)
        env_origins[:] = origins[row_indices, col_indices]
        return env_origins


    def shuffle_terrain_origins(self, num_envs: int):
        """Shuffle the environment origins."""


        self.env_origins = self._compute_env_origins_random(num_envs, self.terrain_origins)
