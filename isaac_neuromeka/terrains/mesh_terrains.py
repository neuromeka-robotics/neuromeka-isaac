from __future__ import annotations

import numpy as np
import torch
import trimesh
from typing import TYPE_CHECKING

from isaaclab.utils import *  # noqa: F401, F403

if TYPE_CHECKING:
    from . import mesh_terrain_cfg



def random_boxes_trimesh(
    difficulty: float, cfg: mesh_terrain_cfg.MeshGridTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:


    # check to ensure square terrain
    if cfg.size[0] != cfg.size[1]:
        raise ValueError(f"The terrain must be square. Received size: {cfg.size}.")
    # resolve the terrain configuration
    # grid_height = cfg.grid_height_range[0] + difficulty * (cfg.grid_height_range[1] - cfg.grid_height_range[0])

    grid_max_height =  difficulty * cfg.grid_height_range[1]
    grid_min_height =  difficulty * cfg.grid_height_range[0]
    

    # initialize list of meshes
    meshes_list = list()
    # compute the number of boxes in each direction
    num_boxes_x = int(cfg.size[0] / cfg.grid_width)
    num_boxes_y = int(cfg.size[1] / cfg.grid_width)
    
    # constant parameters
    terrain_height = 1.0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # create a template grid of terrain height
    grid_dim = [cfg.grid_width, cfg.grid_width, terrain_height]
    grid_position = [0.5 * cfg.grid_width, 0.5 * cfg.grid_width, -terrain_height / 2]
    template_box = trimesh.creation.box(grid_dim, trimesh.transformations.translation_matrix(grid_position))
    # extract vertices and faces of the box to create a template
    template_vertices = template_box.vertices  # (8, 3)
    template_faces = template_box.faces

    # repeat the template box vertices to span the terrain (num_boxes_x * num_boxes_y, 8, 3)
    vertices = torch.tensor(template_vertices, device=device).repeat(num_boxes_x * num_boxes_y, 1, 1)
    
    # create a meshgrid to offset the vertices
    x = torch.arange(0, num_boxes_x, device=device)
    y = torch.arange(0, num_boxes_y, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    xx = xx.flatten().view(-1, 1)
    yy = yy.flatten().view(-1, 1)
    xx_yy = torch.cat((xx, yy), dim=1)
    # offset the vertices
    offsets = cfg.grid_width * xx_yy 
    vertices[:, :, :2] += offsets.unsqueeze(1)

    # add noise to the vertices to have a random height over each grid cell
    num_boxes = len(vertices)
    
    # Sample height offsets
    grid_low_max_height = grid_min_height * (1 - cfg.low_height_ratio) + grid_max_height * cfg.low_height_ratio
    
    h_offset = torch.zeros((num_boxes, 3), device=device)
    h_offset[:, 2].uniform_(grid_low_max_height, grid_max_height)
    
    box_centers = vertices[:, :4, :].mean(dim=1) # shape = [num_boxes, 3]
    
    # if box center is close to the center of the terrain, set the height to 0
    center = torch.tensor([cfg.size[0] * 0.5, cfg.size[1] * 0.5], device=device)
    dist_to_center = torch.norm(box_centers[:,:2] - center, dim=1)

    
    center_indicies = dist_to_center < cfg.platform_width
    low_indicies = torch.rand((num_boxes), device=device) > cfg.high_prob
    
    # merge the two conditions
    low_indicies = center_indicies | low_indicies
    h_offset[low_indicies, 2] = torch.zeros((low_indicies.sum()), device=device).uniform_(grid_min_height, grid_low_max_height)
    
    close_to_robot_indicies = dist_to_center < cfg.robot_range_width
    # h_offset[close_to_robot_indicies, 2] = -0.02
    # clip the height to be within the range
    h_offset[close_to_robot_indicies, 2] = torch.clamp(h_offset[close_to_robot_indicies, 2], grid_min_height, -0.01)
    
    # reshape noise to match the vertices (num_boxes, 4, 3)
    # only the top vertices of the box are affected
    vertices_offset = torch.zeros((num_boxes, 4, 3), device=device)
    vertices_offset += h_offset.unsqueeze(1)
    vertices[vertices[:, :, 2] == 0] += vertices_offset.view(-1, 3)
    vertices = vertices.reshape(-1, 3).cpu().numpy()

    # create faces for boxes (num_boxes, 12, 3). Each box has 6 faces, each face has 2 triangles.
    faces = torch.tensor(template_faces, device=device).repeat(num_boxes, 1, 1)
    face_offsets = torch.arange(0, num_boxes, device=device).unsqueeze(1).repeat(1, 12) * 8
    faces += face_offsets.unsqueeze(2)
    # move to numpy
    faces = faces.view(-1, 3).cpu().numpy()
    # convert to trimesh
    grid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    meshes_list.append(grid_mesh)

    # specify the origin of the terrain
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])


    return meshes_list, origin