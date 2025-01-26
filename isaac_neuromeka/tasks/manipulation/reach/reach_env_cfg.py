# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pdb
from dataclasses import MISSING

import numpy as np
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.managers import ActionTermCfg as ActionTerm
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import ManagerTermBase
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from isaac_neuromeka.utils.etc import EmptyCfg
import isaac_neuromeka.mdp as mdp

# Import common environment configuration
from isaac_neuromeka.tasks.manipulation.common.env_cfg_common import *

##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    # )

    # robots
    robot: ArticulationCfg = None
    
    # target object
    obstacle = None

    # contact sensor
    contact_sensors = None
    
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    # sky_light = AssetBaseCfg(
    #     prim_path="/World/skyLight",
    #     spawn=sim_utils.DomeLightCfg(
    #         intensity=750.0,
    #         texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    #     ),
    # )

##
# Environment configuration
##

@configclass
class ReachEnvCfg(NrmkRLEnvCfg): 
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=3.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg | EmptyCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg | EmptyCfg = EventCfg()
    curriculum: CurriculumCfg | EmptyCfg = CurriculumCfg()
    # CMDP settings
    costs: CostsCfg | EmptyCfg = EmptyCfg()
    
    # 
    actor_obs_list: list = ["policy"] # ["proprioception", "point_cloud", "privileged"]
    critic_obs_list: list | None = None # None: same as actor_obs_list
    teacher_obs_list: list | None = None # None: same as actor_obs_list


    def __post_init__(self):
        """Post initialization."""
        # task settings
        self.decimation = 24  # 5hz # 30 Hz (4)
        self.episode_length_s = 12.0
        # viewer settings
        self.viewer.eye = (2.5, 2.5, 2.5)
