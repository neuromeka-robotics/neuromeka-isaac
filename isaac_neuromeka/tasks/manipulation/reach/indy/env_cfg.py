from __future__ import annotations

import math
import numpy as np

from isaaclab.utils import configclass
# from isaaclab.assets import RigidObjectCfg
# # import isaaclab.sim as sim_utils
# from isaaclab.sensors import CameraCfg, ContactSensorCfg
import isaac_neuromeka.mdp as mdp

##
# Pre-defined configs
##
from isaac_neuromeka.assets import INDY7_CFG
from isaac_neuromeka.tasks.manipulation.reach.reach_env_cfg import ReachEnvCfg, ObservationsCfg
from isaac_neuromeka.tasks.manipulation.reach.reach_env_cfg import TeacherObsCfg
from isaac_neuromeka.tasks.manipulation.reach.reach_env_cfg import CostsCfg
from isaac_neuromeka.mdp.actions import CustomJointPositionAction
from isaac_neuromeka.utils.etc import EmptyCfg

##
# Environment configuration
##


@configclass
class Indy7ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to indy
        self.scene.robot = INDY7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["tcp"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["tcp"]
        # self.rewards.end_effector_pose_tracking.params["asset_cfg"].body_names = ["tcp"]
        self.rewards.end_effector_speed.params["asset_cfg"].body_names = ["tcp"]
        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            class_type=CustomJointPositionAction,
            asset_name="robot", joint_names=["joint[0-5]"], scale=0.2, use_default_offset=True
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "tcp"


@configclass
class Indy7ReachTeacherEnvCfg(Indy7ReachEnvCfg):
    observations = TeacherObsCfg()
        
    actor_obs_list: list = ["proprioception", "privileged"]
    critic_obs_list: list | None = None # None: same as actor_obs_list
    teacher_obs_list: list | None = None # unused. No teacher for teacher

@configclass
class Indy7ReachStudentEnvCfg(Indy7ReachTeacherEnvCfg):
    actor_obs_list: list = ["proprioception"]
    teacher_obs_list: list = ["proprioception", "privileged"]

@configclass
class Indy7ReachCMDPEnvCfg(Indy7ReachEnvCfg):
    observations = TeacherObsCfg()
    costs = CostsCfg()
        
    actor_obs_list: list = ["proprioception", "privileged"]
    critic_obs_list: list | None = None 
    teacher_obs_list: list | None = None 

    def __post_init__(self):
        super().__post_init__()
        # self.rewards.joint_vel.weight *= 0.1



from isaaclab.terrains import FlatPatchSamplingCfg, TerrainImporter, TerrainImporterCfg
from isaaclab.terrains  import TerrainGeneratorCfg
from isaac_neuromeka.terrains.mesh_terrain_cfg import MeshBoxTerrainCfg
from isaac_neuromeka.terrains.terrain_importer_no_overlap import TerrainImporterNoOverlap
import isaaclab.sim as sim_utils

@configclass
class IndyReachObstacleEnvCfg(Indy7ReachEnvCfg):
    
    
    def __post_init__(self):
        super().__post_init__()
        
        terain_cfg = TerrainGeneratorCfg(
            curriculum=False,
            size=(2.0, 2.0),
            border_width=0.0,
            num_rows=65,num_cols=65, # for 4000 envs
            difficulty_range=(0.25, 1.0),
            sub_terrains={
                "boxes": MeshBoxTerrainCfg(
                    high_prob=0.1,
                    proportion=0.5,
                    grid_width=0.3,
                    grid_height_range=(-0.05, 0.3),
                    low_height_ratio = 0.3,                   
                    platform_width=0.5,
                    robot_range_width = 0.25
                )                
            },
        )
                
        # remove ground attribute
        delattr(self.scene, "ground")
    
        # replace terrain
        self.scene.terrain = TerrainImporterCfg(
            class_type=TerrainImporterNoOverlap,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terain_cfg,
            max_init_terrain_level=9,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Plastics/Rubber_Smooth.mdl",
                project_uvw=True,
            ),
            debug_vis=False,
        )
