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
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
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
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from isaac_neuromeka.env.rl_task_custom_env import HistoryManager
from isaac_neuromeka.utils.etc import EmptyCfg
import isaac_neuromeka.mdp as mdp


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    class ConFig:
        default_ee_pose = np.array([0.3563, -0.1829,  0.5132], float)
    
    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(6.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # pos_x=(ConFig.default_ee_pose[0], ConFig.default_ee_pose[0] + 0.3),
            # pos_y=(ConFig.default_ee_pose[1] - 0.2, ConFig.default_ee_pose[1] + 0.2),
            # pos_z=(ConFig.default_ee_pose[2] - 0.3, ConFig.default_ee_pose[2]),
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=MISSING,  # depends on end-effector axis
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Gnoise(std=0.05))
        joint_vel = ObsTerm(func=mdp.finite_joint_vel, noise=Gnoise(std=0.5))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})

        joint_pos_history = ObsTerm(func=HistoryManager, params={"name": "joint_pos", "length": 2})
        joint_vel_history = ObsTerm(func=HistoryManager, params={"name": "joint_vel", "length": 2})
        action_history = ObsTerm(func=mdp.action_history)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            

    @configclass
    class DemoPolicyCfg(ObsGroup):
        q = ObsTerm(func=mdp.joint_pos)
        qdot = ObsTerm(func=mdp.finite_joint_vel)
        p = ObsTerm(func=mdp.body_pose_b, params={"body_name": "tcp"})
        pdot = ObsTerm(func=mdp.body_vel_b, params={"body_name": "tcp"})
        op_state = ObsTerm(func=mdp.op_state)
        position_error = ObsTerm(func=mdp.position_error)
        orientation_error = ObsTerm(func=mdp.orientation_error)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg | DemoPolicyCfg = PolicyCfg()

@configclass
class TeacherObsCfg(ObservationsCfg):
    
    @configclass
    class Proprio(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Gnoise(std=0.01))
        joint_vel = ObsTerm(func=mdp.finite_joint_vel, noise=Gnoise(std=0.1))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})

        joint_pos_history = ObsTerm(func=HistoryManager, params={"name": "joint_pos", "length": 2})
        joint_vel_history = ObsTerm(func=HistoryManager, params={"name": "joint_vel", "length": 2})
        action_history = ObsTerm(func=mdp.action_history)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class Privileged(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_friction = ObsTerm(func=mdp.joint_friction)
        joint_damping = ObsTerm(func=mdp.joint_damping)
        action_delay = ObsTerm(func=mdp.action_delay_steps)
        # TODO: action delay
         
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True            
        
    proprioception = Proprio()
    privileged = Privileged()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.5, 1.5),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_range": (0.7, 1.3),
            "operation": "abs",
            "distribution": "uniform"
        }
    )

    randomize_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_range": (94.0, 106.0),  # (100 - 6, 100 + 6)
            "damping_range": (17.0, 23.0),  # (20 - 3, 20 + 3)
            "operation": "abs",  # if use "reset" + "add", the sampled values are added to previous iter values.
            "distribution": "uniform",
        },
    )

    randomize_delay = EventTerm(
        func=mdp.randomize_delay,
        mode="reset",
        params={
            "delay_step_range": {"low": 20, "high": 24}
        }
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.end_effector_position_tracking_bounded,
        weight= 0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose", "distance_max": 0.5},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.end_effector_orientation_tracking_distance_bounded,
        weight= 0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose", "distance_max": 0.25 },
    )

    ## regularizers    
    end_effector_speed = RewTerm(
        func=mdp.end_effector_speed,
        weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING)},
    )
    
    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)

    action_second_rate = RewTerm(func=mdp.action_second_rate_l2, weight=-0.0001)  # -0.00005

    joint_vel = RewTerm(
        func=mdp.finite_joint_vel_l2,
        weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    

@configclass
class CostsCfg:
    """Cost terms for the CMDP."""

    joint_vel = RewTerm(
        func=mdp.joint_vel_cost_relu,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "soft_limit_ratio": 0.9}
    )
    
    
    ee_spd = RewTerm(
        func=mdp.ee_speed_cost_relu,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["tcp"]), "speed_limit": 1.0}   
    )
    
    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    # )


##
# Environment configuration
##

from omni.isaac.lab.envs import ManagerBasedEnvCfg
from omni.isaac.lab.envs.ui import ManagerBasedRLEnvWindow

@configclass
class NrmkRLCfg(ManagerBasedEnvCfg):
    """Configuration for a reinforcement learning environment."""

    # ui settings
    ui_window_class_type: type | None = ManagerBasedRLEnvWindow

    # general settings
    is_finite_horizon: bool = False
    episode_length_s: float = MISSING

    # environment settings
    rewards: object = MISSING
    terminations: object = MISSING
    curriculum: object = MISSING
    commands: object = MISSING
    
    # New for NRMK-RL
    actor_obs_list: list = ["policy"]
    critic_obs_list: list | None = None
    teacher_obs_list: list | None = None
    
    

