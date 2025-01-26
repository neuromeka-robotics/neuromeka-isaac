from __future__ import annotations

import math
import numpy as np

from omni.isaac.lab.utils import configclass
# from omni.isaac.lab.assets import RigidObjectCfg
# import omni.isaac.lab.sim as sim_utils
# from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg
import isaac_neuromeka.mdp as mdp

##
# Pre-defined configs
##
from isaac_neuromeka.assets import DUAL_ARM_CFG
from isaac_neuromeka.tasks.manipulation.reach.reach_env_cfg import ReachEnvCfg, ObservationsCfg
from isaac_neuromeka.tasks.manipulation.reach.reach_env_cfg import TeacherObsCfg
from isaac_neuromeka.tasks.manipulation.reach.reach_env_cfg import CostsCfg
from isaac_neuromeka.mdp.actions import CustomJointPositionAction
from isaac_neuromeka.utils.etc import EmptyCfg
from omni.isaac.lab.managers import ActionTermCfg as ActionTerm

##
# Environment configuration
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = mdp.JointPositionActionCfg(
            class_type=CustomJointPositionAction,
            asset_name="robot", joint_names=[".*_arm_.*"], scale=1.0, use_default_offset=True
        )
    base_action: ActionTerm = mdp.JointPositionActionCfg(
            class_type=CustomJointPositionAction,
            asset_name="robot", joint_names=["base_joint"], scale=0.5, use_default_offset=True
        )

@configclass
class DualArmReachEnvCfg(ReachEnvCfg):

    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to indy
        self.scene.robot = DUAL_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["left_arm_tcp"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["left_arm_tcp"]
        # self.rewards.end_effector_pose_tracking.params["asset_cfg"].body_names = ["tcp"]
        self.rewards.end_effector_speed.params["asset_cfg"].body_names = ["left_arm_tcp"]
        # override actions

        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "left_arm_tcp"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)
        self.commands.ee_pose.debug_vis = False

