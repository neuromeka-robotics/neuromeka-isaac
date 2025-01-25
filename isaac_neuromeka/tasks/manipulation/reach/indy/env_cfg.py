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
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)
        # self.observations.policy.enable_corruption = False



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


@configclass
class Indy7ReachPlayEnvCfg(Indy7ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # override simulation time
        self.episode_length_s = 120.
        # overide observation
        self.observations.policy.enable_corruption = False
        self.observations.robot = self.observations.DemoPolicyCfg()

        # override command generator to use keyboard inputs
        self.commands.ee_pose = mdp.DefaultUniformPoseCommandCfg(
            class_type=mdp.KeyboardPoseCommand,
            default_ee_pose=np.array([*self.commands.ConFig.default_ee_pose, 0., math.pi, 0.]),
            asset_name="robot",
            body_name="tcp",
            resampling_time_range=(self.episode_length_s, self.episode_length_s),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(self.commands.ConFig.default_ee_pose[0], self.commands.ConFig.default_ee_pose[0] + 0.3),
                pos_y=(self.commands.ConFig.default_ee_pose[1] - 0.2, self.commands.ConFig.default_ee_pose[1] + 0.2),
                pos_z=(self.commands.ConFig.default_ee_pose[2] - 0.3, self.commands.ConFig.default_ee_pose[2]),
                roll=(0.0, 0.0),
                pitch=(math.pi, math.pi),
                yaw=(-3.14, 3.14),
            ),
        )


@configclass
class Indy7ReachDemoEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to indy
        self.scene.robot = INDY7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override
        self.observations.policy = self.observations.DemoPolicyCfg()
        self.rewards = EmptyCfg()
        self.curriculum = EmptyCfg()
        self.events.randomize_joint_friction = None
        self.events.randomize_joint_stiffness_and_damping = None
        self.events.randomize_delay = None  # modeling real-robot delay
        self.episode_length_s = 6000.

        self.actions.arm_action = mdp.JointPositionActionCfg(
            class_type=CustomJointPositionAction,
            asset_name="robot", joint_names=["joint[0-5]"], scale=1., use_default_offset=False
        )

        # override command generator to use keyboard inputs
        self.commands.ee_pose = mdp.UniformPoseCommandCfg(
            class_type=mdp.EmptyPoseCommand,
            asset_name="robot",
            body_name="tcp",
            resampling_time_range=(self.episode_length_s, self.episode_length_s),
            debug_vis=True,
        )
