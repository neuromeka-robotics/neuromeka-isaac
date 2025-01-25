# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from NRMK-RL."""

from __future__ import annotations

import pdb
import time
from datetime import datetime

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.orbit.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with NRMK-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append NRMK-RL cli arguments
cli_args.add_nrmk_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
if args_cli.headless:
    args_cli.experience = "omni.isaac.sim.python.gym.headless.kit"
else:
    args_cli.experience = "omni.isaac.sim.python.kit"

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import os
import sys

import torch
import gymnasium as gym
import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import get_checkpoint_path, parse_env_cfg

# add NRMK-RL path
from nrmk_rl.runners import *

# Import extensions to set up environment tasks
import isaac_neuromeka.tasks  # noqa: F401
from isaac_neuromeka.learning.runner_cfg import NrmkRlOnPolicyRunnerCfg
from isaac_neuromeka.env.vecenv_wrapper import  NrmkRlVecEnvWrapper

from isaac_neuromeka.utils.exporter import (
    export_policy_as_onnx,
    _OnnxBaseMLPPolicyExporter
)
from isaac_neuromeka.utils.helper import DataLogger


def main():
    """Play with NRMK-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    agent_cfg: NrmkRlOnPolicyRunnerCfg = cli_args.parse_nrmk_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for nrmk-rl
    env = NrmkRlVecEnvWrapper(env)

    # specify directory for logging experiments
    import importlib.util
    pkg_name = "isaac_neuromeka"
    spec = importlib.util.find_spec(pkg_name)
    if spec is not None:
        pkg_path = spec.origin
    else:
        print(f"Package '{pkg_name}' not found.")
    pkg_path = os.path.join(pkg_path[:pkg_path.rfind(pkg_name)], pkg_name)
    log_root_path = os.path.join(pkg_path, "logs", "nrmk_rl", agent_cfg.experiment_name)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create runner from nrmk-rl
    runner_cls = eval(agent_cfg.runner_cls_name)
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
 
    # load previously trained model
    runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)



    # control dt
    control_dt = env.unwrapped.step_dt

    # set logger
    use_logger = False
    if use_logger:
        logger = DataLogger(
            folder=f"robot_logs/{datetime.now().strftime('%Y-%m-%d')}",
            file=f"{datetime.now().strftime('%H-%M-%S')}",
            data_types=["qpos", "qvel", "command",
                        "qpos_history", "qvel_history", "action_history",
                        "nn_action", "position_error", "orientation_error"]
        )

    # reset environment
    obs, infos = env.get_observations()
    
    
    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(
        actor_critic=runner.alg.actor_critic,
        input_dim = obs.shape[1],
        Exporter=_OnnxBaseMLPPolicyExporter,
        path=export_model_dir, filename="policy.onnx", device="cuda")


    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            start = time.time()
    
            # agent stepping
            actions = policy(obs)
            obs_np = obs[0].cpu().numpy()

            # update logger
            if use_logger:
                logger.append(
                    obs_np[:6],
                    obs_np[6:12],
                    obs_np[12:19],
                    obs_np[19:31],
                    obs_np[31:43],
                    obs_np[43:55],
                    actions[0].cpu().numpy(),
                    infos["observations"]["robot"]["position_error"].cpu().numpy()[0],
                    infos["observations"]["robot"]["orientation_error"].cpu().numpy()[0]
                )

            # env stepping
            obs, _, _, infos = env.step(actions)

            end = time.time()
            wait_time = control_dt - (end - start)
            if wait_time > 0:
                time.sleep(wait_time)

    # close the simulator
    env.close()

    # save logger
    if use_logger:
        logger.save_to_csv()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
