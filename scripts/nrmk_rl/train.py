"""Script to train RL agent with NRMK-RL."""

from __future__ import annotations

import pdb

"""Launch Isaac Sim Simulator first."""


import argparse
import os

from omni.isaac.orbit.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with NRMK-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
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

import sys
from datetime import datetime
from shutil import copyfile

import gymnasium as gym
import torch

# add NRMK-RL path
from nrmk_rl.runners import *

import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.utils.dict import print_dict
from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml
from omni.isaac.orbit_tasks.utils import get_checkpoint_path, parse_env_cfg

import isaac_neuromeka.tasks  # import extensions
from isaac_neuromeka.learning.runner_cfg import NrmkRlOnPolicyRunnerCfg
from isaac_neuromeka.env.vecenv_wrapper import NrmkRlVecEnvWrapper


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Train with NRMK-RL agent."""
    # parse configuration
    env_cfg: RLTaskEnvCfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: NrmkRlOnPolicyRunnerCfg = cli_args.parse_nrmk_rl_cfg(args_cli.task, args_cli)

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
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for nrmk-rl
    env = NrmkRlVecEnvWrapper(env)

    # create runner from nrmk-rl
    runner_cls = eval(agent_cfg.runner_cls_name)
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    copyfile(agent_cfg.policy, os.path.join(log_dir, "params", agent_cfg.policy.split("/")[-1]))
    
    if hasattr(agent_cfg, "teacher_policy"):
        copyfile(agent_cfg.teacher_policy, os.path.join(log_dir, "params", agent_cfg.teacher_policy.split("/")[-1]))

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
