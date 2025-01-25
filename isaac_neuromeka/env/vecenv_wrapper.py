# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`RLTaskEnv` instance to NRMK-RL vectorized environment.

The following example shows how to wrap an environment for NRMK-RL:

.. code-block:: python

    from omni.isaac.orbit_tasks.utils.wrappers.nrmk_rl import NrmkRlVecEnvWrapper

    env = NrmkRlVecEnvWrapper(env)

"""
import pdb

import gymnasium as gym
import torch
from nrmk_rl.env import VecEnv
from omni.isaac.orbit.envs import RLTaskEnv


class NrmkRlVecEnvWrapper(VecEnv):
    """Wraps around Orbit environment for NRMK-RL library"""

    def __init__(self, env: RLTaskEnv):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the NRMK-RL runner does not call reset.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`RLTaskEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, RLTaskEnv):
            raise ValueError(f"The environment must be inherited from RLTaskEnv. Environment type: {type(env)}")
        # initialize the wrapper
        self.env = env
        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        # -- privileged observations
        if "critic" in self.unwrapped.observation_manager.group_obs_dim:
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        else:
            self.num_privileged_obs = 0
        # reset at the start since the NRMK-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> RLTaskEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    @property
    def num_cost_terms(self) -> int:
        """Returns the number of cost terms in the environment."""
        return self.env.cost_manager.num_cost_terms
    
    """
    Properties
    """

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        obs_dict = self.unwrapped.observation_manager.compute()
        return self.get_flat_observation(obs_dict), {"observations": obs_dict}

    def get_flat_observation(self, obs_dict) -> torch.Tensor:
        if hasattr(self.cfg, "actor_obs_list"):
            return torch.cat([obs_dict[key] for key in self.cfg.actor_obs_list], dim=-1)
        if isinstance(obs_dict["policy"], dict):
            return torch.cat(list(obs_dict["policy"].values()), dim=-1)
        else:
            return obs_dict["policy"]
        
        
    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in NRMK-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        # reset the environment
        obs_dict, extras = self.env.reset()
        
        obs = self.get_flat_observation(obs_dict)
        extras["observations"] = obs_dict
        return obs, extras

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with NRMK-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        
        obs = self.get_flat_observation(obs_dict)

        extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        return obs, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()
