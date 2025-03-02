# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import pdb
from collections.abc import Sequence
import warnings
import carb
from collections.abc import Callable

import numpy as np
import torch
# from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventManager,
    ObservationManager,
    CommandManager,
    CurriculumManager,
    RewardManager,
    TerminationManager,
    RecorderManager
)


from isaaclab.managers.manager_base import ManagerBase, ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.utils import configclass
from dataclasses import MISSING

from isaac_neuromeka.env.managers import*

class CustomManagerBasedRLEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        # -- container for extra information
        self.extra_data = dict()
        super().__init__(cfg, render_mode, **kwargs)
        # -- container for delay randomization
        self.delay_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
    def step(self, action: torch.Tensor):
        # process actions
        self.action_manager.process_action(action)
        # perform physics stepping
        for i in range(self.cfg.decimation):
            # set actions into buffers w/ delay
            execute_env_ids = torch.where(self.delay_steps <= i)[0]
            self.action_manager.apply_action(execute_env_ids)
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
        # perform rendering if gui is enabled
        if self.sim.has_gui() or self.sim.has_rtx_sensors():
            self.sim.render()

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        
        # -- cost computation
        self.cost_buf = self.cost_manager.compute(dt=self.step_dt)
        
        self.extras["costs"] = self.cost_buf

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def load_managers(self):
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)
        # call the parent class to load the managers for observations and actions.


        # prepare the managers
        # -- recorder manager
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        # -- action manager
        self.action_manager = CustomActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        # -- observation manager
        self.observation_manager = CustomObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)
        # -- event manager
        self.event_manager = EventManager(self.cfg.events, self)
        print("[INFO] Event Manager: ", self.event_manager)

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = CustomRewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)
        
        # CMDP specific
        self.cost_manager = CostManager(self.cfg.costs, self)
        print("[INFO] Cost Manager: ", self.cost_manager)


    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        cost_info = self.cost_manager.reset(env_ids)
        self.extras["log"].update(cost_info)
        
    def close(self):
        if not self._is_closed:
            del self.cost_manager
        super().close()
    



