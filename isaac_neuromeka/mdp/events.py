import numpy as np
import torch
from omni.isaac.orbit.envs import RLTaskEnv
from typing import Dict

def randomize_delay(
        env: RLTaskEnv,
        env_ids: torch.Tensor | None,
        delay_step_range: Dict[str, int]
):
    """
    Available delay: 0 - (decimation - 1)
    """
    env.delay_steps = torch.randint(
        low=np.clip(delay_step_range.get("low", 0), a_min=0, a_max=env.cfg.decimation-1),
        high=np.clip(delay_step_range.get("high", 0), a_min=0, a_max=env.cfg.decimation),
        size=(env.num_envs,), device=env.device)
