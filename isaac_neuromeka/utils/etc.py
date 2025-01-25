from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlPpoActorCriticCfg
from omni.isaac.orbit.utils import configclass


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks."""

    rnn_type: str = "gru"



@configclass
class EmptyCfg:
    """ Empty configuration file """
