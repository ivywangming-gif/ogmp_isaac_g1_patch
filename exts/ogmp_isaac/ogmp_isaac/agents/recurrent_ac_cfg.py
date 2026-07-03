from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    rnn_type: str = MISSING
    """The type of RNN to use."""

    rnn_hidden_size: int = MISSING
    """The hidden size of the RNN."""

    rnn_num_layers: int = MISSING
    """The number of layers of the RNN."""
