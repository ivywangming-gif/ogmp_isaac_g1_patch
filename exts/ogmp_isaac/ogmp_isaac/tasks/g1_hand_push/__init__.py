import gymnasium as gym
import ogmp_isaac.agents as agents

from .short_hand_push_env import FlatBoxEnv, FlatBoxEnvCfg

gym.register(
    id="G1-Hand-ShortPush-v0",
    entry_point="ogmp_isaac.tasks.g1_hand_push:FlatBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FlatBoxEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.PPORunnerCfg,
    },
)
