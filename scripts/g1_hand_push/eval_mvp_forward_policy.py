"""Headless eval for G1-Hand-ShortPush MVP policy.

Uses the same RslRlVecEnvWrapper path as train.py/play.py.
Prints box displacement so WSL/headless can evaluate without GUI.
"""

import argparse
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="logs/g1-hand-shortpush/mvp_forward/G1_DC/model_299.pt")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--print_every", type=int, default=25)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import ogmp_isaac.tasks  # noqa: F401
from ogmp_isaac.tasks.g1_hand_push.short_hand_push_env import FlatBoxEnvCfg
from ogmp_isaac.agents.rsl_rl_ppo_cfg import PPORunnerCfg


def fmt(x):
    return [round(float(v), 4) for v in x.detach().cpu().tolist()]


def make_cfg():
    cfg = FlatBoxEnvCfg()
    cfg.scene.num_envs = 1
    cfg.robot_model = "G1_DC"
    cfg.oracle["params"]["nominal_height"] = 0.74

    # Match MVP training.
    cfg.omni_direction_lim = [0.0, 0.0]
    cfg.box_start = 0.75
    cfg.target = 1.45
    cfg.box_yaw_lim = [0.0, 0.0]
    cfg.goal_yaw_lim = [0.0, 0.0]

    cfg.debug_contact_modes = False
    cfg.debug_contact_modes_max_prints = 0
    cfg.debug_hand_target_errors = True
    cfg.select_reachable_contact_mode = True
    cfg.allow_contact_target_swap = True
    cfg.visualize_markers = False

    if hasattr(args_cli, "device") and args_cli.device is not None:
        cfg.sim.device = args_cli.device

    return cfg


def make_agent_cfg():
    cfg = PPORunnerCfg()
    cfg.policy.class_name = "ActorCriticRecurrent"
    cfg.policy.init_noise_std = 1.0
    cfg.policy.actor_hidden_dims = [200, 100]
    cfg.policy.critic_hidden_dims = [200, 100]
    cfg.policy.activation = "elu"
    cfg.policy.rnn_type = "lstm"
    cfg.policy.rnn_hidden_size = 128
    cfg.policy.rnn_num_layers = 1
    return cfg


def main():
    env_cfg = make_cfg()

    gym_env = gym.make("G1-Hand-ShortPush-v0", cfg=env_cfg, render_mode=None)
    raw_env = gym_env.unwrapped
    env = RslRlVecEnvWrapper(gym_env)

    try:
        agent_cfg = make_agent_cfg()
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        obs = env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]

        box0 = raw_env.box.data.root_pos_w[0, :3].clone()
        robot0 = raw_env.robot.data.root_pos_w[0, :3].clone()

        print(
            "[MVP_EVAL_START] "
            f"checkpoint={args_cli.checkpoint} "
            f"box0={fmt(box0)} robot0={fmt(robot0)} "
            f"target={fmt(raw_env.target_pos[0])}",
            flush=True,
        )

        for step in range(args_cli.steps):
            with torch.inference_mode():
                action = policy(obs)

            out = env.step(action)
            if len(out) == 4:
                obs, rew, dones, info = out
                terminated = dones
                truncated = torch.zeros_like(dones)
            elif len(out) == 5:
                obs, rew, terminated, truncated, info = out
            else:
                raise RuntimeError(f"Unexpected env.step output length: {len(out)}")

            if hasattr(policy, "reset"):
                policy.reset(terminated | truncated)

            if step % args_cli.print_every == 0 or step == args_cli.steps - 1:
                box = raw_env.box.data.root_pos_w[0, :3].clone()
                robot = raw_env.robot.data.root_pos_w[0, :3].clone()
                box_disp = box - box0
                rb_dist = torch.linalg.norm(robot[:2] - box[:2])

                print(
                    "[MVP_EVAL_STEP] "
                    f"step={step:03d} "
                    f"reward={float(rew[0].detach().cpu()):+.4f} "
                    f"box={fmt(box)} "
                    f"box_disp={fmt(box_disp)} "
                    f"robot={fmt(robot)} "
                    f"robot_box_xy_dist={float(rb_dist.detach().cpu()):.4f} "
                    f"terminated={bool(terminated[0].detach().cpu())} "
                    f"truncated={bool(truncated[0].detach().cpu())}",
                    flush=True,
                )

        box1 = raw_env.box.data.root_pos_w[0, :3].clone()
        robot1 = raw_env.robot.data.root_pos_w[0, :3].clone()
        disp = box1 - box0

        print(
            "[MVP_EVAL_RESULT] "
            f"box0={fmt(box0)} box1={fmt(box1)} "
            f"box_disp={fmt(disp)} "
            f"forward_x={float(disp[0].detach().cpu()):+.4f} "
            f"lateral_y={float(disp[1].detach().cpu()):+.4f} "
            f"robot0={fmt(robot0)} robot1={fmt(robot1)}",
            flush=True,
        )
        print("MVP_FORWARD_POLICY_EVAL_OK", flush=True)

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
