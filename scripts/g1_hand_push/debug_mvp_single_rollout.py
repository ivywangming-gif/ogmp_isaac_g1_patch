"""Single-rollout MVP diagnostic. Run one mode per Isaac process."""

import argparse
import torch
import gymnasium as gym
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["default", "zero", "policy"], required=True)
parser.add_argument("--checkpoint", type=str, default="logs/g1-hand-shortpush/mvp_rearface/G1_DC/model_9.pt")
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--print_every", type=int, default=5)
parser.add_argument("--box_h", type=float, default=0.50)
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
    return [round(float(v), 4) for v in x.detach().cpu().flatten().tolist()]


def palm_pos(raw):
    body = getattr(raw.robot.data, "body_link_pos_w", None)
    if body is None:
        body = raw.robot.data.body_pos_w
    return body[:, raw.left_hand_body_idx, :3], body[:, raw.right_hand_body_idx, :3]


def make_cfg():
    cfg = FlatBoxEnvCfg()
    cfg.scene.num_envs = 1
    cfg.robot_model = "G1_DC"
    cfg.observation_space = 113
    cfg.num_observations = 113

    cfg.use_longbox_asset = True
    cfg.longbox_dims = [1.6, 0.8, float(args_cli.box_h)]
    cfg.longbox_source_size = 0.5

    cfg.omni_direction_lim = [0.0, 0.0]
    cfg.box_height = 0.5
    cfg.box_start = 1.05
    cfg.target = 1.65
    cfg.box_yaw_lim = [0.0, 0.0]
    cfg.goal_yaw_lim = [0.0, 0.0]

    cfg.debug_contact_modes = True
    cfg.debug_contact_modes_max_prints = 1
    cfg.debug_hand_target_errors = True
    cfg.select_reachable_contact_mode = True
    cfg.allow_contact_target_swap = True
    cfg.visualize_markers = False

    cfg.oracle["params"]["nominal_height"] = 0.74

    cfg.rewards = {
        "base_pos": {"weight": 0.20, "exp_scale": 3.0},
        "base_ori": {"weight": 0.20, "exp_scale": 5.0},
        "base_lin_vel": {"weight": 0.10, "exp_scale": 2.0},
        "hand_target_tracking": {"weight": 2.0, "exp_scale": 3.0},
        "hand_target_close": {"weight": 1.0, "threshold": 0.18},
        "box_forward_progress": {"weight": 8.0, "max_progress": 0.60},
        "box_forward_vel": {"weight": 2.0, "max_vel": 0.50},
        "torque_exp_norm": {"weight": 0.05, "exp_scale": 0.05},
        "action_norm": {"weight": 0.05, "exp_scale": 1.0},
    }

    cfg.terminations = {}

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


def get_obs(env):
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    return obs


def main():
    gym_env = gym.make("G1-Hand-ShortPush-v0", cfg=make_cfg(), render_mode=None)
    raw = gym_env.unwrapped
    env = RslRlVecEnvWrapper(gym_env)

    try:
        policy = None
        if args_cli.mode == "policy":
            agent_cfg = make_agent_cfg()
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(args_cli.checkpoint)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

        obs = get_obs(env)
        box0 = raw.box.data.root_pos_w[0, :3].clone()
        robot0 = raw.robot.data.root_pos_w[0, :3].clone()

        print(f"=== SINGLE_ROLLOUT_START mode={args_cli.mode} ===")
        print(f"obs_type={type(obs)} obs_shape={getattr(obs, 'shape', None)}")
        print(f"box0={fmt(box0)} robot0={fmt(robot0)} target={fmt(raw.target_pos[0])}")
        print(f"mode_name={raw._contact_mode_names[int(raw.contact_mode_ids[0].detach().cpu())]}")
        print(f"left_target={fmt(raw.left_contact_target_w[0])} right_target={fmt(raw.right_contact_target_w[0])}")

        done_count = 0

        for step in range(args_cli.steps):
            if args_cli.mode == "default":
                action = raw.robot.data.default_joint_pos.clone()
            elif args_cli.mode == "zero":
                action = torch.zeros_like(raw.robot.data.default_joint_pos)
            else:
                with torch.inference_mode():
                    action = policy(obs)

            out = env.step(action)
            if len(out) == 4:
                obs, rew, done, info = out
                terminated = done
                truncated = torch.zeros_like(done)
            else:
                obs, rew, terminated, truncated, info = out

            done = terminated | truncated
            done_count += int(done[0].detach().cpu())

            if policy is not None and hasattr(policy, "reset"):
                policy.reset(done)

            if step % args_cli.print_every == 0 or step == args_cli.steps - 1 or bool(done[0]):
                box = raw.box.data.root_pos_w[0, :3].clone()
                robot = raw.robot.data.root_pos_w[0, :3].clone()
                lhand, rhand = palm_pos(raw)
                le = torch.linalg.norm(lhand[0] - raw.left_contact_target_w[0])
                re = torch.linalg.norm(rhand[0] - raw.right_contact_target_w[0])
                disp = box - box0

                print(
                    f"[{args_cli.mode}] step={step:03d} "
                    f"rew={float(rew[0].detach().cpu()):+.4f} "
                    f"box={fmt(box)} disp={fmt(disp)} "
                    f"robot={fmt(robot)} "
                    f"left_err={float(le.detach().cpu()):.4f} "
                    f"right_err={float(re.detach().cpu()):.4f} "
                    f"done={bool(done[0].detach().cpu())}",
                    flush=True,
                )

        box1 = raw.box.data.root_pos_w[0, :3].clone()
        robot1 = raw.robot.data.root_pos_w[0, :3].clone()
        lhand, rhand = palm_pos(raw)
        le = torch.linalg.norm(lhand[0] - raw.left_contact_target_w[0])
        re = torch.linalg.norm(rhand[0] - raw.right_contact_target_w[0])
        disp = box1 - box0

        print(
            f"=== SINGLE_ROLLOUT_RESULT mode={args_cli.mode} "
            f"forward_x={float(disp[0].detach().cpu()):+.4f} "
            f"robot_z={float(robot1[2].detach().cpu()):+.4f} "
            f"left_err={float(le.detach().cpu()):.4f} "
            f"right_err={float(re.detach().cpu()):.4f} "
            f"done_count={done_count} ===",
            flush=True,
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
