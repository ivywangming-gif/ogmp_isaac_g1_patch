"""Per-joint sensitivity debug: which joints move each palm."""

import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=25)
parser.add_argument("--delta", type=float, default=0.10)
parser.add_argument("--topk", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from ogmp_isaac.tasks.g1_hand_push.short_hand_push_env import FlatBoxEnv, FlatBoxEnvCfg


def names_or_die(obj, attr):
    x = getattr(obj, attr, None)
    if x is None:
        raise RuntimeError(f"missing {attr}")
    return list(x)


def joint_names(env):
    return list(getattr(env.robot, "joint_names", None) or getattr(env.robot.data, "joint_names"))


def body_pos_w(env):
    x = getattr(env.robot.data, "body_link_pos_w", None)
    return x if x is not None else env.robot.data.body_pos_w


def palm_pos(env):
    x = body_pos_w(env)
    return (
        x[:, env.left_hand_body_idx, :3].clone(),
        x[:, env.right_hand_body_idx, :3].clone(),
    )


def run(env, action, steps):
    for _ in range(steps):
        env.step(action)


def clamp(env, action):
    lim = getattr(env, "motor_pos_cmd_limits", None)
    if lim is None:
        return action
    return torch.max(torch.min(action, lim[:, 1]), lim[:, 0])


def main():
    cfg = FlatBoxEnvCfg()
    cfg.scene.num_envs = 1
    cfg.robot_model = "G1_DC"
    cfg.oracle["params"]["nominal_height"] = 0.74
    cfg.debug_contact_modes = True
    cfg.debug_contact_modes_max_prints = 1
    cfg.visualize_markers = False
    cfg.rewards = {}
    cfg.terminations = {}

    if hasattr(args_cli, "device"):
        cfg.sim.device = args_cli.device

    env = FlatBoxEnv(cfg, render_mode=None)

    try:
        jnames = joint_names(env)
        default = env.robot.data.default_joint_pos.clone()

        rows = []
        for jid, jname in enumerate(jnames):
            env.reset()
            run(env, default, 8)
            l0, r0 = palm_pos(env)

            action = default.clone()
            action[:, jid] += args_cli.delta
            action = clamp(env, action)

            run(env, action, args_cli.steps)
            l1, r1 = palm_pos(env)

            lm = float(torch.linalg.norm(l1 - l0, dim=-1)[0].detach().cpu())
            rm = float(torch.linalg.norm(r1 - r0, dim=-1)[0].detach().cpu())
            rows.append((jid, jname, lm, rm, lm - rm))

        print("[G1_HAND_JOINT_SENSITIVITY] top_left")
        for jid, name, lm, rm, diff in sorted(rows, key=lambda x: x[2], reverse=True)[: args_cli.topk]:
            print(f"  {jid:02d} {name:40s} left_move={lm:.4f} right_move={rm:.4f} left_minus_right={diff:+.4f}")

        print("[G1_HAND_JOINT_SENSITIVITY] top_right")
        for jid, name, lm, rm, diff in sorted(rows, key=lambda x: x[3], reverse=True)[: args_cli.topk]:
            print(f"  {jid:02d} {name:40s} left_move={lm:.4f} right_move={rm:.4f} left_minus_right={diff:+.4f}")

        print("[G1_HAND_JOINT_SENSITIVITY] asymmetric_left")
        for jid, name, lm, rm, diff in sorted(rows, key=lambda x: x[4], reverse=True)[: args_cli.topk]:
            print(f"  {jid:02d} {name:40s} left_move={lm:.4f} right_move={rm:.4f} left_minus_right={diff:+.4f}")

        print("[G1_HAND_JOINT_SENSITIVITY] asymmetric_right")
        for jid, name, lm, rm, diff in sorted(rows, key=lambda x: x[4])[: args_cli.topk]:
            print(f"  {jid:02d} {name:40s} left_move={lm:.4f} right_move={rm:.4f} left_minus_right={diff:+.4f}")

        print("G1_HAND_JOINT_SENSITIVITY_DEBUG_OK", flush=True)

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
