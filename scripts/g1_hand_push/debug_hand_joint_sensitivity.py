"""Arm-only, baseline-subtracted per-joint palm sensitivity debug."""

import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=25)
parser.add_argument("--settle_steps", type=int, default=8)
parser.add_argument("--delta", type=float, default=0.10)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from ogmp_isaac.tasks.g1_hand_push.short_hand_push_env import FlatBoxEnv, FlatBoxEnvCfg


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


def arm_joint_ids(jnames, side):
    prefix = "left_" if side == "left" else "right_"
    keep = ("shoulder", "elbow", "wrist")
    return [i for i, n in enumerate(jnames) if n.lower().startswith(prefix) and any(k in n.lower() for k in keep)]


def rollout_delta(env, base_action, jid=None, delta=0.0):
    env.reset()
    run(env, base_action, args_cli.settle_steps)
    l0, r0 = palm_pos(env)

    action = base_action.clone()
    if jid is not None:
        action[:, jid] += delta
        action = clamp(env, action)

    run(env, action, args_cli.steps)
    l1, r1 = palm_pos(env)
    return l1 - l0, r1 - r0


def norm1(x):
    return float(torch.linalg.norm(x, dim=-1)[0].detach().cpu())


def fmtvec(x):
    return [round(float(v), 4) for v in x[0].detach().cpu().tolist()]


def main():
    cfg = FlatBoxEnvCfg()
    cfg.scene.num_envs = 1
    cfg.robot_model = "G1_DC"
    cfg.oracle["params"]["nominal_height"] = 0.74
    cfg.debug_contact_modes = False
    cfg.visualize_markers = False
    cfg.rewards = {}
    cfg.terminations = {}

    if hasattr(args_cli, "device"):
        cfg.sim.device = args_cli.device

    env = FlatBoxEnv(cfg, render_mode=None)

    try:
        jnames = joint_names(env)
        left_ids = arm_joint_ids(jnames, "left")
        right_ids = arm_joint_ids(jnames, "right")
        ids = left_ids + right_ids

        print("[G1_ARM_IDS_LEFT]")
        for i in left_ids:
            print(f"  {i:02d} {jnames[i]}")
        print("[G1_ARM_IDS_RIGHT]")
        for i in right_ids:
            print(f"  {i:02d} {jnames[i]}")

        base_action = env.robot.data.default_joint_pos.clone()
        base_l, base_r = rollout_delta(env, base_action, None, 0.0)

        rows = []
        for jid in ids:
            plus_l, plus_r = rollout_delta(env, base_action, jid, +args_cli.delta)
            minus_l, minus_r = rollout_delta(env, base_action, jid, -args_cli.delta)

            # Remove common default drift, then use symmetric magnitude.
            eff_plus_l = plus_l - base_l
            eff_plus_r = plus_r - base_r
            eff_minus_l = minus_l - base_l
            eff_minus_r = minus_r - base_r

            left_mag = 0.5 * (norm1(eff_plus_l) + norm1(eff_minus_l))
            right_mag = 0.5 * (norm1(eff_plus_r) + norm1(eff_minus_r))

            rows.append((jid, jnames[jid], left_mag, right_mag, left_mag - right_mag, fmtvec(eff_plus_l), fmtvec(eff_plus_r)))

        print("[G1_HAND_JOINT_SENSITIVITY_BASELINE_SUBTRACTED]")
        for jid, name, lm, rm, diff, lvec, rvec in sorted(rows, key=lambda x: max(x[2], x[3]), reverse=True):
            print(
                f"  {jid:02d} {name:40s} "
                f"left_eff={lm:.5f} right_eff={rm:.5f} left_minus_right={diff:+.5f} "
                f"plus_left_vec={lvec} plus_right_vec={rvec}"
            )

        print("G1_HAND_JOINT_SENSITIVITY_BASELINE_SUBTRACTED_OK", flush=True)

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
