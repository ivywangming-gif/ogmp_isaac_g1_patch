"""Check whether arm action targets actually move G1 arm joints."""

import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--settle_steps", type=int, default=20)
parser.add_argument("--delta", type=float, default=0.30)
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


def arm_ids(jnames):
    ids = []
    for i, n in enumerate(jnames):
        low = n.lower()
        if (low.startswith("left_") or low.startswith("right_")) and (
            "shoulder" in low or "elbow" in low or "wrist" in low
        ):
            ids.append(i)
    return ids


def f(x):
    return float(x.detach().cpu())


def fmtv(x):
    return [round(float(v), 5) for v in x.detach().cpu().tolist()]


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
        ids = arm_ids(jnames)
        default = env.robot.data.default_joint_pos.clone()

        print(f"[JOINT_TRACK_DEBUG] num_joints={len(jnames)} action_shape={tuple(default.shape)}")
        print("[JOINT_TRACK_DEBUG] arm_ids")
        for jid in ids:
            lim = env.motor_pos_cmd_limits[jid]
            print(
                f"  {jid:02d} {jnames[jid]:35s} "
                f"default={f(default[0, jid]):+.4f} "
                f"lim=[{f(lim[0]):+.4f},{f(lim[1]):+.4f}]"
            )

        print("[JOINT_TRACK_DEBUG] per_joint_tracking")

        for jid in ids:
            env.reset()
            run(env, default, args_cli.settle_steps)

            q0 = env.robot.data.joint_pos[:, jid].clone()
            l0, r0 = palm_pos(env)

            action = default.clone()
            action[:, jid] += args_cli.delta
            action = clamp(env, action)
            target = action[:, jid].clone()

            run(env, action, args_cli.steps)

            q1 = env.robot.data.joint_pos[:, jid].clone()
            l1, r1 = palm_pos(env)

            left_move = torch.linalg.norm(l1 - l0, dim=-1)
            right_move = torch.linalg.norm(r1 - r0, dim=-1)

            print(
                f"  {jid:02d} {jnames[jid]:35s} "
                f"q0={f(q0[0]):+.5f} target={f(target[0]):+.5f} q1={f(q1[0]):+.5f} "
                f"dq={f(q1[0]-q0[0]):+.5f} target_err={f(target[0]-q1[0]):+.5f} "
                f"left_move={f(left_move[0]):.5f} right_move={f(right_move[0]):.5f} "
                f"left0={fmtv(l0[0])} left1={fmtv(l1[0])} "
                f"right0={fmtv(r0[0])} right1={fmtv(r1[0])}",
                flush=True,
            )

        print("G1_JOINT_TARGET_TRACKING_DEBUG_OK", flush=True)

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
