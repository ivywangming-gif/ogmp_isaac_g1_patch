"""Kinematic FK sensitivity: write joint state directly, no dynamic rollout."""

import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
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


def arm_joint_ids(jnames, side):
    prefix = "left_" if side == "left" else "right_"
    keep = ("shoulder", "elbow", "wrist")
    return [i for i, n in enumerate(jnames) if n.lower().startswith(prefix) and any(k in n.lower() for k in keep)]


def set_robot_state(env, q):
    env_ids = env.robot._ALL_INDICES
    root = env.robot.data.default_root_state[env_ids].clone()
    root[:, :3] += env.scene.env_origins[env_ids]
    qd = torch.zeros_like(q)

    env.robot.write_root_pose_to_sim(root[:, :7], env_ids)
    env.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
    env.robot.write_joint_state_to_sim(q, qd, None, env_ids)

    # Push state through PhysX/USD buffers without dynamic rollout.
    env.sim.forward()
    env.scene.update(0.0)


def f(x):
    return float(x.detach().cpu())


def fmt(x):
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
        env.reset()

        jnames = joint_names(env)
        left_ids = arm_joint_ids(jnames, "left")
        right_ids = arm_joint_ids(jnames, "right")
        ids = left_ids + right_ids

        q0 = env.robot.data.default_joint_pos.clone()
        set_robot_state(env, q0)
        l0, r0 = palm_pos(env)

        print("[G1_HAND_FK_DEBUG] left_arm_ids")
        for i in left_ids:
            print(f"  {i:02d} {jnames[i]}")
        print("[G1_HAND_FK_DEBUG] right_arm_ids")
        for i in right_ids:
            print(f"  {i:02d} {jnames[i]}")

        print(
            "[G1_HAND_FK_DEBUG] baseline "
            f"left={fmt(l0[0])} right={fmt(r0[0])}",
            flush=True,
        )

        rows = []
        for jid in ids:
            q = q0.clone()
            q[:, jid] += args_cli.delta
            set_robot_state(env, q)
            lp, rp = palm_pos(env)

            q = q0.clone()
            q[:, jid] -= args_cli.delta
            set_robot_state(env, q)
            lm, rm = palm_pos(env)

            left_plus = lp - l0
            right_plus = rp - r0
            left_minus = lm - l0
            right_minus = rm - r0

            left_mag = 0.5 * (
                torch.linalg.norm(left_plus, dim=-1)
                + torch.linalg.norm(left_minus, dim=-1)
            )
            right_mag = 0.5 * (
                torch.linalg.norm(right_plus, dim=-1)
                + torch.linalg.norm(right_minus, dim=-1)
            )

            rows.append(
                (
                    jid,
                    jnames[jid],
                    f(left_mag[0]),
                    f(right_mag[0]),
                    fmt(left_plus[0]),
                    fmt(right_plus[0]),
                )
            )

        print("[G1_HAND_FK_SENSITIVITY]")
        for jid, name, lm, rm, lv, rv in sorted(rows, key=lambda x: max(x[2], x[3]), reverse=True):
            print(
                f"  {jid:02d} {name:35s} "
                f"left_fk={lm:.5f} right_fk={rm:.5f} "
                f"plus_left_vec={lv} plus_right_vec={rv}",
                flush=True,
            )

        print("G1_HAND_FK_SENSITIVITY_DEBUG_OK", flush=True)

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
