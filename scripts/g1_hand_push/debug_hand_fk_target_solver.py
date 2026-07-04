"""Kinematic FK IK smoke: solve G1 palms toward sampled contact targets."""

import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=20)
parser.add_argument("--fd_eps", type=float, default=0.05)
parser.add_argument("--damping", type=float, default=0.05)
parser.add_argument("--step_scale", type=float, default=0.7)
parser.add_argument("--max_dq", type=float, default=0.20)
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
    keep = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch")
    return [i for i, n in enumerate(jnames) if n.lower().startswith(prefix) and any(k in n.lower() for k in keep)]


def set_robot_state(env, q):
    env_ids = env.robot._ALL_INDICES
    root = env.robot.data.default_root_state[env_ids].clone()
    root[:, :3] += env.scene.env_origins[env_ids]
    qd = torch.zeros_like(q)

    env.robot.write_root_pose_to_sim(root[:, :7], env_ids)
    env.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
    env.robot.write_joint_state_to_sim(q, qd, None, env_ids)

    env.sim.forward()
    env.scene.update(0.0)


def fmt(x):
    return [round(float(v), 4) for v in x.detach().cpu().tolist()]


def damped_ls_step(J, err, damping):
    # J: 3 x n, err: 3
    eye = torch.eye(3, device=J.device, dtype=J.dtype)
    dq = J.T @ torch.linalg.solve(J @ J.T + damping * damping * eye, err)
    return dq


def solve_side(env, q, joint_ids, side):
    target = env.left_contact_target_w[0] if side == "left" else env.right_contact_target_w[0]

    set_robot_state(env, q)
    left, right = palm_pos(env)
    cur = left[0] if side == "left" else right[0]
    err = target - cur

    J_cols = []
    for jid in joint_ids:
        q_plus = q.clone()
        q_plus[:, jid] += args_cli.fd_eps
        set_robot_state(env, q_plus)
        lp, rp = palm_pos(env)
        pos_plus = lp[0] if side == "left" else rp[0]

        q_minus = q.clone()
        q_minus[:, jid] -= args_cli.fd_eps
        set_robot_state(env, q_minus)
        lm, rm = palm_pos(env)
        pos_minus = lm[0] if side == "left" else rm[0]

        J_cols.append((pos_plus - pos_minus) / (2.0 * args_cli.fd_eps))

    J = torch.stack(J_cols, dim=1)
    dq = damped_ls_step(J, err, args_cli.damping)
    dq = args_cli.step_scale * torch.clamp(dq, -args_cli.max_dq, args_cli.max_dq)

    for local_i, jid in enumerate(joint_ids):
        q[:, jid] += dq[local_i]

    return q, cur, target, err, dq


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
        env.reset()

        jnames = joint_names(env)
        left_ids = arm_joint_ids(jnames, "left")
        right_ids = arm_joint_ids(jnames, "right")

        print("[G1_HAND_FK_IK_DEBUG] left_ids")
        for i in left_ids:
            print(f"  {i:02d} {jnames[i]}")
        print("[G1_HAND_FK_IK_DEBUG] right_ids")
        for i in right_ids:
            print(f"  {i:02d} {jnames[i]}")

        q = env.robot.data.default_joint_pos.clone()
        set_robot_state(env, q)

        for it in range(args_cli.iters):
            q, lcur, ltgt, lerr, ldq = solve_side(env, q, left_ids, "left")
            q, rcur, rtgt, rerr, rdq = solve_side(env, q, right_ids, "right")

            set_robot_state(env, q)
            left, right = palm_pos(env)
            lnorm = torch.linalg.norm(env.left_contact_target_w[0] - left[0])
            rnorm = torch.linalg.norm(env.right_contact_target_w[0] - right[0])

            print(
                "[G1_HAND_FK_IK_DEBUG] "
                f"iter={it:02d} "
                f"left_err={float(lnorm.detach().cpu()):.4f} "
                f"right_err={float(rnorm.detach().cpu()):.4f} "
                f"left_pos={fmt(left[0])} "
                f"left_target={fmt(env.left_contact_target_w[0])} "
                f"right_pos={fmt(right[0])} "
                f"right_target={fmt(env.right_contact_target_w[0])}",
                flush=True,
            )

        print("G1_HAND_FK_TARGET_SOLVER_DEBUG_OK", flush=True)

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
