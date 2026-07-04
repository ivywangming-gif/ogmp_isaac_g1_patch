"""Debug whether G1 arm joint actions move left/right palm links."""

import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=30)
parser.add_argument("--delta", type=float, default=0.15)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from ogmp_isaac.tasks.g1_hand_push.short_hand_push_env import FlatBoxEnv, FlatBoxEnvCfg


def get_joint_names(env):
    names = getattr(env.robot, "joint_names", None)
    if names is None:
        names = getattr(env.robot.data, "joint_names", None)
    if names is None:
        raise RuntimeError("Cannot find robot joint_names")
    return list(names)


def get_body_pos_w(env):
    body_pos_w = getattr(env.robot.data, "body_link_pos_w", None)
    if body_pos_w is None:
        body_pos_w = env.robot.data.body_pos_w
    return body_pos_w


def palm_pos(env):
    body_pos_w = get_body_pos_w(env)
    left = body_pos_w[:, env.left_hand_body_idx, :3].clone()
    right = body_pos_w[:, env.right_hand_body_idx, :3].clone()
    return left, right


def resolve_arm_joint_ids(joint_names, side):
    side_tokens = ["left", "l_"] if side == "left" else ["right", "r_"]
    keep_tokens = ["shoulder", "elbow", "wrist"]
    ids = []
    for i, name in enumerate(joint_names):
        low = name.lower()
        if any(tok in low for tok in side_tokens) and any(tok in low for tok in keep_tokens):
            ids.append(i)
    return ids


def clamp_actions(env, actions):
    limits = getattr(env, "motor_pos_cmd_limits", None)
    if limits is not None:
        actions = torch.max(torch.min(actions, limits[:, 1]), limits[:, 0])
    return actions


def run_action(env, actions, steps):
    for _ in range(steps):
        obs, rew, terminated, truncated, info = env.step(actions)
    return obs


def fmt(x):
    return [round(float(v), 4) for v in x.detach().cpu().tolist()]


def main():
    cfg = FlatBoxEnvCfg()
    cfg.scene.num_envs = 1
    cfg.robot_model = "G1_DC"
    cfg.oracle["params"]["nominal_height"] = 0.74
    cfg.debug_contact_modes = True
    cfg.debug_contact_modes_max_prints = 1
    cfg.visualize_markers = False

    # Debug only.
    cfg.rewards = {}
    cfg.terminations = {}

    if hasattr(args_cli, "device"):
        cfg.sim.device = args_cli.device

    env = FlatBoxEnv(cfg, render_mode=None)

    try:
        env.reset()

        joint_names = get_joint_names(env)
        left_ids = resolve_arm_joint_ids(joint_names, "left")
        right_ids = resolve_arm_joint_ids(joint_names, "right")
        arm_ids = left_ids + right_ids

        print("[G1_HAND_ACTION_DEBUG] selected_left_arm_joints:")
        for i in left_ids:
            print(f"  {i}: {joint_names[i]}")
        print("[G1_HAND_ACTION_DEBUG] selected_right_arm_joints:")
        for i in right_ids:
            print(f"  {i}: {joint_names[i]}")

        if not arm_ids:
            print("[G1_HAND_ACTION_DEBUG] all_joint_names:")
            for i, n in enumerate(joint_names):
                print(f"  {i}: {n}")
            raise RuntimeError("No arm joint ids resolved")

        default_action = env.robot.data.default_joint_pos.clone()
        run_action(env, default_action, 10)
        base_left, base_right = palm_pos(env)

        print(
            "[G1_HAND_ACTION_DEBUG] baseline "
            f"left={fmt(base_left[0])} right={fmt(base_right[0])}",
            flush=True,
        )

        tests = [
            ("both_arm_plus", arm_ids, +args_cli.delta),
            ("both_arm_minus", arm_ids, -args_cli.delta),
            ("left_arm_plus", left_ids, +args_cli.delta),
            ("right_arm_plus", right_ids, +args_cli.delta),
        ]

        for name, ids, delta in tests:
            env.reset()
            run_action(env, default_action, 10)
            before_left, before_right = palm_pos(env)

            action = default_action.clone()
            action[:, ids] += delta
            action = clamp_actions(env, action)

            run_action(env, action, args_cli.num_steps)
            after_left, after_right = palm_pos(env)

            left_move = torch.linalg.norm(after_left - before_left, dim=-1)
            right_move = torch.linalg.norm(after_right - before_right, dim=-1)

            print(
                "[G1_HAND_ACTION_DEBUG] "
                f"test={name} delta={delta:+.3f} steps={args_cli.num_steps} "
                f"left_before={fmt(before_left[0])} left_after={fmt(after_left[0])} "
                f"left_move={float(left_move[0].detach().cpu()):.4f} "
                f"right_before={fmt(before_right[0])} right_after={fmt(after_right[0])} "
                f"right_move={float(right_move[0].detach().cpu()):.4f}",
                flush=True,
            )

        print("G1_HAND_ACTION_RESPONSE_DEBUG_OK", flush=True)

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
