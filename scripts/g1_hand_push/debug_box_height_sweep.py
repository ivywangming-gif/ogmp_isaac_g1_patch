"""Sweep longbox height for default-action rear-face push feasibility.

Important: do not name the argument --height because AppLauncher treats
--height as SimulationApp viewport height.
"""

import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--box_h", type=float, required=True)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--print_every", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from ogmp_isaac.tasks.g1_hand_push.short_hand_push_env import FlatBoxEnv, FlatBoxEnvCfg


def fmt(x):
    return [round(float(v), 4) for v in x.detach().cpu().flatten().tolist()]


def palm_pos(env):
    body = getattr(env.robot.data, "body_link_pos_w", None)
    if body is None:
        body = env.robot.data.body_pos_w
    return body[:, env.left_hand_body_idx, :3], body[:, env.right_hand_body_idx, :3]


def main():
    h = float(args_cli.box_h)

    cfg = FlatBoxEnvCfg()
    cfg.scene.num_envs = 1
    cfg.robot_model = "G1_DC"
    cfg.observation_space = 113
    cfg.num_observations = 113

    cfg.use_longbox_asset = True
    cfg.longbox_dims = [1.6, 0.8, h]
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

    # Feasibility diagnostic only.
    cfg.rewards = {}
    cfg.terminations = {}

    if hasattr(args_cli, "device") and args_cli.device is not None:
        cfg.sim.device = args_cli.device

    env = FlatBoxEnv(cfg, render_mode=None)

    try:
        env.reset()

        box0 = env.box.data.root_pos_w[0, :3].clone()
        robot0 = env.robot.data.root_pos_w[0, :3].clone()
        l0, r0 = palm_pos(env)
        le0 = torch.linalg.norm(l0[0] - env.left_contact_target_w[0])
        re0 = torch.linalg.norm(r0[0] - env.right_contact_target_w[0])

        print(
            "[HEIGHT_SWEEP_START] "
            f"h={h:.2f} "
            f"box0={fmt(box0)} robot0={fmt(robot0)} "
            f"left_target={fmt(env.left_contact_target_w[0])} "
            f"right_target={fmt(env.right_contact_target_w[0])} "
            f"left_err0={float(le0.detach().cpu()):.4f} "
            f"right_err0={float(re0.detach().cpu()):.4f}",
            flush=True,
        )

        action = env.robot.data.default_joint_pos.clone()

        for step in range(args_cli.steps):
            env.step(action)

            if step % args_cli.print_every == 0 or step == args_cli.steps - 1:
                box = env.box.data.root_pos_w[0, :3].clone()
                robot = env.robot.data.root_pos_w[0, :3].clone()
                l, r = palm_pos(env)
                le = torch.linalg.norm(l[0] - env.left_contact_target_w[0])
                re = torch.linalg.norm(r[0] - env.right_contact_target_w[0])
                disp = box - box0

                print(
                    "[HEIGHT_SWEEP_STEP] "
                    f"h={h:.2f} step={step:03d} "
                    f"forward_x={float(disp[0].detach().cpu()):+.4f} "
                    f"lateral_y={float(disp[1].detach().cpu()):+.4f} "
                    f"robot_z={float(robot[2].detach().cpu()):+.4f} "
                    f"left_err={float(le.detach().cpu()):.4f} "
                    f"right_err={float(re.detach().cpu()):.4f}",
                    flush=True,
                )

        box1 = env.box.data.root_pos_w[0, :3].clone()
        robot1 = env.robot.data.root_pos_w[0, :3].clone()
        l1, r1 = palm_pos(env)
        le1 = torch.linalg.norm(l1[0] - env.left_contact_target_w[0])
        re1 = torch.linalg.norm(r1[0] - env.right_contact_target_w[0])
        disp = box1 - box0

        print(
            "[HEIGHT_SWEEP_RESULT] "
            f"h={h:.2f} "
            f"forward_x={float(disp[0].detach().cpu()):+.4f} "
            f"lateral_y={float(disp[1].detach().cpu()):+.4f} "
            f"robot_z={float(robot1[2].detach().cpu()):+.4f} "
            f"left_err={float(le1.detach().cpu()):.4f} "
            f"right_err={float(re1.detach().cpu()):.4f}",
            flush=True,
        )
        print("HEIGHT_SWEEP_OK", flush=True)

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
