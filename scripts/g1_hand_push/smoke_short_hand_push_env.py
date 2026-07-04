"""Smoke test for env-level G1 hand short-push contact target debug."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test G1-Hand-ShortPush contact target debug.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_resets", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from ogmp_isaac.tasks.g1_hand_push.short_hand_push_env import FlatBoxEnv, FlatBoxEnvCfg


def main():
    cfg = FlatBoxEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.robot_model = "G1_DC"
    cfg.oracle["params"]["nominal_height"] = 0.74
    cfg.debug_contact_modes = True
    cfg.debug_contact_modes_max_prints = args_cli.num_resets
    cfg.visualize_markers = False

    # Smoke only: bypass reward/termination registry while testing reset geometry.
    cfg.rewards = {}
    cfg.terminations = {}

    if hasattr(args_cli, "device"):
        cfg.sim.device = args_cli.device

    env = FlatBoxEnv(cfg, render_mode=None)

    try:
        for _ in range(args_cli.num_resets):
            env.reset()

        print("G1_HAND_SHORT_PUSH_ENV_SMOKE_OK", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
