import os
import torch
import math

import isaaclab.sim as sim_utils
from ogmp_isaac.tasks.g1_hand_push.contact_modes import (
    get_mode_names,
    local_contacts_to_world,
)
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from ogmp_isaac.tasks.base_env import BaseEnv, BaseEnvCfg

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "objects")


@configclass
class FlatBoxEnvCfg(BaseEnvCfg):
    # Isaac Lab 5.x requires explicit Gym spaces.
    # For G1 PushBox:
    # obs = base_z(1) + base_ori(4) + joint_pos(37) + base_lin_vel(3)
    #     + base_ang_vel(3) + joint_vel(37) + box_dist(2)
    #     + target_dist(2) + sinusoid_phase(2) = 91
    # action = 37 G1 joints
    observation_space = 109
    action_space = 37
    state_space = 0
    num_observations = 109
    num_actions = 37
    marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.05, 0.05, 0.05),
            ),
            "base": sim_utils.CuboidCfg(
                size=(0.125, 0.19, 0.248),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.0)),
            ),
        },
    )

    rewards = {
        "base_pos": {"weight": 0.4, "exp_coeff": 3.0},
        "base_ori": {"weight": 0.23, "exp_coeff": 5.0},
        "base_lin_vel": {"weight": 0.3, "exp_coeff": 2.0},
        "box_closeness": {"weight": 0.5, "threshold": 0.5},
        "preference": {"weight": -1.0,},
        "torque_exp_norm": {"weight": 0.15, "exp_coeff": 0.05},
        "action": {"weight": 0.15, "exp_coeff": 1.0},
    }
    observations = [
        "base_z",
        "base_ori",
        "joint_pos",
        "base_lin_vel",
        "base_ang_vel",
        "joint_vel",
        "box_dist",
        "target_dist",
        "sinusoid_phase",
        "hand_contact_target_delta",
        "hand_contact_target_error",
        "contact_mode_onehot",
        "box_goal_yaw",
    ]
    terminations = {
        "base_pos_x": 0.4,
        "base_pos_y": 0.4,
        "base_pos_z": 0.1,
        "box_pos_x": 0.2,
        "box_pos_y": 0.2,
    }
    oracle = {
        "name": "PushBoxOracle",
        "params": {
            "speed": 0.8,
            "reach_thresh": 0.4,
            "detach_thresh": 0.4,
        },
    }
    omni_direction_lim = [0.0, 360.0]
    box_height = 0.5
    height_to_file_name = {
        "0.5": "box_0p5m.usd",
        "1.0": "box_1m.usd",
        "1.5": "box_1p5m.usd",
    }
    box_start = 1.0
    target = 3.0

    # Env-level debug only: sample SE(2) goal and P1 two-hand contact mode.
    debug_contact_modes = True
    debug_contact_modes_max_prints = 5
    debug_hand_target_errors = True
    select_reachable_contact_mode = True
    allow_contact_target_swap = True
    box_yaw_lim = [-3.14159265, 3.14159265]
    goal_yaw_lim = [-3.14159265, 3.14159265]


class FlatBoxEnv(BaseEnv):
    cfg: FlatBoxEnvCfg

    def __init__(self, cfg: FlatBoxEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.target_pos = torch.zeros((self.num_envs, 2), device=self.sim.device)
        self.heading_angles = torch.zeros((self.num_envs,), device=self.sim.device)
        self.target_yaw = torch.zeros((self.num_envs,), device=self.sim.device)
        self.box_yaw = torch.zeros((self.num_envs,), device=self.sim.device)

        self.contact_mode_ids = torch.zeros((self.num_envs,), dtype=torch.long, device=self.sim.device)
        self.left_contact_target_w = torch.zeros((self.num_envs, 3), device=self.sim.device)
        self.right_contact_target_w = torch.zeros((self.num_envs, 3), device=self.sim.device)
        self.contact_target_swapped = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.sim.device)
        self._contact_mode_names = get_mode_names()
        self._contact_debug_print_count = 0
        self.left_hand_body_idx, self.left_hand_body_name = self._resolve_debug_hand_body("left")
        self.right_hand_body_idx, self.right_hand_body_name = self._resolve_debug_hand_body("right")
        if self.cfg.debug_contact_modes:
            print(
                f"[G1_HAND_BODY_DEBUG] left={self.left_hand_body_name}[{self.left_hand_body_idx}] "
                f"right={self.right_hand_body_name}[{self.right_hand_body_idx}]",
                flush=True,
            )
        self.start_angle = torch.deg2rad(torch.tensor(self.cfg.omni_direction_lim[0], device=self.sim.device))
        self.end_angle = torch.deg2rad(torch.tensor(self.cfg.omni_direction_lim[1], device=self.sim.device))
        if self.cfg.visualize_markers:
            self.marker = VisualizationMarkers(self.cfg.marker_cfg)
        self.oracle.box_com = self.box.data.default_root_state[0, 2]
        self.cfg.box_start += self.cfg.box_height - 0.5
        self.cfg.target += self.cfg.box_height - 0.5
        # reach thresh is the half-diagonal of the box + 0.05
        self.oracle.reach_thresh = torch.tensor(0.05 + math.sqrt(2) * self.cfg.box_height / 2, device=self.sim.device)

    def _setup_scene(self):
        box_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/box",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(ASSETS_DIR, self.cfg.height_to_file_name[str(self.cfg.box_height)]),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(2.0, 0.0, self.cfg.box_height / 2),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
            ),
        )
        self.box = RigidObject(box_cfg)
        self.scene.rigid_objects["box"] = self.box
        super()._setup_scene()

    def _get_rewards(self) -> torch.Tensor:
        # Current yaw is okay, no penalty
        yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)[2]
        self.oracle.reference.base_ori[self.env_indices, self.oracle.phase.squeeze()] = quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        )
        return super()._get_rewards()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or env_ids.numel() == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.box.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        num_reset_envs = env_ids.numel()

        # Randomize heading
        self.heading_angles[env_ids] = (
            torch.rand((num_reset_envs,), device=self.sim.device) * (self.end_angle - self.start_angle)
            + self.start_angle
        )
        cos_heading = torch.cos(self.heading_angles[env_ids])
        sin_heading = torch.sin(self.heading_angles[env_ids])

        self.box_yaw[env_ids] = (
            torch.rand((num_reset_envs,), device=self.sim.device)
            * (self.cfg.box_yaw_lim[1] - self.cfg.box_yaw_lim[0])
            + self.cfg.box_yaw_lim[0]
        )

        # Randomize box position
        box_start_state = self.box.data.default_root_state[env_ids].clone()
        box_start_state[:, 0] = cos_heading * self.cfg.box_start
        box_start_state[:, 1] = sin_heading * self.cfg.box_start
        zeros = torch.zeros_like(self.box_yaw[env_ids])
        box_start_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, self.box_yaw[env_ids])
        box_start_state[:, :3] += self.scene.env_origins[env_ids]
        self.box.write_root_pose_to_sim(box_start_state[:, :7], env_ids)
        self.box.write_root_velocity_to_sim(box_start_state[:, 7:], env_ids)

        # Randomize target position
        self.target_pos[env_ids, 0] = cos_heading * self.cfg.target
        self.target_pos[env_ids, 1] = sin_heading * self.cfg.target
        self.target_pos[env_ids, :2] += self.scene.env_origins[env_ids, :2]

        self.target_yaw[env_ids] = (
            torch.rand((num_reset_envs,), device=self.sim.device)
            * (self.cfg.goal_yaw_lim[1] - self.cfg.goal_yaw_lim[0])
            + self.cfg.goal_yaw_lim[0]
        )

        if self.cfg.select_reachable_contact_mode:
            body_pos_w = getattr(self.robot.data, "body_link_pos_w", None)
            if body_pos_w is None:
                body_pos_w = self.robot.data.body_pos_w

            left_hand_w = body_pos_w[env_ids, self.left_hand_body_idx, :3]
            right_hand_w = body_pos_w[env_ids, self.right_hand_body_idx, :3]

            all_costs = []
            all_left_targets = []
            all_right_targets = []
            all_swaps = []

            for mode_id in range(len(self._contact_mode_names)):
                mode_ids = torch.full(
                    (num_reset_envs,),
                    mode_id,
                    device=self.sim.device,
                    dtype=torch.long,
                )
                cand_left_w, cand_right_w = local_contacts_to_world(
                    box_start_state[:, :3],
                    self.box_yaw[env_ids],
                    mode_ids,
                )

                direct_cost = (
                    torch.linalg.norm(cand_left_w - left_hand_w, dim=-1)
                    + torch.linalg.norm(cand_right_w - right_hand_w, dim=-1)
                )
                swap_cost = (
                    torch.linalg.norm(cand_right_w - left_hand_w, dim=-1)
                    + torch.linalg.norm(cand_left_w - right_hand_w, dim=-1)
                )

                use_swap = self.cfg.allow_contact_target_swap & (swap_cost < direct_cost)
                best_cost = torch.where(use_swap, swap_cost, direct_cost)
                best_left = torch.where(use_swap.unsqueeze(-1), cand_right_w, cand_left_w)
                best_right = torch.where(use_swap.unsqueeze(-1), cand_left_w, cand_right_w)

                all_costs.append(best_cost)
                all_left_targets.append(best_left)
                all_right_targets.append(best_right)
                all_swaps.append(use_swap)

            costs = torch.stack(all_costs, dim=1)
            best_mode = torch.argmin(costs, dim=1)

            left_stack = torch.stack(all_left_targets, dim=1)
            right_stack = torch.stack(all_right_targets, dim=1)
            swap_stack = torch.stack(all_swaps, dim=1)

            gather_xyz = best_mode.view(-1, 1, 1).expand(-1, 1, 3)
            gather_bool = best_mode.view(-1, 1)

            self.contact_mode_ids[env_ids] = best_mode
            self.left_contact_target_w[env_ids] = torch.gather(left_stack, 1, gather_xyz).squeeze(1)
            self.right_contact_target_w[env_ids] = torch.gather(right_stack, 1, gather_xyz).squeeze(1)
            self.contact_target_swapped[env_ids] = torch.gather(swap_stack, 1, gather_bool).squeeze(1)
        else:
            self.contact_mode_ids[env_ids] = torch.randint(
                low=0,
                high=len(self._contact_mode_names),
                size=(num_reset_envs,),
                device=self.sim.device,
                dtype=torch.long,
            )

            left_w, right_w = local_contacts_to_world(
                box_start_state[:, :3],
                self.box_yaw[env_ids],
                self.contact_mode_ids[env_ids],
            )
            self.left_contact_target_w[env_ids] = left_w
            self.right_contact_target_w[env_ids] = right_w
            self.contact_target_swapped[env_ids] = False

        self._print_contact_debug(env_ids, box_start_state[:, :3])

    def _resolve_debug_hand_body(self, side: str) -> tuple[int, str]:
        names = getattr(self.robot, "body_names", None)
        if names is None:
            names = self.robot.data.body_names
        names = list(names)

        side_tokens = ["left", "l_"] if side == "left" else ["right", "r_"]
        prefs = ["palm", "hand", "wrist_yaw", "wrist_roll", "wrist_pitch", "wrist"]

        for pref in prefs:
            for i, name in enumerate(names):
                low = name.lower()
                if any(tok in low for tok in side_tokens) and pref in low:
                    return i, name

        raise RuntimeError(
            f"Could not resolve {side} hand/wrist body. Available body names: {names}"
        )

    def _fmt_debug_vec(self, x: torch.Tensor) -> list[float]:
        return [round(float(v), 4) for v in x.detach().cpu().tolist()]

    def _print_contact_debug(self, env_ids: torch.Tensor, box_pos_w_for_debug: torch.Tensor):
        if not self.cfg.debug_contact_modes:
            return
        if self._contact_debug_print_count >= self.cfg.debug_contact_modes_max_prints:
            return

        local_idx = 0
        env_id = int(env_ids[local_idx].item())
        mode_id = int(self.contact_mode_ids[env_id].item())

        body_pos_w = getattr(self.robot.data, "body_link_pos_w", None)
        if body_pos_w is None:
            body_pos_w = self.robot.data.body_pos_w

        left_hand_w = body_pos_w[env_id, self.left_hand_body_idx, :3]
        right_hand_w = body_pos_w[env_id, self.right_hand_body_idx, :3]
        left_err = torch.linalg.norm(left_hand_w - self.left_contact_target_w[env_id])
        right_err = torch.linalg.norm(right_hand_w - self.right_contact_target_w[env_id])

        print(
            "[G1_HAND_CONTACT_DEBUG] "
            f"reset={self._contact_debug_print_count} "
            f"env={env_id} "
            f"box_pos={self._fmt_debug_vec(box_pos_w_for_debug[local_idx, :3])} "
            f"box_yaw={float(self.box_yaw[env_id].detach().cpu()):+.3f} "
            f"goal_xy={self._fmt_debug_vec(self.target_pos[env_id, :2])} "
            f"goal_yaw={float(self.target_yaw[env_id].detach().cpu()):+.3f} "
            f"mode_id={mode_id} "
            f"mode={self._contact_mode_names[mode_id]} "
            f"swapped={bool(self.contact_target_swapped[env_id].detach().cpu())} "
            f"left_w={self._fmt_debug_vec(self.left_contact_target_w[env_id])} "
            f"right_w={self._fmt_debug_vec(self.right_contact_target_w[env_id])} "
            f"left_body={self.left_hand_body_name} "
            f"left_pos={self._fmt_debug_vec(left_hand_w)} "
            f"left_err={float(left_err.detach().cpu()):.3f} "
            f"right_body={self.right_hand_body_name} "
            f"right_pos={self._fmt_debug_vec(right_hand_w)} "
            f"right_err={float(right_err.detach().cpu()):.3f}",
            flush=True,
        )
        self._contact_debug_print_count += 1

    def _get_feedback(self):
        feedback = {
            "robot_pos": self.robot.data.root_pos_w[:, :3],
            "box_pos": self.box.data.root_pos_w[:, :3],
            "box_yaw": self.box_yaw,
            "target_pos": self.target_pos,
            "target_yaw": self.target_yaw,
            "heading": self.heading_angles,
            "contact_mode_ids": self.contact_mode_ids,
            "left_contact_target_w": self.left_contact_target_w,
            "right_contact_target_w": self.right_contact_target_w,
            "contact_target_swapped": self.contact_target_swapped,
        }
        return feedback

    def render_marker_visualization(self):

        # robot base pose traj
        ref_base_pos_traj = self.oracle.reference.base_pos.view(-1, 3).clone()
        ref_base_ori_traj = self.oracle.reference.base_ori.view(-1, 4).clone()

        # box pose
        ref_box_pos_traj = self.oracle.reference.box_pos.view(-1, 3).clone()
        ref_box_ori_traj = torch.zeros((self.num_envs * self.oracle.prediction_horizon, 4), device=self.sim.device)
        ref_box_ori_traj[:, 3] = 1.0

        # get current robot base pose ref
        ref_base_pos = torch.gather(
            self.oracle.reference.base_pos, 1, self.oracle.phase.unsqueeze(-1).expand(-1, -1, 3)
        ).squeeze(1)
        ref_base_ori = torch.gather(
            self.oracle.reference.base_ori, 1, self.oracle.phase.unsqueeze(-1).expand(-1, -1, 4)
        ).squeeze(1)
        # get current box pose ref
        ref_box_pos = torch.gather(
            self.oracle.reference.box_pos, 1, self.oracle.phase.unsqueeze(-1).expand(-1, -1, 3)
        ).squeeze(1)
        ref_box_ori = torch.zeros((self.num_envs, 4), device=self.sim.device)
        ref_box_ori[:, 3] = 1.0

        # stack the current base pose ref
        marker_pos = torch.cat((ref_base_pos_traj, ref_base_pos, ref_box_pos_traj, ref_box_pos), dim=0)
        marker_ori = torch.cat((ref_base_ori_traj, ref_base_ori, ref_box_ori_traj, ref_box_ori), dim=0)

        marker_indices = torch.arange(2 * (self.oracle.prediction_horizon + 1), device=self.sim.device).repeat(
            self.num_envs
        )
        marker_indices *= 0  # frames
        marker_indices[
            self.oracle.prediction_horizon * self.num_envs : self.oracle.prediction_horizon * self.num_envs
            + self.num_envs
        ] = 1  # base
        marker_indices[2 * self.oracle.prediction_horizon * self.num_envs + self.num_envs :] = 2  # box

        self.marker.visualize(marker_pos, marker_ori, marker_indices=marker_indices)
