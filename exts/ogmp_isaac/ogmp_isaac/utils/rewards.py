import torch


def rew_base_pos(env):
    target_pos = torch.gather(
        env.oracle.reference.base_pos, 1, env.oracle.phase.unsqueeze(-1).expand(-1, -1, 3)
    ).squeeze(1)
    error_pos = torch.linalg.vector_norm(env.robot.data.root_pos_w - target_pos, dim=-1)
    error_pos = env.cfg.rewards["base_pos"]["weight"] * torch.exp(-env.cfg.rewards["base_pos"]["exp_scale"] * error_pos)
    return error_pos


def rew_base_ori(env):
    target_ori = torch.gather(
        env.oracle.reference.base_ori, 1, env.oracle.phase.unsqueeze(-1).expand(-1, -1, 4)
    ).squeeze(1)
    error_ori = 1 - torch.einsum("ij,ij->i", env.robot.data.root_quat_w, target_ori) ** 2
    error_ori = env.cfg.rewards["base_ori"]["weight"] * torch.exp(-env.cfg.rewards["base_ori"]["exp_scale"] * error_ori)
    return error_ori


def rew_base_lin_vel(env):
    target_lin_vel = torch.gather(
        env.oracle.reference.base_lin_vel, 1, env.oracle.phase.unsqueeze(-1).expand(-1, -1, 3)
    ).squeeze(1)
    error_lin_vel = torch.linalg.vector_norm(env.robot.data.root_lin_vel_w - target_lin_vel, dim=-1)
    error_lin_vel = env.cfg.rewards["base_lin_vel"]["weight"] * torch.exp(
        -env.cfg.rewards["base_lin_vel"]["exp_scale"] * error_lin_vel
    )
    return error_lin_vel


def rew_base_ang_vel(env):
    target_ang_vel = torch.gather(
        env.oracle.reference.base_ang_vel, 1, env.oracle.phase.unsqueeze(-1).expand(-1, -1, 3)
    ).squeeze(1)
    error_ang_vel = torch.linalg.vector_norm(env.robot.data.root_ang_vel_w - target_ang_vel, dim=-1)
    error_ang_vel = env.cfg.rewards["base_ang_vel"]["weight"] * torch.exp(
        -env.cfg.rewards["base_ang_vel"]["exp_scale"] * error_ang_vel
    )
    return error_ang_vel


def rew_ball_closeness(env):
    error_ball_pos = (
        env.cfg.rewards["ball_closeness"]["weight"]
        * (env.oracle.modes == 1).float()
        * (
            torch.linalg.vector_norm(env.ball.data.root_pos_w - env.robot.data.root_pos_w, dim=-1)
            <= env.nominal_height
        ).float()
    )
    return error_ball_pos


def rew_box_closeness(env):
    error_box_pos = (
        env.cfg.rewards["box_closeness"]["weight"]
        * (env.oracle.modes == 1).float()
        * (
            torch.linalg.vector_norm(env.box.data.root_pos_w - env.robot.data.root_pos_w, dim=-1)
            <= env.cfg.box_height
        ).float()
    )
    return error_box_pos


def rew_preference(env):
    preference = (env.oracle.modes - env.max_modes < 0).float()
    return env.cfg.rewards["preference"]["weight"] * preference


def rew_ball_pos(env):
    target_pos = torch.gather(
        env.oracle.reference.ball_pos, 1, env.oracle.phase.unsqueeze(-1).expand(-1, -1, 3)
    ).squeeze(1)
    ball_pos_error = torch.linalg.vector_norm(env.ball.data.root_pos_w - target_pos, dim=-1)
    ball_pos_error = (
        env.cfg.rewards["ball_pos"]["weight"]
        * (env.oracle.modes >= 2).float()
        * torch.exp(-env.cfg.rewards["ball_pos"]["exp_scale"] * ball_pos_error)
    )
    return ball_pos_error


def rew_box_pos(env):
    target_pos = torch.gather(
        env.oracle.reference.box_pos, 1, env.oracle.phase.unsqueeze(-1).expand(-1, -1, 3)
    ).squeeze(1)
    box_pos_error = torch.linalg.vector_norm(env.box.data.root_pos_w - target_pos, dim=-1)
    box_pos_error = (
        env.cfg.rewards["box_pos"]["weight"]
        * (env.oracle.modes >= 2).float()
        * torch.exp(-env.cfg.rewards["box_pos"]["exp_scale"] * box_pos_error)
    )
    return box_pos_error


def rew_ball_vel(env):
    target_vel = torch.gather(
        env.oracle.reference.ball_vel, 1, env.oracle.phase.unsqueeze(-1).expand(-1, -1, 3)
    ).squeeze(1)
    ball_vel_error = torch.linalg.vector_norm(env.ball.data.root_lin_vel_w - target_vel, dim=-1)
    ball_vel_error = (
        env.cfg.rewards["ball_vel"]["weight"]
        * (env.oracle.modes >= 2).float()
        * torch.exp(-env.cfg.rewards["ball_vel"]["exp_scale"] * ball_vel_error)
    )
    return ball_vel_error


def rew_box_vel(env):
    target_vel = torch.gather(
        env.oracle.reference.box_vel, 1, env.oracle.phase.unsqueeze(-1).expand(-1, -1, 3)
    ).squeeze(1)
    box_vel_error = torch.linalg.vector_norm(env.box.data.root_lin_vel_w - target_vel, dim=-1)
    box_vel_error = (
        env.cfg.rewards["box_vel"]["weight"]
        * (env.oracle.modes >= 2).float()
        * torch.exp(-env.cfg.rewards["box_vel"]["exp_scale"] * box_vel_error)
    )
    return box_vel_error


def rew_penalize_ball_rest(env):
    ball_vel_norm = torch.linalg.vector_norm(env.ball.data.root_lin_vel_w, dim=-1)
    # penalize if ball is at rest,  only in manipulation mode
    ball_rest = (ball_vel_norm <= env.cfg.rewards["penalize_ball_rest"]["threshold"]).float() * (
        env.oracle.modes == 1
    ).float()
    return env.cfg.rewards["penalize_ball_rest"]["weight"] * ball_rest


def rew_penalize_box_rest(env):
    box_vel_norm = torch.linalg.vector_norm(env.box.data.root_lin_vel_w, dim=-1)
    box_rest = (box_vel_norm <= env.cfg.rewards["penalize_box_rest"]["threshold"]).float() * (
        env.oracle.modes == 1
    ).float()
    return env.cfg.rewards["penalize_box_rest"]["weight"] * box_rest


def rew_torque_exp_norm(env):
    torque = torch.linalg.vector_norm(env.robot.data.applied_torque, dim=-1) / env.max_joint_torque
    error_torque = env.cfg.rewards["torque_exp_norm"]["weight"] * torch.exp(
        -env.cfg.rewards["torque_exp_norm"]["exp_scale"] * torque
    )
    return error_torque


def rew_torque_rate_norm(env):
    torque_rate = torch.linalg.vector_norm(env.robot.data.applied_torque - env.previous_torques, dim=-1) / env.max_joint_torque
    return env.cfg.rewards["torque_rate_norm"]["weight"] * torch.exp(
        -env.cfg.rewards["torque_rate_norm"]["exp_scale"] * torque_rate
    )


def rew_action_norm(env):
    error_action = torch.linalg.vector_norm(env.actions - env.previous_actions, dim=-1) / env.max_joint_action
    error_action = env.cfg.rewards["action_norm"]["weight"] * torch.exp(
        -env.cfg.rewards["action_norm"]["exp_scale"] * error_action
    )
    return error_action


def rew_joint_vel_norm(env):
    jvel_mag = torch.linalg.vector_norm(env.robot.data.joint_vel, dim=-1) / env.max_total_joint_vel
    return env.cfg.rewards["joint_vel_norm"]["weight"] * torch.exp(
        -env.cfg.rewards["joint_vel_norm"]["exp_scale"] * jvel_mag
    )


def rew_default_joint_pos(env):
    return env.cfg.rewards["default_joint_pos"]["weight"] * torch.sum(
        torch.abs(env.robot.data.joint_pos - env.robot.data.default_joint_pos), dim=-1
    )


def rew_feet_air_time(env):
    reward = torch.zeros((env.num_envs), device=env.sim.device)
    for i, sensor in enumerate(env.feet_contact_sensors):
        first_contact = sensor.compute_first_contact(env.step_dt)
        last_air_time = sensor.data.last_air_time
        reward += (
            env.cfg.rewards["feet_air_time"]["weight"]
            * ((last_air_time - env.cfg.rewards["feet_air_time"]["threshold"]) * first_contact).squeeze()
        )
    reward /= i + 1
    return reward


def _g1_hand_push_palm_pos(env):
    body_pos_w = getattr(env.robot.data, "body_link_pos_w", None)
    if body_pos_w is None:
        body_pos_w = env.robot.data.body_pos_w
    left = body_pos_w[:, env.left_hand_body_idx, :3]
    right = body_pos_w[:, env.right_hand_body_idx, :3]
    return left, right

def rew_hand_target_tracking(env):
    left, right = _g1_hand_push_palm_pos(env)
    left_err = torch.linalg.vector_norm(left - env.left_contact_target_w, dim=-1)
    right_err = torch.linalg.vector_norm(right - env.right_contact_target_w, dim=-1)
    err = left_err + right_err
    return env.cfg.rewards["hand_target_tracking"]["weight"] * torch.exp(
        -env.cfg.rewards["hand_target_tracking"]["exp_scale"] * err
    )

def rew_hand_target_close(env):
    left, right = _g1_hand_push_palm_pos(env)
    left_err = torch.linalg.vector_norm(left - env.left_contact_target_w, dim=-1)
    right_err = torch.linalg.vector_norm(right - env.right_contact_target_w, dim=-1)
    close = ((left_err < env.cfg.rewards["hand_target_close"]["threshold"]) &
             (right_err < env.cfg.rewards["hand_target_close"]["threshold"])).float()
    return env.cfg.rewards["hand_target_close"]["weight"] * close

def rew_box_forward_progress(env):
    start_x = env.scene.env_origins[:, 0] + env.cfg.box_start
    progress = env.box.data.root_pos_w[:, 0] - start_x
    progress = torch.clamp(progress, min=-0.05, max=env.cfg.rewards["box_forward_progress"]["max_progress"])
    return env.cfg.rewards["box_forward_progress"]["weight"] * progress

def rew_box_forward_vel(env):
    vx = torch.clamp(env.box.data.root_lin_vel_w[:, 0], min=0.0, max=env.cfg.rewards["box_forward_vel"]["max_vel"])
    return env.cfg.rewards["box_forward_vel"]["weight"] * vx


# ---- G1 hand-push upright guarded MVP rewards ----

def rew_base_height_floor(env):
    z = env.robot.data.root_pos_w[:, 2]
    min_h = env.cfg.rewards["base_height_floor"]["min_height"]
    target_h = env.cfg.rewards["base_height_floor"]["target_height"]
    score = torch.clamp((z - min_h) / max(target_h - min_h, 1e-6), min=0.0, max=1.0)
    return env.cfg.rewards["base_height_floor"]["weight"] * score


def rew_default_action_tracking(env):
    err = torch.linalg.vector_norm(env.actions - env.robot.data.default_joint_pos, dim=-1) / env.max_joint_action
    return env.cfg.rewards["default_action_tracking"]["weight"] * torch.exp(
        -env.cfg.rewards["default_action_tracking"]["exp_scale"] * err
    )
