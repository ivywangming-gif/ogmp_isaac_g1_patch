"""Contact mode library for G1 two-hand short-horizon long-box pushing.

Box:
- length = 1.6 m, half-length hx = 0.8
- width  = 0.8 m, half-width  hy = 0.4
- height = 0.5 m

Coordinate convention:
- Box local x: long axis
- Box local y: short axis / long-side normal
- Box local z: vertical
- Contact points are in box local frame.

First version:
- Use only P1 same-long-side modes.
- Do not use opposite-side clamping modes yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ContactMode:
    name: str
    left_xyz: tuple[float, float, float]
    right_xyz: tuple[float, float, float]
    mode_type: str
    yaw_authority: str


# P1 modes only.
# z = 0.35 is chosen for H=0.5 box: above center, below top surface.
P1_CONTACT_MODES: tuple[ContactMode, ...] = (
    ContactMode(
        name="same_long_side_center_y_plus",
        left_xyz=(-0.25, +0.40, +0.35),
        right_xyz=(+0.25, +0.40, +0.35),
        mode_type="same_side_translation",
        yaw_authority="low",
    ),
    ContactMode(
        name="same_long_side_center_y_minus",
        left_xyz=(-0.25, -0.40, +0.35),
        right_xyz=(+0.25, -0.40, +0.35),
        mode_type="same_side_translation",
        yaw_authority="low",
    ),
    ContactMode(
        name="same_long_side_front_bias_y_plus",
        left_xyz=(+0.25, +0.40, +0.35),
        right_xyz=(+0.70, +0.40, +0.35),
        mode_type="same_side_yaw",
        yaw_authority="high",
    ),
    ContactMode(
        name="same_long_side_back_bias_y_plus",
        left_xyz=(-0.70, +0.40, +0.35),
        right_xyz=(-0.25, +0.40, +0.35),
        mode_type="same_side_yaw",
        yaw_authority="high",
    ),
    ContactMode(
        name="same_long_side_front_bias_y_minus",
        left_xyz=(+0.25, -0.40, +0.35),
        right_xyz=(+0.70, -0.40, +0.35),
        mode_type="same_side_yaw",
        yaw_authority="high",
    ),
    ContactMode(
        name="same_long_side_back_bias_y_minus",
        left_xyz=(-0.70, -0.40, +0.35),
        right_xyz=(-0.25, -0.40, +0.35),
        mode_type="same_side_yaw",
        yaw_authority="high",
    ),
)


def get_mode_names() -> list[str]:
    return [m.name for m in P1_CONTACT_MODES]


def get_mode_local_points(device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
    """Return left/right local contact points.

    Returns:
        left_points:  tensor shape (num_modes, 3)
        right_points: tensor shape (num_modes, 3)
    """
    left = torch.tensor([m.left_xyz for m in P1_CONTACT_MODES], device=device, dtype=dtype)
    right = torch.tensor([m.right_xyz for m in P1_CONTACT_MODES], device=device, dtype=dtype)
    return left, right


def yaw_to_rot2d(yaw: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    row0 = torch.stack([c, -s], dim=-1)
    row1 = torch.stack([s, c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def local_contacts_to_world(
    box_pos_w: torch.Tensor,
    box_yaw_w: torch.Tensor,
    mode_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert local contact mode points to world-frame targets.

    Args:
        box_pos_w: shape (N, 3)
        box_yaw_w: shape (N,)
        mode_ids: shape (N,), values in [0, num_modes)

    Returns:
        left_w, right_w: each shape (N, 3)
    """
    device = box_pos_w.device
    dtype = box_pos_w.dtype

    left_local_all, right_local_all = get_mode_local_points(device=device, dtype=dtype)
    left_local = left_local_all[mode_ids]
    right_local = right_local_all[mode_ids]

    R = yaw_to_rot2d(box_yaw_w)

    left_xy = box_pos_w[:, :2] + torch.einsum("nij,nj->ni", R, left_local[:, :2])
    right_xy = box_pos_w[:, :2] + torch.einsum("nij,nj->ni", R, right_local[:, :2])

    left_z = box_pos_w[:, 2] + left_local[:, 2]
    right_z = box_pos_w[:, 2] + right_local[:, 2]

    left_w = torch.cat([left_xy, left_z.unsqueeze(-1)], dim=-1)
    right_w = torch.cat([right_xy, right_z.unsqueeze(-1)], dim=-1)
    return left_w, right_w


def demo() -> None:
    print("=== P1 contact modes ===")
    for i, m in enumerate(P1_CONTACT_MODES):
        print(f"{i}: {m.name}")
        print(f"   L={m.left_xyz} R={m.right_xyz} type={m.mode_type} yaw={m.yaw_authority}")

    box_pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]], dtype=torch.float32)
    box_yaw = torch.tensor([0.0, torch.pi / 2], dtype=torch.float32)
    mode_ids = torch.tensor([0, 2], dtype=torch.long)

    left, right = local_contacts_to_world(box_pos, box_yaw, mode_ids)

    print("=== demo world contacts ===")
    print("box_pos:")
    print(box_pos)
    print("box_yaw:")
    print(box_yaw)
    print("mode_ids:")
    print(mode_ids)
    print("left_w:")
    print(left)
    print("right_w:")
    print(right)
    print("separation:")
    print(torch.linalg.norm(left - right, dim=-1))


if __name__ == "__main__":
    demo()
