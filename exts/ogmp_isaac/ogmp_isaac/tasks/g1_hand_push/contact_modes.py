"""Contact mode library for G1-Hand-ShortPush longbox MVP.

Box convention:
- Box local frame is relative to root/COM.
- Longbox size: x=1.6m, y=0.8m, z=0.5m.
- Half extents: x=0.8, y=0.4, z=0.25.
- Contact z local=0.20 -> world z≈0.45 when box COM z=0.25.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ContactMode:
    name: str
    left_local: tuple[float, float, float]
    right_local: tuple[float, float, float]


Z = 0.20

CONTACT_MODES: tuple[ContactMode, ...] = (
    # Forward push from rear face. Box center x ahead of robot; rear face is local x=-0.8.
    ContactMode("rear_face_center", (-0.80, +0.18, Z), (-0.80, -0.18, Z)),
    ContactMode("rear_face_wide", (-0.80, +0.28, Z), (-0.80, -0.28, Z)),
    ContactMode("rear_face_y_plus_bias", (-0.80, +0.12, Z), (-0.80, +0.34, Z)),
    ContactMode("rear_face_y_minus_bias", (-0.80, -0.34, Z), (-0.80, -0.12, Z)),

    # Same-long-side modes for later yaw / side-push experiments.
    ContactMode("same_long_side_center_y_plus", (-0.25, +0.40, Z), (+0.25, +0.40, Z)),
    ContactMode("same_long_side_center_y_minus", (-0.25, -0.40, Z), (+0.25, -0.40, Z)),
    ContactMode("same_long_side_front_bias_y_plus", (+0.25, +0.40, Z), (+0.70, +0.40, Z)),
    ContactMode("same_long_side_back_bias_y_plus", (-0.70, +0.40, Z), (-0.25, +0.40, Z)),
    ContactMode("same_long_side_front_bias_y_minus", (+0.25, -0.40, Z), (+0.70, -0.40, Z)),
    ContactMode("same_long_side_back_bias_y_minus", (-0.70, -0.40, Z), (-0.25, -0.40, Z)),
)


def get_mode_names() -> list[str]:
    return [m.name for m in CONTACT_MODES]


def get_local_contacts(mode_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = mode_ids.device
    dtype = torch.float32
    left = torch.tensor([m.left_local for m in CONTACT_MODES], device=device, dtype=dtype)
    right = torch.tensor([m.right_local for m in CONTACT_MODES], device=device, dtype=dtype)
    return left[mode_ids], right[mode_ids]


def local_contacts_to_world(
    box_pos_w: torch.Tensor,
    box_yaw_w: torch.Tensor,
    mode_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    left_l, right_l = get_local_contacts(mode_ids)

    cos_yaw = torch.cos(box_yaw_w)
    sin_yaw = torch.sin(box_yaw_w)

    def transform(p_l: torch.Tensor) -> torch.Tensor:
        x = cos_yaw * p_l[:, 0] - sin_yaw * p_l[:, 1]
        y = sin_yaw * p_l[:, 0] + cos_yaw * p_l[:, 1]
        z = p_l[:, 2]
        return box_pos_w + torch.stack((x, y, z), dim=-1)

    return transform(left_l), transform(right_l)
