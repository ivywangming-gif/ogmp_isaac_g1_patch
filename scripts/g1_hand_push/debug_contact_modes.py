from pathlib import Path
import importlib.util
import sys
import torch

CONTACT_MODES_PATH = (
    Path(__file__).resolve().parents[2]
    / "exts/ogmp_isaac/ogmp_isaac/tasks/g1_hand_push/contact_modes.py"
)

spec = importlib.util.spec_from_file_location("contact_modes", CONTACT_MODES_PATH)
contact_modes = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = contact_modes
spec.loader.exec_module(contact_modes)

get_mode_names = contact_modes.get_mode_names
local_contacts_to_world = contact_modes.local_contacts_to_world


def main():
    device = "cpu"
    names = get_mode_names()
    n = len(names)

    base = torch.tensor(
        [
            [0.00, 0.00, 0.25],
            [1.00, 0.00, 0.25],
            [0.00, 1.00, 0.25],
            [1.00, 1.00, 0.25],
            [-1.00, 0.50, 0.25],
            [0.50, -1.00, 0.25],
        ],
        dtype=torch.float32,
        device=device,
    )
    box_pos_w = base[torch.arange(n) % base.shape[0]].clone()

    yaw_base = torch.tensor(
        [0.0, 0.523599, 1.047198, 1.570796, -0.785398, 3.141593],
        dtype=torch.float32,
        device=device,
    )
    box_yaw_w = yaw_base[torch.arange(n) % yaw_base.shape[0]].clone()

    mode_ids = torch.arange(n, dtype=torch.long, device=device)
    left_w, right_w = local_contacts_to_world(box_pos_w, box_yaw_w, mode_ids)

    print("=== G1-Hand-ShortPush contact mode debug ===")
    print(f"num_modes={n}")

    for i in range(n):
        sep = torch.linalg.norm(left_w[i] - right_w[i]).item()
        print(f"\nmode {i}: {names[i]}")
        print(f"  box_pos = {box_pos_w[i].tolist()}")
        print(f"  box_yaw = {box_yaw_w[i].item():+.3f} rad")
        print(f"  left_w  = {left_w[i].tolist()}")
        print(f"  right_w = {right_w[i].tolist()}")
        print(f"  hand_separation = {sep:.3f} m")

    assert left_w.shape == (n, 3)
    assert right_w.shape == (n, 3)
    assert torch.all(torch.isfinite(left_w))
    assert torch.all(torch.isfinite(right_w))
    print("\nCONTACT_MODE_DEBUG_OK")


if __name__ == "__main__":
    main()
