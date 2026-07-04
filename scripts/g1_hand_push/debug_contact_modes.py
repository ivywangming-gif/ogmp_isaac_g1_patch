from pathlib import Path
import importlib.util
import sys
import torch

# Load contact_modes.py directly to avoid importing the full ogmp_isaac package,
# which would require Isaac Sim AppLauncher / pxr.
CONTACT_MODES_PATH = (
    Path(__file__).resolve().parents[2]
    / "exts/ogmp_isaac/ogmp_isaac/tasks/g1_hand_push/contact_modes.py"
)

spec = importlib.util.spec_from_file_location("contact_modes", CONTACT_MODES_PATH)
contact_modes = importlib.util.module_from_spec(spec)
assert spec.loader is not None
# Required for dataclass modules loaded via importlib.
sys.modules[spec.name] = contact_modes
spec.loader.exec_module(contact_modes)

get_mode_names = contact_modes.get_mode_names
local_contacts_to_world = contact_modes.local_contacts_to_world


def main():
    device = "cpu"

    box_pos_w = torch.tensor(
        [
            [0.0, 0.0, 0.25],
            [1.0, 0.0, 0.25],
            [0.0, 1.0, 0.25],
            [1.0, 1.0, 0.25],
            [-1.0, 0.5, 0.25],
            [0.5, -1.0, 0.25],
        ],
        dtype=torch.float32,
        device=device,
    )

    box_yaw_w = torch.tensor(
        [0.0, 0.523599, 1.047198, 1.570796, -0.785398, 3.141593],
        dtype=torch.float32,
        device=device,
    )

    mode_ids = torch.arange(6, dtype=torch.long, device=device)

    left_w, right_w = local_contacts_to_world(box_pos_w, box_yaw_w, mode_ids)
    names = get_mode_names()

    print("=== G1-Hand-ShortPush contact mode debug ===")
    for i in range(6):
        sep = torch.linalg.norm(left_w[i] - right_w[i]).item()
        print(f"\nmode {i}: {names[i]}")
        print(f"  box_pos = {box_pos_w[i].tolist()}")
        print(f"  box_yaw = {box_yaw_w[i].item():+.3f} rad")
        print(f"  left_w  = {left_w[i].tolist()}")
        print(f"  right_w = {right_w[i].tolist()}")
        print(f"  hand_separation = {sep:.3f} m")

    assert left_w.shape == (6, 3)
    assert right_w.shape == (6, 3)
    assert torch.all(torch.isfinite(left_w))
    assert torch.all(torch.isfinite(right_w))

    print("\nCONTACT_MODE_DEBUG_OK")


if __name__ == "__main__":
    main()
