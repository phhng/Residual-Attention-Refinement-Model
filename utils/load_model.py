import torch
from model.MSR import MSR

DEAD_PREFIXES = ("cross_scale_attention.",)  # Built but never used in forward


def load_model(model_path, num_classes=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MSR(layers=50, num_classes=num_classes)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Remove DataParallel prefix
    state_dict = {
        k.removeprefix("module."): v
        for k, v in state_dict.items()
    }

    state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith(DEAD_PREFIXES)
    }

    model_dict = model.state_dict()

    compatible_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict, strict=True)

    skipped = set(state_dict) - set(compatible_dict)

    if skipped:
        print("\nSkipped checkpoint parameters:")
        for k in sorted(skipped):
            print(f"  {k}")

    missing = set(model_dict) - set(compatible_dict)

    if missing:
        print("\nCurrent model parameters not loaded from checkpoint:")
        for k in sorted(missing):
            print(f"  {k}")

    model.to(device).eval()

    print("\nCustom model loaded successfully!")
    return model, device