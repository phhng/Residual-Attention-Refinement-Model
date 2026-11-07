import torch
from model.MSR import MSR

DEAD_PREFIXES = ("cross_scale_attention.",)  # built but never used in forward

def load_model(model_path, num_classes=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MSR(layers=50, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith(DEAD_PREFIXES)}

    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    print("Custom model loaded successfully!")
    return model, device