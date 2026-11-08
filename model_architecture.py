from main import MODEL_PATH
from utils.load_model import load_model

MODEL_PATH = "weights/eca_LN_kermany42.pth"
OUT_FILE = "architecture.txt"

model, device = load_model(MODEL_PATH, num_classes=2)

lines = [str(model), "", "Parameters per top-level module:"]

for name, module in model.named_children():
    params = list(module.parameters())

    n = sum(p.numel() for p in params)
    t = sum(p.numel() for p in params if p.requires_grad)

    lines.append(
        f"  {name:<22} total={n:>12,}  trainable={t:>12,}"
    )

# Count unique parameters across the entire model
all_params = list(model.parameters())

total = sum(p.numel() for p in all_params)
trainable = sum(p.numel() for p in all_params if p.requires_grad)

lines += [
    "",
    f"Total: {total:,}",
    f"Trainable: {trainable:,}"
]

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Saved to {OUT_FILE}")