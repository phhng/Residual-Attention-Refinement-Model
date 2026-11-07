
from main import MODEL_PATH
from utils.load_model import load_model

OUT_FILE = "architecture.txt"

model, device = load_model(MODEL_PATH, num_classes=2)

lines = [str(model), "", "Parameters per top-level module:"]
total = trainable = 0
for name, module in model.named_children():
    n = sum(p.numel() for p in module.parameters())
    t = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total += n
    trainable += t
    lines.append(f"  {name:<22} total={n:>12,}  trainable={t:>12,}")
lines += ["", f"Total: {total:,}", f"Trainable: {trainable:,}"]

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Saved to {OUT_FILE}")