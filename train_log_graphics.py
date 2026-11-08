# %% [markdown]
# # Training logs -> grouped SVG figure
# Expected layout (dataset = first-level folder, each run's log inside it):
# ```
# logs/
#   kermany/  resnet50.csv  vgg16.csv
#   cohen/    resnet50.csv  vgg16.csv
# ```
# Supported log files: `.csv` (e.g. Keras CSVLogger) and `.json` (Keras `history.history`
# as dict-of-lists, or a list of per-epoch dicts).

# %% Cell 1 - config & imports
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import SVG, display

LOG_ROOT = Path("results/logs")            # <-- change to your logs folder
OUT_SVG  = Path("results/graphics/training_results.svg")

# accepted column names for each metric (case-insensitive)
ALIASES = {
    "train_acc":  ["accuracy", "acc", "train_acc", "train_accuracy", "training_accuracy"],
    "val_acc":    ["val_accuracy", "val_acc", "valid_acc", "validation_accuracy"],
    "train_loss": ["loss", "train_loss", "training_loss"],
    "val_loss":   ["val_loss", "valid_loss", "validation_loss"],
}

# %% Cell 2 - read logs
def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    out = {}
    for key, names in ALIASES.items():
        for n in names:
            if n in cols:
                out[key] = df[cols[n]].astype(float).values
                break
    return pd.DataFrame(out)

def read_log(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        df = pd.DataFrame(data)          # works for dict-of-lists and list-of-dicts
    else:
        raise ValueError(f"Unsupported log type: {path}")
    return _standardize(df)

# %% Cell 3 - group logs by dataset (folder name)
def load_all(root: Path) -> dict:
    """returns {dataset_name: {run_name: DataFrame}}"""
    grouped = {}
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = {}
        for f in sorted(ds_dir.rglob("*")):
            if f.suffix.lower() in {".csv", ".json"}:
                df = read_log(f)
                if not df.empty:
                    # run name = file name, prefixed by subfolder if nested
                    rel = f.relative_to(ds_dir).with_suffix("")
                    runs[str(rel).replace("\\", "/")] = df
        if runs:
            grouped[ds_dir.name] = runs
    return grouped

logs = load_all(LOG_ROOT)
for ds, runs in logs.items():
    print(f"{ds}: {len(runs)} run(s) -> {list(runs)}")

# %% Cell 4 - plot: row 1 = train/val accuracy, row 2 = train/val loss; one column per dataset
SHOW_NUMBERS = False   # False -> no numeric tick labels on the axes
TRAIN_COLORS = ["#1f77b4", "#08306b", "#6baed6", "#17becf"]   # blues, one per run
VAL_COLORS   = ["#ff7f0e", "#d95f02", "#fdae61", "#e6550d"]   # oranges, one per run

import math

STACK = True   # True: accuracy above loss inside each dataset cell; False: side by side

def grid_shape(n: int):
    """Near-square grid of datasets. Even n splits exactly (4 -> 2x2, 6 -> 2x3, 8 -> 2x4);
    odd n uses ceil(sqrt(n)) columns and leaves the last cell(s) empty."""
    if n % 2 == 0:
        rows = max(d for d in range(1, int(math.isqrt(n)) + 1) if n % d == 0)
        return rows, n // rows
    cols = math.ceil(math.sqrt(n))
    return math.ceil(n / cols), cols

def plot_logs(logs: dict, out_path: Path):
    plt.rcParams["svg.fonttype"] = "none"      # keep text editable in the SVG
    d_rows, d_cols = grid_shape(len(logs))     # grid of datasets
    sr, sc = (2, 1) if STACK else (1, 2)       # sub-grid per dataset (accuracy, loss)
    fig, axes = plt.subplots(d_rows * sr, d_cols * sc,
                             figsize=(4.4 * d_cols * sc, 3.4 * d_rows * sr), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")                          # unused cells stay empty

    for j, (ds, runs) in enumerate(logs.items()):
        r, c = divmod(j, d_cols)
        for i, (metric, ylabel) in enumerate([("acc", "Accuracy"), ("loss", "Loss")]):
            ax = axes[r * sr + (i if STACK else 0), c * sc + (0 if STACK else i)]
            ax.axis("on")
            for k, (run, df) in enumerate(runs.items()):
                # legend text: "Train accuracy" / "Val accuracy" / "Train loss" / "Val loss"
                # (run name is prefixed only when a dataset has more than one run)
                prefix = f"{run} - " if len(runs) > 1 else ""
                if f"train_{metric}" in df:
                    ax.plot(range(1, len(df) + 1), df[f"train_{metric}"],
                            color=TRAIN_COLORS[k % len(TRAIN_COLORS)], ls="-",
                            label=f"{prefix}Train {ylabel.lower()}")
                if f"val_{metric}" in df:
                    ax.plot(range(1, len(df) + 1), df[f"val_{metric}"],
                            color=VAL_COLORS[k % len(VAL_COLORS)], ls="-",
                            label=f"{prefix}Val {ylabel.lower()}")
            if not SHOW_NUMBERS:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.set_xlabel("Epoch")
            # annotation on every panel: "(a) Kermany - Accuracy" / "(a) Kermany - Loss"
            ax.set_title(f"({chr(97 + j)}) {ds.replace('_', ' ').title()} - {ylabel}",
                         fontweight="bold")
            ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path

svg_path = plot_logs(logs, OUT_SVG)
print("saved:", svg_path.resolve())

# %% Cell 5 - display the SVG
display(SVG(filename=str(svg_path)))