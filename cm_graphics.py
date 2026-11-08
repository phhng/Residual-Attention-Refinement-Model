# %% [markdown]
# # Confusion matrices -> grouped SVG figure
# Expected layout (dataset = first-level folder; one CM file per run, name contains "cm" or "confusion"):
# ```
# logs/
#   kermany/  resnet50_cm.csv  vgg16_cm.csv
#   cohen/    resnet50_cm.csv  vgg16_cm.csv
# ```
# File format: CSV/TSV with a header row and a label column, e.g.
# ```
#                    pred_COVID  pred_Lung_Opacity  pred_Normal  pred_Viral Pneumonia
# true_COVID         325         17                 20           0
# ...
# ```
# Orientation is detected from the `true_` / `pred_` prefixes, so either layout works.
# The output always has TRUE labels on the rows (y-axis) and PREDICTED labels on the columns (x-axis).

# %% Cell 1 - config & imports
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import SVG, display

CM_ROOT     = Path("results/logs")          # <-- change to your folder
CM_KEYWORDS = ("cm", "confusion")   # a file is a CM if its name contains one of these
OUT_SVG     = Path("results/graphics/confusion_matrices.svg")
NORMALIZE   = True                  # colour = % of each true class; cell text = raw count (+ %)

# %% Cell 2 - read a confusion matrix
def _strip(label: str) -> str:
    s = str(label).strip()
    for p in ("true_", "pred_", "true ", "pred "):
        if s.lower().startswith(p):
            return s[len(p):]
    return s

def read_cm(path: Path) -> pd.DataFrame:
    """Returns a DataFrame: rows = true class, columns = predicted class."""
    df = pd.read_csv(path, sep=None, engine="python", index_col=0)  # sniffs , ; or tab
    df.columns = [str(c).strip() for c in df.columns]
    df.index   = [str(i).strip() for i in df.index]

    cols_pred = all(c.lower().startswith("pred") for c in df.columns)
    cols_true = all(c.lower().startswith("true") for c in df.columns)
    if cols_true and not cols_pred:      # columns are TRUE -> transpose so rows = TRUE
        df = df.T
    df.columns = [_strip(c) for c in df.columns]
    df.index   = [_strip(i) for i in df.index]
    return df.astype(float)

# %% Cell 3 - group CMs by dataset (folder name)
def load_cms(root: Path) -> dict:
    """returns {dataset_name: {run_name: DataFrame}}"""
    grouped = {}
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = {}
        for f in sorted(ds_dir.rglob("*")):
            if f.suffix.lower() in {".csv", ".tsv", ".txt"} and \
               any(k in f.stem.lower() for k in CM_KEYWORDS):
                rel = f.relative_to(ds_dir).with_suffix("")
                runs[str(rel).replace("\\", "/")] = read_cm(f)
        if runs:
            grouped[ds_dir.name] = runs
    return grouped

cms = load_cms(CM_ROOT)
for ds, runs in cms.items():
    print(f"{ds}: {list(runs)}")

# %% Cell 4 - plot: one row per dataset, one column per run
SHOW_VALUES = True   # count (and % of true class when NORMALIZE=True) inside each cell
import math

def grid_shape(n: int):
    """Near-square grid. Even n splits exactly (4 -> 2x2, 6 -> 2x3, 8 -> 2x4, 12 -> 3x4);
    odd n uses ceil(sqrt(n)) columns and leaves the last cell(s) empty."""
    if n % 2 == 0:
        rows = max(d for d in range(1, int(math.isqrt(n)) + 1) if n % d == 0)
        return rows, n // rows
    cols = math.ceil(math.sqrt(n))
    return math.ceil(n / cols), cols

def plot_cms(cms: dict, out_path: Path):
    plt.rcParams["svg.fonttype"] = "none"
    # flatten to a list of panels in dataset order
    panels = []
    for i, (ds, runs) in enumerate(cms.items()):
        letter = f"({chr(97 + i)}) {ds.replace('_', ' ').title()}"
        for run, cm in runs.items():
            panels.append((f"{letter} - {run}" if len(runs) > 1 else letter, cm))

    n_rows, n_cols = grid_shape(len(panels))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows),
                             squeeze=False)

    for ax in axes.flat:
        ax.axis("off")                          # unused cells stay empty

    for ax, (title, cm) in zip(axes.flat, panels):
        ax.axis("on")
        counts = cm.values
        pct = counts / counts.sum(axis=1, keepdims=True).clip(min=1) * 100
        ax.imshow(pct if NORMALIZE else counts, cmap="Blues",
                  vmin=0, vmax=100 if NORMALIZE else None)
        if SHOW_VALUES:
            shade = pct if NORMALIZE else counts
            thresh = 50 if NORMALIZE else counts.max() / 2
            for r in range(counts.shape[0]):
                for c in range(counts.shape[1]):
                    txt = f"{int(counts[r, c])}"
                    if NORMALIZE:
                        txt += f"\n{pct[r, c]:.1f}%"
                    ax.text(c, r, txt, ha="center", va="center", fontsize=6,
                            color="white" if shade[r, c] > thresh else "black")
        ax.set_xticks(range(len(cm.columns)))
        ax.set_xticklabels(cm.columns, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(cm.index)))
        ax.set_yticklabels(cm.index, fontsize=6)
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("True", fontsize=7)
        ax.set_title(title, fontweight="bold", fontsize=8)

    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path

svg_path = plot_cms(cms, OUT_SVG)
print("saved:", svg_path.resolve())

# %% Cell 5 - display
display(SVG(filename=str(svg_path)))