# %% [markdown]
# # Confusion matrices -> grouped SVG figure
# Cross-dataset confusion-matrix CSVs are grouped by their first-level folder.
# ```
# logs/
#   kermany/  resnet50_cm.csv  vgg16_cm.csv
#   cohen/    resnet50_cm.csv  vgg16_cm.csv
# ```
# File format: CSV with a header row and a label column, e.g.
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

# CM_ROOT     = Path("results/logs")
CM_ROOT     = Path("results/cross-dataset_eval")
OUT_SVG     = Path("results/graphics/confusion_matrices.svg")
NORMALIZE   = True

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

# %% Cell 3 - group confusion matrices by dataset (folder name)
def load_cms(root: Path) -> dict:
    """Load confusion matrices, excluding training logs and augmented runs."""
    grouped = {}
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = {}
        for f in sorted(ds_dir.rglob("*")):
            if (f.suffix.lower() == ".csv" and f.stem.casefold() != "training_log"
                    and "aug" not in f.name.casefold()):
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
    if n < 1:
        raise ValueError("cannot create confusion-matrix grid: no CSV panels were selected")
    if n % 2 == 0:
        rows = max(d for d in range(1, int(math.isqrt(n)) + 1) if n % d == 0)
        return rows, n // rows
    cols = math.ceil(math.sqrt(n))
    return math.ceil(n / cols), cols


def panel_title(dataset: str, run: str) -> str:
    run_name = Path(run).name
    dataset_name = dataset.replace("_", " ").title()
    if run_name.casefold() == "confusion_matrix":
        return dataset_name

    model_names = {
        "eca": "Model A",
        "full": "Model B",
    }
    model_key, separator, variant = run_name.partition("_")
    model_name = model_names.get(model_key, run_name)
    has_norahman_marker = "norahman" in run_name.casefold()
    if separator and not has_norahman_marker:
        model_name += f" ({variant})"
    if has_norahman_marker:
        dataset_name += "*"
    return f"{dataset_name} - {model_name}"


def plot_cms(cms: dict, out_path: Path):
    plt.rcParams["svg.fonttype"] = "none"
    panels = []
    for ds, runs in cms.items():
        for run, cm in runs.items():
            panels.append((panel_title(ds, run), cm))

    if not panels:
        raise ValueError("no confusion-matrix CSVs found after filtering training logs and augmented runs")
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