#!/usr/bin/env python
"""
tensor_probe.py  -  show the tensors the MSR network REALLY receives, plus the input and output of every block,
for the five masking set-ups.

Put this file in the project root (next to model/ and utils/, same place you run model_architecture.py) and run:

    python tensor_probe.py --image <path>/COVID/images/COVID-1.png \
        --weights model/weights/eca_LN_kermany42.pth --num-classes 2

With no --image, all samples described by samples/**/manifest.json are processed. Masks are found from the
manifest, *_mask sibling filenames, or the Rahman layout (<class>/images/X.png -> <class>/masks/X.png).
Checkpoints are selected automatically from model/weights for each masking setup and sample class count;
--weights, --num-classes, --ckpt, and --n-classes override those choices.

    --ckpt input_mask=model/weights/rahman_in.pth --n-classes input_mask=4 \
    --ckpt feature_mask=model/weights/rahman_feat.pth --n-classes feature_mask=4 ...

What it reproduces (config.py flags -> what is fed to model(x, lung_mask=...)):
    no_mask              USE_MASK=False                          x = img                lung_mask = None
    input_mask           USE_MASK, INPUT_MASKING                 x = img * mask         lung_mask = None
    feature_mask         USE_MASK, FEATURE_MASKING               x = img                lung_mask = mask
    input_mask_inverted  USE_MASK, INPUT_MASKING, INVERTED       x = img * (1 - mask)   lung_mask = None
    mask_only            MASK_ONLY                               x = mask (3 channels)  lung_mask = None
Pre-processing mirrors data/joint_dataset.py: bicubic resize to IMAGE_SIZE, values in [0, 1], NO mean/std.
Masks go through the same bicubic resize and are NOT thresholded (that is what JointTransform returns, so the
edges are soft); MASK_ONLY thresholds at > 0.5 exactly like XrayDataset. If your training loop binarizes the mask
before using it, pass --binarize-mask. The script re-runs your own transforms and prints a parity check.
The flags in config.py are NOT touched (ENCODER_CONFIG is read from it, since it changes the architecture).

Output (in --out, default tensor_probe_out/):
    overview.png            one row per set-up: model input, masks, features before/after masking, probabilities
    summary.csv             prediction + input statistics + sanity checks per set-up
    <setup>/model_input.npy|png   the exact tensor handed to the backbone (float32 [3,H,W] / 8-bit preview)
    <setup>/tensors.pt      input, logits, probs and EVERY hooked block's inputs/outputs (a plain dict, torch.load it)
    <setup>/layer_io_stats.csv    shape / min / max / mean / std / fraction-of-zeros of every captured tensor
    <setup>/pipeline.png    input/output of every top-level block (channel-mean maps for feature maps)
"""
import argparse
import csv
import io
import json
import math
from collections import OrderedDict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = getattr(Image, "Resampling", Image)  # Pillow >= 9.1 has Image.Resampling; older has Image.BICUBIC
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASSIFIER_WEIGHTS = {
    2: Path("model/weights/eca_LN_kermany42.pth"),
    3: Path("model/weights/eca_LN_cohen42.pth"),
    4: Path("model/weights/eca_LN_rahman42.pth"),
}
MASK_WEIGHTS = {
    "input_mask": Path("model/weights/mask_exp/eca_input_mask.pth"),
    "feature_mask": Path("model/weights/mask_exp/eca_feature_mask.pth"),
    "input_mask_inverted": Path("model/weights/mask_exp/eca_inverted_mask.pth"),
    "mask_only": Path("model/weights/mask_exp/eca_mask_img_only.pth"),
}

# ------------------------------------------------------------------------------------------------------------
# The five set-ups. To probe another combination (e.g. input + feature masking) add a line here.
# ------------------------------------------------------------------------------------------------------------
FLAG_DEFAULTS = dict(use_mask=False, input_masking=False, feature_masking=False, inverted=False, mask_only=False)
SCENARIOS = OrderedDict([
    ("no_mask",             dict(title="USE_MASK = False")),
    ("input_mask",          dict(title="USE_MASK + INPUT_MASKING", use_mask=True, input_masking=True)),
    ("feature_mask",        dict(title="USE_MASK + FEATURE_MASKING", use_mask=True, feature_masking=True)),
    ("input_mask_inverted", dict(title="USE_MASK + INPUT_MASKING + USE_INVERTED_MASK",
                                 use_mask=True, input_masking=True, inverted=True)),
    ("mask_only",           dict(title="MASK_ONLY (mask is the image)", mask_only=True)),
])


# ------------------------------------------------------------------------------------------------------------
# Pre-processing and per-set-up inputs (pure numpy, NCHW float32)
# ------------------------------------------------------------------------------------------------------------
def load_image(path, size_hw):
    """Image -> bicubic resize -> float32 [1,3,H,W] in [0,1]. No mean/std (the model was trained without it)."""
    h, w = size_hw
    img = Image.open(path).convert("RGB").resize((w, h), RES.BICUBIC)
    return np.ascontiguousarray((np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None])


def load_mask(path, size_hw):
    """Mask exactly as JointTransform returns it: 'L' image -> bicubic resize -> /255 -> float32 [1,1,H,W].
    NOT thresholded (soft edges). Assumes 0/255 masks like Rahman's."""
    h, w = size_hw
    a = np.asarray(Image.open(path).convert("L").resize((w, h), RES.BICUBIC), dtype=np.float32) / 255.0
    return np.ascontiguousarray(a[None, None])


def binarize(mask):
    return (mask > 0.5).astype(np.float32)


def infer_mask_path(image_path):
    """Find a paired *_mask file or the Rahman <class>/images -> masks layout."""
    p = Path(image_path)
    if p.parent.exists():
        matches = sorted(
            candidate for candidate in p.parent.iterdir()
            if candidate.is_file() and candidate.stem.casefold() == (p.stem + "_mask").casefold()
        )
        if matches:
            return matches[0]
    if p.parent.name.lower() == "images":
        cand = p.parent.parent / "masks" / p.name
        if cand.exists():
            return cand
    return None


def discover_samples(samples_dir):
    """Load image/mask pairs and class metadata from sample manifests when available."""
    jobs = []
    described_images = set()
    for manifest_path in sorted(Path(samples_dir).rglob("manifest.json")):
        with open(manifest_path, encoding="utf-8") as fh:
            entries = json.load(fh)
        class_names = {
            entry["label"]: entry.get("class_folder", str(entry["label"]))
            for entry in entries if "label" in entry
        }
        n_classes = len(class_names) or None
        for entry in entries:
            image_path = manifest_path.parent / entry["image"]
            if not image_path.is_file():
                raise FileNotFoundError("sample image listed in %s does not exist: %s" % (
                    manifest_path, image_path))
            described_images.add(image_path.resolve())
            mask_name = entry.get("mask")
            mask_path = manifest_path.parent / mask_name if mask_name else infer_mask_path(image_path)
            if mask_path is not None and not mask_path.is_file():
                raise FileNotFoundError("sample mask listed in %s does not exist: %s" % (
                    manifest_path, mask_path))
            jobs.append(dict(image=image_path, mask=mask_path, dataset=manifest_path.parent.name,
                             num_classes=n_classes, class_names=class_names))

    for image_path in sorted(Path(samples_dir).rglob("*")):
        if (image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
                and "mask" not in image_path.stem.casefold()
                and image_path.resolve() not in described_images):
            jobs.append(dict(image=image_path, mask=infer_mask_path(image_path),
                             dataset=image_path.parent.name, num_classes=None, class_names={}))
    if not jobs:
        raise FileNotFoundError("no image samples found under %s" % samples_dir)
    return jobs


def infer_num_classes(image_path):
    """Use a nearby manifest for a single image, if present."""
    for parent in (Path(image_path).parent, *Path(image_path).parents):
        manifest_path = parent / "manifest.json"
        if manifest_path.is_file():
            with open(manifest_path, encoding="utf-8") as fh:
                entries = json.load(fh)
            labels = {entry["label"] for entry in entries if "label" in entry}
            if labels:
                return len(labels)
    return None


def build_model_inputs(flags, img, mask, binarize_mask=False):
    """Returns (x for the backbone, lung_mask argument for model.forward or None, mask actually used or None).
    mask is the soft mask from load_mask. MASK_ONLY always thresholds at > 0.5 (as XrayDataset does);
    the other set-ups use the soft mask unless binarize_mask is set."""
    used = None
    if flags["use_mask"] or flags["mask_only"]:
        base = binarize(mask) if (flags["mask_only"] or binarize_mask) else mask
        used = 1.0 - base if flags["inverted"] else base
    if flags["mask_only"]:
        return np.repeat(used, 3, axis=1), None, used
    x, lung_mask = img, None
    if flags["use_mask"]:
        if flags["input_masking"]:
            x = img * used
        if flags["feature_masking"]:
            lung_mask = used
    return x, lung_mask, used


# ------------------------------------------------------------------------------------------------------------
# Recording the input and output of every block
# ------------------------------------------------------------------------------------------------------------
def _grab(obj, base):
    """Flatten tensors out of (nested) tuples / lists / dicts -> {'in[0]': tensor, 'out[1]': tensor, ...} (copies)."""
    found = OrderedDict()

    def rec(o, path):
        if torch.is_tensor(o):
            found[path] = o.detach().cpu().clone()
        elif isinstance(o, dict):
            for k, v in o.items():
                rec(v, "%s[%s]" % (path, k))
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                rec(v, "%s[%d]" % (path, i))

    rec(obj, base)
    return found


class IORecorder:
    """Pre-hook copies a block's inputs BEFORE it runs (matters for in-place ReLUs), post-hook copies its outputs.
    calls[module_name] = [ {'in': {...}, 'out': {...}}, ... ]   (one entry per call, in execution order)."""

    def __init__(self, model, depth):
        self.calls = OrderedDict()
        self._handles = []
        for name, mod in model.named_modules():
            if not name or name.count(".") + 1 > depth:
                continue
            self._handles.append(mod.register_forward_pre_hook(self._pre(name), with_kwargs=True))
            self._handles.append(mod.register_forward_hook(self._post(name), with_kwargs=True))

    def _pre(self, name):
        def hook(module, args, kwargs):
            entry = {"in": _grab(args, "in"), "out": OrderedDict()}
            for k, v in kwargs.items():
                entry["in"].update(_grab(v, "in:%s" % k))
            self.calls.setdefault(name, []).append(entry)
        return hook

    def _post(self, name):
        def hook(module, args, kwargs, output):
            self.calls[name][-1]["out"] = _grab(output, "out")
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def check_parity(image_path, mask_path, size_hw, img_np, mask_np):
    """Run the project's own transforms (data/joint_dataset.py) on the same files and compare with ours."""
    try:
        from data.joint_dataset import JointTransform, transform_val
    except Exception as e:  # not run from the project root, etc.
        print("note: parity check skipped (cannot import data.joint_dataset: %s)" % e)
        return []
    image = Image.open(image_path).convert("RGB")
    checks = [("image == transform_val(image)", bool(np.array_equal(to_np(transform_val(image))[None], img_np)))]
    if mask_path is not None:
        mask = Image.open(mask_path).convert("L")
        j_img, j_mask = JointTransform(size_hw, train=False)(image, mask)
        checks.append(("image == JointTransform image", bool(np.array_equal(to_np(j_img)[None], img_np))))
        checks.append(("mask == JointTransform mask (bicubic, soft)", bool(np.array_equal(to_np(j_mask)[None], mask_np))))
        only = (to_np(transform_val(mask)) > 0.5).astype(np.float32)[None]
        checks.append(("MASK_ONLY mask == (transform_val(mask) > 0.5)", bool(np.array_equal(only, binarize(mask_np)))))
    return checks


def run_scenario(model, device, x_np, lm_np, depth):
    x = torch.from_numpy(np.ascontiguousarray(x_np)).clone().to(device)
    lm = torch.from_numpy(np.ascontiguousarray(lm_np)).clone().to(device) if lm_np is not None else None
    rec = IORecorder(model, depth)
    try:
        with torch.no_grad():
            logits = model(x, lung_mask=lm)
    finally:
        rec.remove()
    return logits.detach().cpu(), rec.calls


def _c(calls, module, kind, key):
    return calls[module][0][kind][key]


def feature_views(lm_np, calls):
    """Feature maps before / after the (optional) feature-level mask, rebuilt exactly as MSR.forward does."""
    pre_cat = torch.cat([_c(calls, "conv_red_1", "out", "out"), _c(calls, "conv_red_2", "out", "out")], dim=1)
    pre_3 = _c(calls, "conv_red_3", "out", "out")
    views = dict(pre_cat=pre_cat, pre_3=pre_3,
                 in_cat=_c(calls, "enhancement_cat", "in", "in[0]"),
                 in_3=_c(calls, "enhancement_3", "in", "in[0]"), m_cat=None, m_3=None)
    if lm_np is not None:
        lm = torch.from_numpy(np.ascontiguousarray(lm_np))
        views["m_cat"] = F.interpolate(lm, size=tuple(pre_cat.shape[-2:]), mode="nearest")
        views["m_3"] = F.interpolate(lm, size=tuple(pre_3.shape[-2:]), mode="nearest")
    return views


def verify(flags, x_np, used, views, calls):
    """Hard checks that what was captured inside the network is what we think was delivered."""
    checks = []
    got = _c(calls, "backbone", "in", "in[0]")
    checks.append(("backbone input == saved model_input", bool(torch.equal(got, torch.from_numpy(x_np)))))
    if flags["use_mask"] and flags["input_masking"]:
        checks.append(("input is exactly 0 where the mask is 0", float(np.abs(x_np * (used == 0)).max()) == 0.0))
    if flags["mask_only"]:
        checks.append(("input is binary {0,1}", bool(np.isin(x_np, (0.0, 1.0)).all())))
    if views["m_3"] is None:
        checks.append(("no feature mask: enhancement inputs == conv_red outputs",
                       bool(torch.equal(views["in_cat"], views["pre_cat"]) and torch.equal(views["in_3"], views["pre_3"]))))
    else:
        checks.append(("feature mask: enhancement_cat input == conv_red cat * mask",
                       bool(torch.equal(views["in_cat"], views["pre_cat"] * views["m_cat"]))))
        checks.append(("feature mask: enhancement_3 input == conv_red_3 * mask",
                       bool(torch.equal(views["in_3"], views["pre_3"] * views["m_3"]))))
    return checks


# ------------------------------------------------------------------------------------------------------------
# Statistics + saving
# ------------------------------------------------------------------------------------------------------------
def to_np(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def tensor_stats(t):
    a = to_np(t)
    f = a.astype(np.float64)
    if f.size == 0:
        return dict(shape="x".join(map(str, a.shape)), dtype=str(a.dtype), min="", max="", mean="", std="",
                    frac_zero="", n_nan=0)
    return dict(shape="x".join(map(str, a.shape)), dtype=str(a.dtype), min=float(np.nanmin(f)),
                max=float(np.nanmax(f)), mean=float(np.nanmean(f)), std=float(np.nanstd(f)),
                frac_zero=float((f == 0).mean()), n_nan=int(np.isnan(f).sum()))


def write_io_stats(path, calls):
    cols = ["module", "call", "direction", "key", "shape", "dtype", "min", "max", "mean", "std", "frac_zero", "n_nan"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for mod, entries in calls.items():
            for i, e in enumerate(entries):
                for direction in ("in", "out"):
                    for key, t in e[direction].items():
                        w.writerow(dict(module=mod, call=i, direction=direction, key=key, **tensor_stats(t)))


def save_scenario(sdir, name, flags, x_np, lm_np, logits, probs, calls, views):
    sdir.mkdir(parents=True, exist_ok=True)
    np.save(sdir / "model_input.npy", x_np[0])
    preview = (x_np[0].transpose(1, 2, 0) * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(preview).save(sdir / "model_input.png")
    if lm_np is not None:
        np.save(sdir / "lung_mask_arg.npy", lm_np[0, 0])
    payload = {
        "scenario": name, "flags": dict(flags),
        "model_input": torch.from_numpy(np.ascontiguousarray(x_np)),
        "lung_mask_arg": torch.from_numpy(np.ascontiguousarray(lm_np)) if lm_np is not None else None,
        "logits": logits, "probs": probs,
        "features_before_mask": {"cat": views["pre_cat"], "3": views["pre_3"]},
        "features_after_mask": {"cat": views["in_cat"], "3": views["in_3"]},
        "feature_mask_resized": {"cat": views["m_cat"], "3": views["m_3"]},
        "io": {k: v for k, v in calls.items()},
    }
    torch.save(payload, sdir / "tensors.pt")
    write_io_stats(sdir / "layer_io_stats.csv", calls)


# ------------------------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------------------------
def draw_tensor(ax, t, title, vmin=None, vmax=None):
    """Draw any tensor: RGB image, channel-mean map, matrix, or vector. Batch size is 1 so singleton dims are squeezed."""
    raw = to_np(t)
    a = np.squeeze(raw.astype(np.float32))
    if a.ndim == 3 and a.shape[0] == 3:
        rgb = a.transpose(1, 2, 0)
        lo, hi = float(rgb.min()), float(rgb.max())
        if lo < 0 or hi > 1:
            rgb = (rgb - lo) / (hi - lo + 1e-8)
        ax.imshow(np.clip(rgb, 0, 1))
    elif a.ndim == 3:
        ax.imshow(a.mean(0), cmap="viridis", vmin=vmin, vmax=vmax)
    elif a.ndim == 2:
        ax.imshow(a, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    elif a.ndim == 1:
        if a.size <= 8:
            ax.bar(range(a.size), a)
        else:
            ax.plot(a, lw=0.8)
    else:
        ax.text(0.5, 0.5, "%.4g" % float(a), ha="center", va="center")
    if a.ndim >= 2:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("%s\n%s" % (title, "x".join(map(str, raw.shape))), fontsize=11)
    ax.tick_params(labelsize=9)


def pipeline_figure(path, name, calls):
    panels = []
    for mod, entries in calls.items():
        if "." in mod:  # sub-blocks live in tensors.pt / csv; the figure shows the top-level blocks
            continue
        e = entries[0]
        if e["in"]:
            k = next(iter(e["in"]))
            panels.append(("%s.%s" % (mod, k), e["in"][k]))
        panels += [("%s.%s" % (mod, k), v) for k, v in e["out"].items()]
    ncols = 6
    nrows = max(1, math.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.1 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, t) in zip(axes.ravel(), panels):
        ax.axis("on")
        draw_tensor(ax, t, title)
    fig.suptitle("%s - input[0] and every output of each top-level block" % name, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def overview_figure(path, results, class_names_for):
    heads = ["model input\n(tensor given to the backbone)", "mask applied to the input",
             "feature mask (resized, branch 3)", "conv_red_3 output\n(channel mean, before mask)",
             "enhancement_3 input\n(channel mean, as delivered)", "class probabilities"]
    n = len(results)
    fig, axes = plt.subplots(n, 6, figsize=(20, 3.4 * n), squeeze=False)
    for r, (name, d) in enumerate(results.items()):
        row = axes[r]
        draw_tensor(row[0], d["x"], "")
        for ax, tensor, text in ((row[1], d["input_mask"], "none"), (row[2], d["m_3"], "none")):
            if tensor is None:
                ax.text(0.5, 0.5, text, ha="center", va="center", color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                draw_tensor(ax, tensor, "", vmin=0, vmax=1)
        pre = np.squeeze(to_np(d["pre_3"])).mean(0)
        post = np.squeeze(to_np(d["in_3"])).mean(0)
        vmax = float(max(pre.max(), post.max(), 1e-8))
        draw_tensor(row[3], d["pre_3"], "", vmin=0, vmax=vmax)
        draw_tensor(row[4], d["in_3"], "", vmin=0, vmax=vmax)
        probs = to_np(d["probs"]).ravel()
        row[5].bar(range(probs.size), probs, color="tab:blue")
        row[5].set_ylim(0, 1)
        row[5].set_xticks(range(probs.size))
        row[5].set_xticklabels(class_names_for(name), fontsize=10)
        for i, p in enumerate(probs):
            row[5].text(i, min(p + 0.02, 0.9), "%.3f" % p, ha="center", fontsize=9)
        row[0].set_ylabel("%s\n%s" % (name, d["title"]), fontsize=10)
    for c, h in enumerate(heads):
        shape_line = axes[0, c].get_title().strip()      # keep the tensor-shape caption of the first row
        axes[0, c].set_title(h + ("\n" + shape_line if shape_line else ""), fontsize=11)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------------------
def parse_kv(items, cast=str):
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep:
            raise SystemExit("expected NAME=VALUE, got '%s'" % it)
        out[k.strip()] = cast(v.strip())
    return out


def get_model(cache, load_model, path, n_classes):
    key = (str(path), n_classes)
    if key not in cache:
        cache.clear()  # keep one model in memory at a time
        buf = io.StringIO()
        with redirect_stdout(buf):
            model, device = load_model(str(path), num_classes=n_classes)
        text = buf.getvalue()
        if "Skipped checkpoint" in text or "not loaded from checkpoint" in text:
            print("WARNING: load_model could not load every parameter from %s with num_classes=%d "
                  "(wrong class count? the outputs would be meaningless):\n%s" % (path, n_classes, text))
        cache[key] = (model, device)
    return cache[key]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=None, help="one chest X-ray; omit to process --samples-dir")
    ap.add_argument("--mask", default=None, help="lung mask; default: <class>/masks/<name> for the Rahman layout")
    ap.add_argument("--samples-dir", default="samples", help="sample tree to process when --image is omitted")
    ap.add_argument("--weights", default=None, help="override the automatically selected checkpoint")
    ap.add_argument("--num-classes", type=int, default=None, help="override the inferred checkpoint class count")
    ap.add_argument("--ckpt", action="append", default=[], metavar="SETUP=PATH", help="checkpoint for one set-up")
    ap.add_argument("--n-classes", action="append", default=[], metavar="SETUP=N", help="class count for one set-up")
    ap.add_argument("--scenarios", default="all", help="comma list of: %s (default all)" % ", ".join(SCENARIOS))
    ap.add_argument("--binarize-mask", action="store_true",
                    help="threshold the mask at > 0.5 for the non-MASK_ONLY set-ups (use if your training loop does)")
    ap.add_argument("--classes", default=None, help="class names, comma separated, in label order")
    ap.add_argument("--depth", type=int, default=2, help="1 = top-level blocks, 2 = + their children (default)")
    ap.add_argument("--out", default="tensor_probe_out")
    args = ap.parse_args()

    names = list(SCENARIOS) if args.scenarios == "all" else [s.strip() for s in args.scenarios.split(",")]
    for n in names:
        if n not in SCENARIOS:
            ap.error("unknown set-up '%s'; choose from %s" % (n, ", ".join(SCENARIOS)))
    if args.image and not Path(args.image).is_file():
        ap.error("image does not exist: %s" % args.image)
    if args.mask and not args.image:
        ap.error("--mask can only be used together with --image")

    ckpts, ncls = parse_kv(args.ckpt), parse_kv(args.n_classes, int)

    from utils.load_model import load_model          # imports utils.config (kagglehub) exactly like your scripts
    from utils.config import IMAGE_SIZE, ENCODER_CONFIG

    if args.image:
        image_path = Path(args.image)
        mask_path = Path(args.mask) if args.mask else infer_mask_path(image_path)
        if mask_path is not None and not mask_path.is_file():
            ap.error("mask does not exist: %s" % mask_path)
        jobs = [dict(image=image_path, mask=mask_path, dataset=image_path.parent.name,
                     num_classes=infer_num_classes(image_path), class_names={})]
    else:
        jobs = discover_samples(args.samples_dir)

    if ENCODER_CONFIG is False:
        print("NOTE: ENCODER_CONFIG=False -> Transformer.decoder has 0 layers, so it only applies LayerNorm to the "
              "pooled query; the encoder output ('memory') is computed but does not reach the classifier.")

    needs_mask = [n for n in names if SCENARIOS[n].get("use_mask") or SCENARIOS[n].get("mask_only")]
    processed = 0
    for job in jobs:
        image_path, mask_path = job["image"], job["mask"]
        if needs_mask and mask_path is None and args.image:
            ap.error("set-ups %s need a lung mask: pass --mask or provide a paired *_mask file" %
                     ", ".join(needs_mask))
        img = load_image(image_path, IMAGE_SIZE)
        mask = load_mask(mask_path, IMAGE_SIZE) if mask_path else None

        if mask is not None:
            cov = float(binarize(mask).mean())
            soft = float(((mask > 0) & (mask < 1)).mean())
            print("image %s | mask %s | mask covers %.1f%% of the %dx%d frame | %.1f%% of pixels are soft edge values (0<m<1)%s" % (
                image_path, mask_path, 100 * cov, IMAGE_SIZE[0], IMAGE_SIZE[1], 100 * soft,
                " -> binarized by --binarize-mask" if args.binarize_mask else ""))
            if cov in (0.0, 1.0) or mask.max() < 0.5:
                print("WARNING: the mask is empty, full or 0/1-valued (the pipeline divides by 255) - check the mask")
        parity = check_parity(image_path, mask_path, tuple(IMAGE_SIZE), img, mask)
        if parity:
            print("pre-processing parity with data/joint_dataset.py:")
            for label, ok in parity:
                print("  [%s] %s" % ("OK  " if ok else "FAIL", label))

        if args.image:
            out = Path(args.out)
        else:
            out = Path(args.out) / job["dataset"] / image_path.stem
        out.mkdir(parents=True, exist_ok=True)
        cache, results, rows = {}, OrderedDict(), []
        for name in names:
            cfg = SCENARIOS[name]
            flags = dict(FLAG_DEFAULTS, **{k: v for k, v in cfg.items() if k != "title"})
            if (flags["use_mask"] or flags["mask_only"]) and mask is None:
                print("Skipping %s for %s: no paired lung mask" % (name, image_path))
                continue

            inferred_classes = 4 if name in MASK_WEIGHTS else (job["num_classes"] or 2)
            n_classes = ncls.get(name, args.num_classes or inferred_classes)
            if args.weights:
                default_ckpt = args.weights
            elif name in MASK_WEIGHTS:
                default_ckpt = MASK_WEIGHTS[name]
            else:
                default_ckpt = CLASSIFIER_WEIGHTS.get(n_classes, CLASSIFIER_WEIGHTS[2])
            ckpt = ckpts.get(name, default_ckpt)
            model, device = get_model(cache, load_model, ckpt, n_classes)

            x_np, lm_np, used = build_model_inputs(flags, img, mask, args.binarize_mask)
            logits, calls = run_scenario(model, device, x_np, lm_np, max(1, args.depth))
            probs = torch.softmax(logits, dim=1)[0]
            views = feature_views(lm_np, calls)
            checks = verify(flags, x_np, used, views, calls)
            save_scenario(out / name, name, flags, x_np, lm_np, logits, probs, calls, views)
            pipeline_figure(out / name / "pipeline.png", name, calls)

            st = tensor_stats(x_np)
            pred = int(probs.argmax())
            print("\n=== %s  [%s]  weights=%s (%d classes)" % (name, cfg["title"], ckpt, n_classes))
            print("model input  %s  min %.3f  max %.3f  mean %.3f  exact-zero %.1f%%" % (
                st["shape"], st["min"], st["max"], st["mean"], 100 * st["frac_zero"]))
            print("lung_mask arg to forward(): %s" % ("none" if lm_np is None else "yes (%s)" % "x".join(map(str, views["m_3"].shape))))
            print("logits %s | probs %s | predicted class %d" % (
                np.round(logits[0].numpy(), 4).tolist(), np.round(probs.numpy(), 4).tolist(), pred))
            for label, ok in checks:
                print("  [%s] %s" % ("OK  " if ok else "FAIL", label))

            results[name] = dict(title=cfg["title"], x=x_np,
                                 input_mask=used if (flags["input_masking"] or flags["mask_only"]) else None,
                                 m_3=views["m_3"], pre_3=views["pre_3"], in_3=views["in_3"], probs=probs)
            rows.append(dict(setup=name, title=cfg["title"], weights=ckpt, n_classes=n_classes, pred=pred,
                             probs=";".join("%.5f" % p for p in probs.tolist()),
                             logits=";".join("%.5f" % v for v in logits[0].tolist()),
                             input_min=st["min"], input_max=st["max"], input_mean=st["mean"],
                             input_frac_zero=st["frac_zero"], feature_mask_applied=lm_np is not None,
                             checks_passed=all(ok for _, ok in checks)))

        if not results:
            ap.error("no requested scenarios could run for %s" % image_path)
        names_arg = [c.strip() for c in args.classes.split(",")] if args.classes else None
        manifest_names = [job["class_names"][i] for i in sorted(job["class_names"])]

        def class_names_for(setup):
            n = len(results[setup]["probs"])
            if names_arg and len(names_arg) == n:
                return names_arg
            if len(manifest_names) == n:
                return manifest_names
            return [str(i) for i in range(n)]

        overview_figure(out / "overview.png", results, class_names_for)
        with open(out / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("\nSaved everything to %s/  (start with overview.png and summary.csv)" % out)
        processed += 1

    print("\nProcessed %d sample(s)." % processed)


if __name__ == "__main__":
    main()