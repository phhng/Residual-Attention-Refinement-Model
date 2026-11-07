from data.joint_dataset import *
from utils.config import *
from utils.log import log
import json
import os
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def make_loader(df, transform, shuffle):
    return DataLoader(XrayDataset(df, transform), batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=2, pin_memory=True, persistent_workers=True)

root = os.path.join(DATA_PATH, DATASETS[DATASET]["root"])
IMG_EXT = {".png", ".jpg", ".jpeg"}

def _images(folder):
    return sorted(f for f in os.listdir(folder)
                  if Path(f).suffix.lower() in IMG_EXT and os.path.isfile(os.path.join(folder, f)))

def extract_patient_id(dataset, filename):
    """Best-effort patient ID from a file name; None when the dataset does not encode it."""
    stem = Path(filename).stem
    if dataset == "kermany":
        m = re.match(r"(person\d+)_(?:bacteria|virus)_\d+", stem, re.I)       # pneumonia
        if m:
            return m.group(1).lower()
        m = re.match(r"((?:NORMAL2-)?IM-\d{4})-\d{4}", stem, re.I)             # normal
        if m:
            return m.group(1).upper()
    return None

def _row(dataset, root, class_name, folder, fname, orig_split=None):
    path = os.path.join(folder, fname)
    return dict(dataset=dataset, class_name=class_name, filename=fname, image_path=path,
                rel_path=os.path.relpath(path, root), patient_id=extract_patient_id(dataset, fname),
                orig_split=orig_split)

def scan_rahman(root):
    rows = []
    for cls in sorted(os.listdir(root)):
        img_dir, mask_dir = os.path.join(root, cls, "images"), os.path.join(root, cls, "masks")
        if not os.path.isdir(img_dir):
            continue
        masks = set(os.listdir(mask_dir)) if os.path.isdir(mask_dir) else set()
        for f in _images(img_dir):
            r = _row("rahman", root, cls, img_dir, f)
            r["mask_path"] = os.path.join(mask_dir, f) if f in masks else None   # masks share the image file name
            rows.append(r)
    return rows

def scan_kermany(root):
    wanted = {"train", "val", "test"}
    split_dirs = lambda p: [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d)) and d.lower() in wanted]
    found = split_dirs(root)
    if not found:
        for sub in sorted(os.listdir(root)):
            sub_path = os.path.join(root, sub)
            if os.path.isdir(sub_path) and split_dirs(sub_path):
                root, found = sub_path, split_dirs(sub_path)
                break
    rows = []
    for split in sorted(found):
        split_dir = os.path.join(root, split)
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                rows += [_row("kermany", root, cls, cls_dir, f, split.lower()) for f in _images(cls_dir)]
    return rows

def scan_cohen(root):
    rows = []
    for split in ("train", "test"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                rows += [_row("cohen", root, cls, cls_dir, f, split) for f in _images(cls_dir)]
    return rows

def make_splits(df, seed):
    """Stratified 80 / 10 / 10 split (identical procedure for every dataset)."""
    tr, tmp = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label"])
    va, ts = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=tmp["label"])
    return tr.reset_index(drop=True), va.reset_index(drop=True), ts.reset_index(drop=True)

def split_summary(splits: dict, class_names):
    """Class counts per split + patient overlap between splits (when patient IDs exist)."""
    rows = []
    for name, d in splits.items():
        row = {"split": name, "n_images": len(d)}
        row.update({c: int((d["class_name"] == c).sum()) for c in class_names})
        row["n_patients"] = d["patient_id"].nunique() if d["patient_id"].notna().any() else None
        rows.append(row)
    summary = pd.DataFrame(rows)
    leak = {}
    if splits["train"]["patient_id"].notna().any():
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = set(splits[a]["patient_id"].dropna()) & set(splits[b]["patient_id"].dropna())
            leak[f"{a}&{b}"] = dict(shared_patients=len(shared),
                                    images_in_second_split=int(splits[b]["patient_id"].isin(shared).sum()))
    return summary, leak

def load_split_csvs(csv_dir, dataset, seed, root):
    """Reload the train/val/test CSVs of an earlier run. If the saved image paths do not exist on this machine
    (e.g. different Kaggle mount), they are rebuilt from the saved relative paths and the current dataset root."""
    splits = {}
    for name in ("train", "val", "test"):
        f = Path(csv_dir) / f"{dataset}_seed{seed}_{name}.csv"
        if not f.exists():
            raise FileNotFoundError(f"Split CSV not found: {f}")
        d = pd.read_csv(f)
        moved = ~d["image_path"].map(os.path.exists)
        if moved.any():
            d.loc[moved, "image_path"] = d.loc[moved, "rel_path"].map(lambda r: os.path.join(root, r))
            if "mask_path" in d:
                has_mask = moved & d["mask_path"].notna()
                d.loc[has_mask, "mask_path"] = d.loc[has_mask, "rel_path"].map(
                    lambda r: os.path.join(root, r.replace(os.sep + "images" + os.sep, os.sep + "masks" + os.sep)))
        missing = ~d["image_path"].map(os.path.exists)
        if missing.any():
            raise FileNotFoundError(f"{missing.sum()} images of '{name}' split not found, e.g. {d.loc[missing, 'image_path'].iloc[0]}")
        splits[name] = d
    return splits

def split_data():
    if SPLIT_CSV_DIR is None:
        rows = {"rahman": scan_rahman, "kermany": scan_kermany, "cohen": scan_cohen}[DATASET](root)
        original_df = pd.DataFrame(rows)
        class_names = sorted(original_df["class_name"].unique())
        label_map = {name: i for i, name in enumerate(class_names)}
        original_df["label"] = original_df["class_name"].map(label_map)
        original_df.insert(0, "row_id", range(len(original_df)))
        original_df.to_csv(DATA_DIR / f"{DATASET}_original.csv", index=False)
        tr_df, val_df, ts_df = make_splits(original_df, SPLIT_SEED)
        for name, d in {"train": tr_df, "val": val_df, "test": ts_df}.items():
            d.to_csv(DATA_DIR / f"{DATASET}_seed{SPLIT_SEED}_{name}.csv", index=False)
        log.info(f"Original data: {len(original_df)} images | stratified 80/10/10 split, seed {SPLIT_SEED}")
    else:                                           # ---- reuse a saved split ----
        loaded = load_split_csvs(SPLIT_CSV_DIR, DATASET, SPLIT_SEED, root)
        tr_df, val_df, ts_df = loaded["train"], loaded["val"], loaded["test"]
        original_df = pd.concat(loaded.values()).sort_values("row_id").reset_index(drop=True)
        class_names = sorted(original_df["class_name"].unique())
        label_map = {name: i for i, name in enumerate(class_names)}
        assert (original_df["class_name"].map(label_map) == original_df["label"]).all(), "label ids in CSV do not match class names"
        assert not set(tr_df["row_id"]) & set(ts_df["row_id"]) and not set(val_df["row_id"]) & set(ts_df["row_id"]) \
            and not set(tr_df["row_id"]) & set(val_df["row_id"]), "splits overlap"
        log.info(f"Loaded split CSVs from {SPLIT_CSV_DIR} (no re-splitting, no new split CSVs written)")

    splits = {"train": tr_df, "val": val_df, "test": ts_df}
    (OUT_DIR / "class_mapping.json").write_text(json.dumps(label_map, indent=2))
    log.info(f"{len(original_df)} images | label map {label_map}")
    log.info("Patient IDs: " + (f"{original_df['patient_id'].nunique()} unique patients extracted from file names"
                                if original_df["patient_id"].notna().any() else "not available for this dataset"))

    summary, leak = split_summary(splits, class_names)
    summary.to_csv(DATA_DIR / f"{DATASET}_seed{SPLIT_SEED}_split_summary.csv", index=False)
    log.info(f"Split sizes:\n{summary.to_string(index=False)}")
    for pair, v in leak.items():
        log.info(f"Patient overlap {pair}: {v}")

    return tr_df, val_df, ts_df, class_names

def load_data(tr_df, val_df, ts_df):

    _drop = lambda d: d if (USE_MASK or MASK_ONLY) else d.drop(
        columns=["mask_path"], errors="ignore"
    )
    tf_train = JointTransform(IMAGE_SIZE, train=True) if USE_MASK else transform_train
    tf_val = JointTransform(IMAGE_SIZE, train=False) if USE_MASK else transform_val

    train_loader = make_loader(_drop(tr_df), tf_train, shuffle=True)
    val_loader = make_loader(_drop(val_df), tf_val, shuffle=False)
    test_loader = make_loader(_drop(ts_df), tf_val, shuffle=False)
    log.info(f"Loaders: train {len(train_loader.dataset)} | val {len(val_loader.dataset)} | test {len(test_loader.dataset)}"
            f" | batch {BATCH_SIZE} | masks {'ON' if USE_MASK else 'OFF'}")
    return train_loader, val_loader, test_loader