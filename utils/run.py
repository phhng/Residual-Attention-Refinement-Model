from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset_loader import split_data, load_data
from utils.log import *
from utils.config import *
from utils.train import train, evaluate
from model.MSR import *
from utils.report_results import *

def run_experiment(seed, run_dir):
    
    tr_df, val_df, ts_df, class_names = split_data()
    train_loader, val_loader,test_loader = load_data(tr_df, val_df, ts_df)
    num_classes = len(class_names)
    """Train with `seed` on the fixed seed-42 split, evaluate on the test split, write all logs/reports to run_dir."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"===== Experiment: seed {seed} -> {run_dir} =====")
    set_seed(seed)

    model = nn.DataParallel(MSR(layers=LAYERS, num_classes=num_classes, use_eca=USE_ECA, use_dsa=USE_DSA)).to(device)
    # model = MSR(layers=LAYERS, num_classes=num_classes, use_eca=USE_ECA, use_dsa=USE_DSA).to(device)
    criterion = FocalLossCE()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=NUM_WARMUP)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - NUM_WARMUP, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[NUM_WARMUP])

    ckpt = run_dir / "best_model.pth"
    history = train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, NUM_EPOCHS,
                    ckpt, run_dir / "training_log.csv", feature_masking=FEATURE_MASKING,
                    input_masking=INPUT_MASKING)
    torch.save(model.state_dict(), run_dir / "last_model.pth")
    plot_training_curves(history, run_dir / "training_curves.png")

    if EVAL_CHECKPOINT == "best":
        model.load_state_dict(torch.load(ckpt, map_location=device))
    log.info(f"Evaluating the {EVAL_CHECKPOINT} model on the test split")
    test_loss, test_acc, labels, preds, probs = evaluate(model, test_loader, criterion, device,
                                                         invert_mask=USE_INVERTED_MASK,
                                                         feature_masking=FEATURE_MASKING,
                                                         input_masking=INPUT_MASKING)
    log.info(f"Test loss {test_loss:.4f} | test accuracy {test_acc:.2f}%")
    metrics = report_results(labels, preds, probs, class_names, ts_df, run_dir, test_loss)
    return model, metrics

def run_multi_seed(seeds=(42, 123, 7, 1, 777)):
    rows = []
    for s in seeds:
        _, m = run_experiment(s, OUT_DIR / f"seed_{s}")
        rows.append(dict(seed=s, **m))
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "multi_seed_results.csv", index=False)
    cols = [c for c in res.columns if c not in ("seed", "test_loss")]
    summary = pd.DataFrame({"mean": res[cols].mean() * 100, "std": res[cols].std(ddof=1) * 100})
    summary.to_csv(OUT_DIR / "multi_seed_summary.csv")
    log.info("Mean ± std over seeds (%):\n" + "\n".join(f"{k}: {r['mean']:.2f} ± {r['std']:.2f}" for k, r in summary.iterrows()))
    return res, summary


# model, metrics = run_experiment(SEED, OUT_DIR)
# res, summary = run_multi_seed()