import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from utils.log import log

def plot_training_curves(history, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["epoch"], history["train_acc"], label="Train")
    ax[0].plot(history["epoch"], history["val_acc"], label="Validation")
    ax[0].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
    ax[1].plot(history["epoch"], history["train_loss"], label="Train")
    ax[1].plot(history["epoch"], history["val_loss"], label="Validation")
    ax[1].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.show(); plt.close(fig)


def compute_metrics(labels, preds, probs, n_cls):
    probs = probs.astype(np.float64)
    probs = probs / probs.sum(1, keepdims=True)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    m = dict(accuracy=accuracy_score(labels, preds), macro_precision=p, macro_recall=r, macro_f1=f1)
    if n_cls == 2:
        m["auc"] = roc_auc_score(labels, probs[:, 1])
    else:
        y = label_binarize(labels, classes=list(range(n_cls)))
        fpr, tpr, _ = roc_curve(y.ravel(), probs.ravel())
        m["auc_micro_ovr"] = auc(fpr, tpr)
        m["auc_macro_ovr"] = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    return m


def plot_roc(labels, probs, class_names, path):
    n_cls = len(class_names)
    plt.figure(figsize=(8, 6))
    if n_cls == 2:
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr, tpr):.4f}")
    else:
        y = label_binarize(labels, classes=list(range(n_cls)))
        for i, name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y[:, i], probs[:, i])
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc(fpr, tpr):.4f})")
        fpr, tpr, _ = roc_curve(y.ravel(), probs.ravel())
        plt.plot(fpr, tpr, "--", lw=2, label=f"Micro-average (AUC = {auc(fpr, tpr):.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim(0, 1); plt.ylim(0, 1.05); plt.grid(True)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.legend(loc="lower right")
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.show(); plt.close()


def report_results(labels, preds, probs, class_names, test_df, run_dir, test_loss=None):
    """Saves + logs: classification report, confusion matrix, metrics.json, ROC, per-image predictions."""
    run_dir, n_cls = Path(run_dir), len(class_names)
    idx = list(range(n_cls))

    # classification report
    report_txt = classification_report(labels, preds, labels=idx, target_names=class_names, digits=4, zero_division=0)
    report_csv = pd.DataFrame(classification_report(labels, preds, labels=idx, target_names=class_names,
                                                    output_dict=True, zero_division=0)).T
    (run_dir / "classification_report.txt").write_text(report_txt)
    report_csv.to_csv(run_dir / "classification_report.csv")
    log.info("Classification report:\n" + report_txt)

    # confusion matrix (counts + row-normalised)
    cm = confusion_matrix(labels, preds, labels=idx)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in class_names], columns=[f"pred_{c}" for c in class_names])
    cm_df.to_csv(run_dir / "confusion_matrix.csv")
    log.info("Confusion matrix (rows = true, cols = predicted):\n" + cm_df.to_string())
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(1, 2, figsize=(5 + 3 * n_cls, 4 + 0.6 * n_cls))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax[0])
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", vmin=0, vmax=1, xticklabels=class_names,
                yticklabels=class_names, ax=ax[1])
    for a, t in zip(ax, ("Confusion matrix (counts)", "Confusion matrix (row-normalised)")):
        a.set(xlabel="Predicted", ylabel="True", title=t)
        a.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(run_dir / "confusion_matrix.png", dpi=200); plt.show(); plt.close(fig)

    # metrics, ROC, per-image predictions
    metrics = compute_metrics(labels, preds, probs, n_cls)
    if test_loss is not None:
        metrics["test_loss"] = test_loss
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("Test metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    plot_roc(labels, probs, class_names, run_dir / "roc_curve.png")

    pred_df = test_df[["row_id", "class_name", "filename", "image_path", "patient_id"]].copy()
    pred_df["pred_class"] = [class_names[i] for i in preds]
    pred_df["correct"] = labels == preds
    for i, c in enumerate(class_names):
        pred_df[f"prob_{c}"] = probs[:, i]
    pred_df.to_csv(run_dir / "test_predictions.csv", index=False)
    return metrics