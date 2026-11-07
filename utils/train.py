import time
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from utils.config import *
from utils.log import log

def make_inputs(batch, dataset, device, invert_mask=False, feature_masking=False, input_masking=False):
    """Returns (images, targets, masks). Masking happens on the GPU, after the batch is moved to `device`.
    input_masking   : images = images * masks
    feature_masking : masks are returned and passed to the model (masks the backbone features); else None
    invert_mask     : use the background (1 - mask) instead of the lungs"""
    masks = None
    if len(batch) == 3:
        images, masks, targets = (tensor.to(device) for tensor in batch)
        if invert_mask:
            masks = 1 - masks
        masks = (masks > 0.5).float()
    else:
        images, targets = (tensor.to(device) for tensor in batch)
        unet = getattr(dataset, "unet_model", None)
        if unet is not None:
            unet.eval()
            with torch.no_grad():
                mask_device = next(unet.parameters()).device
                masks = torch.sigmoid(unet(images.to(mask_device))).to(device)
            masks = TF.gaussian_blur(masks, [5, 5], sigma=2.0)
            masks = (masks > 0.5).float()
            if invert_mask:
                masks = 1 - masks
    if masks is not None and input_masking:
        images = images * masks
    return images, targets, (masks if feature_masking else None)


@torch.no_grad()
def evaluate(model, loader, criterion, device, invert_mask=False, feature_masking=FEATURE_MASKING,
             input_masking=INPUT_MASKING):
    """Returns loss (mean over batches), accuracy (%), labels, predictions, softmax probabilities [N, C]."""
    model.eval()
    running_loss, labels, preds, probs = 0.0, [], [], []
    for batch in loader:
        inputs, targets, masks = make_inputs(batch, loader.dataset, device, invert_mask,
                                             feature_masking=feature_masking,
                                             input_masking=input_masking)
        outputs = model(inputs, lung_mask=masks) if masks is not None else model(inputs)

        running_loss += criterion(outputs, targets).item()
        probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
        preds.append(outputs.argmax(1).cpu().numpy())
        labels.append(targets.cpu().numpy())
    labels, preds, probs = np.concatenate(labels), np.concatenate(preds), np.concatenate(probs)
    return running_loss / len(loader), 100.0 * (labels == preds).mean(), labels, preds, probs


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs,
          checkpoint_path, history_csv, feature_masking=FEATURE_MASKING, input_masking=INPUT_MASKING):
    """Trains, logs every epoch (text log + CSV), keeps the best-val-accuracy checkpoint. Returns history DataFrame."""
    best_val_acc, best_epoch, history = 0.0, 0, []
    t_start = time.time()
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            inputs, targets, masks = make_inputs(batch, train_loader.dataset, device,
                                                 invert_mask=USE_INVERTED_MASK,
                                                 feature_masking=feature_masking,
                                                 input_masking=input_masking)
            optimizer.zero_grad()
            outputs = model(inputs, lung_mask=masks) if masks is not None else model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total += targets.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
        train_loss, train_acc = running_loss / len(train_loader), 100.0 * correct / total

        val_loss, val_acc, *_ = evaluate(model, val_loader, criterion, device,
                                         invert_mask=USE_INVERTED_MASK,
                                         feature_masking=feature_masking,
                                         input_masking=input_masking)
        scheduler.step()
        torch.cuda.empty_cache()

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), checkpoint_path)
        history.append(dict(epoch=epoch, lr=lr, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss,
                            val_acc=val_acc, epoch_time_s=time.time() - t0, is_best=is_best))
        pd.DataFrame(history).to_csv(history_csv, index=False)
        log.info(f"Epoch [{epoch:02d}/{num_epochs}] lr {lr:.2e} | train loss {train_loss:.4f} acc {train_acc:.2f}% | "
                 f"val loss {val_loss:.4f} acc {val_acc:.2f}% | {time.time() - t0:.0f}s" + (" | best saved" if is_best else ""))
    log.info(f"Training finished in {(time.time() - t_start) / 60:.1f} min | best val acc {best_val_acc:.2f}% (epoch {best_epoch})")
    return pd.DataFrame(history)