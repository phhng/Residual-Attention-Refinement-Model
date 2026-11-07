import json, logging, random, sys
from pathlib import Path
from utils.config import *
import numpy as np
import torch

def setup_logging(out_dir: Path) -> logging.Logger:
    """Log to <out_dir>/run.log and to stdout."""
    logger = logging.getLogger("run")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(out_dir / "run.log", mode="a"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

log = setup_logging(OUT_DIR)
CONFIG = dict(
    run_name=RUN_NAME, dataset=DATASET, split_seed=SPLIT_SEED, split_csv_dir=str(SPLIT_CSV_DIR), seed=SEED, 
    use_mask=USE_MASK, input_masking=INPUT_MASKING, feature_masking=FEATURE_MASKING, use_inverted_mask=USE_INVERTED_MASK, 
    layers=LAYERS, use_eca=USE_ECA, use_dsa=USE_DSA, augment=AUGMENT,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
    warmup_epochs=NUM_WARMUP, lr=LR, weight_decay=WEIGHT_DECAY, eval_checkpoint=EVAL_CHECKPOINT,
    focal_gamma=2.0, focal_alpha=None, optimizer="AdamW",
    scheduler="LinearLR(1e-6->1) warm-up + CosineAnnealing(eta_min=1e-6)",
)
(OUT_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2))
log.info(f"Run: {RUN_NAME}")
log.info(f"Config: {json.dumps(CONFIG)}")
log.info(f"torch {torch.__version__} | device {device} | GPUs {torch.cuda.device_count()}"
         + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
log.info(f"Dataset '{DATASET}' downloaded to {DATA_PATH}")
set_seed(SEED)