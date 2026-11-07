from pathlib import Path
import kagglehub
import torch

# ============================== CONFIG ==============================
DATASET = "kermany"         # "rahman" (4 classes) | "kermany" (2 classes) | "cohen" (3 classes)
SPLIT_SEED = 42
SEED = 42
SPLIT_CSV_DIR = "./csv/kermany"

AUGMENT = False

ENCODER_CONFIG = False

USE_MASK = False            # enabled mask use
INPUT_MASKING = False       # mask the image on the GPU: img * mask (default way to use the mask)
FEATURE_MASKING = False     # mask the reduced backbone features instead / as well
USE_INVERTED_MASK = False   # True: background-only (1 - lung mask) for train, val and test
MASK_ONLY = False

LAYERS = 50
USE_ECA, USE_DSA = True, False

NO_DUPE = False

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 128            # use 64 for single GPU test
NUM_EPOCHS = 30
NUM_WARMUP = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4

EVAL_CHECKPOINT = "last"

DATASETS = {
    "rahman":  dict(kaggle="tawsifurrahman/covid19-radiography-database", root="COVID-19_Radiography_Dataset"),
    "kermany": dict(kaggle="paultimothymooney/chest-xray-pneumonia",      root="chest_xray"),
    "cohen":   dict(kaggle="prashant268/chest-xray-covid19-pneumonia",    root="Data"),
}
assert DATASET in DATASETS
assert not USE_MASK or INPUT_MASKING or FEATURE_MASKING, "USE_MASK needs INPUT_MASKING and/or FEATURE_MASKING"
assert not (USE_MASK and DATASET != "rahman"), "masks only exist for the Rahman dataset"

MASK_TAG = ("-mask" + ("-in" if INPUT_MASKING else "") + ("-feat" if FEATURE_MASKING else "")
            + ("-inv" if USE_INVERTED_MASK else "")) if USE_MASK else ""
RUN_NAME = f"fe-transres50-{DATASET}{MASK_TAG}-seed{SEED}"
BASE_OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
OUT_DIR = BASE_OUT / RUN_NAME
DATA_DIR = OUT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = kagglehub.dataset_download(DATASETS[DATASET]["kaggle"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")