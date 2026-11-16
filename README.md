# Lung disease classification from chest X-ray images with residual attention refinement on multi-scale ResNet features

This repository releases the code, environment, split manifests, and instructions needed to regenerate the results of **Lung disease classification from chest X-ray images with residual attention refinement on multi-scale ResNet features**

- **Proposed implementation (this repository):** [GitHub - phhng/Residual-Attention-Refinement-Model · GitHub](https://github.com/phhng/Residual-Attention-Refinement-Model)
- **Baseline implementation we build on:** [GitHub - amirrezafateh/Multi-Scale-Transformer-Pneumonia · GitHub](https://github.com/amirrezafateh/Multi-Scale-Transformer-Pneumonia)

---
## Repository layout

```
csv/
	cohen/ # contain train-val-test csv
	kermany/
	rahman/
	...
model/
		weights/ # containt .pth model weight
		...
    Backbone.py
    Transformer.py
    MSR.py
data/ # contain data loader and splits from .csv file/original dataset
	dataset_loader.py
	joint_dataset.py
	patient_split.py
model/
	/weights # trained weights folder
	Backbone.py
	MSR.py
	Transformer.py
model_details_svg/ # contain model architecture
	...
notebooks/
	cm_graphics.py # generate confusion matrices from .csv files in results
	cross-dataset-eval.ipynb # cross-dataset evaluation
	gradcam_compare.ipynb # generate heatmap example from loaded weight from conv_3
	train_log_graphics # generate graphics from training logs
results/ # contain results in .csv file across experiments
utils/
    complexity.py
    config.py
    load_model.py # load model from check point and strip unused component in current model configuration
    log.py # init before all run
    report_results.py 
    run.py
    train.py
   
main.py # run this to run the full experiment, default run is single seed
model_architecture.py # return model's architecture along with in and output
probabilities_per_class.py # compute raw softmax probabilities from cross-dataset evaluation
requirements.txt
tensor_check.py # check what model receives after every masking experiment
```



---
## Environment

| Item           | Version                              |
| -------------- | ------------------------------------ |
| Python         | 3.13.16                              |
| PyTorch        | 2.11.0+cu130                         |
| torchvision    | 0.26.0+cu130                         |
| CUDA / cuDNN   | 13.0 / 92700                         |
| OS             | Linux-6.18.55+-x86_64-with-glibc2.39 |
| Other packages | see `requirements.txt`               |

**Hardware used**

| Item      | Value                                  |
| --------- | -------------------------------------- |
| GPU       | NVIDIA T4 x 2 on Kaggle                |
| CPU / RAM | Intel(R) Xeon(R) CPU @ 2.00GHz / 31 GB |

---
## Data

Three public Kaggle datasets are used. **No images are redistributed.** The repository contains only download code, split manifests (image identifiers and labels, included only where the source terms permit), and SHA-256 hashes of each split file.

| ID        | Dataset                            | Kaggle slug                                   | Classes | Patient IDs | Masks provided |
| --------- | ---------------------------------- | --------------------------------------------- | ------- | ----------- | -------------- |
| `rahman`  | COVID-19 Radiography Database      | `tawsifurrahman/covid19-radiography-database` | 4       | No          | Yes            |
| `cohen`   | Chest X-ray (Covid-19 & Pneumonia) | `prashant268/chest-xray-covid19-pneumonia`    | 3       | No          | No             |
| `kermany` | Chest X-Ray Images (Pneumonia)     | `paultimothymooney/chest-xray-pneumonia`      | 2       | Yes         | No             |

### Sources, licenses, citations

For each dataset, copy the license and citation text **from its Kaggle page on the day of release**, since terms can change, and keep the access date.

| ID        | License / terms (copy from page)                          | Original citation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Accessed  | Version |
| --------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------- |
| `rahman`  | Data files © Original Authors                             | Available on [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)<br>[Exploring the effect of image enhancement techniques on COVID-19 detection using chest X-ray images](https://www.sciencedirect.com/science/article/pii/S001048252100113X?via%3Dihub)<br>[Can AI Help in Screening Viral and COVID-19 Pneumonia?](https://www.researchgate.net/publication/343094700_Can_AI_Help_in_Screening_Viral_and_COVID-19_Pneumonia)                                                   | 7/11/2026 | 5       |
| `cohen`   | Unknown                                                   | Available on [Kaggle](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)<br>[Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)<br>[Covid Chest X-ray dataset - github: ieee8023](https://github.com/ieee8023/covid-chestxray-dataset) - [Paper: COVID-19 image data collection](https://arxiv.org/abs/2003.11597) <br>[COVID-chestxray-dataset - github: agchung](https://github.com/agchung/Figure1-COVID-chestxray-dataset) | 7/11/2026 | 2       |
| `kermany` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Available on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)<br>[Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)<br>[Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification](https://data.mendeley.com/datasets/rscbjbr9sj/2)                                                                                                                           | 7/11/2026 | 2       |

### Download

Requires a Kaggle account and API token (`~/.kaggle/kaggle.json`, or Kaggle notebook inputs). Accept each dataset's terms on its page first.

```bash
pip install kaggle
mkdir -p data
kaggle datasets download -d tawsifurrahman/covid19-radiography-database -p data/rahman --unzip
kaggle datasets download -d prashant268/chest-xray-covid19-pneumonia   -p data/chest_covid_pneumonia --unzip
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia     -p data/chest_pneumonia --unzip
```

Otherwise, modify `utils/config.py` to choose dataset to run. Then run `py ./main/py` to start the full experiment (including loading from .csv file, train and evaluate model)

### Splits

Splits are loaded from CSV manifests and not regenerated, so every run sees identical partitions.

| Dataset   | Split rule                                                 | Split seed | Manifest            |
| --------- | ---------------------------------------------------------- | ---------- | ------------------- |
| `rahman`  | class-stratified, 80/10/10 (16,932 / 2,116 / 2,117 images) | 42         | `csv/rahman/*.csv`  |
| `cohen`   | class-stratified, 80/10/10 (5,145/643/644 images)          | 42         | `csv/cohen/*.csv`   |
| `kermany` | class-stratified, 80/10/10 (4684/586/586 images)           | 42         | `csv/kermany/*.csv` |

- `rahman` has no patient identifiers, so its split is image-level and images from the same patient may appear in different partitions.
- `cohen` is a composite of Kermany, Cohen and additional dataset by agchung.
- `kermany` has patient identifiers, however, it was not applied in our main study. grouped patient split can be found in `csv/grouped_kermany/*`

### Class dictionaries and counts

```python
# rahman
CLASS_TO_IDX = {
  "COVID": 0,
  "Lung_Opacity": 1,
  "Normal": 2,
  "Viral Pneumonia": 3
}
# cohen
CLASS_TO_IDX = {
  "COVID19": 0,
  "NORMAL": 1,
  "PNEUMONIA": 2
}
# kermany
CLASS_TO_IDX = {
  "NORMAL": 0,
  "PNEUMONIA": 1
}
```

| Dataset   | Split | n      | per-class counts                                                        |
| --------- | ----- | ------ | ----------------------------------------------------------------------- |
| `rahman`  | train | 16,932 | COVID 2,893 / LUNG OPACITY 4,809 / NORMAL 8,154 / VIRAL PNEUMONIA 1,076 |
| `rahman`  | val   | 2,116  | 361 / 601 / 1,019 / 135                                                 |
| `rahman`  | test  | 2,117  | 362 / 602 / 1,019 / 134                                                 |
| `cohen`   | train | 5,145  | COVID19 461 / NORMAL 1266 / PNEUMONIA 3418                              |
| `cohen`   | val   | 643    | 58 / 158 / 427                                                          |
| `cohen`   | test  | 644    | 57 / 159 / 428                                                          |
| `kermany` | train | 4684   | NORMAL 1266 / PNEUMONIA 3418                                            |
| `kermany` | val   | 586    | 159 / 427                                                               |
| `kermany` | test  | 586    | 158 / 428                                                               |

### Lung masks

- `rahman` can be run with 4 configuration, modify in `/utils/config.py`: 
	- If use mask, turn `USE_MASK = True`:
		- Input masking (`img * mask`): `INPUT_MASK = True`
		- Feature-level masking (`red_cat = red_cat * m_cat; red_3 = red_3 * m_3`): `FEATURE_MASK = True`
		- Inverted mask (background only `img * (1 - mask)`): `USE_INVERTED_MASK = True`
	- Otherwise keep all config = `False`
- For other datasets included in this study, masks were not provided nor generated with U-Net
- Where used, the mask is resized with nearest-neighbour interpolation to feature resolution and multiplied into the reduced features before the enhancement blocks (feature-level masking only). 
	- Otherwise, after resizing the masks are converted to tensor and loaded with respective images.

---

## Preprocessing

| Step               | Setting                                              |
| ------------------ | ---------------------------------------------------- |
| Channel handling   | convert("RGB")                                       |
| Resize             | (224, 224), interpolation = BICUBIC                  |
| Mask preprocessing | resized with nearest-neighbour to feature resolution |

### Augmentation applied in training, in order

1. RandomHorizontalFlip(p=0.5)
2. RandomResizedCrop(degrees=15)
3. ColorJitter(brightness=0.2, contrast=0.2)
4. ToTensor

Validation/test: resize -> ToTensor.
If images and masks are augmented, applied a joint transformation to keep mask and images match augmentation.

---

## Model Init

Implementation: `model/MSR.py`, `model/Transformer.py`, `model/Backbone.py`.

### Backbone and pretrained weights

- Architecture: `resnet{layers}` with `layers` = 50 (the `conv_red_*` input widths 512/1024/2048 correspond to stage outputs of a bottleneck ResNet).
- Pretrained weights: `IMAGENET1K_V1`
- `dilation=[False, True, True]`, `return_interm_layers=True`.
- `train_backbone=False`: the **backbone is frozen**.
- **Batch-norm mode in the backbone:** `FrozenBatchNorm2d`

### Trainable vs frozen

| Module                                   | Trainable | BN mode                              |
| ---------------------------------------- | --------- | ------------------------------------ |
| `backbone`                               | No        | `FrozenBatchNorm2d`                  |
| `conv_red_1/2/3` (1x1 conv + BN + ReLU)  | Yes       | train / eval follows `model.train()` |
| `enhancement_cat`, `enhancement_3` (FEB) | Yes       | BN follows `model.train()`           |
| `transformer_cat`, `transformer_3`       | Yes       | `LayerNorm`                          |
| `classifier`                             | Yes       | follows `model.train()`              |

### Key hyperparameters

| Component                        | Setting                                                                                                                                                           |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `reduce_dim`                     | 256                                                                                                                                                               |
| `conv_red_1`, `conv_red_2`       | 512 -> 128, 1024 -> 128 (concatenated to 256); `conv_red_3`: 2048 -> 256                                                                                          |
| FEB                              | two 3x3 conv + BN residual block with 1x1 skip when widths differ, then CRAM; the block output is added to its input in `MSR.forward`                             |
| ECA                              | 1-D conv, `k_size=3`. **Note:** `ECALayer` returns `x * w` and `CRAM` multiplies by `x` again, so the channel step is `x * x * w`. This is the variant evaluated. |
| DSA                              | 3x3 conv on [avg, max] descriptors, then 7x7 dilated conv (dilation 3), sigmoid map                                                                               |
| Transformer encoder              | `num_encoder_layers=1`, `nhead=4`, `dim_feedforward=64`, `dropout=0.1`, post-norm (`normalize_before=False`), ReLU                                                |
| Positional encoding              | `PositionEmbeddingSine(num_pos_feats=128, normalize=True)`                                                                                                        |
| Pooling (ENCODER_CONFIG = True)  | masked mean over tokens (all tokens kept, since no mask is passed), then `out_norm` LayerNorm                                                                     |
| Pooling (ENCODER_CONFIG = False) | `AdaptiveAvgPool2d(1)` then `PooledNorm` (LayerNorm)                                                                                                              |
| Classifier                       | Linear(512,256)-BN1d-ReLU-Dropout(0.5)-Linear(256,128)-ReLU-Dropout(0.3)-Linear(128,`num_classes`)                                                                |

---

## Training

| Setting              | Value                                                                                                                                                             |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Loss                 | `FocalLossCE(gamma=2.0, alpha=None, reduction="mean")`: `loss = (1 - pt)^gamma * CE`, `pt = exp(-CE)`, per-sample CE with `reduction="none"`, no class weighting. |
| Optimizer            | `AdamW, lr=1e-3, weight decay = 1e-4`                                                                                                                             |
| Scheduler            | `LinearLR(1e-6->1) warm-up + CosineAnnealing(eta_min=1e-6)`                                                                                                       |
| Scheduler stepping   | `warmup_epochs=10`                                                                                                                                                |
| Batch size           | 64 (128 for x 2 GPUs)                                                                                                                                             |
| Epochs               | 30                                                                                                                                                                |
| Checkpoint criterion | val_acc                                                                                                                                                           |
| Test-set use         | Test set evaluated once with the latest model                                                                                                                     |

### Seeds and determinism

```python
SEED = 42
SPLIT_SEED = 42
def set_seed(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms(True)
```

Seeds used: 42, 1, 123, 7, 777. Results are reported as mean ± SD over 5 seeds.

Note: some earlier runs used a code version that instantiated extra unused parameters. Those runs are statistically comparable but not bit-for-bit reproducible with this code.

---

## Reproducing results

### Commands

```bash
# Setup
pip install -r requirements.txt

# Configure settings in utils/config.py
# To run the experiment, run the following command. With train_model.py default to training single seed (42)
py .\train_model.py

```

