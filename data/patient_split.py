from sklearn.model_selection import StratifiedGroupKFold
from data.dataset_loader import split_summary
from utils.log import *
from utils.config import *

def make_patient_grouped_splits(df, seed=42, no_dupe=False):
    data = df.copy().reset_index(drop=True)

    # Validate patient IDs
    if data["patient_id"].isna().any():
        missing = data["patient_id"].isna().sum()
        raise ValueError(
            f"{missing} images have no patient ID. "
            "Patient-level splitting cannot be guaranteed."
        )

    # Ensure each patient has only one label
    patient_label_counts = data.groupby("patient_id")["label"].nunique()

    if (patient_label_counts > 1).any():
        raise ValueError(
            "Some patient IDs have multiple labels. "
            "Review patient identification and label assignments."
        )

    # Optional: keep only one image per patient
    if no_dupe:
        data = (
            data.sort_values("filename")
                .drop_duplicates(subset="patient_id", keep="first")
                .reset_index(drop=True)
        )

        print(f"Duplicate prevention enabled.")
        print(f"Images remaining: {len(data)}")
        print(f"Unique patients: {data['patient_id'].nunique()}")

    # Patient-grouped train/test split
    X = data.index.to_numpy()
    y = data["label"].to_numpy()
    groups = data["patient_id"].to_numpy()

    outer_cv = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=seed
    )

    dev_idx, test_idx = next(outer_cv.split(X, y, groups))

    dev_data = data.iloc[dev_idx].reset_index(drop=True)

    # Patient-grouped train/validation split
    inner_cv = StratifiedGroupKFold(
        n_splits=9,
        shuffle=True,
        random_state=seed
    )

    dev_X = dev_data.index.to_numpy()
    dev_y = dev_data["label"].to_numpy()
    dev_groups = dev_data["patient_id"].to_numpy()

    train_idx, val_idx = next(
        inner_cv.split(dev_X, dev_y, dev_groups)
    )

    train_df = dev_data.iloc[train_idx].reset_index(drop=True)
    val_df = dev_data.iloc[val_idx].reset_index(drop=True)
    test_df = data.iloc[test_idx].reset_index(drop=True)

    # Verify patient separation
    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    test_patients = set(test_df["patient_id"])

    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)

    # Verify one image per patient when enabled
    if no_dupe:
        assert train_df["patient_id"].is_unique
        assert val_df["patient_id"].is_unique
        assert test_df["patient_id"].is_unique

    return train_df, val_df, test_df

# Create a separate patient-grouped experiment
def patient_split_experiment(df, seed=SPLIT_SEED, no_dupe=NO_DUPE, original_df=None, class_names=None):
    patient_train, patient_val, patient_test = (
        make_patient_grouped_splits(
            original_df,
            seed=SPLIT_SEED,
            no_dupe=NO_DUPE
        )
    )

    patient_splits = {
        "train": patient_train,
        "val": patient_val,
        "test": patient_test
    }

    # Save separately; do not overwrite existing CSVs
    for name, data in patient_splits.items():
        data.to_csv(
            DATA_DIR / f"kermany_patient_grouped_seed{SPLIT_SEED}_{name}.csv",
            index=False
        )

    # Generate split summary
    summary, leak = split_summary(
        patient_splits,
        class_names
    )

    summary.to_csv(
        DATA_DIR / f"kermany_patient_grouped_seed{SPLIT_SEED}_summary.csv",
        index=False
    )

    print(summary.to_string(index=False))
    print("Patient overlap:", leak)

# patient_split_experiment(original_df, seed=SPLIT_SEED, no_dupe=NO_DUPE, original_df=original_df, class_names=class_names)