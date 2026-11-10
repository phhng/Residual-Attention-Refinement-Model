#!/usr/bin/env python
"""Write a confusion-matrix-style table of mean model probabilities per true class.

Reads three-class Cohen per-image softmax files, including ``norahman`` variants,
from ``results/softmax`` and writes a percentage-only matrix beside them. The
grouped means describe the model's average probability distribution, not accuracy
or calibration.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "results" / "softmax"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "probabilities_per_class_matrix.csv"
RAW_COLUMNS = ("raw_0", "raw_1", "raw_2", "raw_3")
OUTPUT_CLASSES = ("COVID", "Lung Opacity", "Normal", "Viral Pneumonia")
COHEN_SUFFIX = "_cohen3_test__softmax_probabilities.csv"
NORAHMAN_SUFFIX = "_cohen3_test_norahman__softmax_probabilities.csv"


def label_names_for(path):
    """Return external label names for datasets represented in the filename."""
    name = path.name.casefold()
    if "cohen3" in name:
        return {0: "COVID-19", 1: "Normal", 2: "Pneumonia"}
    if "kermany" in name:
        return {0: "Normal", 1: "Pneumonia"}
    raise ValueError("cannot determine external label mapping from filename: %s" % path.name)


def summarize_file(path):
    label_names = label_names_for(path)
    totals = defaultdict(lambda: [0, [0.0] * len(RAW_COLUMNS)])
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = set(RAW_COLUMNS + ("label",)) - set(reader.fieldnames or ())
        if missing:
            raise ValueError("%s is missing required columns: %s" % (path, ", ".join(sorted(missing))))

        for line_number, row in enumerate(reader, start=2):
            try:
                label = int(row["label"])
                probabilities = [float(row[column]) for column in RAW_COLUMNS]
            except (TypeError, ValueError) as exc:
                raise ValueError("%s:%d has an invalid label or probability" % (path, line_number)) from exc
            if label not in label_names:
                raise ValueError("%s:%d has unknown external label %d" % (path, line_number, label))
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
                raise ValueError("%s:%d has a probability outside [0, 1]" % (path, line_number))

            count, sums = totals[label]
            totals[label] = [count + 1, [total + value for total, value in zip(sums, probabilities)]]

    if path.name.endswith(NORAHMAN_SUFFIX):
        model_name = path.name[:-len(NORAHMAN_SUFFIX)] + "_norahman"
    elif path.name.endswith(COHEN_SUFFIX):
        model_name = path.name[:-len(COHEN_SUFFIX)]
    else:
        raise ValueError("unexpected Cohen softmax filename: %s" % path.name)
    raw_to_output_class = dict(zip(RAW_COLUMNS, OUTPUT_CLASSES))
    rows = []
    for label in sorted(totals):
        count, sums = totals[label]
        raw_means = {column: total / count for column, total in zip(RAW_COLUMNS, sums)}
        row = {"True External Class": label_names[label], "Model": model_name}
        row.update({
            raw_to_output_class[column]: "%.2f%%" % (probability * 100.0)
            for column, probability in raw_means.items()
        })
        rows.append((label, model_name, row))
    return rows


def main():
    files = sorted(
        path for path in DEFAULT_INPUT_DIR.glob("*_cohen3_test*__softmax_probabilities.csv")
        if path.name.endswith((COHEN_SUFFIX, NORAHMAN_SUFFIX))
    )
    if not files:
        raise FileNotFoundError("no three-class Cohen per-image softmax CSV files found in %s" %
                                DEFAULT_INPUT_DIR)

    rows = []
    for path in files:
        rows.extend(summarize_file(path))

    rows.sort(key=lambda item: (item[0], item[1]))
    columns = ["True External Class", "Model", *OUTPUT_CLASSES]
    with DEFAULT_OUTPUT.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(row for _, _, row in rows)

    print("Summarized %d model file(s) across %d true-class/model rows: %s" % (
        len(files), len(rows), DEFAULT_OUTPUT))


if __name__ == "__main__":
    main()