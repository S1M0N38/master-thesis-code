from pathlib import Path
import argparse

# Argparse
parser = argparse.ArgumentParser(
    description="Compute predictions metrics and save to results/predictions.pkl"
)
parser.add_argument(
    "--path_experiment",
    type=Path,
    required=True,
    help="Path to experiment",
)
parser.add_argument(
    "--path_encoding",
    type=Path,
    required=True,
    help="Path to encoding",
)
args = parser.parse_args()

# Check arguments
assert args.path_experiment.exists(), f"{args.path_experiment} does not exists"
assert args.path_encoding.exists(), f"{args.path_encodings} does not exists"
assert args.path_encoding.suffix == ".npy", f"{args.path_encoding} is not a .npy file"


# Paths, dataset, exp, hierarchy, and metrics
path_experiment = args.path_experiment
path_encoding = args.path_encoding
dataset, exp = path_experiment.parts[-2:]
path_root = Path(__file__).parent.parent
path_dataset = path_root / "datasets" / "datasets" / dataset
path_save = path_root / "evals" / dataset / exp / "results" / "predictions.pkl"

# if path_save.exists():
#     print(f"SKIP: {path_save} already exists")
#     quit()

import numpy as np
import pandas as pd
import tqdm

import utils

hierarchy = np.load(path_dataset / "hierarchy" / "hierarchy.npy")
metrics = {
    "error_rate": utils.error_rate,
    "hier_dist_mistake": utils.hier_dist_mistake,
    "hier_dist": utils.hier_dist,
}

index = pd.Index(data=[exp], dtype=str, name="experiments")
levels = range(len(hierarchy) - 1)
columns = pd.MultiIndex.from_product([levels, metrics], names=["level", "metric"])
df = pd.DataFrame(index=index, columns=columns, dtype=float)

# Data
outputs = np.load(path_experiment / "results" / "outputs.npy")
targets = np.load(path_experiment / "results" / "targets.npy")
encodings = np.load(path_encoding)
labels = utils.get_labels(targets, encodings)
predictions = utils.get_predictions(outputs, encodings)

# Calculate metrics
pb_levels = tqdm.tqdm(levels)
for lvl in pb_levels:
    pb_levels.set_description_str(f"Level {lvl + 1}")
    pb_metrics = tqdm.tqdm(metrics.items(), leave=False)
    for metric, func in pb_metrics:
        pb_metrics.set_description_str(metric)
        df.loc[exp, (lvl, metric)] = func(predictions, labels, hierarchy, level=lvl)

# Save Dataframe
path_save.parent.mkdir(exist_ok=True, parents=True)
df.to_pickle(path_save)
