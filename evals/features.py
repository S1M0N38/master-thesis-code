from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from s_dbw import S_Dbw as sdbw_score
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
import tqdm

# Argparse
parser = argparse.ArgumentParser(
    description="Compute features metrics and save to results/features.pkl"
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
path_save = path_root / "evals" / dataset / exp / "results" / "features.pkl"
hierarchy = np.load(path_dataset / "hierarchy" / "hierarchy.npy")
metrics = {
    "silhouette": silhouette_score,
    "calinski_harabasz": calinski_harabasz_score,
    "davies_bouldin": davies_bouldin_score,
    "sdbw": sdbw_score,
}

if path_save.exists():
    quit()

# Dataframe
index = pd.Index(data=[exp], dtype=str, name="experiments")
levels = range(len(hierarchy) - 1)
columns = pd.MultiIndex.from_product([levels, metrics], names=["level", "metric"])
df = pd.DataFrame(index=index, columns=columns, dtype=float)

# Data
features = np.load(path_experiment / "results" / "features.npy")
outputs = np.load(path_experiment / "results" / "outputs.npy")
encodings = np.load(path_encoding)

# From outputs to labels
if outputs.shape[-1] == 1:
    outputs = np.eye(hierarchy.shape[-1])[outputs.squeeze().astype(int)]
outputs /= np.linalg.norm(outputs, axis=1, keepdims=True)
encodings /= np.linalg.norm(encodings, axis=1, keepdims=True)
labels = (outputs @ encodings.T).argmax(axis=-1)

# Calculate metrics
pb_levels = tqdm.tqdm(levels)
for lvl in pb_levels:
    pb_levels.set_description_str(f"Level {lvl + 1}")
    pb_metrics = tqdm.tqdm(metrics.items(), leave=False)
    for metric, func in pb_metrics:
        pb_metrics.set_description_str(metric)
        df.loc[exp, (lvl, metric)] = func(features, hierarchy[lvl][labels])

# Save Dataframe
path_save.parent.parent.mkdir(exist_ok=True)
path_save.parent.mkdir(exist_ok=True)
df.to_pickle(path_save)
