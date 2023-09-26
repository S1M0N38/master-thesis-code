import sys
from pathlib import Path

import numpy as np
import pandas as pd
from s_dbw import S_Dbw as sdbw_score
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from tqdm import tqdm, trange

# Avaialve datasets: CIFAR100, iNaturalist19, tieredImageNet
DATASET = sys.argv[1]

# Path
PATH_ROOT = Path(__file__).parent.parent
PATH_EXPERIMENTS = PATH_ROOT / "experiments" / DATASET
PATH_DATASET = PATH_ROOT / "datasets" / DATASET
PATH_ENCODINGS = PATH_DATASET / "encodings"
PATH_RESULTS = PATH_ROOT / "evaluation" / "results" / DATASET

HIERARCHY = np.load(PATH_DATASET / "hierarchy" / "hierarchy.npy")


def features_labels(exp):
    # Load test results
    data = np.load(PATH_EXPERIMENTS / exp / "features_targets.npz")
    features, targets = data["features"], data["targets"]

    # For one-hot encoding targets are already class index (aka labels)
    # Convert back to one hot encoding to be consistent with others encondings
    if targets.shape[-1] == 1:
        labels = targets.squeeze().astype(int)
        targets = np.eye(HIERARCHY.shape[-1])[labels]

    # Select the encoder matrix
    encs = np.load(exps.loc[exp, "encoding"])

    # Normalize quantites
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    encs /= np.linalg.norm(encs, axis=1, keepdims=True)

    labels = (targets @ encs.T).argmax(axis=-1)
    return features, labels


exps = pd.read_csv(
    PATH_EXPERIMENTS / "experiments.csv",
    index_col="id",
    converters={"encoding": lambda enc: PATH_ENCODINGS / enc},
    comment="#",
)

index = pd.Index(
    data=exps[exps["selected"]].index,
    dtype=str,
    name="experiments",
)

columns = pd.MultiIndex.from_product(
    iterables=[
        range(len(HIERARCHY) - 1),
        ["silhouette", "calinski_harabasz", "davies_bouldin", "sdbw"],
    ],
    names=["level", "metric"],
)

df = pd.DataFrame(index=index, columns=columns, dtype=float)

# Load previous calculations
try:
    old_df = pd.read_pickle(PATH_RESULTS / "features" / "metrics.pkl")
    done = old_df.index.intersection(df.index)
    for idx, row in old_df.iterrows():
        if idx in df.index:
            df.loc[idx] = row
except FileNotFoundError:
    done = pd.Index([], dtype=str)

# Perform new calculations
todo = df.loc[df.index.difference(done)]
pbar = tqdm(todo.iterrows(), total=len(todo), desc="Experiments")  # type: ignore
for exp, row in pbar:  # type: ignore
    f, labels = features_labels(exp)
    levels = trange(len(HIERARCHY) - 1, leave=False, desc=str(exp))
    for lvl in levels:
        labels = HIERARCHY[lvl][labels]
        df.loc[exp, (lvl, "silhouette")] = silhouette_score(f, labels)
        df.loc[exp, (lvl, "calinski_harabasz")] = calinski_harabasz_score(f, labels)
        df.loc[exp, (lvl, "davies_bouldin")] = davies_bouldin_score(f, labels)
        df.loc[exp, (lvl, "sdbw")] = sdbw_score(f, labels)


df.to_pickle(PATH_RESULTS / "features" / "metrics.pkl")
