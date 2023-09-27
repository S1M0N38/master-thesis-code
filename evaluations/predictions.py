import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

# Avaialve datasets: CIFAR100, iNaturalist19, tieredImageNet
DATASET = sys.argv[1]

# Path
PATH_ROOT = Path(__file__).parent.parent
PATH_EXPERIMENTS = PATH_ROOT / "experiments" / DATASET
PATH_DATASET = PATH_ROOT / "datasets" / "datasets" / DATASET
PATH_ENCODINGS = PATH_DATASET / "encodings"
PATH_RESULTS = PATH_ROOT / "evaluations" / "results" / DATASET

HIERARCHY = np.load(PATH_DATASET / "hierarchy" / "hierarchy.npy")


def hierarchy_to_lca(hierarchy: np.ndarray) -> np.ndarray:
    """
    Converts a hierarchy to a Least Common Ancestor (LCA) matrix.

    The LCA matrix is a square matrix where each element (i, j) represents
    the level of the least common ancestor for classes i and j.

    Args:
        hierarchy (np.array or torch.Tensor): A matrix where each row represents
            the ancestor hierarchy of a class.

    Returns:
        A square numpy array containing the LCA matrix.
    """
    # Number of hierarchy levels (L)
    # Number of finer classes (C)
    L, C = hierarchy.shape

    lca = np.full((C, C), L, dtype=int)

    for level in hierarchy:
        for row, coarse in zip(lca, level):
            for index, value in enumerate(level):
                if coarse == value:
                    row[index] -= 1
    return lca


def accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    hierarchy: np.ndarray = HIERARCHY,
    level: int = 0,
    k: int = 1,
) -> float:
    top_k_predictions = np.argsort(predictions, axis=1)[:, -k:]
    labels = labels.reshape(-1, 1)
    corrects = np.any(
        hierarchy[level][top_k_predictions] == hierarchy[level][labels],
        axis=1,
    )
    return np.mean(corrects)


def error_rate(
    predictions: np.ndarray,
    labels: np.ndarray,
    hierarchy: np.ndarray = HIERARCHY,
    level: int = 0,
    k: int = 1,
) -> float:
    top_k_predictions = np.argsort(predictions, axis=1)[:, -k:]
    labels = labels.reshape(-1, 1)
    wrongs = np.all(
        hierarchy[level][top_k_predictions] != hierarchy[level][labels],
        axis=1,
    )
    return np.mean(wrongs)


def hier_dist_mistake(
    predictions: np.ndarray,
    labels: np.ndarray,
    hierarchy: np.ndarray = HIERARCHY,
    level: int = 0,
    k: int = 1,
) -> float:
    top_k_predictions = np.argsort(predictions, axis=1)[:, -k:]
    labels = labels.reshape(-1, 1)
    wrongs = np.all(
        hierarchy[level][top_k_predictions] != hierarchy[level][labels],
        axis=1,
    )
    lca = hierarchy_to_lca(hierarchy[level:])
    lca_heights = lca[top_k_predictions[wrongs], labels[wrongs]]
    return np.mean(lca_heights)  # type: ignore


def hier_dist(
    predictions: np.ndarray,
    labels: np.ndarray,
    hierarchy: np.ndarray = HIERARCHY,
    level: int = 0,
    k: int = 1,
) -> float:
    top_k_predictions = np.argsort(predictions, axis=1)[:, -k:]
    labels = labels.reshape(-1, 1)
    lca_heights = hierarchy_to_lca(hierarchy[level:])[top_k_predictions, labels]
    return np.mean(lca_heights)  # type: ignore


# TODO: change to outputs_labels
def predictions_labels(exp):
    # Load test results
    data = np.load(PATH_EXPERIMENTS / exp / "outputs_targets.npz")
    outputs, targets = data["outputs"], data["targets"]

    # For one-hot encoding targets are already class index (aka labels)
    # Convert back to one hot encoding to be consistent with others encondings
    if targets.shape[-1] == 1:
        labels = targets.squeeze().astype(int)
        targets = np.eye(HIERARCHY.shape[-1])[labels]

    # Select the encoder matrix
    encs = np.load(exps.loc[exp, "encoding"])

    # Normalize quantites
    outputs /= np.linalg.norm(outputs, axis=1, keepdims=True)
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    encs /= np.linalg.norm(encs, axis=1, keepdims=True)

    # Calculate predictions and labels from outputs and targets
    predictions = outputs @ encs.T
    labels = (targets @ encs.T).argmax(axis=-1)

    return predictions, labels


exps = pd.read_csv(
    PATH_RESULTS / "experiments.csv",
    index_col="id",
    converters={"encoding": lambda enc: PATH_ENCODINGS / enc},
    comment="#",
)

index = pd.Index(
    data=exps.index,
    dtype=str,
    name="experiments",
)

# TODO: change number of columns, and namas
columns = pd.MultiIndex.from_product(
    iterables=[
        range(len(HIERARCHY) - 1),
        ["error_rate", "hier_dist_mistake", "hier_dist"],
    ],
    names=["level", "metric"],
)

df = pd.DataFrame(index=index, columns=columns, dtype=float)

# Load previous calculations
try:
    old_df = pd.read_pickle(PATH_RESULTS / "predictions" / "metrics.pkl")
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
    predictions, labels = predictions_labels(exp)
    levels = trange(len(HIERARCHY) - 1, leave=False, desc=str(exp))
    for level in levels:
        df.loc[exp, (level, "error_rate")] = error_rate(
            predictions, labels, level=level, k=1
        )
        df.loc[exp, (level, "hier_dist_mistake")] = hier_dist_mistake(
            predictions, labels, level=level, k=1
        )
        df.loc[exp, (level, "hier_dist")] = hier_dist(
            predictions, labels, level=level, k=1
        )


df.to_pickle(PATH_RESULTS / "predictions" / "metrics.pkl")
