import numpy as np


def get_predictions(outputs: np.ndarray, encodings: np.ndarray) -> np.ndarray:
    outputs /= np.linalg.norm(outputs, axis=1, keepdims=True)
    encodings /= np.linalg.norm(encodings, axis=1, keepdims=True)
    predictions = outputs @ encodings.T
    return predictions


def get_labels(
    targets: np.ndarray,
    encodings: np.ndarray,
) -> np.ndarray:
    # For one-hot encoding targets are already class index (aka labels)
    if targets.shape[-1] == 1:
        labels = targets.squeeze().astype(int)
        return labels

    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    encodings /= np.linalg.norm(encodings, axis=1, keepdims=True)
    labels = (targets @ encodings.T).argmax(axis=-1)
    return labels


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


def corrects(
    predictions: np.ndarray,
    labels: np.ndarray,
    hierarchy: np.ndarray,
    level: int = 0,
    k: int = 1,
) -> np.array:
    top_k_predictions = np.argsort(predictions, axis=1)[:, -k:]
    labels = labels.reshape(-1, 1)
    return np.any(
        hierarchy[level][top_k_predictions] == hierarchy[level][labels],
        axis=1,
    )


def wrongs(
    predictions: np.ndarray,
    labels: np.ndarray,
    hierarchy: np.ndarray,
    level: int = 0,
    k: int = 1,
) -> np.array:
    top_k_predictions = np.argsort(predictions, axis=1)[:, -k:]
    labels = labels.reshape(-1, 1)
    return np.all(
        hierarchy[level][top_k_predictions] != hierarchy[level][labels],
        axis=1,
    )


def accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    hierarchy: np.ndarray,
    level: int = 0,
    k: int = 1,
) -> float:
    return np.mean(corrects(predictions, labels, hierarchy, level, k))


def error_rate(
    predictions: np.ndarray,
    labels: np.ndarray,
    hierarchy: np.ndarray,
    level: int = 0,
    k: int = 1,
) -> float:
    return np.mean(wrongs(predictions, labels, hierarchy, level, k))


def hier_dist_mistake(
    predictions: np.ndarray,
    labels: np.ndarray,
    hierarchy: np.ndarray,
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
    hierarchy: np.ndarray,
    level: int = 0,
    k: int = 1,
) -> float:
    top_k_predictions = np.argsort(predictions, axis=1)[:, -k:]
    labels = labels.reshape(-1, 1)
    lca_heights = hierarchy_to_lca(hierarchy[level:])[top_k_predictions, labels]
    return np.mean(lca_heights)  # type: ignore
