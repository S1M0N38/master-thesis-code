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


def lca_to_hierarchy(lca: np.ndarray) -> np.ndarray:
    """
    Converts a Least Common Ancestor (LCA) matrix to a hierarchy matrix.

    The hierarchy matrix is a matrix where each row represents the ancestor hierarchy
    of a class.

    Args:
        lca (np.array or torch.Tensor): A square matrix where each element (i, j)
            represents the level of the least common ancestor for classes i and j.

    Returns:
        A numpy array containing the hierarchy matrix.
    """
    # Make a copy to avoid inplace operations
    lca = np.array(lca, dtype=int)

    # Number of hierarchy levels (L)
    # Number of finer classes (C)
    L, C = lca.max(), len(lca)

    hierarchy = -np.ones((L, C), dtype=int)

    for level in range(L):
        # Find all siblings at `level`,
        # reverse to be consistence at level 0
        siblings = np.unique(lca == level, axis=0)[::-1]

        # Generate labeler
        labeler = np.arange(len(siblings), dtype=int)

        # Apply labels to siblings with labeler
        labels = labeler @ siblings

        # Add labels to hierarchy
        hierarchy[level] = labels

        # Update lca for next iteration
        lca[lca == level] += 1

    return hierarchy


def hierarchy_to_dot(hierarchy: np.ndarray, shape: str = "point") -> str:
    # This function generate graph using the dot languages from hierarchy
    """
    Generate graph using dot langauage from hierarchy matrix.

    Args:
        hierarchy (np.array or torch.Tensor): A matrix where each row represents
            the ancestor hierarchy of a class.
        shape (str): The shape of the nodes in the graph.

    Returns:
        A string containing the graph in dot language.
    """

    # store nodes and edges in dot format
    nodes = f"\nnode [shape={shape}]\n"
    edges = "\n"

    # generate lca matrix without first level
    lca = hierarchy_to_lca(hierarchy[1:])

    # Number of hierarchy levels in the reduce lca
    L = lca.max()

    for level in range(L):
        # each row is collection of siblings at specific level
        relatives = np.unique(lca == level, axis=0)[::-1]

        # extract parents and childrens from relatives
        parents, childrens = np.where(relatives)

        # get childrens names at specific level
        childrens = hierarchy[level, childrens]

        # remove redundant connection
        parents, childrens = np.unique(np.vstack((parents, childrens)), axis=1)

        for parent, children in zip(parents, childrens):
            nodes += f"{level}.{children} [label={children}]\n"
            edges += f"{level}.{children} -- {level + 1}.{parent} \n"

        # update lca for next iteration
        lca[lca == level] += 1

    # add root node and connections to the last level
    for parent in np.unique(parents):  # type: ignore
        nodes += f"{level + 1}.{parent} [label={parent}]\n"  # type: ignore
        edges += f"{level + 1}.{parent} -- {level + 2}.0 \n"  # type: ignore
    nodes += f"{level + 2}.0 [label=0]\n"  # type: ignore

    return f"graph {{{nodes}{edges}}}"



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
