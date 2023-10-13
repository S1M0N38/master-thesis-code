from pathlib import Path
import re
import numpy as np
import sys
from functools import partial

import pandas as pd
from pandas.io.formats.style import Styler


# Avaialve datasets: CIFAR100, iNaturalist19, tieredImageNet
DATASET = sys.argv[1]
assert DATASET in ["CIFAR100", "iNaturalist19", "tieredImageNet"]

# Path
PATH_ROOT = Path(__file__).parent.parent
PATH_EXPERIMENTS = PATH_ROOT / "experiments" / DATASET
PATH_DATASET = PATH_ROOT / "datasets" / "datasets" / DATASET
PATH_ENCODINGS = PATH_DATASET / "encodings"
PATH_RESULTS = PATH_ROOT / "evaluations" / "results" / DATASET


hierarchy = np.load(PATH_DATASET / "hierarchy" / "hierarchy.npy")

exps = pd.read_csv(
    PATH_RESULTS / "experiments.csv",
    index_col="id",
    converters={"encoding": lambda enc: PATH_ENCODINGS / enc},
    comment="#",
)

predictions = pd.read_pickle(PATH_RESULTS / "predictions" / "metrics.pkl")
features = pd.read_pickle(PATH_RESULTS / "features" / "metrics.pkl")

assert isinstance(predictions, pd.DataFrame)
assert isinstance(features, pd.DataFrame)

metrics = {
    # Predictions metrics
    "error_rate": "Error Rate",
    "hier_dist_mistake": "Hier. Dist. M.",
    "hier_dist": "Hier. Dist.",
    # Features metrics
    "silhouette": "Silhouette",
    "calinski_harabasz": "CH",
    "davies_bouldin": "DB",
    "sdbw": "SDbw",
}


def highlight_predictions(dfs: Styler, axis: int = 0) -> Styler:
    """
    Highlight and format a DataFrame with style for prediction metrics.
    For all metrics predictions dataframe lower is better.

    Args:
        dfs (Styler): The pandas Styler object representing the DataFrame to be styled.
        axis (int, optional): The axis along which to apply the styling
        (0 for rows, 1 for columns). Defaults to 0.

    Returns:
        Styler: A Styler object with the specified styling applied.
    """
    dfs = dfs.background_gradient("Greens_r", axis=axis, low=1)
    dfs = dfs.highlight_min(props="font-weight: bold", axis=axis)
    return dfs


def highlight_features(dfs: Styler, axis: int = 0) -> Styler:
    """
    Highlight and format a DataFrame with style for feature evaluation metrics.
    For `silhouette` and `calinski_harabasz` higher is better.
    For `davies_bouldin` and `sdbw` lower is better.

    Args:
        dfs (Styler): The pandas Styler object representing the DataFrame to be styled.
        axis (int, optional): The axis along which to apply the styling
            (0 for rows, 1 for columns). Defaults to 0.

    Returns:
        Styler: A Styler object with the specified styling applied.
    """
    idx = pd.IndexSlice
    if axis == 0:
        idx_max = idx[:, idx[:, ["silhouette", "calinski_harabasz"]]]
        idx_min = idx[:, idx[:, ["davies_bouldin", "sdbw"]]]
    elif axis == 1:
        idx_max = idx[idx[:, ["silhouette", "calinski_harabasz"]], :]
        idx_min = idx[idx[:, ["davies_bouldin", "sdbw"]], :]
    else:
        raise ValueError("Valid value for axis are 0 and 1.")
    dfs = dfs.background_gradient("Greens", axis=axis, high=1, subset=idx_max)
    dfs = dfs.background_gradient("Greens_r", axis=axis, low=1, subset=idx_min)
    dfs = dfs.highlight_max(props="font-weight: bold", axis=axis, subset=idx_max)
    dfs = dfs.highlight_min(props="font-weight: bold", axis=axis, subset=idx_min)
    return dfs


def table_html(df: pd.DataFrame, path: str) -> Styler:
    """Create HTML table with the predictions metrics for each experiment.
    This table is used to select the best experiment for each type.
    This will not be used in the final paper.
    """
    # df = df.filter(like="cd-desc", axis=0)
    dfs = df.style.format(precision=3)
    dfs = dfs.format_index(lambda i: i.split("_")[-1], axis=0)

    if path == "predictions":
        dfs = highlight_predictions(dfs)  # type: ignore
    elif path == "features":
        dfs = highlight_features(dfs)  # type: ignore
    else:
        raise ValueError("Path can be 'predictions' or 'features'.")

    dfs.to_html(PATH_RESULTS / path / "metrics.html")
    return dfs


def table_tex(df: pd.DataFrame, path: str) -> str:
    """Create Tex table with the predictions metrics mean and std for
    each experiment type. This table will be include in the paper.
    """
    global std_index
    idx = pd.IndexSlice
    std_index = 0
    df_mean = df.groupby(exps[exps["selected"]]["name"]).mean()
    df_std = df.groupby(exps[exps["selected"]]["name"]).std().fillna(0)  # type: ignore

    dfs = df_mean.T.style  # Traspose df scales better to more levels

    def fmt(value: float, precision: float):
        global std_index
        std = df_std.T.stack().values[std_index]  # type: ignore
        std_index += 1
        return rf"{value:.{precision}f} \mdseries ± {std:.{precision}f}"

    # Cells Style
    if path == "predictions":
        dfs = highlight_predictions(dfs, axis=1)
        # It's ok to use precision=3 for all predictions metrics
        dfs = dfs.format(partial(fmt, precision=3))
    elif path == "features":
        dfs = highlight_features(dfs, axis=1)
        # Format values with the right number of digits
        dfs = dfs.format(
            partial(fmt, precision=3),
            subset=idx[idx[:, ["silhouette"]], :],
        )
        dfs = dfs.format(
            partial(fmt, precision=0),
            subset=idx[idx[:, ["calinski_harabasz"]], :],
        )
        dfs = dfs.format(
            partial(fmt, precision=2),
            subset=idx[idx[:, ["davies_bouldin"]], :],
        )
        dfs = dfs.format(
            partial(fmt, precision=3),
            subset=idx[idx[:, ["sdbw"]], :],
        )
    else:
        raise ValueError("Path can be 'predictions' or 'features'.")

    # Headers Style
    dfs = dfs.hide(names=True, axis=0)  # type: ignore
    dfs = dfs.hide(names=True, axis=1)
    dfs = dfs.format_index(lambda m: metrics[m], axis=0, level=1)
    dfs = dfs.format_index(
        lambda m: exps[exps["name"] == m]["tex"].iloc[0], axis=1  # type: ignore
    )

    # Convert to TeX
    dfs = dfs.to_latex(
        hrules=True,
        column_format=f"X r *{{{len(df_mean)}}}{{c}}",
        convert_css=True,
        multirow_align="c",
        clines="skip-last;index",
    )
    dfs = dfs.replace(
        r"\cline{1-2}",  # clines don't work with colored background
        rf"\hhline{{{'-' * (len(df_mean) + 2)}}}",  # so replace with hhline
        len(hierarchy) - 2,  # on n occurences replace only the first n - 1
        # because the last one it replaced by ""
    ).replace(r"\cline{1-2}", "")
    # Exract tabular enviroment.
    # I prefer to work with tabular instead of table env for the following reasons:
    # - Better control of table position and dimensions
    # - Better control of caption position
    pattern = r"\\begin\{tabular\}(.*?)\\end\{tabular\}"
    dfs = re.search(pattern, dfs, re.DOTALL)
    assert dfs is not None, "Pattern not found"
    dfs = r"\begin{tabularx}{\linewidth}" + dfs.group(1) + r"\end{tabularx}"

    with open(PATH_RESULTS / path / "metrics.tex", "w") as f:
        f.write(dfs)

    return dfs


if __name__ == "__main__":
    table_html(predictions, "predictions")
    table_html(features, "features")
    table_tex(predictions, "predictions")
    table_tex(features, "features")
