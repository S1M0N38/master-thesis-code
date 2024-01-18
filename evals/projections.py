from pathlib import Path
import argparse


# Argparse
parser = argparse.ArgumentParser(
    description="Compute projections and save to results/projections.pkl"
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
parser.add_argument(
    "--umap_n_neighbors",
    type=int,
    required=True,
    help="UMAP number of neighbors",
)
parser.add_argument(
    "--umap_min_dist",
    type=float,
    required=True,
    help="UMAP min distance",
)
parser.add_argument(
    "--umap_supervised",
    action="store_true",
    help="UMAP supervised, use ground truth labels",
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
path_hierarchy = path_dataset / "hierarchy" / "hierarchy.npy"

# Save path
filename = (
    f"projections_"
    f"{args.umap_n_neighbors:04}_"
    f"{args.umap_min_dist:.2f}"
    f"{'_supervised' if args.umap_supervised else ''}.npy"
)

path_save = path_root / "evals" / dataset / exp / "results" / filename
path_save.parent.mkdir(parents=True, exist_ok=True)
if path_save.exists():
    print(f"SKIP: {path_save} already exists")
    quit()


import numpy as np
import utils
from umap import UMAP


# Data
hierarchy = np.load(path_dataset / "hierarchy" / "hierarchy.npy")
features = np.load(path_experiment / "results" / "features.npy")
targets = np.load(path_experiment / "results" / "targets.npy")
encodings = np.load(path_encoding)
labels = utils.get_labels(targets, encodings)


# Calculate projections
umap = UMAP(
    n_components=2,
    metric="cosine",
    verbose=True,
    random_state=42,
    min_dist=args.umap_min_dist,
    n_neighbors=args.umap_n_neighbors,
)

if args.umap_supervised:
    projections = umap.fit(features, y=labels)
else:
    projections = umap.fit_transform(features)
assert isinstance(projections, np.ndarray)

np.save(path_save, projections)
