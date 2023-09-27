import copy
import logging
import sys
from pathlib import Path

import numpy as np
import toml
import torch
import torchmetrics
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

import dataloaders
import losses
import metrics
import models


class Tester:
    def __init__(self, config: dict, experiemnt: str) -> None:
        cfg = copy.deepcopy(config)
        self.config = config

        # [dataloaders]
        self.dataloader_test = init(dataloaders, cfg["dataloaders"]["test"])

        self.gpus = list(range(cfg.get("num_gpus", 0)))
        self.device = torch.device("cuda:0" if self.gpus else "cpu")

        # [model]
        self.model = init(models, cfg["model"]).to(self.device)
        if len(self.gpus) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus)

        # [loss]
        self.loss = init(losses, cfg["loss"]).to(self.device)
        self.loss_test = torchmetrics.MeanMetric().to(self.device)

        # [metrics]
        self.metrics_test = torchmetrics.MetricCollection(
            {
                name: init(metrics, metric).to(self.device)
                for name, metric in cfg["metrics"]["test"].items()
            }
        )

        # Track and save results
        self.path = Path(cfg.get("path", ".")) / experiemnt
        self.path.mkdir(parents=True, exist_ok=True)
        self.logger = init_logger(self.path / "tester.log")
        self.logger.info(f"Tester initialized using {self.path / 'config.toml'}")
        self.logger.debug(f"Using {self.device} as device.")

    def __str__(self) -> str:
        return toml.dumps(self.config)

    def load(self, path: Path):
        checkpoint = torch.load(path.resolve())
        self.model.load_state_dict(checkpoint["model"])
        return self

    def test(self, save: bool = False) -> dict[str, torch.Tensor]:
        self.model = self.model.eval()
        self.logger.info("Start testing.")

        if save:
            _, y = next(iter(self.dataloader_test))
            bs, len_target = y.shape if y.ndim > 1 else y.unsqueeze(1).shape
            len_output = self.config["model"]["num_classes"]
            len_dataset = len(self.dataloader_test.dataset)
            arr_outputs = torch.empty(len_dataset, len_output).to(self.device)
            arr_targets = torch.empty(len_dataset, len_target).to(self.device)
            self.logger.info(f"Saving results to {self.path / 'outputs_targets.npz'}")

        with torch.no_grad():
            pbar = tqdm(self.dataloader_test, total=len(self.dataloader_test))
            for batch, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)
                self.metrics_test(outputs, targets)
                self.loss_test(loss)
                if save:
                    loc = slice(batch * bs, batch * bs + len(inputs))  # type: ignore
                    targets = targets if targets.ndim > 1 else targets.unsqueeze(1)
                    arr_outputs[loc] = outputs  # type: ignore
                    arr_targets[loc] = targets  # type: ignore

            metrics = self.metrics_test.compute()
            metrics["loss"] = self.loss_test.compute()

        if save:
            np.savez_compressed(
                self.path / "outputs_targets.npz",
                outputs=arr_outputs.cpu().numpy(),  # type: ignore
                targets=arr_targets.cpu().numpy(),  # type: ignore
            )
            self.logger.info(f"Saved results to {self.path / 'outputs_targets.npz'}")

        self.logger.info("Testing completed.")
        return metrics

    def features_extraction(self, node: str, len_features: int) -> None:
        self.logger.info("Create features extractor")
        model = create_feature_extractor(self.model, return_nodes={node: "features"})
        model.eval()
        self.logger.info("Start features extraction.")

        _, y = next(iter(self.dataloader_test))
        bs, len_target = y.shape if y.ndim > 1 else y.unsqueeze(1).shape
        len_dataset = len(self.dataloader_test.dataset)
        arr_features = torch.empty(len_dataset, len_features).to(self.device)
        arr_targets = torch.empty(len_dataset, len_target).to(self.device)
        self.logger.info(f"Saving results to {self.path / 'features_targets.npz'}")

        with torch.no_grad():
            pbar = tqdm(self.dataloader_test, total=len(self.dataloader_test))
            for batch, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                features = model(inputs)["features"]
                loc = slice(batch * bs, batch * bs + len(inputs))  # type: ignore
                targets = targets if targets.ndim > 1 else targets.unsqueeze(1)
                arr_features[loc] = features  # type: ignore
                arr_targets[loc] = targets  # type: ignore

        np.savez_compressed(
            self.path / "features_targets.npz",
            features=arr_features.cpu().numpy(),  # type: ignore
            targets=arr_targets.cpu().numpy(),  # type: ignore
        )
        self.logger.info(f"Saved results to {self.path / 'features_targets.npz'}")
        self.logger.info("Features extractions completed.")


def init(module: object, class_args: dict):
    class_name = class_args.pop("class")
    return getattr(module, class_name)(**class_args)


def init_logger(path: Path):
    logger = logging.getLogger(path.stem)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(levelname)8s - %(asctime)s - %(message)s")
    fh = logging.FileHandler(path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def get_experiements(config: dict) -> list[Path]:
    experiements = []
    print("Select experiement to test:")
    for experiement in Path(config["path"]).iterdir():
        if config["name"] in experiement.name:
            experiements.append(experiement)
            print(f"{len(experiements)}. {experiement.name}")

    if len(experiements) == 0:
        print("No experiements found.")
        quit()

    if len(experiements) > 1:
        print("0. All")

    i = int(input("Select experiement: "))
    if i == 0:
        experiements = experiements
    else:
        experiements = [experiements[i - 1]]

    return experiements


if __name__ == "__main__":
    config = toml.load(Path(sys.argv[1]))
    experiements = get_experiements(config)

    for experiement in experiements:
        print(f"Testing {experiement.name}")
        tester = Tester(config, experiement.name)
        tester.load(tester.path / "checkpoints" / "accuracy-top-1.pt")

        print(f"Progress at {tester.path.parent / '*' / 'tester.log'}")
        print("Features extraction ...")
        tester.features_extraction("model.flatten", 1280)

        print(f"Progress at {tester.path.parent / '*' / 'tester.log'}")
        print("Testing ...")
        results = tester.test(save=True)
        print(results)
