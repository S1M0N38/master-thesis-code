import argparse
import copy
import json
import logging
from collections.abc import Callable
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

    def test(self, save: bool = False) -> dict[str, float]:
        self.model = self.model.eval()
        self.metrics_test.reset()
        self.loss_test.reset()
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
            metrics_ = {k: v.item() for k, v in self.metrics_test.compute().items()}

            metrics_["loss"] = self.loss_test.compute().item()

        if save:
            np.savez_compressed(
                self.path / "outputs_targets.npz",
                outputs=arr_outputs.cpu().numpy(),  # type: ignore
                targets=arr_targets.cpu().numpy(),  # type: ignore
            )
            self.logger.info(f"Saved results to {self.path / 'outputs_targets.npz'}")

        self.logger.info("Testing completed.")
        return metrics_

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

    def attack(
        self, attack: Callable, epsilon: float, save: bool = False
    ) -> dict[str, float]:
        self.model = self.model.eval()
        self.metrics_test.reset()
        self.loss_test.reset()
        self.logger.info("Starting attack.")

        filename = f"uattack-eps{epsilon:.5f}_targets.npz"
        if save:
            _, y = next(iter(self.dataloader_test))
            bs, len_target = y.shape if y.ndim > 1 else y.unsqueeze(1).shape
            len_output = self.config["model"]["num_classes"]
            len_dataset = len(self.dataloader_test.dataset)
            arr_outputs = torch.empty(len_dataset, len_output)
            arr_targets = torch.empty(len_dataset, len_target)
            self.logger.info(f"Saving results to {self.path / filename}")

        pbar = tqdm(self.dataloader_test, total=len(self.dataloader_test))
        for batch, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # Need to derive wrt to inputs, for this reason batch size must
            # be reduce in order to fit the data into memory
            inputs.requires_grad = True

            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)
            self.model.zero_grad()
            loss.backward()

            # x are denormalized inputs
            x = self.dataloader_test.denorm(inputs)
            x_perturbed = attack(x, inputs.grad.data, epsilon)
            inputs_perturbed = self.dataloader_test.norm(x_perturbed)

            # Evaluate model on perturded inputs
            outputs_perturbed = self.model(inputs_perturbed)
            loss = self.loss(outputs, targets)
            self.metrics_test(outputs_perturbed, targets)
            self.loss_test(loss)

            if save:
                loc = slice(batch * bs, batch * bs + len(inputs))  # type: ignore
                targets = targets if targets.ndim > 1 else targets.unsqueeze(1)
                arr_outputs[loc] = outputs_perturbed.detach().cpu()  # type: ignore
                arr_targets[loc] = targets.detach().cpu()  # type: ignore

        metrics_ = {k: v.item() for k, v in self.metrics_test.compute().items()}
        metrics_["loss"] = self.loss_test.compute().item()

        if save:
            np.savez_compressed(
                self.path / filename,
                outputs=arr_outputs.numpy(),  # type: ignore
                targets=arr_targets.numpy(),  # type: ignore
            )
            self.logger.info(f"Saved results to {self.path / filename}")

        self.logger.info("Attacked completed.")
        return metrics_


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


def get_experiements(config: dict, all: bool) -> list[Path]:
    experiements = []
    print("Experiments found:")
    for experiement in Path(config["path"]).iterdir():
        if config["name"] in experiement.name:
            experiements.append(experiement)
            print(f"{len(experiements)}. {experiement.name}")
    if len(experiements) == 0:
        print("No experiements found.")
        quit()
    if all:
        print("Test all experiments.")
        return experiements
    else:
        i = int(input("Select experiement: "))
        return [experiements[i - 1]]


def fgsm(inputs_denorm, inputs_grad, epsilon) -> torch.Tensor:
    """
    FGSM attack: very simple, but effective
    x' = x + eps * sign(dL/dx)
    """
    inputs_grad_sign = inputs_grad.sign()
    inputs_denorm_perturbed = inputs_denorm + epsilon * inputs_grad_sign
    inputs_denorm_perturbed = torch.clamp(inputs_denorm_perturbed, 0, 1)
    return inputs_denorm_perturbed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Test model in various ways:\n"
            "  - classification performance (e.g top 1 accuracy)\n"
            "  - features extraction\n"
            "  - against adversarial attack (e.g. fgsm)\n"
        )
    )

    parser.add_argument(
        "config",
        help="TOML configuration for the model",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run tests for all experiments with the same config",
    )

    parser.add_argument(
        "--predictions",
        action="store_true",
        help="Evaluate predictions metrics and save model output",
    )

    parser.add_argument(
        "--features",
        action="store_true",
        help="Extract feature from the penultime layer and save",
    )

    parser.add_argument(
        "--uattack",
        action="store_true",
        help="Perform FGSM untargeted atttack",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        action="store",
        help="Epsilon used in the attacks",
    )

    args = parser.parse_args()

    config = toml.load(Path(args.config))
    experiements = get_experiements(config, args.all)

    for experiement in experiements:
        print(f"Testing {experiement.name}")
        tester = Tester(config, experiement.name)
        tester.load(tester.path / "checkpoints" / "accuracy-top-1.pt")
        print(f"Progress at {tester.path.parent / '*' / 'tester.log'}")

        if args.predictions:
            print("Predictions ...")
            results = tester.test(save=False)
            with open(tester.path / "predictions.json", "w") as json_file:
                json.dump(results, json_file)

        if args.features:
            print("Features extraction ...")
            tester.features_extraction("model.flatten", 1280)

        if args.uattack:
            print("Untargeted Attack ...")
            results = tester.attack(attack=fgsm, epsilon=args.epsilon, save=True)
            filename = f"uattack-eps{args.epsilon:.5f}.json"
            with open(tester.path / filename, "w") as json_file:
                json.dump(results, json_file)

        else:
            print("No testing selected")
