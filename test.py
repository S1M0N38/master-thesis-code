import argparse
import copy
import logging
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
        self.metrics_test_adv = torchmetrics.MetricCollection(
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

    def _save_batch(self, dim):
        _, y = next(iter(self.dataloader_test))
        bs, targets_dim = y.shape if y.ndim > 1 else y.unsqueeze(1).shape
        len_dataset = len(self.dataloader_test.dataset)
        tensor = torch.empty(len_dataset, dim, device="cpu")

        def save(batch_idx, values):
            loc = slice(batch_idx * bs, batch_idx * bs + len(values))
            values = values if values.ndim > 1 else values.unsqueeze(1)
            tensor[loc] = values.detach().cpu()
            return tensor

        return save, bs, targets_dim

    def test(
        self,
        outputs_node: str,
        outputs_dim: int,
        extract_features: str,
        features_node: str,
        features_dim: int,
        attack_eps: float,
        attack_target: torch.Tensor | None = None,
        attack_target_name: str = "_",
    ):
        self.metrics_test.reset()
        self.metrics_test_adv.reset()
        self.model.eval()
        self.logger.info("Start testing.")

        save_outputs, *_, targets_dim = self._save_batch(outputs_dim)
        save_outputs_adv, *_ = self._save_batch(outputs_dim)
        save_targets, *_ = self._save_batch(targets_dim)

        path_results = self.path / "results"
        path_results_adv = self.path / "results" / attack_target_name
        path_results.mkdir(exist_ok=True)
        path_results_adv.mkdir(exist_ok=True)

        if extract_features:
            save_features, *_ = self._save_batch(features_dim)
            save_features_adv, *_ = self._save_batch(features_dim)
            return_nodes = {outputs_node: "outputs", features_node: "features"}
            model = create_feature_extractor(self.model, return_nodes=return_nodes)
        else:
            model = self.model

        model.eval()

        if attack_target is not None:
            self.logger.info(f"FGSM in targeted mode ({attack_target_name}).")
            attack_target = attack_target.to(self.device)
            attack_eps = -attack_eps
        else:
            self.logger.info("FGSM in untargeted mode.")

        pbar = tqdm(self.dataloader_test, total=len(self.dataloader_test))
        for batch, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            tens_targets = save_targets(batch, targets)
            inputs.requires_grad = True

            # Evaluate on inputs and Save
            results = model(inputs)
            if extract_features:
                outputs = results["outputs"]
                features = results["features"]
                tens_features = save_features(batch, features)  # type: ignore
            else:
                outputs = results
            tens_outputs = save_outputs(batch, outputs)
            self.metrics_test(outputs, targets)

            # Set the loss to optimize for in adversarial attack
            if attack_target is not None:
                repeat = attack_target.repeat(len(outputs), 1)
                repeat = repeat.squeeze(1) if len(attack_target) == 1 else repeat
                loss = self.loss(outputs, repeat)
            else:
                loss = self.loss(outputs, targets)

            model.zero_grad()
            loss.backward()

            # FGSM
            x = self.dataloader_test.denorm(inputs)
            x_ = x + attack_eps * inputs.grad.data.sign()
            x_ = torch.clamp(x_, 0, 1)
            inputs_adv = self.dataloader_test.norm(x_)

            # Evaluate on adversarial inputs and Save
            results_adv = model(inputs_adv)
            if extract_features:
                outputs_adv = results_adv["outputs"]
                features_adv = results_adv["features"]
                tens_features_adv = save_features_adv(batch, features_adv)  # type: ignore
            else:
                outputs_adv = results_adv
            tens_outputs_adv = save_outputs_adv(batch, outputs_adv)
            self.metrics_test_adv(outputs_adv, targets)

        np.save(
            path_results / "targets.npy",
            tens_targets.numpy(),  # type: ignore
        )
        np.save(
            path_results / "outputs.npy",
            tens_outputs.numpy(),  # type: ignore
        )
        np.save(
            path_results_adv / f"outputs-{abs(attack_eps):.5f}.npy",
            tens_outputs_adv.numpy(),  # type: ignore
        )
        if extract_features:
            np.save(
                path_results / "features.npy",
                tens_features.numpy(),  # type: ignore
            )
            np.save(
                path_results_adv / f"features-{abs(attack_eps):.5f}.npy",
                tens_features_adv.numpy(),  # type: ignore
            )

        return (
            {k: v.item() for k, v in self.metrics_test.compute().items()},
            {k: v.item() for k, v in self.metrics_test_adv.compute().items()},
        )


def init(module: object, class_args: dict):
    class_args = copy.deepcopy(class_args)
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


def get_attack_target(config: dict, idx: int | None) -> tuple[torch.Tensor | None, str]:
    path_dataset = Path(config["dataloaders"]["test"]["path"]).parent

    with open(path_dataset / "classes" / "classes.txt") as f:
        classes = [cls.strip() for cls in f.readlines()]

    if idx is None:
        for i, cls in enumerate(classes, 1):
            print(f"{i:>4}. {cls}")
        print("Select a target for FGSM attack. 0 for untargeted attack")
        idx = int(input("Select target: "))

    if idx == 0:
        attack_target = None
        attack_target_name = "_"
    else:
        if path_embeddings := config["dataloaders"]["test"].get("path_embeddings"):
            embedding = np.load(path_embeddings)[idx - 1]
        else:
            embedding = np.array([idx - 1])
        attack_target = torch.from_numpy(embedding)
        attack_target_name = classes[idx - 1]

    return attack_target, attack_target_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model on test dataset")

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
        "--features",
        action="store_true",
        help="Extract features vectors and save them",
    )

    parser.add_argument(
        "--epsilon",
        action="store",
        type=float,
        required=True,
        help="Epsilon used in the attacks",
    )

    parser.add_argument(
        "--target",
        action="store",
        type=int,
        help=(
            "Class index of target for FGSM attack (1-based). "
            "Use 0 for untargeted attack"
        ),
    )

    args = parser.parse_args()
    config = toml.load(Path(args.config))
    attack_target, attack_target_name = get_attack_target(config, args.target)
    experiements = get_experiements(config, args.all)

    for experiement in experiements:
        tester = Tester(config, experiement.name)
        tester.load(tester.path / "checkpoints" / "accuracy-top-1.pt")
        print(f"Progress at {tester.path.parent / '*' / 'trainer.log'}")
        print("Testing ...")

        tester.test(
            outputs_node="model.classifier.1",
            outputs_dim=config["model"]["num_classes"],
            extract_features=args.features,
            features_node="model.flatten",
            features_dim=1280,
            attack_eps=args.epsilon,
            attack_target=attack_target,
            attack_target_name=attack_target_name,
        )
