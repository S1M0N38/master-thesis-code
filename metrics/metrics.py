import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics import classification

Accuracy = torchmetrics.Accuracy


class EmbeddingsAccuracy(classification.MulticlassAccuracy):
    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        labels = targets.argmax(dim=-1)
        super().update(outputs, labels)


class AlignmentAccuracy(torchmetrics.Metric):
    def __init__(self, path_embeddings: str, top_k: int = 1) -> None:
        super().__init__()
        embeddings = np.load(path_embeddings)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = torch.from_numpy(embeddings).to("cuda")
        self.top_k = top_k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        outputs = F.normalize(outputs, p=2, dim=1)
        targets = F.normalize(targets, p=2, dim=1)
        _, preds = (outputs @ self.embeddings.T).topk(self.top_k)
        labels = (targets @ self.embeddings.T).argmax(dim=-1)
        self.correct += (preds == labels.unsqueeze(-1)).any(dim=-1).sum()
        self.total += labels.numel()  # type: ignore

    def compute(self) -> torch.Tensor:
        return self.correct / self.total  # type: ignore
