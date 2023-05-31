import torchmetrics
from torchmetrics import classification

Accuracy = torchmetrics.Accuracy
MulticlassAccuracy = classification.MulticlassAccuracy


class EmbeddingsAccuracy(classification.MulticlassAccuracy):
    def update(self, outputs, targets):
        targets = targets.argmax(dim=-1)
        super().update(outputs, targets)
