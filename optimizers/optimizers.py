import torch


class Adam(torch.optim.Adam):
    def __init__(
        self,
        model: torch.nn.Module,
        lr_features: None | float = None,
        lr_classifier: None | float = None,
        *args,
        **kwargs,
    ):
        if lr_features is not None and lr_classifier is not None:
            params = [
                {"params": model.model.features.parameters(), "lr": lr_features},
                {"params": model.model.classifier.parameters(), "lr": lr_classifier},
            ]
        else:
            params = model.parameters()

        super().__init__(params, *args, **kwargs)
