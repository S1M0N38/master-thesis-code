import torch.nn as nn
import torch.nn.functional as F
import torchvision


class LeNet5(nn.Module):
    """
    LeNet-5 is a classic convolutional neural network architecture that was
    introduced in 1998 by Yann LeCun et al. It was designed for handwritten
    digit recognition and achieved state-of-the-art performance on the MNIST
    dataset at the time.

    Args:
        num_classes (int): The number of classes in the classification task.
            Default is 10, which corresponds to the number of digits in MNIST.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer, which has 6 output
            channels, a kernel size of 5x5, and a padding of 2.
        conv2 (nn.Conv2d): The second convolutional layer, which has 16 output
            channels and a kernel size of 5x5.
        fc1 (nn.Linear): The first fully connected layer, which has 120 output
            features.
        fc2 (nn.Linear): The second fully connected layer, which has 84 output
            features.
        fc3 (nn.Linear): The final fully connected layer, which produces the
            output logits for each class.

    Methods:
        forward(x): Computes the forward pass of the LeNet-5 module on the
            input tensor x. Returns the output logits for each class.

    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class EfficientNetB0(nn.Module):
    """
    EfficientNetB0 is a convolutional neural network architecture that was
    introduced in 2019 by Mingxing Tan et al. in "EfficientNet: Rethinking Model
    Scaling for Convolutional Neural Networks" (https://arxiv.org/abs/1905.11946)

    Args:
        num_classes (int): The number of classes in the classification task.
            Default is 1000, which corresponds to the number of classes in
            ImageNet.
        weights (str): Path to the pre-trained weights file. If None, the
            model will be initialized with random weights. Default is None.
        dropout (float): The dropout probability for the fully connected layer
            of the model. Default is 0, which means no dropout will be applied.
        TODO: add l2_norm and freeze

    Attributes:
        model (torchvision.models.EfficientNet): The EfficientNetB0 model from
            the torchvision library, with the final fully connected layer
            replaced by a new one that produces the output logits for each class.

    Methods:
        forward(x): Computes the forward pass of the EfficientNetB0 module on
            the input tensor x. Returns the output logits for each class.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        weights: str | None = None,
        dropout: float = 0,
        l2_norm: bool = False,
        freeze: bool = False,
    ):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0(weights=weights)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(1280, num_classes),
        )
        self.l2_norm = l2_norm

    def forward(self, x):
        x = self.model(x)
        if self.l2_norm:
            x = F.normalize(x, dim=-1)
        return x


class ResNet18(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        weights: str | None = None,
    ):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=weights)
        self.model.fc = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)
