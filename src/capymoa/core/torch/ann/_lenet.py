from torch import nn, Tensor


class LeNet5(nn.Module):
    """LeNet-5 [Lecun1998]_ convolutional neural network for 28x28 grayscale images.

    .. [Lecun1998] Y. Lecun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning
       applied to document recognition," in Proceedings of the IEEE, vol. 86,
       no. 11, pp. 2278-2324, Nov. 1998, doi: 10.1109/5.726791
    """

    def __init__(self, num_classes: int, in_shape: tuple[int, int, int] = (1, 28, 28)):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act_fn = nn.ReLU()
        self.flatten = nn.Flatten()
        self.in_shape = in_shape

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the LeNet-5 model."""
        x = x.view(-1, *self.in_shape)
        x = self.pool(self.act_fn(self.conv1(x)))
        x = self.pool(self.act_fn(self.conv2(x)))
        x = self.flatten(x)
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        x = self.fc3(x)
        return x
