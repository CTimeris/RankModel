import torch.nn as nn
import torch


class LR(nn.Module):
    """Logistic Regression Module. It is the one Non-linear
    transformation for input feature.

    Args:
        input_dim (int): input size of Linear module.
        sigmoid (bool): whether to add sigmoid function before output.

    Shape:
        - Input: `(batch_size, input_dim)`
        - Output: `(batch_size, 1)`
    """

    def __init__(self, input_dim, sigmoid=False):
        super().__init__()
        self.sigmoid = sigmoid
        self.fc = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        if self.sigmoid:
            return torch.sigmoid(self.fc(x))
        else:
            return self.fc(x)