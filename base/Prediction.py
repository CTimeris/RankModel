import torch
import torch.nn as nn


class PredictionLayer(nn.Module):
    """Prediction Layer.

    Args:
        task_type (str): if `task_type='classification'`, then return sigmoid(x),
                    change the input logits to probability. if`task_type='regression'`, then return x.
    """

    def __init__(self, task_type='classification'):
        super(PredictionLayer, self).__init__()
        if task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be classification or regression")
        self.task_type = task_type

    def forward(self, x):
        if self.task_type == "classification":
            x = torch.sigmoid(x)
        return x
