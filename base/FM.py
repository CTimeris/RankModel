import torch.nn as nn
import torch


class FM(nn.Module):
    """
    Args:
        reduce_sum (bool): whether to sum in embed_dim (default = `True`).
    Shape:
        - Input: `(batch_size, num_features, embed_dim)`
        - Output: `(batch_size, 1)`` or ``(batch_size, embed_dim)`
    """

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1)**2      # 和的平方，推导后的公式前半部分
        sum_of_square = torch.sum(x**2, dim=1)      # 平方的和，公式后半部分
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix