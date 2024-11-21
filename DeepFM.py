import torch
from base.FM import FM
from base.LR import LR
from base.MLP import MLP
from base.Embedding import EmbeddingLayer


class DeepFM(torch.nn.Module):
    """Deep Factorization Machine Model

    Args:
        deep_features (list): the list of `Feature Class`, training by the deep part module.
        fm_features (list): the list of `Feature Class`, training by the fm part module.
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
    """

    def __init__(self, deep_features, fm_features, mlp_params):
        super(DeepFM, self).__init__()
        self.deep_features = deep_features
        self.fm_features = fm_features
        self.deep_dims = sum([fea.embed_dim for fea in deep_features])
        self.fm_dims = sum([fea.embed_dim for fea in fm_features])
        self.linear = LR(self.fm_dims)  # 1-odrder interaction  一阶交互用LR
        self.fm = FM(reduce_sum=True)  # 2-odrder interaction   二阶交互用FM
        self.embedding = EmbeddingLayer(deep_features + fm_features)
        self.mlp = MLP(self.deep_dims, **mlp_params)

    def forward(self, x):
        input_deep = self.embedding(x, self.deep_features, squeeze_dim=True)  #[batch_size, deep_dims]
        input_fm = self.embedding(x, self.fm_features, squeeze_dim=False)  #[batch_size, num_fields, embed_dim]

        y_linear = self.linear(input_fm.flatten(start_dim=1))
        y_fm = self.fm(input_fm)
        y_deep = self.mlp(input_deep)  #[batch_size, 1]
        y = y_linear + y_fm + y_deep
        return torch.sigmoid(y.squeeze(1))