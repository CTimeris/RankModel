import torch
import torch.nn as nn
from base.Embedding import EmbeddingLayer
from base.MLP import MLP


class DIN(nn.Module):
    """Deep Interest Network
    Args:
        features (list): the list of `Feature Class`. training by MLP. 用户特征.
        history_features (list): the list of `Feature Class`,training by ActivationUnit. 用户交互历史序列.
        target_features (list): the list of `Feature Class`, training by ActivationUnit. 目标特征.用于算注意力
        mlp_params (dict): {"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
        attention_mlp_params (dict): {"dims":list, "activation":str, "dropout":float, "use_softmax":bool`}
    """

    def __init__(self, features, history_features, target_features, mlp_params, attention_mlp_params):
        super().__init__()
        self.features = features    # 用户特征
        self.history_features = history_features    # 历史交互序列
        self.target_features = target_features      # 目标广告
        self.num_history_features = len(history_features)
        self.all_dims = sum([fea.embed_dim for fea in features + history_features + target_features])

        self.embedding = EmbeddingLayer(features + history_features + target_features)
        self.attention_layers = nn.ModuleList(
            [ActivationUnit(fea.embed_dim, **attention_mlp_params) for fea in self.history_features])
        self.mlp = MLP(self.all_dims, activation="dice", **mlp_params)

    def forward(self, x):
        embed_x_features = self.embedding(x, self.features)  #(batch_size, num_features, emb_dim)
        embed_x_history = self.embedding(
            x, self.history_features)  #(batch_size, num_history_features, seq_length, emb_dim)
        embed_x_target = self.embedding(x, self.target_features)  #(batch_size, num_target_features, emb_dim)
        attention_pooling = []
        for i in range(self.num_history_features):
            attention_seq = self.attention_layers[i](embed_x_history[:, i, :, :], embed_x_target[:, i, :])
            attention_pooling.append(attention_seq.unsqueeze(1))  #(batch_size, 1, emb_dim)
        attention_pooling = torch.cat(attention_pooling, dim=1)  #(batch_size, num_history_features, emb_dim)

        mlp_in = torch.cat([
            attention_pooling.flatten(start_dim=1),
            embed_x_target.flatten(start_dim=1),
            embed_x_features.flatten(start_dim=1)
        ], dim=1)  #(batch_size, N)

        y = self.mlp(mlp_in)
        return torch.sigmoid(y.squeeze(1))


class ActivationUnit(nn.Module):
    """Activation Unit Layer mentioned in DIN paper, it is a Target Attention method.

    Args:
        embed_dim (int): the length of embedding vector.
        history (tensor):
    Shape:
        - Input: `(batch_size, seq_length, emb_dim)`
        - Output: `(batch_size, emb_dim)`
    """

    def __init__(self, emb_dim, dims=None, activation="dice", use_softmax=False):
        super(ActivationUnit, self).__init__()
        if dims is None:
            dims = [36]
        self.emb_dim = emb_dim
        self.use_softmax = use_softmax
        self.attention = MLP(4 * self.emb_dim, dims=dims, activation=activation)

    def forward(self, history, target):
        seq_length = history.size(1)
        target = target.unsqueeze(1).expand(-1, seq_length, -1)  #batch_size,seq_length,emb_dim
        # 论文里只有逐元素相减，代码里有点积
        att_input = torch.cat([target, history, target - history, target * history],
                              dim=-1)  # batch_size,seq_length,4*emb_dim
        att_weight = self.attention(att_input.view(-1, 4 * self.emb_dim))  #  #(batch_size*seq_length,4*emb_dim)
        att_weight = att_weight.view(-1, seq_length)  #(batch_size*seq_length, 1) -> (batch_size,seq_length)
        if self.use_softmax:
            att_weight = att_weight.softmax(dim=-1)
        # (batch_size, seq_length, 1) * (batch_size, seq_length, emb_dim)
        output = (att_weight.unsqueeze(-1) * history).sum(dim=1)  #(batch_size,emb_dim)
        return output