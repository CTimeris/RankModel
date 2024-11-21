import torch.nn as nn
import torch
import paddle
import math
import torch.nn.functional as F


class FM(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field):
        super(FM, self).__init__()
        self.sparse_feature_number = sparse_feature_number  # 1000001
        self.sparse_feature_dim = sparse_feature_dim  # 9
        self.dense_feature_dim = dense_feature_dim  # 13
        self.dense_emb_dim = self.sparse_feature_dim  # 9
        self.sparse_num_field = sparse_num_field  # 26
        self.init_value_ = 0.1
        use_sparse = True
        # sparse coding
        # Embedding(1000001, 1, padding_idx=0, sparse=True)
        self.embedding_one = paddle.nn.Embedding(
            sparse_feature_number,
            1,
            padding_idx=0,
            sparse=use_sparse,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                        math.sqrt(float(self.sparse_feature_dim)))))
        # Embedding(1000001, 9, padding_idx=0, sparse=True)
        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=use_sparse,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                        math.sqrt(float(self.sparse_feature_dim)))))

        # dense coding
        """
        Tensor(shape=[13], dtype=float32, place=CPUPlace, stop_gradient=False,
        [-0.00486396,  0.02755001, -0.01340683,  0.05218775,  0.00938804,  0.01068084,  0.00679830,  
        0.04791596, -0.04357519,  0.06603041, -0.02062148, -0.02801327, -0.04119579]))
        """
        self.dense_w_one = paddle.create_parameter(
            shape=[self.dense_feature_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0,
                std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim))))

        # Tensor(shape=[1, 13, 9])
        self.dense_w = paddle.create_parameter(
            shape=[1, self.dense_feature_dim, self.dense_emb_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0,
                std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim))))

    def forward(self, sparse_inputs, dense_inputs):
        # -------------------- first order term  --------------------
        """
        sparse_inputs: list, length:26, list[tensor], each tensor shape: [2, 1]
        dense_inputs: Tensor(shape=[2, 13]), 2 --> train_batch_size
        """
        # Tensor(shape=[2, 26])
        sparse_inputs_concat = paddle.concat(sparse_inputs, axis=1)
        # Tensor(shape=[2, 26, 1])
        sparse_emb_one = self.embedding_one(sparse_inputs_concat)
        # dense_w_one: shape=[13], dense_inputs: shape=[2, 13]
        # dense_emb_one: shape=[2, 13]
        dense_emb_one = paddle.multiply(dense_inputs, self.dense_w_one)
        # shape=[2, 13, 1]
        dense_emb_one = paddle.unsqueeze(dense_emb_one, axis=2)
        # paddle.sum(sparse_emb_one, 1): shape=[2, 1]
        # paddle.sum(dense_emb_one, 1): shape=[2, 1]
        # y_first_order: shape=[2, 1]
        y_first_order = paddle.sum(sparse_emb_one, 1) + paddle.sum(
            dense_emb_one, 1)
        # -------------------- second order term  --------------------
        # Tensor(shape=[2, 26, 9])
        sparse_embeddings = self.embedding(sparse_inputs_concat)
        # Tensor(shape=[2, 13, 1])
        dense_inputs_re = paddle.unsqueeze(dense_inputs, axis=2)
        # dense_inputs_re: Tensor(shape=[2, 13, 1])
        # dense_w: Tensor(shape=[1, 13, 9])
        # dense_embeddings: Tensor(shape=[2, 13, 9])
        dense_embeddings = paddle.multiply(dense_inputs_re, self.dense_w)
        # Tensor(shape=[2, 39, 9])
        feat_embeddings = paddle.concat([sparse_embeddings, dense_embeddings],
                                        1)
        # sum_square part
        # Tensor(shape=[2, 9])
        # \sum_{i=1}^n(v_{i,f}x_i) ---> for each embedding element: e_i, sum all feature's e_i
        summed_features_emb = paddle.sum(feat_embeddings,
                                         1)  # None * embedding_size
        # Tensor(shape=[2, 9]) 2-->batch_size
        summed_features_emb_square = paddle.square(
            summed_features_emb)  # None * embedding_size
        # square_sum part
        # Tensor(shape=[2, 39, 9])
        squared_features_emb = paddle.square(
            feat_embeddings)  # None * num_field * embedding_size
        # Tensor(shape=[2, 9]) 2-->batch_size
        squared_sum_features_emb = paddle.sum(squared_features_emb,
                                              1)  # None * embedding_size
        # Tensor(shape=[2, 1])
        y_second_order = 0.5 * paddle.sum(
            summed_features_emb_square - squared_sum_features_emb,
            1,
            keepdim=True)  # None * 1

        return y_first_order, y_second_order, feat_embeddings


class DNN(paddle.nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, num_field, layer_sizes):
        super(DNN, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes
        # [351, 512, 256, 128, 32, 1]
        sizes = [sparse_feature_dim * num_field] + self.layer_sizes + [1]
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)

    def forward(self, feat_embeddings):
        """
        feat_embeddings: Tensor(shape=[2, 39, 9])
        """
        # Tensor(shape=[2, 351]) --> 351=39*9,
        # 39 is the number of features(category feature+ continous feature), 9 is embedding size
        y_dnn = paddle.reshape(feat_embeddings,
                               [-1, self.num_field * self.sparse_feature_dim])
        for n_layer in self._mlp_layers:
            y_dnn = n_layer(y_dnn)
        return y_dnn


class DeepFM(nn.Module):
    def __init__(self, sparse_feature_number, sparse_feature_dim, dense_feature_dim, sparse_num_field, num_field,
                 layer_sizes):
        super().__init__()
        self.fm = FM(sparse_feature_number, sparse_feature_dim, dense_feature_dim, sparse_num_field)
        self.dnn = DNN(sparse_feature_number, sparse_feature_dim, dense_feature_dim, num_field, layer_sizes)

    def forward(self, sparse_inputs, dense_inputs):

        y_first_order, y_second_order, feat_embeddings = self.fm.forward(
            sparse_inputs, dense_inputs)
        # feat_embeddings: Tensor(shape=[2, 39, 9])
        # y_dnn: Tensor(shape=[2, 1])
        y_dnn = self.dnn.forward(feat_embeddings)
        print("y_dnn:", y_dnn)

        predict = F.sigmoid(y_first_order + y_second_order + y_dnn)

        return predict
