"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import torch
from torch.nn import Module, Dropout
# import torch.nn as nn

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        # import ipdb; ipdb.set_trace()
        # QK = torch.einsum("nlhd,nshd->nlsh", Q, K)
        # QK_raw = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        # print(f'Q: {Q.shape}, K: {K.shape}')
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        # return queried_values.contiguous(), QK
        return queried_values.contiguous()

# class SelfAttention(nn.Module):
    # def __init__(self, in_channels):
    #     super(SelfAttention, self).__init__()
    #     self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
    #     self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
    #     self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    #     self.gamma = nn.Parameter(torch.zeros(1))

    # def forward(self, x):
    #     batch_size, C, width, height = x.size()
    #     proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
    #     proj_key = self.key(x).view(batch_size, -1, width * height)
    #     energy = torch.bmm(proj_query, proj_key)
    #     attention = torch.softmax(energy, dim=-1)
    #     proj_value = self.value(x).view(batch_size, -1, width * height)

    #     out = torch.bmm(proj_value, attention.permute(0, 2, 1))
    #     out = out.view(batch_size, C, width, height)
    #     out = self.gamma * out + x
    #     return out
