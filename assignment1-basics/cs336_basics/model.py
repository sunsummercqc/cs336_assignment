import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean = 0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean = 0, std=1, a=-3, b=3 )
    def forward(self, token_ids):
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1,keepdim=True) + self.eps)
        result = x / rms * self.weight
        return result.to(in_dtype)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        positions = torch.arange(max_seq_len, device=device)
        angles = positions[:, None] * freq[None, :]
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
    #token_positon是位置索引
    def forward(self, x, token_positions):
        #拆 even odd
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        # 查表
        c = self.cos[token_positions]
        s = self.sin[token_positions]
        # 旋转
        new_even = x_even*c - x_odd*s
        new_odd = x_even*s + x_odd*c
        result = torch.empty_like(x)
        result[..., 0::2] = new_even
        result[..., 1::2] = new_odd
        return result
