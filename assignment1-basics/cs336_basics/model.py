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

#前馈网络 升维 —— 激活 —— 降维
class Swiglu(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
    def forward(self,x):
        a = self.w1(x)
        b = self.w3(x) * torch.sigmoid(a)
        # 在 SwiGLU 里，我们不是对原始输入做 SiLU，而是对 W1·x（线性变换后的结果）做 SiLU
        return self.w2(b*a)

        

class RotaryPositionalEmbedding(nn.Module):
    #原理：
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        # 0～ d_k/2-1 之间的频率
        freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        # 0～ max_seq_len-1 之间的位置索引
        positions = torch.arange(max_seq_len, device=device)
        # 计算角度矩阵， None用于增加维度，使得 freq 和 positions 可以广播相乘
        angles = positions[:, None] * freq[None, :]
        # 计算 cos 和 sin 矩阵，并注册为 buffer，这样它们不会被视为模型参数
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

class scaled_dot_product_attention(nn.Module):
    # 初始化 q, k, v, mask,进行缩放点积注意力计算
    def __init__(self, device=None, dtype=None):
        super().__init__()
    def forward(self, Q, K, V, mask): 
        # Q(...,q,d_k) , K (..., k, d_k)
        # 计算score (...,q,k)
        # 根号d_k 缩放的作用：防止点积过大导致softmax输出过于尖锐
        score = Q@K.transpose(-1,-2)/ ((Q.size(-1)) ** 0.5)
        # 若掩码非空，mask中为t的表示可以看的，为false是需要掩盖的，后续需要把这部分赋值为-inf， 将非空掩码反转,并赋值-inf
        #以下是取反后的值 
        # [t,t,t]
        # [f,t,t]
        # [f,f,t]
        #torch.masked_fill(input, mask, value)
        if mask is not None:
            score = score.masked_fill(~mask, float('-inf'))
        #沿最后一维做softmax
        weight = torch.softmax(score, dim = -1) @ V
        return weight

class MultiheadSelfAttention(nn.Module):
    #多头注意力机制 每个头独自计算sdpa 然后再concat
    # 计算投影 q k v , 计算 attention
    def __init__(self, d_model, h,max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        # q,k,v,oNone
        self.d_model = d_model
        self.h = h
        self.d_k = self.d_model // self.h
        self.q = Linear(self.d_model, self.d_model)
        self.k = Linear(self.d_model, self.d_model)
        self.v = Linear(self.d_model, self.d_model)
        # o
        self.o = Linear (self.d_model, self.d_model)
        #给 q k加上旋转矩阵
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None
        self.attention = scaled_dot_product_attention(device=device, dtype=dtype)
        
        
    def forward(self, x, token_positions=None):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        # 拆头Q （batch, seq_len, d_model) -> (batch, h, seq_len, d_k)
        # reshape 和transpose的区别是什么 
        Q = Q.reshape(Q.size(0), Q.size(1), self.h, self.d_k).transpose(1, 2)
        K = K.reshape(K.size(0), K.size(1), self.h, self.d_k).transpose(1, 2)
        V = V.reshape(V.size(0), V.size(1), self.h, self.d_k).transpose(1, 2)
        # q k 加上旋转矩阵,token_position怎么找
        if self.rope:
            if token_positions is None:
                token_positions = torch.arange(x.size(1), device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        # 计算掩码
        mask = torch.tril(torch.ones((x.size(1), x.size(1)), device=x.device)).bool()
        # 计算 scaled dot product attention
        # 合并头
        # self.o输出投影 score (b, h, s, d_k) -> (b, s, h, d_k) -> (b, q, d_model)
        score = self.attention(Q, K, V, mask)
        score = score.transpose(1, 2).reshape(score.size(0), score.size(2), self.d_model)
        return self.o(score)



