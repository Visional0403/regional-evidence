import math

import torch
from torch import nn


class DropPath(nn.Module):

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class PVTv2SRAAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, sr_ratio=1, qkv_bias=True, drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.sr_ratio = sr_ratio
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.key = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.value = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.query(x)
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_)
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
        else:
            x_ = x
        reduced_tokens = x_.shape[1]
        k = self.key(x_)
        v = self.value(x_)
        k = k.reshape(B, reduced_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, reduced_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PVTv2MixFFN(nn.Module):

    def __init__(self, embed_dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(embed_dim, hidden_dim))
        self.layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim))
        self.layers.append(nn.GELU())
        self.layers.append(nn.Dropout(drop))
        self.layers.append(nn.Linear(hidden_dim, embed_dim))

    def forward(self, x, H, W):
        x = self.layers[0](x)
        batch, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch, channels, H, W)
        x = self.layers[1](x).flatten(2).transpose(1, 2)
        x = self.layers[2](x)
        x = self.layers[3](x)
        x = self.layers[4](x)
        x = self.layers[3](x)
        return x
