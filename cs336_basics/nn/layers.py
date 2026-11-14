import math
from turtle import forward
from unittest import result
from sympy import tensorproduct
import torch
import torch.nn as nn
from torch import LongTensor, Tensor
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device:torch.device=None, dtype:torch.dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(
            torch.empty((out_features, in_features ), device=device, dtype=dtype)
        )

        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.weights, "... d_in , d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device:torch.device = None, dtype:torch.dtype = None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embeddings_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(
            torch.empty((num_embeddings,embedding_dim), device=device, dtype=dtype)
        )

        
        nn.init.trunc_normal_(self.weights, mean=0.0, std=1, a=-3.0, b=3.0)

    def forward(self, token_ids: LongTensor) -> Tensor:
        return self.weights[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float = 1e-5, device:torch.device=None , dtype:torch.dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(
            torch.ones((d_model), device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)

        # Normalize and scale elementwise by learnable gain
        rms = torch.sqrt(mean_square + self.eps)
        result = (x / rms) * self.weights.to(torch.float32)

        return result.to(in_dtype)

