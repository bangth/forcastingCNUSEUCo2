# src/model/GMAN/model.py 
# phiên bản đơn giản hóa, thử trước, nếu không được sẽ viết bản đủ 

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TemporalEmbedding, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, t):
        # t shape: (B, T, N, in_dim)
        return self.linear(t)


class SpatialAttention(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(SpatialAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # x shape: (B, N, F) → reshape as (B*N, T, F)
        B, T, N, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        out, _ = self.attn(x, x, x)
        out = out.mean(dim=1)  # Average over time
        out = out.view(B, N, F)
        return out


class GMAN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, num_heads=4):
        super(GMAN, self).__init__()
        self.device = device
        self.temporal_emb = TemporalEmbedding(input_size, hidden_size)
        self.spatial_att = SpatialAttention(hidden_size, num_heads)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        # x shape: (B, T, N, F)
        B, T, N, F = x.shape
        x = self.temporal_emb(x)  # shape: (B, T, N, hidden)
        x = self.spatial_att(x)   # shape: (B, N, hidden)
        out = self.predictor(x)   # shape: (B, N, 1)
        out = out.squeeze(-1)     # shape: (B, N)
        return out
