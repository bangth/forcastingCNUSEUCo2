import torch
import torch.nn as nn
import torch.nn.functional as F

class MixProp(nn.Module):
    def __init__(self, in_dim, out_dim, gdep, dropout, alpha):
        super(MixProp, self).__init__()
        self.nconv = nn.Linear(in_dim, out_dim)
        self.mlp = nn.Sequential(
            nn.Linear((gdep + 1) * out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
        self.gdep = gdep
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        adj = adj / adj.sum(1, keepdim=True)

        out = [self.nconv(x)]
        a = x
        for _ in range(self.gdep):
            a = self.alpha * x + (1 - self.alpha) * torch.einsum("ij,bjk->bik", adj, a)
            out.append(self.nconv(a))
        h = torch.cat(out, dim=-1)
        return self.mlp(h)


class MTGNN(nn.Module):
    def __init__(self, num_nodes, input_dim, horizon, gcn_depth=2, dropout=0.3, alpha=0.05, hidden_dim=64):
        super(MTGNN, self).__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        self.start_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))

        self.gconv1 = MixProp(hidden_dim, hidden_dim, gdep=gcn_depth, dropout=dropout, alpha=alpha)
        self.gconv2 = MixProp(hidden_dim, hidden_dim, gdep=gcn_depth, dropout=dropout, alpha=alpha)

        self.temporal_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1))
        self.temporal_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1))

        self.end_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, horizon, kernel_size=(1, 1))
        )

    def forward(self, x, adj):
        # x: (B, T, N, D)
        x = x.permute(0, 3, 2, 1)  # (B, D, N, T)
        x = self.start_conv(x)  # (B, H, N, T)

        # GCN layers
        x = x.permute(0, 3, 2, 1)  # (B, T, N, H)
        B, T, N, H = x.shape
        x = x.reshape(B * T, N, H)
        x = self.gconv1(x, adj)
        x = self.gconv2(x, adj)
        x = x.reshape(B, T, N, H).permute(0, 3, 2, 1)  # (B, H, N, T)

        # Temporal convolution
        x = self.temporal_conv1(x)
        x = F.relu(x)
        x = self.temporal_conv2(x)

        # Final output
        x = self.end_conv(x)  # (B, horizon, N, T)
        x = x.mean(dim=-1)  # mean over time dimension
        x = x.permute(0, 2, 1)  # (B, N, horizon)
        return x
