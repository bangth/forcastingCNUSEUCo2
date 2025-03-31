import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input, adj):
        support = self.fc(input)
        output = torch.einsum('ij,bjp->bip', adj, support)
        return output


class MISTAGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, K=2):
        super(MISTAGCNBlock, self).__init__()
        self.K = K
        self.gc_layers = nn.ModuleList()
        for _ in range(K):
            self.gc_layers.append(GraphConvolution(in_channels, out_channels))

    def forward(self, x, adj):
        out = 0
        for k in range(self.K):
            out += self.gc_layers[k](x, adj)
        out = F.relu(out)
        return out


class MISTAGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes, horizon, K=2):
        super(MISTAGCN, self).__init__()
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.gcn1 = MISTAGCNBlock(input_dim, hidden_dim, K=K)
        self.gcn2 = MISTAGCNBlock(hidden_dim, hidden_dim, K=K)
        self.output = nn.Linear(hidden_dim, horizon)

    def forward(self, x, adj):
        # x: (batch_size, input_length, num_nodes, input_dim)
        batch_size, input_len, num_nodes, input_dim = x.shape

        x = x.reshape(batch_size * input_len, num_nodes, input_dim)
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)

        x = x.reshape(batch_size, input_len, num_nodes, -1)
        x = x.mean(dim=1)  # temporal average pooling

        x = self.output(x)  # (batch_size, num_nodes, horizon)
        x = x.permute(0, 2, 1)  # (batch_size, horizon, num_nodes)
        return x


def get_model(args):
    return MISTAGCN(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.horizon,
        num_nodes=args.num_nodes,
        horizon=args.horizon,
        K=args.K if hasattr(args, 'K') else 2,
    )