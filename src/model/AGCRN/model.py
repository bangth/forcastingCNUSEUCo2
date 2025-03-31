import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGraphLearner(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super(AdaptiveGraphLearner, self).__init__()
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)

    def forward(self):
        # A = ReLU(E x E^T) để đảm bảo không âm
        A = F.relu(torch.matmul(self.node_embeddings, self.node_embeddings.transpose(0, 1)))
        # Chuẩn hoá hàng
        A = F.softmax(A, dim=1)
        return A

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        # x: (B, N, input_dim), adj: (N, N)
        x = torch.einsum('bnf,nm->bmf', x, adj)  # Aggregation qua đồ thị
        return self.linear(x)

class AGCRNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes):
        super(AGCRNCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.gate = GraphConv(input_dim + hidden_dim, 2 * hidden_dim)
        self.update = GraphConv(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, h_prev, adj):
        input_and_state = torch.cat([x, h_prev], dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, adj))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        h_tilde = torch.tanh(self.update(torch.cat([x, r * h_prev], dim=-1), adj))
        h = (1 - z) * h_prev + z * h_tilde
        return h

class AGCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim=64, embed_dim=10, horizon=1, num_layers=2):
        super(AGCRN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.num_layers = num_layers

        self.adaptive_graph = AdaptiveGraphLearner(num_nodes, embed_dim)
        self.rnn_cells = nn.ModuleList()
        for _ in range(num_layers):
            cell = AGCRNCell(input_dim if _ == 0 else hidden_dim, hidden_dim, num_nodes)
            self.rnn_cells.append(cell)

        self.projection = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        # x: (B, T, N, F) → cần reshape về (B, T, N, F)
        B, T, N, F = x.shape
        adj = self.adaptive_graph()  # (N, N)

        h = [torch.zeros((B, N, self.hidden_dim), device=x.device) for _ in range(self.num_layers)]

        for t in range(T):
            x_t = x[:, t, :, :]  # (B, N, F)
            for l in range(self.num_layers):
                h[l] = self.rnn_cells[l](x_t, h[l], adj)
                x_t = h[l]

        out = self.projection(h[-1])  # (B, N, H)
        out = out.permute(0, 2, 1)  # (B, H, N)
        return out[:, -1, :]  # Lấy dự báo tại horizon cuối cùng
