import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, input_len, num_nodes, input_dim, hidden_dim=64, output_dim=1):
        """
        input_len: số bước thời gian đầu vào (lags, ví dụ 7)
        num_nodes: số lượng node (địa phương)
        input_dim: số feature mỗi node mỗi bước (ví dụ: emissions, trend,...)
        hidden_dim: số chiều ẩn trong MLP
        output_dim: số bước dự báo (mặc định 1)
        """
        super(MLPModel, self).__init__()
        self.input_size = input_len * num_nodes * input_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes * output_dim)
        )
        self.num_nodes = num_nodes
        self.output_dim = output_dim

    def forward(self, x):
        """
        x shape: [batch_size, input_len, num_nodes, input_dim]
        return: [batch_size, num_nodes, output_dim]
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # flatten
        out = self.net(x)
        out = out.view(batch_size, self.num_nodes, self.output_dim)
        return out
