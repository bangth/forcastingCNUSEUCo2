# src/model/ELM/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ELM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(ELM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device

        # Random hidden weights and biases (fixed)
        self.hidden_weights = nn.Parameter(
            torch.empty(input_size, hidden_size), requires_grad=False
        )
        self.hidden_bias = nn.Parameter(
            torch.empty(hidden_size), requires_grad=False
        )
        self.output_weights = nn.Parameter(
            torch.zeros(hidden_size, output_size), requires_grad=True
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.hidden_weights)
        nn.init.zeros_(self.hidden_bias)

    def forward(self, x):
        # x shape: (B, T, N, F)
        B, T, N, F = x.shape
        x = x.reshape(B * T * N, F)  # (B*T*N, F)

        H = torch.tanh(x @ self.hidden_weights + self.hidden_bias)  # (B*T*N, hidden)
        Y = H @ self.output_weights  # (B*T*N, output)

        Y = Y.view(B, T, N, self.output_size)
        Y = Y[:, -1, :, 0]  # (B, N) → last time step only
        return Y
