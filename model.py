from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    sequence_length: int = 28
    input_features: int = 28
    feature_dim: int = 128
    state_dim: int = 128
    channels: int = 1
    dt_min: float = 0.001
    dt_max: float = 0.1
    num_blocks: int = 6
    dropout_p: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    embedding: bool = False
    vocab_size: int = 50
    output_size: int = 10


class LinearStateSpaceLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        A, B = self.hippo(config.state_dim)    # HiPPO matrices help with long term memory
        self.A = nn.Parameter(A.to(config.device), requires_grad=False)    # A is fixed (LSSL-f) for reducing trainable params
        self.B = nn.Parameter(B.to(config.device))
        self.C = nn.Parameter(
            torch.Tensor(config.channels, config.state_dim).to(config.device)
        )
        torch.nn.init.xavier_normal_(self.C)
        self.D = nn.Parameter(
            torch.ones(config.channels, 1).to(config.device)
        )    # Using ones for initialization following Annotated S4
        # Delta t is learned in log space
        log_dt_min = torch.log(torch.tensor(config.dt_min))
        log_dt_max = torch.log(torch.tensor(config.dt_max))
        self.log_dt = nn.Parameter(
            (log_dt_min + torch.rand(1) * (log_dt_max - log_dt_min)).to(config.device), requires_grad=False
        )    # Fixed delta t (LSSL-f) for reduced trainable params
        # Precomputing powers of A bar for faster learning
        self.dt = torch.exp(self.log_dt)
        I = torch.eye(self.A.shape[0]).to(config.device)
        self.BL = torch.linalg.inv(I - (self.dt / 2.0) * self.A)
        self.A_bar = self.BL @ (I + (self.dt / 2.0) * self.A)
        self.A_bar_pow = torch.stack(
            [torch.matrix_power(self.A_bar, config.sequence_length - 1 - i) for i in range(config.sequence_length)],
            dim=2
        )

    def forward(self, u):
        b, l = u.shape
        B_bar = (self.BL * self.dt) @ self.B
        if self.training:
            # Use convolutional view for faster training
            k_conv = torch.tensor([self.C @ self.A_bar_pow[:, :, i] @ B_bar for i in range(l)], device=self.config.device).reshape(self.config.channels, 1, l)
            return F.conv1d(u[:, None, :], k_conv, padding=l - 1)[:, :, : l] + (u.unsqueeze(-1) @ self.D.T).transpose(-1, -2)
        else:
            # Use recurrent view for faster inference
            x = torch.zeros((b, self.config.state_dim)).to(self.config.device)
            ys = []
            for i in range(l):
                x = x @ self.A_bar.T + u[:, i].unsqueeze(-1) @ B_bar.T
                y = x @ self.C.T + u[:, i].unsqueeze(-1) @ self.D.T
                ys.append(y)
            return torch.cat(ys, dim=1)

    @staticmethod
    def hippo(state_dim):
        P = torch.sqrt(1 + 2 * torch.arange(state_dim))
        A = P[:, None] @ P[None, :]
        A = torch.tril(A) - torch.diag(torch.arange(state_dim))
        B = torch.sqrt(2 * torch.arange(state_dim) + 1.0)
        return -A, B[:, None]


class StackedLSSL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = [LinearStateSpaceLayer(config) for _ in range(config.feature_dim)]
        self.mix = nn.Linear(config.feature_dim * config.channels, config.feature_dim, device=config.device)

    def forward(self, u):
        b, l, h = u.shape
        stacked = torch.stack([layer(u[:, :, i]) for (i, layer) in enumerate(self.layers)], dim=2)
        stacked = torch.reshape(stacked, (b, l, self.config.feature_dim * self.config.channels))
        mixed = self.mix(stacked)
        return mixed


class SSMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.slssl = StackedLSSL(config)
        self.norm = nn.LayerNorm(config.feature_dim, device=config.device)
        self.dropout = nn.Dropout(config.dropout_p)

    def forward(self, u):
        skip = u
        u = self.norm(u)
        u = self.slssl(u)
        u = self.dropout(F.silu(u)) + skip
        return u


class SSMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = [SSMBlock(config) for _ in range(config.num_blocks)]
        self.embedding = nn.Embedding(config.vocab_size, config.feature_dim, device=config.device)
        self.expand = nn.Linear(config.input_features, config.feature_dim, device=config.device)
        self.output = nn.Linear(config.feature_dim, config.output_size, bias=False, device=config.device)

    def forward(self, u):
        if self.config.embedding:
            u = self.embedding(u)
        else:
            u = self.expand(u)
        for i in range(len(self.blocks)):
            u = self.blocks[i](u)
        return self.output(u)
