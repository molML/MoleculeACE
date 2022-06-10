"""
Author: Derek van Tilborg -- TU/e -- 23-05-2022

Attentive Fingerprint [1] graph neural network for regression.

[1] Xiong et al. (2020). Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention
Mechanism

"""

import torch
from torch_geometric.nn.models import AttentiveFP
from MoleculeACE.models.utils import GNN


class AFP(GNN):
    def __init__(self, in_channels: int = 37, edge_dim: int = 6, hidden_channels: int = 128, num_layers: int = 4,
                 num_timesteps: int = 3, dropout: int = 0.2, lr: float = 0.0005, epochs: int = 300, *args, **kwargs):
        super().__init__()

        self.model = AttentiveFP(in_channels=in_channels, edge_dim=edge_dim, hidden_channels=hidden_channels,
                                 out_channels=1, num_layers=num_layers, num_timesteps=num_timesteps,
                                 dropout=dropout)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.name = 'AFP'

        # Move the whole model to the gpu
        self.model = self.model.to(self.device)

    def __repr__(self):
        return f"{self.model}"
