"""
Author: Derek van Tilborg -- TU/e -- 23-05-2022

Basic Multi-Layer Perceptron

"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from MoleculeACE.models.utils import NN
import os


class MLP(NN):
    def __init__(self, feat_in: int = 1024, n_layers: int = 3, hidden: int = 512, dropout: float = 0,
                 lr: float = 0.00005, save_path: str = '.', epochs: int = 500, *args, **kwargs):
        super().__init__()

        self.model = MLPmodel(feat_in=feat_in, n_layers=n_layers, hidden=hidden, dropout=dropout)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.save_path = os.path.join(save_path, 'best_model.pkl')
        self.epochs = epochs
        self.name = 'MLP'

        # Move the whole model to the gpu
        self.model = self.model.to(self.device)


class MLPmodel(torch.nn.Module):
    def __init__(self, feat_in: int = 1024, n_layers: int = 3, hidden: int = 128, dropout: float = 0.2, *args,
                 **kwargs):
        super().__init__()

        # fully connected layer(s)
        self.fc = torch.nn.ModuleList()
        self.fc.append(Linear(feat_in, hidden))

        for k in range(n_layers-1):
            self.fc.append(Linear(hidden, hidden))

        self.dropout = Dropout(dropout)

        # Output layer
        self.out = Linear(hidden, 1)

    def forward(self, x):

        h = F.relu(self.fc[0](x))
        h = self.dropout(h)
        # Apply a fully connected layer.
        for k in range(len(self.fc)-1):
            h = F.relu(self.fc[k+1](h))
            h = self.dropout(h)

        # Apply a final (linear) classifier.
        out = self.out(h)

        return out
