"""
Author: Derek van Tilborg -- TU/e -- 23-05-2022

Message Passing Neural Network for molecular property prediction using continuous kernel-based convolution
(edge-conditioned convolution) [1] and global graph pooling using a graph multiset transformer [2] instead of
the Set2Set method used in [1]. I added dropout to make a more robust

[1] Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry
[2] Baek et al. (2021). Accurate Learning of Graph Representations with Graph Multiset Pooling

"""

import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential, Dropout
from torch_geometric.nn import NNConv, GraphMultisetTransformer
from MoleculeACE.benchmark.const import RANDOM_SEED
from MoleculeACE.models.utils import GNN


class MPNN(GNN):
    def __init__(self, node_in_feats: int = 37, node_hidden: int = 64, edge_in_feats: int = 6,
                 edge_hidden: int = 128, message_steps: int = 3, dropout: float = 0.2,
                 transformer_heads: int = 8, transformer_hidden: int = 128, seed: int = RANDOM_SEED,
                 fc_hidden: int = 64, n_fc_layers: int = 1, lr: float = 0.0005, epochs: int = 300, *args, **kwargs):
        super().__init__()

        self.model = MPNNmodel(node_in_feats=node_in_feats, node_hidden=node_hidden, edge_in_feats=edge_in_feats,
                               edge_hidden=edge_hidden, message_steps=message_steps, dropout=dropout,
                               transformer_heads=transformer_heads, transformer_hidden=transformer_hidden, seed=seed,
                               fc_hidden=fc_hidden, n_fc_layers=n_fc_layers)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.name = 'MPNN'

        # Move the whole model to the gpu
        self.model = self.model.to(self.device)

    def __repr__(self):
        return f"{self.model}"


class MPNNmodel(torch.nn.Module):
    def __init__(self, node_in_feats: int = 37, node_hidden: int = 64, edge_in_feats: int = 6,
                 edge_hidden: int = 128, message_steps: int = 3, dropout: float = 0.2,
                 transformer_heads: int = 8, transformer_hidden: int = 128, seed: int = RANDOM_SEED,
                 fc_hidden: int = 64, n_fc_layers: int = 1, *args, **kwargs):

        # Init parent
        super().__init__()
        torch.manual_seed(seed)

        self.seed = seed
        self.node_hidden = node_hidden
        self.messsage_steps = message_steps
        self.node_in_feats = node_in_feats

        # Layer to project node features to hidden features
        self.project_node_feats = Sequential(Linear(node_in_feats, node_hidden), ReLU())

        # The 'learnable message function'
        edge_network = Sequential(Linear(edge_in_feats, edge_hidden), ReLU(),
                                  Linear(edge_hidden, node_hidden * node_hidden))

        # edge-conditioned convolution layer
        self.gnn_layer = NNConv(in_channels=node_hidden,
                                out_channels=node_hidden,
                                nn=edge_network,
                                aggr='add')

        # The GRU as used in [1]
        self.gru = GRU(node_hidden, node_hidden)

        # Global pooling using a transformer
        self.transformer = GraphMultisetTransformer(in_channels=node_hidden,
                                                    hidden_channels=transformer_hidden,
                                                    out_channels=fc_hidden,
                                                    num_heads=transformer_heads)

        # fully connected layer(s)
        self.fc = torch.nn.ModuleList()
        for k in range(n_fc_layers):
            self.fc.append(Linear(fc_hidden, fc_hidden))

        self.dropout = Dropout(dropout)
        self.lin2 = torch.nn.Linear(fc_hidden, 1)

    def forward(self, x, edge_index, edge_attr, batch):

        # Project node features to node hidden dimensions, which are also used as h_0 for the GRU
        node_feats = self.project_node_feats(x)
        hidden_feats = node_feats.unsqueeze(0)

        # Perform n message passing steps using edge-conditioned convolution as pass it through a GRU
        for _ in range(self.messsage_steps):
            node_feats = F.relu(self.gnn_layer(node_feats, edge_index, edge_attr))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)

        # perform global pooling using a multiset transformer to get graph-wise hidden embeddings
        out = self.transformer(node_feats, batch, edge_index)

        # Apply a fully connected layer.
        for k in range(len(self.fc)):
            out = F.relu(self.fc[k](out))
            out = self.dropout(out)

        out = self.lin2(out)

        return out
