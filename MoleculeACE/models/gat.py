"""
Author: Derek van Tilborg -- TU/e -- 23-05-2022

Basic graph attention network [1] using a transformer as global pooling [2].

1. Veličković et al. (2018). Graph Attention Networks
2. Baek et al. (2021). Accurate Learning of Graph Representations with Graph Multiset Pooling

"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GraphMultisetTransformer, GATConv, GATv2Conv
from MoleculeACE.benchmark.const import RANDOM_SEED
from MoleculeACE.models.utils import GNN


class GAT(GNN):
    def __init__(self, node_feat_in: int = 37, n_fc_layers: int = 1, node_hidden: int = 128,
                 fc_hidden: int = 128, n_gat_layers: int = 3, transformer_hidden: int = 128,
                 n_gat_attention_heads: int = 8, dropout: float = 0.2, seed: int = RANDOM_SEED, gatv2: bool = False,
                 lr: float = 0.0005, epochs: int = 300, *args, **kwargs):
        super().__init__()

        self.model = GATmodel(node_feat_in=node_feat_in, n_fc_layers=n_fc_layers, node_hidden=node_hidden,
                              fc_hidden= fc_hidden, n_gat_layers=n_gat_layers, transformer_hidden=transformer_hidden,
                              dropout=dropout, seed=seed, gatv2=gatv2, n_gat_attention_heads=n_gat_attention_heads)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.name = 'GAT'

        # Move the whole model to the gpu
        self.model = self.model.to(self.device)

    def __repr__(self):
        return f"{self.model}"


class GATmodel(torch.nn.Module):
    def __init__(self, node_feat_in: int = 37, n_fc_layers: int = 1, node_hidden: int = 128,
                 n_gat_attention_heads: int = 8, fc_hidden: int = 128, n_gat_layers: int = 3,
                 transformer_hidden: int = 128, dropout: float = 0.2, seed: int = RANDOM_SEED, gatv2: bool = False,
                 *args, **kwargs):

        # Init parent
        super().__init__()
        torch.manual_seed(seed)

        # GAT layer(s)
        Conv = GATConv if not gatv2 else GATv2Conv
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(Conv(node_feat_in, node_hidden, heads=n_gat_attention_heads, concat=False,
                                     dropout=dropout))
        for k in range(n_gat_layers-1):
            self.conv_layers.append(Conv(node_hidden, node_hidden, heads=n_gat_attention_heads, concat=False,
                                         dropout=dropout))

        # Global pooling
        self.transformer = GraphMultisetTransformer(in_channels=node_hidden,
                                                    hidden_channels=transformer_hidden,
                                                    out_channels=fc_hidden,
                                                    num_heads=8)

        # fully connected layer(s)
        self.fc = torch.nn.ModuleList()
        for k in range(n_fc_layers):
            self.fc.append(Linear(fc_hidden, fc_hidden))

        self.dropout = Dropout(dropout)

        # Output layer
        self.out = Linear(fc_hidden, 1)

    def forward(self, x, edge_index, edge_attr, batch):

        # Conv layers
        h = F.relu(self.conv_layers[0](x.float(), edge_index))
        for k in range(len(self.conv_layers) - 1):
            h = F.relu(self.conv_layers[k+1](h, edge_index))

        # Global graph pooling with a transformer
        h = self.transformer(h, batch, edge_index)

        # Apply a fully connected layer.
        for k in range(len(self.fc)):
            h = F.relu(self.fc[k](h))
            h = self.dropout(h)

        # Apply a final (linear) classifier.
        out = self.out(h)

        return out


#
#
# import pandas as pd
# from MoleculeACE.benchmark.featurization import Featurizer
# from MoleculeACE.models.gcn import GCN
# from MoleculeACE.models.attentivefp import AFP
# from MoleculeACE.models.mpnn import MPNN
#
# df = pd.read_csv(f"MoleculeACE/Data/benchmark_data/CHEMBL233_Ki.csv")
# smiles = df['smiles'].tolist()
# y = df['y'].tolist()
# #
# feat = Featurizer()
# x = feat.tokens(smiles)
#
# # def __init__(self, node_feat_in: int = 37, n_fc_layers: int = 1, node_hidden: int = 128,
# #              fc_hidden: int = 128, n_gat_layers: int = 3, transformer_hidden: int = 128,
# #              n_gat_attention_heads: int = 8, dropout: float = 0.2, seed: int = RANDOM_SEED, gatv2: bool = False,
# #              lr: float = 0.005):
#
# f2 = GAT(n_gat_layers=2, n_fc_layers=1, n_gat_attention_heads=4, transformer_hidden=64, node_hidden=64, lr=0.0005)
# f2 = AFP(hidden_channels=64, num_layers=2, num_timesteps=2, dropout=0.1, lr=0.0005)
# f2 = MPNN(lr=0.0005)
#
# for i in range(40):
#     f2.train(x, y, x_val=x, y_val=y, epochs=2)
#     f2.scatter(x, y)
# f2.plot_losses()
# y_hat = f2.predict(x)
# y_true = y
#
# torch.sqrt(torch.mean(torch.square(y_hat - torch.unsqueeze(torch.tensor(y_true), -1))))
#
#
