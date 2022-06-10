"""
Author: Derek van Tilborg -- TU/e -- 19-05-2022

Basic graph convolutional network [1] using a transformer as global pooling [2].

1. Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks
2. Baek et al. (2021). Accurate Learning of Graph Representations with Graph Multiset Pooling

"""

from MoleculeACE.benchmark.const import RANDOM_SEED
from MoleculeACE.models.utils import GNN
from torch_geometric.nn import GCNConv, GraphMultisetTransformer
from torch.nn import Linear, Dropout
import torch.nn.functional as F
import torch


class GCN(GNN):
    def __init__(self, node_feat_in: int = 37, node_hidden: int = 128, transformer_hidden: int = 128,
                 n_conv_layers: int = 4, n_fc_layers: int = 4, fc_hidden: int = 128, lr: float = 0.0005,
                 epochs: int = 300, dropout: float = 0.2, *args, **kwargs):
        super().__init__()

        self.model = GCNmodel(node_feat_in=node_feat_in, node_hidden=node_hidden, n_conv_layers=n_conv_layers,
                              transformer_hidden=transformer_hidden, fc_hidden=fc_hidden, n_fc_layers=n_fc_layers,
                              dropout=dropout)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.name = 'GCN'

        # Move the whole model to the gpu
        self.model = self.model.to(self.device)

    def __repr__(self):
        return f"{self.model}"


class GCNmodel(torch.nn.Module):
    def __init__(self, node_feat_in: int = 37, n_conv_layers: int = 4, n_fc_layers: int = 1, node_hidden: int = 128,
                 fc_hidden: int = 128, transformer_hidden: int = 128, dropout: float = 0.2, seed: int = RANDOM_SEED,
                 *args, **kwargs):

        # Init parent
        super().__init__()
        torch.manual_seed(seed)

        # GCN layer(s)
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GCNConv(node_feat_in, node_hidden))
        for k in range(n_conv_layers-1):
            self.conv_layers.append(GCNConv(node_hidden, node_hidden))

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






# import pandas as pd
# from MoleculeACE.benchmark.featurization import Featurizer
#
# df = pd.read_csv(f"MoleculeACE/Data/benchmark_data/CHEMBL233_Ki.csv")
# smiles = df['smiles'].tolist()
# y = df['y'].tolist()
#
# feat = Featurizer()
# x = feat.graphs(smiles)
#
# f = GCN(n_conv_layers=3, n_fc_layers=2)
# f.train(x, y, x_val=x, y_val=y, epochs=2)
#
#
# afp = AFP()
# afp.train(x, y, x_val=x, y_val=y, epochs=20)
#
#
# print(f"{f.model}")
#
# afp.predict(x)
# afp.plot_losses()
#
# train_losses_float = [float(loss.detach().numpy()) for loss in f.train_losses]
# val_losses_float = f.val_losses
#
# loss_indices = [i for i, l in enumerate(train_losses_float)]
#
# plt.figure()
# plt.plot(loss_indices, train_losses_float, val_losses_float)
# plt.show()
#
# from torch_geometric.nn.models import AttentiveFP
# afp = AttentiveFP(in_channels=37, out_channels=1, hidden_channels=32, edge_dim=6, num_layers=2, num_timesteps=2)
# afp(x[0].x.float(), x[0].edge_index, x[0].edge_attr, torch.tensor([0]))
#
# gcn = GCN_model()
#
#
#
# gcn(batch.x.float(), batch.edge_index, batch.edge_attr, batch.batch)
#
# DataLoader(x)
#
# x[0].edge_attr.shape
#
# torch.tensor([[0]]).shape

