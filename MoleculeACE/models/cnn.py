"""
Author: Derek van Tilborg -- TU/e -- 24-05-2022

Basic 1-D Convolutional Neural Network based on the architecture of [1]

[1] Kimber et al. (2021). Maxsmi: Maximizing molecular property prediction performance with confidence estimation using
    SMILES augmentation and deep learning

"""

import torch
import torch.nn.functional as F
from MoleculeACE.models.utils import NN


class CNN(NN):
    def __init__(self, nchar_in: int = 41, seq_len_in: int = 202, kernel_size: int = 10, hidden: int = 128,
                 lr: float = 0.0005, epochs: int = 300, *args, **kwargs):
        super().__init__()

        self.model = CNNmodel(nchar_in=nchar_in, seq_len_in=seq_len_in, kernel_size=kernel_size, hidden=hidden)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.name = 'CNN'

        # Move the whole model to the gpu
        self.model = self.model.to(self.device)

    def __repr__(self):
        return f"1-D Convolutional Neural Network"


class CNNmodel(torch.nn.Module):
    def __init__(self, nchar_in: int = 41, seq_len_in: int = 202, kernel_size: int = 10, hidden: int = 128, *args,
                 **kwargs):
        """

        :param nchar_in: (int) number of unique characters in the SMILES sequence (default = 41)
        :param seq_len_in: (int) length of the SMILES sequence
        :param kernel_size: (int) convolution kernel size
        :param hidden: (int) number of neurons in the hidden layer
        """
        super().__init__()

        self.conv0 = torch.nn.Conv1d(in_channels=seq_len_in, out_channels=nchar_in, kernel_size=kernel_size)

        conv_out_size = (nchar_in-kernel_size+1)*nchar_in
        self.fc0 = torch.nn.Linear(conv_out_size, hidden)

        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, x):

        h = F.relu(self.conv0(x))
        h = torch.flatten(h, 1)
        h = F.relu(self.fc0(h))
        out = self.out(h)

        return out
