"""
Author: Derek van Tilborg -- TU/e -- 23-05-2022

Functions to run a ChemBerta transformer for bioactivity prediction. We used pre-trained weights of ChemBerta trained
on 10 million SMILES strings from PubChem.

    - RobertaRegressionHead:    regression head that is slapped on the frozen pre-trained model
    - Transformer:              class of Transformer model
        - train()
        - test()
        - predict()
    - ChemBertaDataset:         dataset class to work with the Transformer model class
    - chemberta_loader():       function that returns a dataloader

"""

from MoleculeACE.benchmark.const import CONFIG_PATH_TRANS, CONFIG_PATH_GENERAL
from MoleculeACE.benchmark.utils import get_config
from transformers import AutoModel
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torch import nn
import pickle
import os
import torch

trans_settings = get_config(CONFIG_PATH_TRANS)


class RobertaRegressionHead(nn.Module):
    """Head for singletask regression models."""

    def __init__(self, config):
        super(RobertaRegressionHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Transformer:
    def __init__(self, lr: int = 0.0005, batch_size: int = 32, freeze_core: bool = True, save_path='.',
                 epochs: int = 100, *args, **kwargs):
        """

        :param lr: (float) learning rate
        :param batch_size: (int) batch_size
        :param freeze_core: (bool) Freeze the core of the transformer (aka only train the regression head)
        :param save_path: (str) path where intermediate models are stored
        :param epochs: (int) train for n epochs (is the default for train() if no epoch argument is given there)
        """

        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.epoch = 0
        self.save_path = os.path.join(save_path, 'best_model.pkl')
        self.name = 'Transformer'

        # Load the model and init the regression head
        self.model = AutoModel.from_pretrained('seyonec/PubChem10M_SMILES_BPE_450k')

        if freeze_core:
            # Freeze all ChemBerta params
            for param in self.model.parameters():
                param.requires_grad = False

        # Get rid of the pooler head of Chemberta and replace it by a regression head
        self.model._modules['pooler'] = RobertaRegressionHead(self.model.config)

        # create the infrastructure needed to train the model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

        self.train_losses, self.val_losses = [], []
        self._total_params = sum(p.numel() for p in self.model.parameters())

        # Move model to the gpu (if applicable)
        self.model = self.model.to(self.device)

    def train(self, x_train: Dict[Tensor, Tensor], y_train: List[float], x_val: Dict[Tensor, Tensor] = None,
              y_val: List[float] = None, early_stopping_patience: int = None, epochs: int = None,
              print_every_n: int = 100, *args, **kwargs):
        """

        :param x_train: (Dict[Tensor, Tensor]) dict from ChemBerta tokenizer with train data
        :param y_train: (List[float]): train bioactivity
        :param x_val: (Dict[Tensor, Tensor]) dict from ChemBerta tokenizer with val data
        :param y_val: (List[float]): validation bioactivity
        :param early_stopping_patience: (int) stop training if the model doesn't improve after n epochs
        :param epochs: (int) epochs to train
        :param print_every_n: (int) printout training progress every nth epoch
        """

        if epochs is None:
            epochs = self.epochs
        patience = None if early_stopping_patience is None else 0
        train_loader = chemberta_loader(x_train, y_train, batch_size=self.batch_size)

        for epoch in range(epochs):

            # If we reached the end of our patience, load the best model and stop training
            if patience is not None and patience >= early_stopping_patience:

                print('Stopping training early')
                try:
                    with open(self.save_path, 'rb') as handle:
                        self.model = pickle.load(handle)

                    os.remove(self.save_path)
                except Warning:
                    print('Could not load best model, keeping the current weights instead')

                break

            # As long as the model is still improving, continue training
            else:
                self.model.train(True)
                loss = self._one_epoch(train_loader)
                self.train_losses.append(loss)

                val_loss = 0
                if x_val is not None:
                    # val_pred, val_y = self.test(x_val)
                    val_pred = self.predict(x_val)
                    val_loss = self.loss_fn(val_pred, torch.tensor(y_val))
                self.val_losses.append(val_loss)

                self.epoch += 1

                # Pickle model if its the best
                if val_loss <= min(self.val_losses):
                    with open(self.save_path, 'wb') as handle:
                        pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    patience = 0
                else:
                    patience += 1

                # Printout training info
                if self.epoch % print_every_n == 0:
                    print(f"Epoch {self.epoch} | Train Loss {loss} | Val Loss {val_loss} | Patience {patience}")

    def _one_epoch(self, train_loader):

        # Enumerate over the data
        for idx, batch in enumerate(train_loader):

            # Move batch to gpu
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            y = batch[2].to(self.device)

            # Transform the batch of graphs with the model
            self.optimizer.zero_grad()
            self.model.train(True)
            y_hat = self.model(input_ids=input_ids, attention_mask=attention_mask).pooler_output

            # Calculating the loss
            loss = self.loss_fn(y_hat.squeeze(), y.squeeze())

            if not loss > 0:
                print(idx)

            # Autodiff
            loss.backward()

            # Update using the gradients
            self.optimizer.step()

        return loss

    def test(self, x_test: Dict[Tensor, Tensor], y_test: List[float]):
        """ Perform testing

        :param data_loader:  Torch geometric data loader with test data
        :return: A tuple of two 1D-tensors (predicted, true)
        """
        loader = chemberta_loader(x_test, y_test, batch_size=self.batch_size)

        y_pred, y_true = [], []
        with torch.no_grad():
            for batch in loader:
                # Move batch to gpu
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                y = batch[2].to(self.device)

                # Predict
                y_hat = self.model(input_ids=input_ids, attention_mask=attention_mask).pooler_output

                # Append predictions and true values to list
                y_true.extend([i for i in y.squeeze().tolist()])
                y_pred.extend([i for i in y_hat.squeeze().tolist()])

        return torch.tensor(y_pred), torch.tensor(y_true)

    def predict(self, x_test: Dict[Tensor, Tensor]):
        """ Perform prediction

        :param x_test:  dict from the ChemBerta Tokenizer
        :return: A 1D-tensor with predicted bioactivities
        """

        loader = chemberta_loader(x_test, batch_size=self.batch_size)

        y_pred = []
        with torch.no_grad():
            for batch in loader:
                # Move batch to gpu
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)

                # Predict
                y_hat = self.model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
                y_hat = y_hat.squeeze().tolist()
                if type(y_hat) is list:
                    y_pred.extend(y_hat)
                else:
                    y_pred.append(y_hat)

        return torch.tensor(y_pred)

    def __repr__(self):
        return f"ChemBerta Transformer with a regression head. Total params: {self._total_params}"


class ChemBertaDataset(Dataset):
    def __init__(self, tokens, labels: List[float] = None):
        """ Create a dataset for the ChemBerta transformer using a pretrained tokenizer """

        if labels is None:
            labels = [0]*len(tokens['input_ids'])
        self.labels = torch.tensor(labels).unsqueeze(1)
        # self.chemical_tokenizer = AutoTokenizer.from_pretrained('seyonec/PubChem10M_SMILES_BPE_450k')
        self.tokens = tokens

    def __getitem__(self, idx):
        return self.tokens['input_ids'][idx], self.tokens['attention_mask'][idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


def chemberta_loader(x: Dict[Tensor, Tensor], y: List[float] = None, batch_size: int = 32):
    dataset = ChemBertaDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size)
