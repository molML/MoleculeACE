
from torch import nn
from transformers import  AutoModel
import matplotlib.pyplot as plt
import torch


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
    def __init__(self, lr: int = 0.0005, batch_size: int = 32, freeze_core: bool = True):
        """

        :param lr: (float) learning rate
        :param batch_size: (int) batch_size
        :param freeze_core: (bool) Freeze the core of the transformer (aka only train the regression head)
        """

        self.lr = lr
        self.batch_size = batch_size
        self.epoch = 0

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
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

        self.train_losses, self.val_losses = [], []

        self.model = self.model.to(self.device)

        self._total_params = sum(p.numel() for p in self.model.parameters())

    def train(self, train_loader, val_loader=None, epochs: int = 100):
        """ Train a model for n epochs

        :param train_loader: Torch geometric data loader with training data
        :param val_loader: Torch geometric data loader with validation data (optional)
        :param epochs: (int) number of epochs to train
        """

        for epoch in range(epochs):
            self.model.train(True)
            loss = self._one_epoch(train_loader)
            self.train_losses.append(loss)

            val_loss = 0
            if val_loader is not None:
                val_pred, val_y = self.test(val_loader)
                val_loss = self.loss_fn(val_pred, val_y)
            self.val_losses.append(val_loss)

            self.epoch += 1

            if self.epoch % 2 == 0:
                print(f"Epoch {self.epoch} | Train Loss {loss} | Val Loss {val_loss}")
                self.scatter(train_loader)

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
            loss = self.loss_fn(y_hat, y)

            if not loss > 0:
                print(idx)

            # Autodiff
            loss.backward()

            # Update using the gradients
            self.optimizer.step()

        return loss

    def test(self, loader):
        """ Perform testing

        :param data_loader:  Torch geometric data loader with test data
        :return: A tuple of two 1D-tensors (predicted, true)
        """
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
                y_true.extend([i for i in y.tolist()])
                y_pred.extend([i for i in y_hat.tolist()])

        return torch.tensor(y_pred), torch.tensor(y_true)

    def predict(self, loader):
        """ Perform prediction

        :param data_loader:  Torch geometric data loader with data
        :return: A 1D-tensor
        """
        y_pred = []
        with torch.no_grad():
            for batch in loader:
                # Move batch to gpu
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)

                # Predict
                y_hat = self.model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
                y_pred.extend([i for i in y_hat.tolist()])

        return torch.tensor(y_pred)

    def __repr__(self):
        return f"ChemBerta Transformer with a regression head. Total params: {self._total_params}"

    def plot_loss(self):
        train_losses_float = [float(loss.cpu().detach().numpy()) for loss in self.train_losses]
        val_losses_float = [float(loss.cpu().detach().numpy()) for loss in self.val_losses]
        loss_indices = [i for i, l in enumerate(train_losses_float)]

        plt.figure()
        plt.plot(loss_indices, train_losses_float, val_losses_float)
        plt.show()

    def scatter(self, loader, min_axis_val: float = -5, max_axis_val: float = 1):

        y_hat, y = self.test(loader)

        plt.figure()
        plt.scatter(x=y_hat, y=y, alpha=0.5)
        plt.axline((0, 0), slope=1, color="black", linestyle=(1, (5, 5)))
        plt.xlim(min_axis_val, max_axis_val)
        plt.ylim(min_axis_val, max_axis_val)
        plt.xlabel("predicted")
        plt.ylabel("true")
        plt.show()

